"""Edit route: counterfactual reasoning plus instruction-based image editing."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from .api_client import MLLMClient, extract_json_object
from .edit_client import EditClient, EditConfig
from .reason_agent import build_reason_context, finalize_edit_prompt, run_reason_agent
from .schemas import EditIteration, EditPipelineResult, ReasonResult, VQACheck, ensure_output_dir


DEFAULT_REFINE_PROMPT = Path("prompts/edit_refine.txt")
DEFAULT_CANDIDATE_PROMPT = Path("prompts/edit_candidate.txt")


def run_edit_pipeline(
    image_path: str | Path,
    instruction: str,
    output_dir: str | Path,
    iterations: int = 2,
    min_iterations: int = 2,
    candidates: int = 2,
    score_threshold: float = 0.85,
    dry_run: bool | None = None,
    seed: int | None = None,
) -> EditPipelineResult:
    """Run edit with VQA-driven verify loop and per-round candidate selection.

    Each iteration generates multiple edit_prompt candidates, edits the source
    image with each, VQA-scores them, and keeps the best result as image_iter_N.
    When min_iterations=2 (default), the pipeline always performs at least one
    refine-and-reedit cycle even if the first VQA score already passes.
    """

    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"Input image not found: {source}")

    out_dir = ensure_output_dir(output_dir)
    client = MLLMClient()
    use_dry_run = (not client.configured) if dry_run is None else dry_run
    editor = EditClient(EditConfig(backend="dry_run") if use_dry_run else None)
    after_suffix = ".svg" if editor.config.backend == "dry_run" else ".png"

    before_path = out_dir / f"image_before{source.suffix or '.png'}"
    shutil.copy2(source, before_path)

    reason = run_reason_agent(
        image_path=before_path,
        instruction=instruction,
        mode="edit",
        dry_run=use_dry_run,
        client=client,
    )
    checklist = reason.vqa_checklist
    reason_context = build_reason_context(reason)
    current_prompt = finalize_edit_prompt(reason) or instruction
    latest_vqa: dict[str, object] | None = None
    history: list[EditIteration] = []

    write_reason_analysis(reason, reason_context, out_dir)

    total_steps = max(iterations, 1)
    min_steps = min(max(min_iterations, 1), total_steps)
    num_candidates = max(candidates, 1)
    for step in range(1, total_steps + 1):
        reference_image = before_path if not history else Path(history[-1].image_path or before_path)
        candidate_prompts = generate_candidate_edit_prompts(
            client=client,
            instruction=instruction,
            current_prompt=current_prompt,
            checklist=checklist,
            image_path=reference_image,
            history=history,
            latest_vqa=latest_vqa,
            candidates=num_candidates,
            use_dry_run=use_dry_run,
            force_refinement=step > 1 and (step - 1) < min_steps,
            reason_context=reason_context,
        )
        best = select_best_edit_candidate(
            editor=editor,
            client=client,
            before_path=before_path,
            candidate_prompts=candidate_prompts,
            instruction=instruction,
            checklist=checklist,
            output_dir=out_dir,
            step=step,
            after_suffix=after_suffix,
            seed=seed,
            use_dry_run=use_dry_run,
            reason_context=reason_context,
        )

        current_prompt = str(best["prompt"])
        latest_vqa = best.get("analysis") if isinstance(best.get("analysis"), dict) else latest_vqa
        score = best.get("score")
        if not isinstance(score, (int, float)):
            score = None
        else:
            score = float(score)

        history.append(
            EditIteration(
                iteration=step,
                edit_prompt=current_prompt,
                analysis=format_edit_iteration_analysis(step, best),
                score=score,
                image_path=str(best["image_path"]),
            )
        )
        (out_dir / f"edit_prompt_iter_{step}.txt").write_text(current_prompt + "\n", encoding="utf-8")

        if should_stop_verify_loop(score, score_threshold, step, total_steps, min_steps):
            break

    best = select_best_iteration(history)
    final_path = out_dir / f"image_after{after_suffix}"
    shutil.copy2(best.image_path or before_path, final_path)

    result = EditPipelineResult(
        final_image=str(final_path),
        final_prompt=best.edit_prompt,
        route="edit",
        reasoning_chain=reason.reasoning_chain,
        image_before=str(before_path),
        instruction=instruction,
        edit_prompt=best.edit_prompt,
        target_objects=reason.target_objects,
        vqa_checklist=checklist,
        vqa_result=extract_vqa_from_analysis(best.analysis),
        iterations=history,
        metadata={
            "dry_run": use_dry_run,
            "edit_backend": editor.config.backend,
            "edit_model": editor.config.model,
            "reasoning_type": reason.reasoning_type,
            "mode": "edit",
            "num_iterations": total_steps,
            "min_iterations": min_steps,
            "num_candidates": num_candidates,
            "score_threshold": score_threshold,
        },
    )
    write_result_files(result, out_dir)
    return result


def should_stop_verify_loop(
    score: float | None,
    score_threshold: float,
    step: int,
    total_steps: int,
    min_iterations: int,
) -> bool:
    if step >= total_steps:
        return True
    if step < min_iterations:
        return False
    if score is None:
        return False
    return score >= score_threshold


def select_best_iteration(history: list[EditIteration]) -> EditIteration:
    scored = [item for item in history if item.score is not None]
    if scored:
        return max(scored, key=lambda item: float(item.score))
    return history[-1]


def extract_score(vqa_result: dict[str, object] | None) -> float | None:
    if not vqa_result:
        return None
    score = vqa_result.get("score")
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            return None
    return float(score) if isinstance(score, (int, float)) else None


def format_vqa_analysis(vqa_result: dict[str, object] | None, step: int) -> str:
    if vqa_result is None:
        return f"Iteration {step}: VQA skipped."
    return f"Iteration {step} VQA. {json.dumps(vqa_result, ensure_ascii=False)}"


def format_edit_iteration_analysis(step: int, best: dict[str, object]) -> str:
    payload = {
        "prompt": best.get("prompt"),
        "image_path": best.get("image_path"),
        "score": best.get("score"),
        "analysis": best.get("analysis"),
        "candidates": best.get("candidates"),
    }
    return f"Edit candidate selection for iteration {step}. {json.dumps(payload, ensure_ascii=False)}"


def extract_vqa_from_analysis(analysis: str) -> dict[str, object] | None:
    if "Edit candidate selection" in analysis:
        marker = ". "
        if marker not in analysis:
            return None
        try:
            payload = json.loads(analysis.split(marker, 1)[1])
            inner = payload.get("analysis")
            if isinstance(inner, dict):
                return inner
            if isinstance(payload.get("score"), (int, float)):
                return {"score": payload["score"]}
        except json.JSONDecodeError:
            return None

    if " VQA. " not in analysis:
        return None
    payload = analysis.split(" VQA. ", 1)[1]
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def verify_edit_result(
    client: MLLMClient,
    image_path: str | Path,
    instruction: str,
    edit_prompt: str,
    checklist: list[VQACheck],
    reason_context: str = "",
) -> dict[str, object]:
    context_block = f"\nReasoning context:\n{reason_context}\n" if reason_context else ""
    user_prompt = f"""Original hypothetical instruction:
{instruction}
{context_block}
Applied edit prompt:
{edit_prompt}

Checklist:
{json.dumps([item.to_dict() for item in checklist], ensure_ascii=False, indent=2)}

Inspect the edited image and return JSON exactly in this shape:
{{
  "score": 0.0,
  "passed": ["checklist items that are satisfied"],
  "errors": ["missing or incorrect visual constraints"],
  "summary": "one short sentence"
}}
Use score from 0 to 1. Be strict: score 1.0 only if every checklist item is fully and unambiguously satisfied.
Penalize physics violations and failure to preserve elements listed in the reasoning context."""
    try:
        raw = client.chat_vision(
            image_path,
            user_prompt,
            system_prompt=(
                "You are a strict visual verification agent for image editing. "
                "Penalize missing objects, wrong interactions, physics violations, and partial edits. "
                "Return strict JSON only."
            ),
            temperature=0.1,
        )
        data = extract_json_object(raw)
        score = data.get("score")
        if isinstance(score, str):
            score = float(score)
        data["score"] = score if isinstance(score, (int, float)) else None
        data.setdefault("passed", [])
        data.setdefault("errors", [])
        return data
    except Exception as exc:
        return {
            "score": None,
            "passed": [],
            "errors": [str(exc)],
            "summary": "Edit VQA fallback used.",
        }


def load_candidate_system_prompt(path: str | Path = DEFAULT_CANDIDATE_PROMPT) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "Generate diverse edit_prompt candidates for image editing. "
        'Return strict JSON: {"candidates": ["..."]}'
    )


def generate_candidate_edit_prompts(
    client: MLLMClient,
    instruction: str,
    current_prompt: str,
    checklist: list[VQACheck],
    image_path: str | Path,
    history: Iterable[EditIteration],
    latest_vqa: dict[str, object] | None,
    candidates: int,
    use_dry_run: bool,
    force_refinement: bool = False,
    reason_context: str = "",
) -> list[str]:
    if candidates <= 1:
        return [current_prompt]

    if use_dry_run:
        return heuristic_candidate_edit_prompts(
            current_prompt,
            latest_vqa or {},
            candidates,
            force_refinement,
        )

    history_text = "\n".join(
        f"- iter {item.iteration}: score={item.score}; edit_prompt={item.edit_prompt}"
        for item in history
    )
    force_note = ""
    if force_refinement:
        force_note = (
            "\nMandatory refinement: generate improved candidates even if the previous score was high."
        )
    vqa_text = json.dumps(latest_vqa or {}, ensure_ascii=False, indent=2)
    context_block = f"\nReasoning context:\n{reason_context}\n" if reason_context else ""
    user_prompt = f"""Original hypothetical instruction:
{instruction}
{context_block}
Current edit_prompt:
{current_prompt}

Checklist:
{json.dumps([item.to_dict() for item in checklist], ensure_ascii=False, indent=2)}

Latest VQA feedback:
{vqa_text}

History:
{history_text or "(none)"}
{force_note}

Generate {candidates} diverse improved edit_prompt candidates for the next edit attempt.
Return JSON exactly:
{{"candidates": ["edit prompt 1", "edit prompt 2"]}}"""
    try:
        raw = client.chat_vision(
            image_path,
            user_prompt,
            system_prompt=load_candidate_system_prompt(),
            temperature=0.4,
        )
        data = extract_json_object(raw)
        prompts = [str(item).strip() for item in data.get("candidates", []) if str(item).strip()]
        if prompts:
            return prompts[: max(candidates, 1)]
    except Exception:
        pass

    refined = refine_edit_prompt(
        client=client,
        instruction=instruction,
        current_prompt=current_prompt,
        image_path=image_path,
        vqa_result=latest_vqa or {},
        use_dry_run=False,
        force_refinement=force_refinement,
        reason_context=reason_context,
    )
    return heuristic_candidate_edit_prompts(refined, latest_vqa or {}, candidates, force_refinement)


def select_best_edit_candidate(
    editor: EditClient,
    client: MLLMClient,
    before_path: Path,
    candidate_prompts: list[str],
    instruction: str,
    checklist: list[VQACheck],
    output_dir: Path,
    step: int,
    after_suffix: str,
    seed: int | None,
    use_dry_run: bool,
    reason_context: str = "",
) -> dict[str, object]:
    scored: list[dict[str, object]] = []
    for index, prompt in enumerate(candidate_prompts, start=1):
        call_prompt = finalize_edit_prompt_from_text(prompt, reason_context)
        candidate_path = editor.edit(
            before_path,
            call_prompt,
            output_dir / f"candidate_iter_{step}_{index}{after_suffix}",
            seed=None if seed is None else seed + step * 100 + index,
        )
        vqa_result: dict[str, object] | None = None
        if checklist and not use_dry_run:
            vqa_result = verify_edit_result(
                client=client,
                image_path=candidate_path,
                instruction=instruction,
                edit_prompt=call_prompt,
                checklist=checklist,
                reason_context=reason_context,
            )
        score = extract_score(vqa_result)
        if not isinstance(score, (int, float)):
            score = heuristic_edit_candidate_score(prompt, checklist)
            if vqa_result is None:
                vqa_result = {
                    "score": score,
                    "passed": [],
                    "errors": ["dry_run: visual analysis skipped"],
                    "summary": "Heuristic score used in dry run.",
                }
            else:
                vqa_result = dict(vqa_result)
                vqa_result["score"] = score
                vqa_result["summary"] = "Heuristic score used because visual scoring was unavailable."
        scored.append(
            {
                "prompt": call_prompt,
                "image_path": str(candidate_path),
                "score": float(score),
                "analysis": vqa_result,
            }
        )

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    best = dict(scored[0])
    best["candidates"] = scored
    final_path = output_dir / f"image_iter_{step}{after_suffix}"
    Path(str(best["image_path"])).replace(final_path)
    best["image_path"] = str(final_path)
    return best


def heuristic_candidate_edit_prompts(
    current_prompt: str,
    vqa_result: dict[str, object],
    candidates: int,
    force_refinement: bool = False,
) -> list[str]:
    base = heuristic_refine_edit_prompt(current_prompt, vqa_result, force_refinement)
    variants = [
        base,
        f"{base.rstrip('.')}. Make spatial layout, object counts, and interactions explicit.",
        f"{base.rstrip('.')}. Emphasize physical plausibility and preserve unchanged background elements.",
        f"{base.rstrip('.')}. Focus on the counterfactual outcome with clear foreground action.",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in variants:
        key = prompt.strip()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    if not deduped:
        deduped = [current_prompt]
    while len(deduped) < candidates:
        deduped.append(f"{deduped[-1].rstrip('.')}. Variant {len(deduped) + 1}.")
    return deduped[: max(candidates, 1)]


def heuristic_edit_candidate_score(prompt: str, checklist: list[VQACheck]) -> float:
    lower = prompt.lower()
    score = 0.3
    important_words: set[str] = set()
    for item in checklist:
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", item.q.lower()):
            if len(word) > 3:
                important_words.add(word)
    if important_words:
        matched = sum(1 for word in important_words if word in lower)
        score += 0.6 * matched / len(important_words)
    if any(token in lower for token in ["exactly", "left", "right", "balanced", "center", "foreground", "background"]):
        score += 0.1
    return min(score, 1.0)


def load_refine_system_prompt(path: str | Path = DEFAULT_REFINE_PROMPT) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "Rewrite edit_prompt to fix VQA errors. Return JSON with edit_prompt only."


def refine_edit_prompt(
    client: MLLMClient,
    instruction: str,
    current_prompt: str,
    image_path: str | Path,
    vqa_result: dict[str, object],
    use_dry_run: bool,
    force_refinement: bool = False,
    reason_context: str = "",
) -> str:
    if use_dry_run:
        return heuristic_refine_edit_prompt(current_prompt, vqa_result, force_refinement)

    force_note = ""
    if force_refinement:
        force_note = (
            "\nThis is a mandatory refinement pass before the next edit attempt. "
            "Improve the prompt even if the previous score was high."
        )

    user_prompt = f"""Original hypothetical instruction:
{instruction}

Reasoning context:
{reason_context or "(none)"}

Current edit_prompt:
{current_prompt}

Latest VQA feedback:
{json.dumps(vqa_result, ensure_ascii=False, indent=2)}
{force_note}

Inspect the latest edited image and return improved JSON with a better edit_prompt."""
    try:
        raw = client.chat_vision(
            image_path,
            user_prompt,
            system_prompt=load_refine_system_prompt(),
            temperature=0.2,
        )
        data = extract_json_object(raw)
        refined = str(data.get("edit_prompt", "")).strip()
        if refined and refined != current_prompt.strip():
            return refined
    except Exception:
        pass
    return heuristic_refine_edit_prompt(current_prompt, vqa_result, force_refinement)


def finalize_edit_prompt_from_text(prompt: str, reason_context: str) -> str:
    text = prompt.strip()
    if not text or not reason_context:
        return text
    marker = "Keep unchanged:"
    if marker in text:
        return text
    preserve_line = next((line for line in reason_context.splitlines() if line.startswith("Must preserve:")), "")
    if preserve_line and preserve_line.split(":", 1)[-1].strip().lower() not in text.lower():
        return f"{text.rstrip('.')}. {preserve_line.replace('Must preserve', 'Keep unchanged')}."
    return text


def heuristic_refine_edit_prompt(
    current_prompt: str,
    vqa_result: dict[str, object],
    force_refinement: bool = False,
) -> str:
    errors = [str(item).strip() for item in vqa_result.get("errors", []) if str(item).strip()]
    if errors:
        fix_text = "; ".join(errors[:3])
        return f"{current_prompt.rstrip('.')}. Fix these issues: {fix_text}."
    if force_refinement:
        summary = str(vqa_result.get("summary", "")).strip()
        suffix = summary if summary else "Make the counterfactual outcome clearer, more physically plausible, and visually explicit."
        return f"{current_prompt.rstrip('.')}. Refine further: {suffix}"
    return f"{current_prompt.rstrip('.')}. Make the counterfactual outcome clearer and more visually explicit."


def write_reason_analysis(reason: ReasonResult, reason_context: str, output_dir: Path) -> None:
    payload = reason.to_dict()
    payload["reason_context"] = reason_context
    (output_dir / "reason_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if reason_context:
        (output_dir / "reason_context.txt").write_text(reason_context + "\n", encoding="utf-8")


def write_result_files(result: EditPipelineResult, output_dir: Path) -> None:
    payload = result.to_dict()
    (output_dir / "edit_final.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "edit_prompt.txt").write_text(result.edit_prompt + "\n", encoding="utf-8")
    if result.reasoning_chain:
        (output_dir / "reasoning_chain.txt").write_text(
            "\n".join(result.reasoning_chain) + "\n",
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReasonGenPilot edit pipeline.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--instruction", required=True, help="Hypothetical edit instruction.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--iterations", type=int, default=2, help="Max edit+VQA verify rounds.")
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=2,
        help="Minimum edit rounds before early stop is allowed (default 2 = always refine once).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.85,
        help="After min_iterations, stop early when VQA score reaches this value.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=2,
        help="Number of edit_prompt candidates to generate and score per iteration.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use configured MLLM and edit APIs instead of dry-run heuristics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_edit_pipeline(
        image_path=args.image,
        instruction=args.instruction,
        output_dir=args.output,
        iterations=args.iterations,
        min_iterations=args.min_iterations,
        candidates=args.candidates,
        score_threshold=args.score_threshold,
        dry_run=not args.real_api,
        seed=args.seed,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
