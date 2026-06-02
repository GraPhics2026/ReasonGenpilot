"""Gen route: prompt optimization plus text-to-image generation."""

from __future__ import annotations

import argparse
import json
import ast
import re
from pathlib import Path
from typing import Iterable

from .api_client import MLLMClient
from .schemas import GenIteration, GenPipelineResult, ensure_output_dir
from .t2i_client import T2IClient


GEN_SYSTEM_PROMPT = """You are the GenPilot prompt optimizer.
Rewrite text-to-image prompts so the image model preserves object counts,
colors, spatial relations, and important attributes. Keep the result concise,
concrete, and directly drawable. Return only the optimized English prompt."""

DECOMPOSE_SYSTEM_PROMPT = """You are the prompt decomposition agent in GenPilot.
Break a text-to-image prompt into checkable visual constraints. Return strict JSON."""

VQA_SYSTEM_PROMPT = """You are the visual error analysis agent in GenPilot.
Compare an image against a prompt checklist. Return strict JSON with a score and errors."""

CANDIDATE_SYSTEM_PROMPT = """You are the prompt refinement agent in GenPilot.
Generate diverse candidate prompts that fix detected image-generation errors."""


def run_gen_pipeline(
    prompt: str,
    output_dir: str | Path,
    iterations: int = 2,
    candidates: int = 2,
    dry_run: bool | None = None,
    seed: int | None = None,
    strategy: str = "genpilot",
) -> GenPipelineResult:
    """Run the member-1 gen path.

    This is a lightweight wrapper compatible with the later full GenPilot Stage
    1/2 integration. It first creates a baseline image, then iteratively
    optimizes the prompt and generates a final image.
    """

    out_dir = ensure_output_dir(output_dir)
    client = MLLMClient()
    t2i = T2IClient()
    use_dry_run = (not client.configured) if dry_run is None else dry_run
    image_suffix = ".svg" if t2i.config.backend == "dry_run" else ".png"

    baseline_path = t2i.generate(prompt, out_dir / f"image_before{image_suffix}", seed=seed)
    current_prompt = prompt
    constraints = decompose_prompt(client, prompt, use_dry_run)
    baseline_analysis = analyze_image_alignment(
        client=client,
        image_path=baseline_path,
        original_prompt=prompt,
        checklist=constraints,
        use_dry_run=use_dry_run,
    )
    history: list[GenIteration] = [
        GenIteration(
            iteration=0,
            prompt=prompt,
            analysis=format_analysis("Baseline generation from the original prompt.", baseline_analysis),
            score=baseline_analysis.get("score"),
            image_path=str(baseline_path),
        )
    ]

    for step in range(1, max(iterations, 0) + 1):
        if use_dry_run:
            optimized = heuristic_optimize_prompt(current_prompt)
            analysis = "Dry-run heuristic: clarified count/color/spatial constraints."
        elif strategy == "simple":
            optimized = optimize_prompt_with_mllm(
                client=client,
                original_prompt=prompt,
                current_prompt=current_prompt,
                history=history,
                candidates=candidates,
            )
            analysis = "Simple MLLM prompt rewrite."
        else:
            candidate_prompts = generate_candidate_prompts(
                client=client,
                original_prompt=prompt,
                current_prompt=current_prompt,
                history=history,
                latest_analysis=history[-1].analysis,
                candidates=candidates,
            )
            best = select_best_candidate(
                client=client,
                t2i=t2i,
                candidate_prompts=candidate_prompts,
                output_dir=out_dir,
                step=step,
                image_suffix=image_suffix,
                original_prompt=prompt,
                checklist=constraints,
                seed=seed,
                use_dry_run=use_dry_run,
            )
            optimized = best["prompt"]
            analysis = format_analysis("GenPilot candidate selection.", best)
        current_prompt = optimized
        if strategy == "genpilot" and not use_dry_run:
            image_path = Path(best["image_path"])
        else:
            image_path = t2i.generate(
                current_prompt,
                out_dir / f"image_iter_{step}{image_suffix}",
                seed=None if seed is None else seed + step,
            )
            image_analysis = analyze_image_alignment(
                client=client,
                image_path=image_path,
                original_prompt=prompt,
                checklist=constraints,
                use_dry_run=use_dry_run,
            )
            analysis = format_analysis(analysis, image_analysis)
        history.append(
            GenIteration(
                iteration=step,
                prompt=current_prompt,
                analysis=analysis,
                score=extract_score_from_analysis(analysis),
                image_path=str(image_path),
            )
        )

    final_image = history[-1].image_path or str(baseline_path)
    result = GenPipelineResult(
        final_image=final_image,
        final_prompt=current_prompt,
        prompt_before=prompt,
        iterations=history,
        metadata={
            "dry_run": use_dry_run,
            "strategy": strategy,
            "num_iterations": iterations,
            "num_candidates": candidates,
            "constraints": constraints,
        },
    )
    write_result_files(result, out_dir)
    return result


def decompose_prompt(client: MLLMClient, prompt: str, use_dry_run: bool) -> list[str]:
    if use_dry_run:
        return heuristic_decompose_prompt(prompt)
    user_prompt = f"""Prompt:
{prompt}

Return JSON exactly in this shape:
{{
  "checklist": [
    "A yes/no visual constraint about object count, color, position, relation, style, or scene."
  ]
}}

Keep each checklist item directly checkable from an image."""
    try:
        data = parse_jsonish(client.chat_text(user_prompt, system_prompt=DECOMPOSE_SYSTEM_PROMPT))
        checklist = data.get("checklist", [])
        return [str(item).strip() for item in checklist if str(item).strip()] or heuristic_decompose_prompt(prompt)
    except Exception as exc:
        return [f"Prompt decomposition failed; fallback checklist used. Error: {exc}", *heuristic_decompose_prompt(prompt)]


def analyze_image_alignment(
    client: MLLMClient,
    image_path: str | Path,
    original_prompt: str,
    checklist: list[str],
    use_dry_run: bool,
) -> dict[str, object]:
    if use_dry_run:
        return {"score": None, "passed": [], "errors": ["dry_run: visual analysis skipped"]}
    user_prompt = f"""Original prompt:
{original_prompt}

Checklist:
{json.dumps(checklist, ensure_ascii=False, indent=2)}

Inspect the image and return JSON exactly in this shape:
{{
  "score": 0.0,
  "passed": ["checklist items that are satisfied"],
  "errors": ["missing, wrong, unclear, or spatially incorrect visual constraints"],
  "summary": "one short sentence"
}}

Use score from 0 to 1 for text-image alignment."""
    try:
        data = parse_jsonish(client.chat_vision(image_path, user_prompt, system_prompt=VQA_SYSTEM_PROMPT))
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
            "errors": [
                "visual analysis unavailable; configure a vision-capable MLLM to enable full VQA feedback",
                str(exc),
            ],
            "summary": "VQA fallback used.",
        }


def generate_candidate_prompts(
    client: MLLMClient,
    original_prompt: str,
    current_prompt: str,
    history: Iterable[GenIteration],
    latest_analysis: str,
    candidates: int,
) -> list[str]:
    history_text = "\n".join(
        f"- iter {item.iteration}: score={item.score}; prompt={item.prompt}; analysis={item.analysis}"
        for item in history
    )
    user_prompt = f"""Original prompt:
{original_prompt}

Current prompt:
{current_prompt}

Latest image analysis:
{latest_analysis}

History:
{history_text}

Generate {candidates} diverse improved prompts. Requirements:
1. Preserve every explicit object, count, color, and spatial relation.
2. Fix the reported image errors.
3. Make spatial layout and object counts unambiguous.
4. Avoid contradicting the original prompt.

Return JSON exactly:
{{"candidates": ["prompt 1", "prompt 2"]}}"""
    data = parse_jsonish(client.chat_text(user_prompt, system_prompt=CANDIDATE_SYSTEM_PROMPT, temperature=0.4))
    prompts = [str(item).strip() for item in data.get("candidates", []) if str(item).strip()]
    if not prompts:
        prompts = [optimize_prompt_with_mllm(client, original_prompt, current_prompt, history, candidates)]
    return prompts[: max(candidates, 1)]


def select_best_candidate(
    client: MLLMClient,
    t2i: T2IClient,
    candidate_prompts: list[str],
    output_dir: Path,
    step: int,
    image_suffix: str,
    original_prompt: str,
    checklist: list[str],
    seed: int | None,
    use_dry_run: bool,
) -> dict[str, object]:
    scored: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_prompts, start=1):
        candidate_image = t2i.generate(
            candidate,
            output_dir / f"candidate_iter_{step}_{index}{image_suffix}",
            seed=None if seed is None else seed + step * 100 + index,
        )
        analysis = analyze_image_alignment(
            client=client,
            image_path=candidate_image,
            original_prompt=original_prompt,
            checklist=checklist,
            use_dry_run=use_dry_run,
        )
        score = analysis.get("score")
        if not isinstance(score, (int, float)):
            score = heuristic_candidate_score(candidate, checklist)
            analysis["score"] = score
            analysis["summary"] = "Heuristic score used because visual scoring was unavailable."
        scored.append(
            {
                "prompt": candidate,
                "image_path": str(candidate_image),
                "score": float(score),
                "analysis": analysis,
            }
        )
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    best = dict(scored[0])
    best["candidates"] = scored
    final_path = output_dir / f"image_iter_{step}{image_suffix}"
    Path(str(best["image_path"])).replace(final_path)
    best["image_path"] = str(final_path)
    return best


def optimize_prompt_with_mllm(
    client: MLLMClient,
    original_prompt: str,
    current_prompt: str,
    history: Iterable[GenIteration],
    candidates: int,
) -> str:
    history_text = "\n".join(
        f"- iter {item.iteration}: {item.prompt}" for item in history
    )
    user_prompt = f"""Original user prompt:
{original_prompt}

Current prompt:
{current_prompt}

Previous prompt history:
{history_text}

Generate one improved prompt. Requirements:
1. Preserve every explicit object, count, color, and spatial relation.
2. Add concrete visual wording only when it helps the image model.
3. Avoid adding new facts that contradict the original.
4. Return only the final English prompt.

Target candidate budget for the full GenPilot stage is {candidates}; for this
wrapper, provide the single best candidate."""
    return client.chat_text(user_prompt, system_prompt=GEN_SYSTEM_PROMPT).strip()


def parse_jsonish(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise


def heuristic_decompose_prompt(prompt: str) -> list[str]:
    text = prompt.strip()
    constraints = [f"The image should depict: {text}"]
    lower = text.lower()
    if any(token in lower for token in ["exactly", "one", "two", "three", "four", "five", "six", "1", "2", "3", "4", "5", "6"]):
        constraints.append("All explicit object counts should be correct.")
    if any(token in lower for token in ["red", "yellow", "blue", "green", "black", "white", "brown", "gray"]):
        constraints.append("All explicit colors should match the prompt.")
    if any(token in lower for token in ["left", "right", "top", "bottom", "center", "under", "above", "behind", "in front"]):
        constraints.append("All explicit spatial relationships should be visible.")
    return constraints


def heuristic_candidate_score(prompt: str, checklist: list[str]) -> float:
    lower = prompt.lower()
    score = 0.3
    important_words = set()
    for item in checklist:
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", item.lower()):
            if len(word) > 3:
                important_words.add(word)
    if important_words:
        matched = sum(1 for word in important_words if word in lower)
        score += 0.6 * matched / len(important_words)
    if any(token in lower for token in ["exactly", "left", "right", "top", "bottom", "center", "foreground", "background"]):
        score += 0.1
    return min(score, 1.0)


def format_analysis(title: str, analysis: dict[str, object]) -> str:
    return f"{title} {json.dumps(analysis, ensure_ascii=False)}"


def extract_score_from_analysis(analysis: str) -> float | None:
    match = re.search(r'"score"\s*:\s*([0-9.]+)', analysis)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def heuristic_optimize_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        return text
    lower = text.lower()
    additions: list[str] = []
    if not re.search(r"\b(highly detailed|clear|sharp|photorealistic)\b", lower):
        additions.append("clear, highly detailed composition")
    if any(token in lower for token in ["six", "6", "three", "3", "two", "2", "exactly"]):
        additions.append("exact object count, no extra duplicated objects")
    if any(token in lower for token in ["red", "yellow", "blue", "green", "black", "white"]):
        additions.append("colors must match the description")
    if any(token in lower for token in ["beside", "next to", "left", "right", "above", "under", "behind"]):
        additions.append("spatial relationships must be visually explicit")
    if not additions:
        additions.append("all described objects visible and unambiguous")
    suffix = "; ".join(additions)
    if suffix.lower() in lower:
        return text
    return f"{text.rstrip('.!?。！？')}. {suffix}."


def write_result_files(result: GenPipelineResult, output_dir: Path) -> None:
    result_json = result.to_dict()
    (output_dir / "prompt_final.json").write_text(
        json.dumps(result_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "final_prompt.txt").write_text(result.final_prompt + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReasonGenPilot gen pipeline.")
    parser.add_argument("--prompt", required=True, help="Text-to-image prompt.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--strategy",
        choices=["genpilot", "simple"],
        default="genpilot",
        help="genpilot uses decomposition, error analysis, candidate generation and selection; simple uses one prompt rewrite.",
    )
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use configured MLLM API instead of dry-run heuristic.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_gen_pipeline(
        prompt=args.prompt,
        output_dir=args.output,
        iterations=args.iterations,
        candidates=args.candidates,
        dry_run=not args.real_api,
        seed=args.seed,
        strategy=args.strategy,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
