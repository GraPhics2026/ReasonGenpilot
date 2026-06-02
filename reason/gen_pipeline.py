"""Gen route: prompt optimization plus text-to-image generation."""

from __future__ import annotations

import argparse
import json
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


def run_gen_pipeline(
    prompt: str,
    output_dir: str | Path,
    iterations: int = 2,
    candidates: int = 3,
    dry_run: bool | None = None,
    seed: int | None = None,
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
    history: list[GenIteration] = [
        GenIteration(
            iteration=0,
            prompt=prompt,
            analysis="Baseline generation from the original prompt.",
            image_path=str(baseline_path),
        )
    ]

    for step in range(1, max(iterations, 0) + 1):
        if use_dry_run:
            optimized = heuristic_optimize_prompt(current_prompt)
            analysis = "Dry-run heuristic: clarified count/color/spatial constraints."
        else:
            optimized = optimize_prompt_with_mllm(
                client=client,
                original_prompt=prompt,
                current_prompt=current_prompt,
                history=history,
                candidates=candidates,
            )
            analysis = "MLLM prompt rewrite using GenPilot-style error analysis."
        current_prompt = optimized
        image_path = t2i.generate(
            current_prompt,
            out_dir / f"image_iter_{step}{image_suffix}",
            seed=None if seed is None else seed + step,
        )
        history.append(
            GenIteration(
                iteration=step,
                prompt=current_prompt,
                analysis=analysis,
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
            "num_iterations": iterations,
            "num_candidates": candidates,
        },
    )
    write_result_files(result, out_dir)
    return result


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
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
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
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
