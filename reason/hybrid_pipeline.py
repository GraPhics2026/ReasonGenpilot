"""Hybrid route: Reason Agent → scene_prompt → T2I generation.

Hybrid pipeline for counterfactual image generation with full scene changes.
Uses the Reason Agent to infer a complete scene description, then feeds it
into GenPilot for T2I generation. Unlike the edit route, this does NOT use
image editing — it regenerates the entire scene from scratch via text-to-image.

This is the member-3 module.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

from .gen_pipeline import run_gen_pipeline
from .reason_agent import build_reason_context, run_reason_agent
from .schemas import GenPipelineResult, ensure_output_dir

logger = logging.getLogger(__name__)


def run_hybrid_pipeline(
    image_path: str | Path,
    instruction: str,
    output_dir: str | Path,
    iterations: int = 1,
    candidates: int = 2,
    dry_run: bool | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Run the hybrid pipeline: reason → scene_prompt → T2I generation.

    The hybrid route regenerates the entire scene from scratch via T2I.
    It does NOT use image editing. For visual identity preservation, use the
    edit route instead.

    Args:
        image_path: Path to the reference image (used for reasoning only, not as image condition).
        instruction: Hypothetical instruction (e.g. "如果房间变成深夜会怎样").
        output_dir: Output directory for this case.
        iterations: Number of GenPilot prompt-optimization iterations.
        candidates: Number of candidate prompts per GenPilot iteration.
        dry_run: Force heuristic mode. Defaults to True when MLLM is unconfigured.
        seed: Optional random seed for T2I.

    Returns:
        A dict with final_image, final_prompt, reasoning_chain, route, and metadata.
    """

    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"Input image not found: {source}")

    out_dir = ensure_output_dir(output_dir)

    # --- Step 1: Reason Agent (hybrid mode) ---
    reason = run_reason_agent(
        image_path=source,
        instruction=instruction,
        mode="hybrid",
        dry_run=dry_run,
    )
    reason_context = build_reason_context(reason)
    scene_prompt = reason.scene_prompt or instruction

    # Validate scene_prompt quality — fix fragmentary prompts before T2I
    scene_prompt = _validate_and_fix_scene_prompt(scene_prompt, reason, instruction)

    write_reason_files(reason, reason_context, scene_prompt, out_dir)

    # --- Step 2: Copy reference image as image_before ---
    before_path = out_dir / f"image_before{source.suffix or '.png'}"
    shutil.copy2(source, before_path)

    # --- Step 3: T2I generation via GenPilot ---
    gen_result: GenPipelineResult = run_gen_pipeline(
        prompt=scene_prompt,
        output_dir=out_dir,
        iterations=iterations,
        candidates=candidates,
        dry_run=dry_run,
        seed=seed,
    )
    final_prompt = gen_result.final_prompt

    # --- Step 4: Copy final image as image_after ---
    suffix = gen_result.final_image.rsplit(".", 1)[-1] if "." in gen_result.final_image else "png"
    after_path = out_dir / f"image_after.{suffix}"
    shutil.copy2(gen_result.final_image, after_path)

    # --- Build result ---
    result: dict[str, object] = {
        "final_image": str(after_path),
        "final_prompt": final_prompt,
        "scene_prompt": scene_prompt,
        "route": "hybrid",
        "reasoning_chain": reason.reasoning_chain,
        "image_before": str(before_path),
        "instruction": instruction,
        "reasoning_type": reason.reasoning_type,
        "visual_cues": reason.visual_cues,
        "physics_implications": reason.physics_implications,
        "target_objects": reason.target_objects,
        "preserve_objects": reason.preserve_objects,
        "vqa_checklist": [item.to_dict() for item in reason.vqa_checklist],
        "metadata": {
            "dry_run": gen_result.metadata.get("dry_run", True),
            "reasoning_type": reason.reasoning_type,
            "num_iterations": iterations,
            "num_candidates": candidates,
        },
    }

    (out_dir / "hybrid_final.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


def _validate_and_fix_scene_prompt(
    scene_prompt: str,
    reason,
    instruction: str,
) -> str:
    """Detect fragmentary scene_prompt and fix it before passing to T2I.

    The MLLM sometimes produces scene_prompts that are just lists of changed objects
    (e.g., "grass, man's sneakers, dog's fur showing the result: snow") instead of
    complete scene descriptions. This causes T2I models to generate images missing
    the main subjects.

    When a fragmentary prompt is detected, we construct a proper scene description
    from the reason agent's visual_cues and physics_implications.
    """
    prompt = scene_prompt.strip()

    # Heuristic: a good scene_prompt should have at least 60 characters
    if len(prompt) < 60:
        logger.warning("scene_prompt is too short (%d chars), likely fragmentary", len(prompt))
        return _build_scene_prompt(reason, instruction, "too short")

    # Detect fragmentary patterns: "X, Y, Z showing the result" or similar
    fragment_patterns = [
        r"^.*,\s+.*,\s+.*\s+(showing|with)\s+(the\s+)?(result|following|change)",
        r"^[a-z\s,]+(showing|depicting)\s+the\s+(result|change)",
        r"^A\s+photorealistic\s+image\s+(depicting|of)\s+[a-z\s,]+(showing|with)\s+the\s+",
    ]
    for pattern in fragment_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            logger.warning("scene_prompt matches fragmentary pattern: %.100s...", prompt)
            return _build_scene_prompt(reason, instruction, "fragmentary pattern detected")

    # Check if the prompt describes a complete scene, not just a list of changed objects
    scene_indicators = ["scene", "background", "sky", "ground", "lighting", "standing",
                        "sitting", "walking", "wearing", "photo", "view", "landscape",
                        "atmosphere", "photorealistic", "surrounding", "environment"]
    has_scene_words = any(word in prompt.lower() for word in scene_indicators)
    if not has_scene_words:
        logger.warning("scene_prompt lacks scene description words: %.100s...", prompt)
        return _build_scene_prompt(reason, instruction, "no scene description indicators")

    return prompt


def _build_scene_prompt(reason, instruction: str, cause: str) -> str:
    """Construct a complete T2I scene description from reason agent output.

    Unlike the edit route's edit_prompt (which is an editing instruction), this
    must be a standalone description that paints the entire scene — because T2I
    models have no access to the original image.
    """
    logger.info("Building scene_prompt from reason agent output (%s)", cause)

    # Collect visual cues for describing subjects and scene
    cues = reason.visual_cues or []
    physics = reason.physics_implications or []
    preserve = reason.preserve_objects or []

    # Build parts of the scene description
    parts: list[str] = []

    # Describe the scene subjects based on visual cues
    if cues:
        subjects = "; ".join(cues[:6])
        parts.append(f"A photorealistic scene. {subjects}.")

    # Apply the hypothetical change
    if physics:
        changes = " ".join(physics[:3])
        parts.append(f"The scene is transformed: {changes}.")

    # Preserve context
    parts.append(f"Context: {instruction.strip()}.")

    # Ensure elements that should stay are mentioned
    if preserve:
        parts.append(f"The following elements remain in the scene: {', '.join(preserve[:5])}.")

    parts.append("Clear composition, highly detailed, professional photography.")

    constructed = " ".join(parts)
    logger.info("Constructed scene_prompt: %.200s...", constructed)
    return constructed


def write_reason_files(
    reason,
    reason_context: str,
    scene_prompt: str,
    output_dir: Path,
) -> None:
    payload = reason.to_dict()
    payload["reason_context"] = reason_context
    (output_dir / "reason_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if reason_context:
        (output_dir / "reason_context.txt").write_text(reason_context + "\n", encoding="utf-8")
    if reason.reasoning_chain:
        (output_dir / "reasoning_chain.txt").write_text(
            "\n\n".join(reason.reasoning_chain) + "\n",
            encoding="utf-8",
        )
    (output_dir / "scene_prompt.txt").write_text(scene_prompt + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReasonGenPilot hybrid pipeline.")
    parser.add_argument("--image", required=True, help="Reference image path.")
    parser.add_argument("--instruction", required=True, help="Hypothetical instruction.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--iterations", type=int, default=1, help="GenPilot optimization iterations.")
    parser.add_argument("--candidates", type=int, default=2, help="Candidates per GenPilot iteration.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use configured MLLM + T2I API instead of dry-run heuristic.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_hybrid_pipeline(
        image_path=args.image,
        instruction=args.instruction,
        output_dir=args.output,
        iterations=args.iterations,
        candidates=args.candidates,
        dry_run=not args.real_api,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()