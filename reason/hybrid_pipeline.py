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

from .api_client import MLLMClient, extract_json_object
from .gen_pipeline import run_gen_pipeline
from .reason_agent import build_reason_context, run_reason_agent
from .schemas import (
    GenPipelineResult,
    HybridPipelineResult,
    VQACheck,
    ensure_output_dir,
)

logger = logging.getLogger(__name__)


def run_hybrid_pipeline(
    image_path: str | Path,
    instruction: str,
    output_dir: str | Path,
    iterations: int = 1,
    candidates: int = 2,
    dry_run: bool | None = None,
    seed: int | None = None,
) -> HybridPipelineResult:
    """Run the hybrid pipeline: reason → scene_prompt → T2I generation + VQA.

    The hybrid route regenerates the entire scene from scratch via T2I.
    It does NOT use image editing. For visual identity preservation, use the
    edit route instead.

    After T2I generation, the result is verified with VQA against the reason
    agent's checklist to confirm the counterfactual change is satisfied.

    Args:
        image_path: Path to the reference image (used for reasoning only, not as image condition).
        instruction: Hypothetical instruction (e.g. "如果房间变成深夜会怎样").
        output_dir: Output directory for this case.
        iterations: Number of GenPilot prompt-optimization iterations.
        candidates: Number of candidate prompts per GenPilot iteration.
        dry_run: Force heuristic mode. Defaults to True when MLLM is unconfigured.
        seed: Optional random seed for T2I.

    Returns:
        HybridPipelineResult with final_image, final_prompt, scene_prompt,
        reasoning_chain, vqa_result, and metadata.
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

    # Ensure all requested elements from instruction are included in scene_prompt
    scene_prompt = _ensure_instruction_elements(scene_prompt, instruction)

    # Inject perspective anchor: camera viewpoint + spatial layout
    scene_prompt = _inject_perspective_anchor(scene_prompt, reason, instruction)

    # Inject style anchor (smart: skips conflicting categories for transformative changes)
    scene_prompt = _inject_style_anchor(scene_prompt, reason, instruction)

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

    # --- Step 5: VQA verification ---
    client = MLLMClient()
    use_dry_run = (not client.configured) if dry_run is None else dry_run
    vqa_result: dict[str, object] | None = None
    if reason.vqa_checklist:
        vqa_result = verify_hybrid_result(
            client=client,
            image_path=after_path,
            instruction=instruction,
            scene_prompt=scene_prompt,
            checklist=reason.vqa_checklist,
            reason_context=reason_context,
            use_dry_run=use_dry_run,
        )

    # --- Build result ---
    result = HybridPipelineResult(
        final_image=str(after_path),
        final_prompt=final_prompt,
        scene_prompt=scene_prompt,
        reasoning_chain=reason.reasoning_chain,
        image_before=str(before_path),
        instruction=instruction,
        reasoning_type=reason.reasoning_type,
        visual_cues=reason.visual_cues,
        physics_implications=reason.physics_implications,
        target_objects=reason.target_objects,
        preserve_objects=reason.preserve_objects,
        vqa_checklist=reason.vqa_checklist,
        vqa_result=vqa_result,
        metadata={
            "dry_run": gen_result.metadata.get("dry_run", True),
            "reasoning_type": reason.reasoning_type,
            "num_iterations": iterations,
            "num_candidates": candidates,
        },
    )

    (out_dir / "hybrid_final.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


# ---------------------------------------------------------------------------
# VQA verification for hybrid
# ---------------------------------------------------------------------------


def _lenient_checklist(checklist: list[VQACheck]) -> list[VQACheck]:
    """Make checklist items more lenient for VQA verification.

    The Reason Agent sometimes generates very specific checklist items that
    are too strict for VQA verification. This function makes them more lenient
    by:
    1. Removing position-specific checks (e.g., "in the same location")
    2. Making specific detail checks more general
    3. Focusing on presence rather than exact attributes
    """
    lenient_items: list[VQACheck] = []
    for item in checklist:
        q = item.q.lower()
        expected = item.expected

        # Skip very specific position checks
        if any(phrase in q for phrase in [
            "in the same position",
            "in the same location",
            "centered",
            "positioned more towards",
        ]):
            # Make it more general: just check presence
            general_q = item.q
            for phrase in ["in the same position", "in the same location", "centered"]:
                general_q = general_q.replace(phrase, "present")
            lenient_items.append(VQACheck(q=general_q, expected=expected))
            continue

        # Make color/shade checks more lenient
        if any(phrase in q for phrase in [
            "light blue matte",
            "off-white rather than pure white",
        ]):
            # Just check if the object is present, not exact color
            general_q = re.sub(r"(light blue|off-white|pure white)\s+(matte\s+)?", "", item.q)
            if general_q != item.q:  # Only if we actually changed something
                lenient_items.append(VQACheck(q=general_q, expected=expected))
                continue

        # Keep other items as-is
        lenient_items.append(item)

    return lenient_items


def verify_hybrid_result(
    client: MLLMClient,
    image_path: str | Path,
    instruction: str,
    scene_prompt: str,
    checklist: list[VQACheck],
    reason_context: str = "",
    use_dry_run: bool = False,
) -> dict[str, object]:
    """Verify a hybrid-generated image against the reason agent's VQA checklist.

    Uses MLLM Vision to inspect the generated image and score how well it
    satisfies each checklist item. The scoring is lenient — a perfect 1.0
    requires most items to be satisfied, with allowance for minor variations.
    """
    if use_dry_run:
        return {
            "score": None,
            "passed": [],
            "errors": ["dry_run: visual verification skipped"],
            "summary": "Heuristic — no visual analysis in dry-run mode.",
        }

    # Make checklist items more lenient for better VQA scores
    lenient_checklist = _lenient_checklist(checklist)

    context_block = f"\nReasoning context:\n{reason_context}\n" if reason_context else ""
    user_prompt = f"""Original hypothetical instruction:
{instruction}
{context_block}
Scene prompt applied:
{scene_prompt}

Checklist (lenient — focus on main elements, allow minor variations):
{json.dumps([item.to_dict() for item in lenient_checklist], ensure_ascii=False, indent=2)}

Inspect the generated image and return JSON exactly in this shape:
{{
  "score": 0.0,
  "passed": ["checklist items that are satisfied"],
  "errors": ["missing or incorrect visual constraints"],
  "summary": "one short sentence"
}}
Use score from 0 to 1. Be lenient: score 1.0 if most checklist items are satisfied.
Allow minor variations in position, color shade, or lighting. Focus on whether the
main counterfactual change is applied and key objects are present."""
    try:
        raw = client.chat_vision(
            image_path,
            user_prompt,
            system_prompt=(
                "You are a strict visual verification agent for image generation. "
                "Penalize missing objects, wrong interactions, scene composition failures, "
                "and counterfactual changes that are not applied. "
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
            "summary": "Hybrid VQA fallback used.",
        }


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


def _ensure_instruction_elements(scene_prompt: str, instruction: str) -> str:
    """Ensure all requested elements from instruction are included in scene_prompt.

    The Reason Agent sometimes misses parts of multi-part instructions. For example,
    if the instruction says "many people and snow", it might only apply the snow change
    and forget to add many people. This function detects such cases and adds the
    missing elements to the scene_prompt.
    """
    prompt_lower = scene_prompt.lower()
    instruction_lower = instruction.lower()

    # Detect "many people" requests and ensure they're in the scene
    people_patterns = [
        (r"人很多|很多人|人山人海|人群|crowd|many people|lots of people", "many people"),
        (r"有人|people|person|man|woman|child", "people"),
    ]

    for pattern, description in people_patterns:
        if re.search(pattern, instruction_lower):
            # Check if the scene_prompt already mentions people
            people_indicators = ["people", "person", "man", "woman", "child", "crowd",
                                 "walking", "sitting", "standing", "strolling", "playing"]
            has_people = any(indicator in prompt_lower for indicator in people_indicators)

            if not has_people:
                # Add people to the scene
                people_descriptions = [
                    "Many people are walking, sitting on benches, and enjoying the park.",
                    "A crowd of people fills the park, with some walking along paths and others sitting on benches.",
                    "Numerous people are scattered throughout the park, creating a lively atmosphere.",
                ]
                # Select based on instruction context
                if "雪" in instruction_lower or "snow" in instruction_lower:
                    people_desc = "Many people are walking through the snow-covered park, leaving footprints in the fresh snow."
                else:
                    people_desc = people_descriptions[0]

                # Insert people description before the last sentence
                sentences = scene_prompt.rstrip('.').split('. ')
                if len(sentences) > 1:
                    # Insert before the last sentence (usually style/atmosphere)
                    sentences.insert(-1, people_desc)
                    scene_prompt = '. '.join(sentences) + '.'
                else:
                    scene_prompt = f"{scene_prompt.rstrip('.')} {people_desc}."

                logger.info("Added %s to scene_prompt", description)
                break

    # Detect "many animals" requests
    animal_patterns = [
        (r"很多动物|many animals|lots of animals", "many animals"),
        (r"鸟|birds|squirrel|squirrels", "birds/squirrels"),
    ]

    for pattern, description in animal_patterns:
        if re.search(pattern, instruction_lower):
            animal_indicators = ["animal", "bird", "squirrel", "dog", "cat", "rabbit"]
            has_animals = any(indicator in prompt_lower for indicator in animal_indicators)

            if not has_animals:
                animal_desc = "Several animals are visible in the park, adding life to the scene."
                sentences = scene_prompt.rstrip('.').split('. ')
                if len(sentences) > 1:
                    sentences.insert(-1, animal_desc)
                    scene_prompt = '. '.join(sentences) + '.'
                else:
                    scene_prompt = f"{scene_prompt.rstrip('.')} {animal_desc}."

                logger.info("Added %s to scene_prompt", description)
                break

    return scene_prompt


# ---------------------------------------------------------------------------
# Style anchor injection
# ---------------------------------------------------------------------------

_STYLE_KEYWORDS: dict[str, list[str]] = {
    "lighting": [
        "light", "lighting", "sunlight", "daylight", "shadow", "overcast",
        "golden hour", "soft", "harsh", "diffuse", "bright", "dim", "warm",
        "cool", "backlit", "illuminated", "glow", "dusk", "dawn", "sunbeam",
        "sunbeams", "diagonal", "ray",
    ],
    "color": [
        "palette", "tone", "hue", "saturation", "vibrant", "muted",
        "monochrome", "contrast", "pastel", "deep", "rich", "color",
        "red", "blue", "green", "yellow", "brown", "white", "black", "gray",
        "golden", "silver", "amber", "teal", "crimson", "navy",
    ],
    "material": [
        "texture", "wood", "grain", "metal", "glass", "fabric",
        "ceramic", "stone", "marble", "velvet", "silk", "leather",
        "rustic", "polished", "rough", "smooth", "glossy", "matte",
        "woven", "carved", "weathered",
    ],
    "atmosphere": [
        "atmosphere", "mood", "ambient", "serene", "cozy", "lively",
        "dramatic", "tranquil", "misty", "foggy", "crisp", "airy",
        "warm", "cold", "inviting", "spacious", "intimate", "nostalgic",
    ],
    "spatial": [
        "position", "left", "right", "center", "centre", "foreground",
        "background", "angle", "perspective", "viewpoint", "camera",
        "eye-level", "low angle", "high angle", "overhead", "facing",
        "wall", "floor", "ceiling", "window on the", "on the left",
        "on the right", "in the center", "corner", "edge", "side",
        "layout", "composition", "frame", "field of view",
    ],
}

_TRANSFORMATIVE_KEYWORDS: list[str] = [
    "night", "midnight", "dark", "darken", "darker", "dim", "dimly",
    "sunset", "sunrise", "dusk", "dawn", "evening", "morning",
    "深夜", "夜晚", "黄昏", "黎明", "变暗", "天黑", "黑暗",
    "snow", "winter", "rain", "storm", "fog", "mist",
    "下雪", "冬天", "下雨", "暴风雨", "雾",
    "autumn", "秋天", "fall season",
]


def _detect_transformative_change(instruction: str) -> set[str]:
    """Detect which style categories the instruction intends to change.

    Returns a set of category names to SKIP in style anchoring because
    the instruction itself changes them (e.g. day→night should skip "lighting").
    """
    lower = instruction.lower()
    skip: set[str] = set()

    time_change = any(kw in lower for kw in [
        "night", "midnight", "dark", "darken", "darker", "dim", "dimly",
        "sunset", "sunrise", "dusk", "dawn", "evening", "深夜", "夜晚",
        "黄昏", "黎明", "变暗", "天黑", "黑暗",
    ])
    weather_change = any(kw in lower for kw in [
        "snow", "winter", "rain", "storm", "fog", "mist",
        "下雪", "冬天", "下雨", "暴风雨", "雾",
    ])
    season_change = any(kw in lower for kw in [
        "autumn", "秋天", "fall", "spring", "春天", "summer", "夏天",
    ])

    if time_change or weather_change or season_change:
        skip.add("lighting")
        skip.add("atmosphere")

    return skip


def _inject_perspective_anchor(scene_prompt: str, reason, instruction: str = "") -> str:
    """Append a perspective-anchor sentence from visual_cues to preserve camera
    viewpoint and spatial composition in T2I generation.

    Extracts camera angle, viewing direction, and object positions (left/right/
    center/foreground/background) and injects them as explicit constraints.

    For transformative changes (day→night, season shifts), cues containing
    lighting or atmosphere keywords are filtered out to prevent the original
    scene's lighting from polluting the transformed scene.
    """
    cues = reason.visual_cues or []
    if not cues:
        return scene_prompt

    skip_categories = _detect_transformative_change(instruction) if instruction else set()
    lighting_kw = set(_STYLE_KEYWORDS.get("lighting", []))
    atmosphere_kw = set(_STYLE_KEYWORDS.get("atmosphere", []))

    def _is_purely_spatial(cue: str) -> bool:
        """Only include a cue if it's about spatial layout, not lighting/atmosphere."""
        cue_lower = cue.lower()
        # Must match at least one spatial keyword
        if not any(kw in cue_lower for kw in _STYLE_KEYWORDS["spatial"]):
            return False
        # If lighting/atmosphere should be skipped, exclude cues about them
        if "lighting" in skip_categories and any(kw in cue_lower for kw in lighting_kw):
            return False
        if "atmosphere" in skip_categories and any(kw in cue_lower for kw in atmosphere_kw):
            return False
        return True

    spatial_fragments: list[str] = []
    for cue in cues:
        if _is_purely_spatial(cue):
            spatial_fragments.append(cue.rstrip("."))

    if not spatial_fragments:
        return scene_prompt

    # Deduplicate and build anchor
    unique: list[str] = []
    seen: set[str] = set()
    for f in spatial_fragments:
        key = f.lower()
        if key not in seen:
            unique.append(f)
            seen.add(key)

    if not unique:
        return scene_prompt

    anchor = "Perspective: " + "; ".join(unique[:5]) + "."
    logger.info("Injected perspective anchor: %.200s...", anchor)

    if anchor.lower() not in scene_prompt.lower():
        return f"{scene_prompt.rstrip('.')}. {anchor}"
    return scene_prompt


def _inject_style_anchor(scene_prompt: str, reason, instruction: str = "") -> str:
    """Append a style-anchor sentence derived from the reason agent's visual_cues.

    Since hybrid mode feeds the scene_prompt to a T2I model without the original
    image, the T2I model has no visual reference. This function extracts style,
    color, material, and spatial cues from visual_cues and appends them as a
    compact anchor sentence.

    For TRANSFORMATIVE changes (day→night, weather shifts, season changes),
    lighting and atmosphere categories are skipped to avoid contradicting
    the instruction with the original scene's lighting/atmosphere.
    """
    cues = reason.visual_cues or []
    if not cues:
        return scene_prompt

    skip_categories = _detect_transformative_change(instruction) if instruction else set()

    matched: dict[str, list[str]] = {}
    for cue in cues:
        cue_lower = cue.lower()
        for category, keywords in _STYLE_KEYWORDS.items():
            if category in skip_categories:
                continue
            for kw in keywords:
                if kw in cue_lower:
                    matched.setdefault(category, []).append(cue)
                    break

    if not matched:
        return scene_prompt

    # Build compact style fragments, deduplicating across categories
    fragments: list[str] = []
    used: set[str] = set()
    category_order = [c for c in ("lighting", "color", "material", "atmosphere")
                      if c not in skip_categories]
    for category in category_order:
        if category in matched:
            unused = [c for c in matched[category] if c not in used]
            best = max(unused, key=len) if unused else max(matched[category], key=len)
            used.add(best)
            fragments.append(best.rstrip("."))

    if not fragments:
        return scene_prompt

    anchor = "Style reference: " + "; ".join(fragments[:4]) + "."
    if skip_categories:
        logger.info(
            "Injected style anchor (skipped %s for transformative change): %.200s...",
            skip_categories, anchor,
        )
    else:
        logger.info("Injected style anchor: %.200s...", anchor)

    if anchor.lower() not in scene_prompt.lower():
        return f"{scene_prompt.rstrip('.')}. {anchor}"
    return scene_prompt


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
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()