"""Reason Agent for edit and hybrid routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .api_client import MLLMClient, extract_json_object
from .schemas import ReasonResult, ReasoningType, VQACheck


DEFAULT_REASON_PROMPT = Path("prompts/reason_system.txt")
VALID_REASONING_TYPES = {"physical", "temporal", "causal", "story"}


def load_reason_system_prompt(path: str | Path = DEFAULT_REASON_PROMPT) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "You are the Reason Agent for counterfactual image editing. "
        "Return strict JSON with reasoning_chain, edit_prompt or scene_prompt, "
        "target_objects, and vqa_checklist."
    )


def run_reason_agent(
    image_path: str | Path,
    instruction: str,
    mode: Literal["edit", "hybrid"] = "edit",
    dry_run: bool | None = None,
    client: MLLMClient | None = None,
) -> ReasonResult:
    """Infer counterfactual visual outcome from an image and instruction."""

    mllm = client or MLLMClient()
    use_dry_run = (not mllm.configured) if dry_run is None else dry_run
    if use_dry_run:
        return heuristic_reason_result(instruction, mode=mode)

    system_prompt = load_reason_system_prompt()
    user_prompt = f"""Mode: {mode}
Hypothetical instruction:
{instruction.strip()}

Inspect the image carefully. Extract fine-grained visual cues before inferring the edit.
For edit mode, fill edit_prompt (English, image-editor ready).
For hybrid mode, fill scene_prompt (English, text-to-image ready) instead of edit_prompt.

CRITICAL for hybrid mode — scene_prompt is a STANDALONE T2I prompt:
It will be sent directly to an image generator WITHOUT the original image.
Therefore scene_prompt MUST describe the ENTIRE scene from scratch, NOT just the changed parts.
- First: describe ALL subjects — every person, animal, object visible in the image.
  Include what they look like (clothing color, hair, pose, size, breed), what they are doing,
  and their spatial positions relative to each other.
- Then: describe the environment (ground, sky, buildings, furniture, vegetation).
- Finally: apply the hypothetical change to the relevant parts while keeping everything else intact.
- The result must read like a complete, flowing English paragraph that paints the full picture.
- BAD example (never do this): "grass, man's sneakers, dog's fur with snow applied"
- GOOD example: "A man in a blue t-shirt and khaki pants stands next to a golden retriever
  in a park on a snowy winter day. Fresh white snow covers the ground. Park benches and bare
  trees in the background under a bright overcast sky. Natural daylight, photorealistic."

Think of it this way: if someone reads your scene_prompt aloud, they should be able to
visualize the ENTIRE image, not guess what it looks like.
"""
    raw = mllm.chat_vision(image_path, user_prompt, system_prompt=system_prompt, temperature=0.2)
    data = extract_json_object(raw)
    return parse_reason_result(data, mode=mode, instruction=instruction)


def parse_reason_result(
    data: dict[str, object],
    mode: Literal["edit", "hybrid"],
    instruction: str,
) -> ReasonResult:
    chain = [str(item).strip() for item in data.get("reasoning_chain", []) if str(item).strip()]
    if not chain:
        chain = [f"Instruction received: {instruction.strip()}"]

    checklist_raw = data.get("vqa_checklist", [])
    checklist: list[VQACheck] = []
    if isinstance(checklist_raw, list):
        for item in checklist_raw:
            if isinstance(item, dict):
                question = str(item.get("q", "")).strip()
                if question:
                    checklist.append(
                        VQACheck(q=question, expected=str(item.get("expected", "yes")).strip() or "yes")
                    )
            elif isinstance(item, str) and item.strip():
                checklist.append(VQACheck(q=item.strip()))

    targets = _string_list(data.get("target_objects"))
    preserve = _string_list(data.get("preserve_objects"))
    visual_cues = _string_list(data.get("visual_cues"))
    physics = _string_list(data.get("physics_implications"))
    reasoning_type = _parse_reasoning_type(data.get("reasoning_type"))
    if reasoning_type is None:
        reasoning_type = infer_reasoning_type(instruction)

    edit_prompt = str(data.get("edit_prompt", "")).strip() or None
    scene_prompt = str(data.get("scene_prompt", "")).strip() or None

    if mode == "edit" and not edit_prompt:
        edit_prompt = fallback_edit_prompt(instruction, targets, physics, preserve)
    if mode == "hybrid" and not scene_prompt:
        scene_prompt = fallback_scene_prompt(instruction, targets, physics)

    if not checklist:
        checklist = default_vqa_checklist(reasoning_type, targets, physics)

    return ReasonResult(
        mode=mode,
        reasoning_chain=chain,
        vqa_checklist=checklist,
        edit_prompt=edit_prompt,
        scene_prompt=scene_prompt,
        target_objects=targets,
        reasoning_type=reasoning_type,
        visual_cues=visual_cues,
        physics_implications=physics,
        preserve_objects=preserve,
    )


def finalize_edit_prompt(result: ReasonResult) -> str:
    """Compose an editor-ready prompt with physics and preservation hints."""

    base = (result.edit_prompt or "").strip()
    if not base:
        return ""
    suffix_parts: list[str] = []
    if result.physics_implications:
        joined = "; ".join(result.physics_implications[:3])
        if joined.lower() not in base.lower():
            suffix_parts.append(f"Expected outcome: {joined}")
    if result.preserve_objects:
        joined = ", ".join(result.preserve_objects[:5])
        if joined.lower() not in base.lower():
            suffix_parts.append(f"Keep unchanged: {joined}")
    if not suffix_parts:
        return base
    return f"{base.rstrip('.')}. {' '.join(suffix_parts)}"


def build_reason_context(result: ReasonResult) -> str:
    parts: list[str] = []
    if result.reasoning_type:
        parts.append(f"Reasoning type: {result.reasoning_type}")
    if result.visual_cues:
        parts.append("Visual cues from source image: " + "; ".join(result.visual_cues[:4]))
    if result.physics_implications:
        parts.append("Expected physical/causal outcome: " + "; ".join(result.physics_implications[:4]))
    if result.preserve_objects:
        parts.append("Must preserve: " + ", ".join(result.preserve_objects[:5]))
    return "\n".join(parts)


def heuristic_reason_result(instruction: str, mode: Literal["edit", "hybrid"]) -> ReasonResult:
    text = instruction.strip()
    lower = text.lower()
    reasoning_type = infer_reasoning_type(text)

    if any(token in lower for token in ["ice", "冰"]):
        return _build_heuristic_result(
            mode=mode,
            text=text,
            reasoning_type="physical",
            visual_cues=["sharp-edged solid ice cubes", "container or plate surface", "ambient lighting"],
            physics_implications=["ice melts into liquid water", "edges soften and volume shrinks"],
            targets=["ice cubes"],
            preserve=["plate or container", "background", "unrelated objects"],
            edit_prompt=(
                "The ice cubes have partially melted into clear water on the plate. "
                "Edges are softened and smaller puddles of water are visible. "
                "Keep the plate and background unchanged. Realistic photo."
            ),
            checklist=[
                VQACheck(q="Are the ice cubes melted or visibly softened?", expected="yes"),
                VQACheck(q="Is there visible liquid water from melting?", expected="yes"),
                VQACheck(q="Are the plate and background unchanged?", expected="yes"),
            ],
        )

    if any(token in lower for token in ["跷跷板", "seesaw"]):
        return _build_heuristic_result(
            mode=mode,
            text=text,
            reasoning_type="physical",
            visual_cues=["large elephant and small squirrel on grass", "open field with sky"],
            physics_implications=[
                "elephant sits on one end of a seesaw",
                "elephant side is lower because it is heavier",
                "squirrel end is raised high",
            ],
            targets=["elephant", "squirrel", "seesaw"],
            preserve=["grass field", "sky", "overall scene style"],
            edit_prompt=(
                "Place the elephant and squirrel on a wooden seesaw in the grassy field. "
                "The elephant sits on one end with its side low near the ground; "
                "the squirrel sits on the other end lifted high. "
                "Preserve the lawn and sky."
            ),
            checklist=[
                VQACheck(q="Are the elephant and squirrel on a seesaw?", expected="yes"),
                VQACheck(q="Is the elephant's side lower than the squirrel's side?", expected="yes"),
                VQACheck(q="Are the grass and sky largely preserved?", expected="yes"),
            ],
        )

    if any(token in lower for token in ["盘子", "plate", "陶瓷", "ceramic", "blue", "蓝"]):
        return _build_heuristic_result(
            mode=mode,
            text=text,
            reasoning_type="physical",
            visual_cues=["plate surface material and color", "food or objects on the plate"],
            physics_implications=["plate material/color changes to the requested counterfactual"],
            targets=["plate"],
            preserve=["food on plate", "table", "background"],
            edit_prompt=(
                "Change the plate to the requested material/color while keeping the food, "
                "table, and background unchanged. Realistic photo."
            ),
            checklist=[
                VQACheck(q="Does the plate show the requested material or color change?", expected="yes"),
                VQACheck(q="Are food and background largely unchanged?", expected="yes"),
            ],
        )

    edit_prompt = fallback_edit_prompt(text, [], [], [])
    checklist = default_vqa_checklist(reasoning_type, [], [])
    return _build_heuristic_result(
        mode=mode,
        text=text,
        reasoning_type=reasoning_type,
        visual_cues=[],
        physics_implications=[f"visible outcome if: {text}"],
        targets=[],
        preserve=["unchanged background and unrelated objects"],
        edit_prompt=edit_prompt,
        checklist=checklist,
    )


def _build_heuristic_result(
    mode: Literal["edit", "hybrid"],
    text: str,
    reasoning_type: ReasoningType,
    visual_cues: list[str],
    physics_implications: list[str],
    targets: list[str],
    preserve: list[str],
    edit_prompt: str,
    checklist: list[VQACheck],
) -> ReasonResult:
    chain = [
        f"Dry-run heuristic ({reasoning_type} reasoning).",
        f"Instruction: {text}",
        "Converted into a structured edit plan.",
    ]
    if mode == "hybrid":
        return ReasonResult(
            mode=mode,
            reasoning_chain=chain,
            scene_prompt=fallback_scene_prompt(text, targets, physics_implications),
            vqa_checklist=checklist,
            target_objects=targets,
            reasoning_type=reasoning_type,
            visual_cues=visual_cues,
            physics_implications=physics_implications,
            preserve_objects=preserve,
        )
    return ReasonResult(
        mode=mode,
        reasoning_chain=chain,
        edit_prompt=edit_prompt,
        vqa_checklist=checklist,
        target_objects=targets,
        reasoning_type=reasoning_type,
        visual_cues=visual_cues,
        physics_implications=physics_implications,
        preserve_objects=preserve,
    )


def infer_reasoning_type(instruction: str) -> ReasoningType:
    lower = instruction.lower()
    if any(token in lower for token in ["sunset", "night", "morning", "later", "after hours", "日落", "夜晚", "时间"]):
        return "temporal"
    if any(token in lower for token in ["land", "collide", "because", "导致", "因此", "interaction", "落在", "碰撞"]):
        return "causal"
    if any(token in lower for token in ["hidden", "secret", "story", "texture", "隐藏", "故事", "纹理"]):
        return "story"
    return "physical"


def default_vqa_checklist(
    reasoning_type: ReasoningType | None,
    targets: list[str],
    physics: list[str],
) -> list[VQACheck]:
    items = [VQACheck(q="Does the image reflect the requested counterfactual change?", expected="yes")]
    if physics:
        items.append(VQACheck(q=f"Is this visible outcome satisfied: {physics[0]}?", expected="yes"))
    if reasoning_type == "physical" and len(items) < 3:
        items.append(VQACheck(q="Are unrelated background elements largely preserved?", expected="yes"))
    if targets and len(items) < 3:
        items.append(VQACheck(q=f"Are the target objects/regions ({', '.join(targets[:3])}) edited as intended?", expected="yes"))
    return items[:4]


def fallback_edit_prompt(
    instruction: str,
    targets: list[str],
    physics: list[str],
    preserve: list[str],
) -> str:
    target_text = ", ".join(targets) if targets else "the relevant region"
    physics_text = physics[0] if physics else instruction.strip()
    preserve_text = ", ".join(preserve) if preserve else "all unrelated objects, layout, and lighting"
    return (
        f"Edit the image so this counterfactual holds: {physics_text}. "
        f"Focus changes on {target_text}. Preserve {preserve_text}."
    )


def fallback_scene_prompt(instruction: str, targets: list[str], physics: list[str]) -> str:
    """Generate a complete, standalone T2I scene description — not an edit instruction.

    This is used when the MLLM fails to produce a valid scene_prompt in hybrid mode.
    The prompt must describe the entire scene from scratch since T2I has no access
    to the original image.
    """
    outcome = physics[0] if physics else instruction.strip()
    target_text = ", ".join(targets) if targets else "relevant scene elements"
    parts = [
        f"A photorealistic image depicting {target_text}.",
        f"The visual change applied: {outcome}.",
        "All original subjects, their appearance, spatial layout, background, and lighting are preserved.",
        f"Context: {instruction.strip()}.",
        "Clear composition, highly detailed, professional photography.",
    ]
    return " ".join(parts)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_reasoning_type(value: object) -> ReasoningType | None:
    raw = str(value or "").strip().lower()
    return raw if raw in VALID_REASONING_TYPES else None
