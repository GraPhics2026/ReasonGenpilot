"""Shared data structures for ReasonGenPilot.

Member 1 owns the base and gen-route schemas. Later members can extend these
classes for edit, hybrid, verification, and router outputs without changing the
pipeline return contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Route = Literal["gen", "edit", "hybrid"]
GenerationBackend = Literal["dry_run", "siliconflow", "dashscope", "comfyui"]
EditBackend = Literal["dry_run", "dashscope"]
ReasoningType = Literal["physical", "temporal", "causal", "story"]


@dataclass(slots=True)
class VQACheck:
    q: str
    expected: str = "yes"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class GenIteration:
    iteration: int
    prompt: str
    analysis: str
    score: float | None = None
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenPipelineResult:
    final_image: str
    final_prompt: str
    route: Route = "gen"
    reasoning_chain: list[str] = field(default_factory=list)
    prompt_before: str | None = None
    iterations: list[GenIteration] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["iterations"] = [item.to_dict() for item in self.iterations]
        return payload


@dataclass(slots=True)
class ReasonResult:
    mode: Literal["edit", "hybrid"]
    reasoning_chain: list[str]
    vqa_checklist: list[VQACheck] = field(default_factory=list)
    edit_prompt: str | None = None
    scene_prompt: str | None = None
    target_objects: list[str] = field(default_factory=list)
    reasoning_type: ReasoningType | None = None
    visual_cues: list[str] = field(default_factory=list)
    physics_implications: list[str] = field(default_factory=list)
    preserve_objects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vqa_checklist"] = [item.to_dict() for item in self.vqa_checklist]
        return payload


@dataclass(slots=True)
class EditIteration:
    iteration: int
    edit_prompt: str
    analysis: str
    score: float | None = None
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EditPipelineResult:
    final_image: str
    final_prompt: str
    route: Route = "edit"
    reasoning_chain: list[str] = field(default_factory=list)
    image_before: str = ""
    instruction: str = ""
    edit_prompt: str = ""
    target_objects: list[str] = field(default_factory=list)
    vqa_checklist: list[VQACheck] = field(default_factory=list)
    vqa_result: dict[str, Any] | None = None
    iterations: list[EditIteration] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vqa_checklist"] = [item.to_dict() for item in self.vqa_checklist]
        payload["iterations"] = [item.to_dict() for item in self.iterations]
        return payload


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
