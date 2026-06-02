"""Edit verify loop: VQA-driven edit_prompt refinement."""

from __future__ import annotations

from pathlib import Path

from .edit_pipeline import run_edit_pipeline
from .schemas import EditPipelineResult


def run_edit_verify_loop(
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
    """Member-3 compatible wrapper around the edit verify loop in edit_pipeline."""

    return run_edit_pipeline(
        image_path=image_path,
        instruction=instruction,
        output_dir=output_dir,
        iterations=iterations,
        min_iterations=min_iterations,
        candidates=candidates,
        score_threshold=score_threshold,
        dry_run=dry_run,
        seed=seed,
    )
