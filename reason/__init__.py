"""ReasonGenPilot base package."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from .hybrid_pipeline import run_hybrid_pipeline
from .edit_pipeline import run_edit_pipeline
from .gen_pipeline import run_gen_pipeline

__all__ = [
    "run_gen_pipeline",
    "run_edit_pipeline",
    "run_hybrid_pipeline",
]

