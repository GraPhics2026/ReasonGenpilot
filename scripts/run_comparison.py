"""Run comparison: edit vs gen-only vs hybrid for each hybrid test case.

Usage:
    python scripts/run_comparison.py --dry-run          # preview with SVG placeholders
    python scripts/run_comparison.py --real-api          # real images (needs API key in config/.env)
    python scripts/run_comparison.py --real-api --cases 1,2  # only specific cases
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from reason.edit_pipeline import run_edit_pipeline  # noqa: E402
from reason.gen_pipeline import run_gen_pipeline  # noqa: E402
from reason.hybrid_pipeline import run_hybrid_pipeline  # noqa: E402


def run_comparison(
    image_path: str,
    instruction: str,
    note: str,
    case_id: int,
    dry_run: bool = True,
    base_output: str = "data/output/hybrid",
) -> dict:
    """Run all three approaches for one case."""

    case_dir = Path(base_output) / f"case_{case_id}"
    results: dict[str, object] = {
        "case": case_id,
        "instruction": instruction,
        "note": note,
        "image_path": image_path,
    }

    # ── 方案A：无 hybrid — 用 edit 硬上 ──
    t0 = time.time()
    try:
        edit_out = case_dir / "comparison_A_edit_attempt"
        edit_out.mkdir(parents=True, exist_ok=True)
        edit_result = run_edit_pipeline(
            image_path=image_path,
            instruction=instruction,
            output_dir=edit_out,
            iterations=1,
            min_iterations=1,
            candidates=2,
            dry_run=dry_run,
        )
        results["A_edit"] = {
            "success": True,
            "elapsed": round(time.time() - t0, 1),
            "final_image": edit_result.final_image,
            "route": edit_result.route,
            "edit_prompt": edit_result.edit_prompt,
            "reasoning_type": edit_result.metadata.get("reasoning_type"),
        }
        print(f"  [A] edit    OK ({results['A_edit']['elapsed']}s)")
    except Exception as exc:
        results["A_edit"] = {"success": False, "error": str(exc), "elapsed": round(time.time() - t0, 1)}
        print(f"  [A] edit    FAIL: {exc}")

    # ── 方案B：无 hybrid — 直接 gen（纯文字指令） ──
    t0 = time.time()
    try:
        gen_out = case_dir / "comparison_B_gen_only"
        gen_out.mkdir(parents=True, exist_ok=True)
        gen_result = run_gen_pipeline(
            prompt=instruction,
            output_dir=gen_out,
            iterations=1,
            dry_run=dry_run,
        )
        results["B_gen"] = {
            "success": True,
            "elapsed": round(time.time() - t0, 1),
            "final_image": gen_result.final_image,
            "route": gen_result.route,
            "final_prompt": gen_result.final_prompt,
        }
        print(f"  [B] gen     OK ({results['B_gen']['elapsed']}s)")
    except Exception as exc:
        results["B_gen"] = {"success": False, "error": str(exc), "elapsed": round(time.time() - t0, 1)}
        print(f"  [B] gen     FAIL: {exc}")

    # ── 方案C：有 hybrid — 完整 pipeline ──
    t0 = time.time()
    try:
        hybrid_out = case_dir / "comparison_C_hybrid"
        hybrid_out.mkdir(parents=True, exist_ok=True)
        hybrid_result = run_hybrid_pipeline(
            image_path=image_path,
            instruction=instruction,
            output_dir=hybrid_out,
            iterations=1,
            candidates=2,
            dry_run=dry_run,
        )
        results["C_hybrid"] = {
            "success": True,
            "elapsed": round(time.time() - t0, 1),
            "final_image": hybrid_result.final_image,
            "route": hybrid_result.route,
            "scene_prompt": hybrid_result.scene_prompt,
            "reasoning_type": hybrid_result.reasoning_type,
            "vqa_score": (
                hybrid_result.vqa_result.get("score")
                if hybrid_result.vqa_result
                else None
            ),
        }
        print(f"  [C] hybrid  OK ({results['C_hybrid']['elapsed']}s, VQA={results['C_hybrid']['vqa_score']})")
    except Exception as exc:
        results["C_hybrid"] = {"success": False, "error": str(exc), "elapsed": round(time.time() - t0, 1)}
        print(f"  [C] hybrid  FAIL: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid comparison: edit vs gen-only vs hybrid")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Use heuristics + SVG placeholders (no API calls).")
    parser.add_argument("--real-api", dest="dry_run", action="store_false",
                        help="Use real MLLM + T2I API (requires config/.env).")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case numbers, e.g. 1,2,3,4")
    parser.add_argument("--output", type=str, default="data/output/hybrid",
                        help="Base output directory.")
    args = parser.parse_args()

    # Load cases
    cases_path = _project_root / "data" / "input" / "hybrid" / "hybrid_cases.jsonl"
    if not cases_path.exists():
        print(f"Error: hybrid_cases.jsonl not found at {cases_path}")
        sys.exit(1)

    all_cases: list[dict] = []
    for line in cases_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            all_cases.append(json.loads(line))

    # Filter by case numbers
    if args.cases:
        indices = {int(x.strip()) for x in args.cases.split(",")}
        selected = [c for i, c in enumerate(all_cases, start=1) if i in indices]
    else:
        selected = all_cases

    print(f"\nRunning comparison: {'DRY-RUN' if args.dry_run else 'REAL API'}")
    print(f"Cases: {len(selected)}")
    print("=" * 60)

    all_results: list[dict] = []
    for idx, case in enumerate(selected, start=1):
        # Determine original case number
        if args.cases:
            actual_idx = sorted({int(x.strip()) for x in args.cases.split(",")})[idx - 1]
        else:
            actual_idx = idx

        print(f"\n--- Case {actual_idx}: {case['instruction']} ---")
        result = run_comparison(
            image_path=case["image"],
            instruction=case["instruction"],
            note=case.get("note", ""),
            case_id=actual_idx,
            dry_run=args.dry_run,
            base_output=args.output,
        )
        all_results.append(result)

    # Write summary
    summary_path = Path(args.output) / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"Summary written to {summary_path}")
    print(f"Results per case under {args.output}/case_N/comparison_*/\n")

    # Quick stats
    a_ok = sum(1 for r in all_results if r.get("A_edit", {}).get("success"))
    b_ok = sum(1 for r in all_results if r.get("B_gen", {}).get("success"))
    c_ok = sum(1 for r in all_results if r.get("C_hybrid", {}).get("success"))
    total = len(all_results)
    print(f"Success rate:  A(edit)={a_ok}/{total}  B(gen)={b_ok}/{total}  C(hybrid)={c_ok}/{total}")


if __name__ == "__main__":
    main()
