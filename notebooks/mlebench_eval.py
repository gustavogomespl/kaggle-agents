#!/usr/bin/env python3
"""
MLE-bench Evaluation Script for Kaggle Agents.

This script provides a streamlined way to evaluate kaggle-agents
on MLE-bench competitions using the new solve_mlebench() function.

Usage:
    python mlebench_eval.py --competition aerial-cactus-identification
    python mlebench_eval.py --lite  # Run all 22 lite competitions
"""

import argparse
import json
import os
import csv
from datetime import datetime
from pathlib import Path

# MLE-bench Lite competitions (22 total)
MLEBENCH_LITE = [
    # Image Classification
    {"id": "aerial-cactus-identification", "type": "binary_classification", "metric": "auc", "size_gb": 0.025},
    {"id": "aptos2019-blindness-detection", "type": "multiclass_classification", "metric": "quadratic_weighted_kappa", "size_gb": 10.22},
    {"id": "dog-breed-identification", "type": "multiclass_classification", "metric": "log_loss", "size_gb": 0.75},
    {"id": "dogs-vs-cats-redux-kernels-edition", "type": "binary_classification", "metric": "log_loss", "size_gb": 0.85},
    {"id": "leaf-classification", "type": "multiclass_classification", "metric": "log_loss", "size_gb": 0.036},
    {"id": "plant-pathology-2020-fgvc7", "type": "multiclass_classification", "metric": "auc", "size_gb": 0.8},
    {"id": "ranzcr-clip-catheter-line-classification", "type": "multilabel_classification", "metric": "auc", "size_gb": 13.13},
    {"id": "siim-isic-melanoma-classification", "type": "binary_classification", "metric": "auc", "size_gb": 116.16},

    # Image To Image / Regression
    {"id": "denoising-dirty-documents", "type": "regression", "metric": "rmse", "size_gb": 0.06},
    {"id": "histopathologic-cancer-detection", "type": "binary_classification", "metric": "auc", "size_gb": 7.76},

    # Text Classification
    {"id": "detecting-insults-in-social-commentary", "type": "binary_classification", "metric": "auc", "size_gb": 0.002},
    {"id": "jigsaw-toxic-comment-classification-challenge", "type": "multilabel_classification", "metric": "auc", "size_gb": 0.06},
    {"id": "random-acts-of-pizza", "type": "binary_classification", "metric": "auc", "size_gb": 0.003},
    {"id": "spooky-author-identification", "type": "multiclass_classification", "metric": "log_loss", "size_gb": 0.002},

    # Tabular
    {"id": "new-york-city-taxi-fare-prediction", "type": "regression", "metric": "rmse", "size_gb": 5.7},
    {"id": "nomad2018-predict-transparent-conductors", "type": "regression", "metric": "rmsle", "size_gb": 0.006},
    {"id": "tabular-playground-series-dec-2021", "type": "regression", "metric": "rmse", "size_gb": 0.7},
    {"id": "tabular-playground-series-may-2022", "type": "regression", "metric": "rmse", "size_gb": 0.57},

    # Audio
    {"id": "mlsp-2013-birds", "type": "multilabel_classification", "metric": "auc", "size_gb": 0.585},
    {"id": "the-icml-2013-whale-challenge-right-whale-redux", "type": "binary_classification", "metric": "auc", "size_gb": 0.29},

    # Seq->Seq
    {"id": "text-normalization-challenge-english-language", "type": "seq2seq", "metric": "accuracy", "size_gb": 0.01},
    {"id": "text-normalization-challenge-russian-language", "type": "seq2seq", "metric": "accuracy", "size_gb": 0.01},
]


def get_competition_info(competition_id: str) -> dict:
    """Get competition info from MLEBENCH_LITE list."""
    for comp in MLEBENCH_LITE:
        if comp["id"] == competition_id:
            return comp
    return {"id": competition_id, "type": "unknown", "metric": "unknown", "size_gb": 0}


def run_evaluation(
    competition_ids: list[str],
    output_dir: str = "./mlebench_results",
    max_iterations: int = 3,
    timeout_per_component: int = 3000,
):
    """
    Run kaggle-agents evaluation on MLE-bench competitions.

    Args:
        competition_ids: List of competition IDs to evaluate
        output_dir: Directory to save results
        max_iterations: Maximum workflow iterations
        timeout_per_component: Timeout per component in seconds
    """
    try:
        from kaggle_agents.mlebench import solve_mlebench
    except ModuleNotFoundError as e:
        if e.name != "kaggle_agents":
            raise
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from kaggle_agents.mlebench import solve_mlebench

    print(f"[mlebench_eval] Starting evaluation at {datetime.now()}", flush=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    start_time = datetime.now()

    print("=" * 70, flush=True)
    print("MLE-BENCH EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Competitions: {len(competition_ids)}", flush=True)
    print(f"Max iterations: {max_iterations}", flush=True)
    print(f"Timeout per component: {timeout_per_component}s", flush=True)
    print("=" * 70, flush=True)

    for idx, comp_id in enumerate(competition_ids, 1):
        print(f"\n{'#' * 70}", flush=True)
        print(f"# [{idx}/{len(competition_ids)}] {comp_id}", flush=True)
        print(f"{'#' * 70}", flush=True)

        comp_info = get_competition_info(comp_id)
        print(f"  Problem type: {comp_info['type']}", flush=True)
        print(f"  Metric: {comp_info['metric']}", flush=True)
        print(f"  Calling solve_mlebench()...", flush=True)

        try:
            result = solve_mlebench(
                competition_id=comp_id,
                problem_type=comp_info["type"],
                evaluation_metric=comp_info["metric"],
                max_iterations=max_iterations,
                timeout_per_component=timeout_per_component,
                enable_checkpoint_recovery=True,
            )

            print(f"  solve_mlebench() returned!", flush=True)
            print(f"  Success: {result.success}", flush=True)
            print(f"  Error: {result.error}", flush=True)

            result_dict = {
                "competition_id": comp_id,
                "success": result.success,
                "valid_submission": result.valid_submission,
                "score": result.score,
                "gold_medal": result.gold_medal,
                "silver_medal": result.silver_medal,
                "bronze_medal": result.bronze_medal,
                "any_medal": bool(result.gold_medal or result.silver_medal or result.bronze_medal),
                "above_median": result.above_median,
                "execution_time": result.execution_time,
                "iterations": result.iterations,
                "components_implemented": result.components_implemented,
                "error": result.error,
            }

            if result.traceback:
                result_dict["traceback"] = result.traceback
                print(f"  Traceback:\n{result.traceback}", flush=True)

        except Exception as e:
            import traceback
            error_tb = traceback.format_exc()
            print(f"  EXCEPTION in solve_mlebench: {e}", flush=True)
            print(f"  Traceback:\n{error_tb}", flush=True)
            result_dict = {
                "competition_id": comp_id,
                "success": False,
                "error": str(e),
                "traceback": error_tb,
            }

        all_results.append(result_dict)

        # Save intermediate results
        with open(output_path / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds()

    summary = {
        "total_competitions": len(competition_ids),
        "successful": sum(1 for r in all_results if r.get("success")),
        "valid_submissions": sum(1 for r in all_results if r.get("valid_submission")),
        "gold_medals": sum(1 for r in all_results if r.get("gold_medal")),
        "silver_medals": sum(1 for r in all_results if r.get("silver_medal")),
        "bronze_medals": sum(1 for r in all_results if r.get("bronze_medal")),
        "any_medals": sum(1 for r in all_results if r.get("any_medal")),
        "above_median": sum(1 for r in all_results if r.get("above_median")),
        "total_time_seconds": total_time,
    }
    total = summary["total_competitions"] or 1
    summary["valid_submission_percentage"] = summary["valid_submissions"] / total
    summary["any_medal_percentage"] = summary["any_medals"] / total

    # Save summary
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV for easy reporting
    csv_path = output_path / "results.csv"
    fieldnames = sorted({k for row in all_results for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total competitions: {summary['total_competitions']}")
    print(f"Successful: {summary['successful']}")
    print(f"Valid submissions: {summary['valid_submissions']}")
    print(f"Gold medals: {summary['gold_medals']}")
    print(f"Silver medals: {summary['silver_medals']}")
    print(f"Bronze medals: {summary['bronze_medals']}")
    print(f"Any medal: {summary['any_medals']} ({summary['any_medal_percentage']:.1%})")
    print(f"Above median: {summary['above_median']}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"\nResults saved to: {output_path}")
    print(f"CSV saved to: {csv_path}")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="MLE-bench Evaluation for Kaggle Agents")
    parser.add_argument(
        "-c", "--competition",
        type=str,
        help="Single competition ID to evaluate"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Run all 22 MLE-bench Lite competitions"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run only small competitions (<1GB)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./mlebench_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum workflow iterations"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3000,
        help="Timeout per component in seconds"
    )

    args = parser.parse_args()

    # Determine competitions to run
    if args.competition:
        competition_ids = [args.competition]
    elif args.lite:
        competition_ids = [c["id"] for c in MLEBENCH_LITE]
    elif args.small:
        competition_ids = [c["id"] for c in MLEBENCH_LITE if c["size_gb"] < 1.0]
    else:
        # Default: run smallest competition as test
        competition_ids = ["aerial-cactus-identification"]
        print("No competition specified. Running default: aerial-cactus-identification")
        print("Use --lite for all 22 competitions, --small for <1GB competitions")

    run_evaluation(
        competition_ids=competition_ids,
        output_dir=args.output,
        max_iterations=args.max_iterations,
        timeout_per_component=args.timeout,
    )


if __name__ == "__main__":
    main()
