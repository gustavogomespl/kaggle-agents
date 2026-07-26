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
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def _run_key(row: dict) -> tuple:
    """Identity of one experimental unit: competition x seed x arm x protocol.

    The protocol fingerprint is part of the identity on purpose. Without it a
    resume happily reuses results produced by a different commit, model,
    provider, budget, or search policy, and the resulting table would mix
    protocols while looking like one experiment.
    """
    return (
        row.get("competition_id"),
        row.get("seed"),
        row.get("arm"),
        row.get("config_fingerprint"),
    )


def is_final_result(row: dict) -> bool:
    """Whether a recorded attempt is terminal and must not be rerun.

    Terminal means the agent got a real, countable outcome -- including a bad
    one. Infrastructure and harness failures are invalid attempts: they stay on
    the ledger but are eligible for rerun, because counting a 401 or a missing
    grader as a failed run would silently lower the reported rate.

    The rule never consults score, medal, or submission validity. Making resume
    depend on the outcome would turn the sweep into a search for good runs.
    """
    if row.get("terminal_status") != "completed":
        return False
    return row.get("failure_origin") in (None, "agent")


def config_fingerprint(**parts) -> str:
    """Short, stable digest of everything that defines the protocol."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_state(repo_root: Path) -> dict:
    """Commit and dirty flag, so a mid-sweep code change invalidates resume."""
    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _write_json_atomic(path: Path, payload) -> None:
    """Write via a temp file and rename, so a crash cannot truncate the ledger."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


# MLE-bench Lite competitions (22 total)
MLEBENCH_LITE = [
    # Image Classification
    {
        "id": "aerial-cactus-identification",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 0.025,
    },
    {
        "id": "aptos2019-blindness-detection",
        "type": "multiclass_classification",
        "metric": "quadratic_weighted_kappa",
        "size_gb": 10.22,
    },
    {
        "id": "dog-breed-identification",
        "type": "multiclass_classification",
        "metric": "log_loss",
        "size_gb": 0.75,
    },
    {
        "id": "dogs-vs-cats-redux-kernels-edition",
        "type": "binary_classification",
        "metric": "log_loss",
        "size_gb": 0.85,
    },
    {
        "id": "leaf-classification",
        "type": "multiclass_classification",
        "metric": "log_loss",
        "size_gb": 0.036,
    },
    {
        "id": "plant-pathology-2020-fgvc7",
        "type": "multiclass_classification",
        "metric": "auc",
        "size_gb": 0.8,
    },
    {
        "id": "ranzcr-clip-catheter-line-classification",
        "type": "multilabel_classification",
        "metric": "auc",
        "size_gb": 13.13,
    },
    {
        "id": "siim-isic-melanoma-classification",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 116.16,
    },
    # Image To Image / Regression
    {"id": "denoising-dirty-documents", "type": "regression", "metric": "rmse", "size_gb": 0.06},
    {
        "id": "histopathologic-cancer-detection",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 7.76,
    },
    # Text Classification
    {
        "id": "detecting-insults-in-social-commentary",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 0.002,
    },
    {
        "id": "jigsaw-toxic-comment-classification-challenge",
        "type": "multilabel_classification",
        "metric": "auc",
        "size_gb": 0.06,
    },
    {
        "id": "random-acts-of-pizza",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 0.003,
    },
    {
        "id": "spooky-author-identification",
        "type": "multiclass_classification",
        "metric": "log_loss",
        "size_gb": 0.002,
    },
    # Tabular
    {
        "id": "new-york-city-taxi-fare-prediction",
        "type": "regression",
        "metric": "rmse",
        "size_gb": 5.7,
    },
    {
        "id": "nomad2018-predict-transparent-conductors",
        "type": "regression",
        "metric": "rmsle",
        "size_gb": 0.006,
    },
    {
        "id": "tabular-playground-series-dec-2021",
        "type": "multiclass_classification",
        "metric": "accuracy",
        "size_gb": 0.7,
    },
    {
        "id": "tabular-playground-series-may-2022",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 0.57,
    },
    # Audio
    {
        "id": "mlsp-2013-birds",
        "type": "multilabel_classification",
        "metric": "auc",
        "size_gb": 0.585,
    },
    {
        "id": "the-icml-2013-whale-challenge-right-whale-redux",
        "type": "binary_classification",
        "metric": "auc",
        "size_gb": 0.29,
    },
    # Seq->Seq
    {
        "id": "text-normalization-challenge-english-language",
        "type": "seq2seq",
        "metric": "accuracy",
        "size_gb": 0.01,
    },
    {
        "id": "text-normalization-challenge-russian-language",
        "type": "seq2seq",
        "metric": "accuracy",
        "size_gb": 0.01,
    },
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
    wall_clock_budget_s: int | None = None,
    resume: bool = True,
):
    """
    Run kaggle-agents evaluation on MLE-bench competitions.

    Args:
        competition_ids: List of competition IDs to evaluate
        output_dir: Directory to save results
        max_iterations: Maximum workflow iterations
        timeout_per_component: Timeout per component in seconds
        wall_clock_budget_s: Cooperative agent budget per competition, in seconds
        resume: Skip competitions already completed for this seed and arm
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
    results_file = output_path / "results.json"

    from kaggle_agents.core.config import get_config, get_run_seed

    seed = get_run_seed()
    toggles = getattr(get_config(), "ablation_toggles", None)
    disabled = toggles.disabled_components() if toggles is not None else []
    arm = "full" if not disabled else "without-" + "-".join(sorted(disabled))

    fingerprint = config_fingerprint(
        git=_git_state(Path(__file__).resolve().parents[1]),
        provider=os.getenv("LLM_PROVIDER"),
        model=os.getenv("LLM_MODEL"),
        role_models={
            role: os.getenv(f"{role}_MODEL")
            for role in ("PLANNER", "DEVELOPER", "EVALUATOR")
        },
        max_iterations=max_iterations,
        timeout_per_component=timeout_per_component,
        wall_clock_budget_s=wall_clock_budget_s,
        arm=arm,
        roster=sorted(competition_ids),
    )

    # Resume: a sweep is the whole GPU budget, so a crash on competition 15
    # must not restart the 14 that already ran.
    all_results = []
    if resume and results_file.exists():
        try:
            with results_file.open(encoding="utf-8") as f:
                all_results = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"[mlebench_eval] Could not read {results_file} ({exc}); starting fresh")
            all_results = []
    done = {_run_key(row) for row in all_results if is_final_result(row)}

    start_time = datetime.now()

    print("=" * 70, flush=True)
    print("MLE-BENCH EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Competitions: {len(competition_ids)}", flush=True)
    print(f"Seed: {seed} | Arm: {arm} | Protocol: {fingerprint}", flush=True)
    print(f"Max iterations: {max_iterations}", flush=True)
    print(f"Timeout per component: {timeout_per_component}s", flush=True)
    print(
        "Wall-clock budget: "
        + (f"{wall_clock_budget_s / 3600:.2f}h" if wall_clock_budget_s else "config default"),
        flush=True,
    )
    if done:
        print(f"Resuming: {len(done)} completed attempt(s) will be skipped", flush=True)
    print("=" * 70, flush=True)

    for idx, comp_id in enumerate(competition_ids, 1):
        print(f"\n{'#' * 70}", flush=True)
        print(f"# [{idx}/{len(competition_ids)}] {comp_id}", flush=True)
        print(f"{'#' * 70}", flush=True)

        if (comp_id, seed, arm, fingerprint) in done:
            print("  Already completed for this seed/arm/protocol - skipping", flush=True)
            continue

        # Earlier invalid attempts stay on the ledger: the protocol requires
        # every attempt to remain visible, including the ones that were retried.
        comp_info = get_competition_info(comp_id)
        print(f"  Problem type: {comp_info['type']}", flush=True)
        print(f"  Metric: {comp_info['metric']}", flush=True)
        print("  Calling solve_mlebench()...", flush=True)

        try:
            result = solve_mlebench(
                competition_id=comp_id,
                problem_type=comp_info["type"],
                evaluation_metric=comp_info["metric"],
                max_iterations=max_iterations,
                timeout_per_component=timeout_per_component,
                enable_checkpoint_recovery=True,
                wall_clock_budget_s=wall_clock_budget_s,
            )

            print("  solve_mlebench() returned!", flush=True)
            print(f"  Success: {result.success}", flush=True)
            print(f"  Error: {result.error}", flush=True)

            # Compact telemetry (full event_log stays in the per-run telemetry.json)
            telemetry = getattr(result, "telemetry", None)
            if isinstance(telemetry, dict):
                telemetry = {k: v for k, v in telemetry.items() if k != "event_log"}

            provenance = (telemetry or {}).get("provenance", {})
            result_dict = {
                "competition_id": comp_id,
                "seed": seed,
                "arm": arm,
                "config_fingerprint": fingerprint,
                "run_id": provenance.get("run_id"),
                "terminal_status": "completed",
                "failure_origin": result.failure_origin,
                "attempted_at": datetime.now().isoformat(),
                "success": result.success,
                "valid_submission": result.valid_submission,
                "score": result.score,
                "gold_medal": result.gold_medal,
                "silver_medal": result.silver_medal,
                "bronze_medal": result.bronze_medal,
                "any_medal": bool(result.gold_medal or result.silver_medal or result.bronze_medal),
                "above_median": result.above_median,
                "execution_time": result.execution_time,
                "agent_execution_time": result.agent_execution_time,
                "deadline_reached": result.deadline_reached,
                "iterations": result.iterations,
                "components_implemented": result.components_implemented,
                "telemetry": telemetry,
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
            # The protocol never ran to completion, so this attempt stays on the
            # ledger but is retryable on resume.
            result_dict = {
                "competition_id": comp_id,
                "seed": seed,
                "arm": arm,
                "config_fingerprint": fingerprint,
                "run_id": None,
                "terminal_status": "harness_exception",
                "failure_origin": "harness",
                "attempted_at": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
                "traceback": error_tb,
            }

        origin = result_dict.get("failure_origin")
        if origin in {"infrastructure", "harness"}:
            print(
                f"  Invalid attempt ({origin}) - stays on the ledger and is "
                "eligible for rerun",
                flush=True,
            )

        all_results.append(result_dict)
        _write_json_atomic(results_file, all_results)

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds()

    # Rates are per seed, arm and protocol: pooling seeds would treat repeated
    # measures on the same competition as independent tasks, and pooling
    # protocols would mix experiments. The numerator is also restricted to the
    # competitions actually requested -- otherwise reusing a directory with 22
    # results while running one competition reports 22 completions against a
    # denominator of 1.
    requested = set(competition_ids)
    arm_rows = [
        r
        for r in all_results
        if r.get("seed") == seed
        and r.get("arm") == arm
        and r.get("config_fingerprint") == fingerprint
        and r.get("competition_id") in requested
    ]
    completed_rows = [r for r in arm_rows if is_final_result(r)]
    # One row per competition: a retried competition must not be counted twice.
    completed_rows = list(
        {r["competition_id"]: r for r in completed_rows}.values()
    )

    summary = {
        "seed": seed,
        "arm": arm,
        "config_fingerprint": fingerprint,
        "total_competitions": len(competition_ids),
        "completed": len(completed_rows),
        "invalid_attempts": len(arm_rows) - len(completed_rows),
        "missing": len(competition_ids) - len(completed_rows),
        "successful": sum(1 for r in completed_rows if r.get("success")),
        "valid_submissions": sum(1 for r in completed_rows if r.get("valid_submission")),
        "gold_medals": sum(1 for r in completed_rows if r.get("gold_medal")),
        "silver_medals": sum(1 for r in completed_rows if r.get("silver_medal")),
        "bronze_medals": sum(1 for r in completed_rows if r.get("bronze_medal")),
        "any_medals": sum(1 for r in completed_rows if r.get("any_medal")),
        "above_median": sum(1 for r in completed_rows if r.get("above_median")),
        "deadline_reached": sum(1 for r in completed_rows if r.get("deadline_reached")),
        "agent_gpu_hours": round(
            sum(float(r.get("agent_execution_time") or 0.0) for r in completed_rows) / 3600, 3
        ),
        "total_time_seconds": total_time,
    }
    total = summary["total_competitions"] or 1
    summary["valid_submission_percentage"] = summary["valid_submissions"] / total
    summary["any_medal_percentage"] = summary["any_medals"] / total

    # Save summary
    _write_json_atomic(output_path / "summary.json", summary)

    # Save CSV for easy reporting (nested/verbose fields stay in results.json)
    csv_path = output_path / "results.csv"
    csv_excluded = {"telemetry", "traceback"}
    all_results_csv = [
        {k: v for k, v in row.items() if k not in csv_excluded} for row in all_results
    ]
    fieldnames = sorted({k for row in all_results_csv for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_csv)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Seed / arm: {summary['seed']} / {summary['arm']}")
    print(f"Total competitions: {summary['total_competitions']}")
    print(
        f"Completed: {summary['completed']} | "
        f"invalid attempts: {summary['invalid_attempts']} | "
        f"missing: {summary['missing']}"
    )
    if summary["missing"]:
        print(
            "  Missing/invalid attempts are eligible for rerun: re-run the same "
            "command to resume."
        )
    print(f"Agent GPU-hours: {summary['agent_gpu_hours']}")
    print(f"Deadline reached: {summary['deadline_reached']}")
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
    parser.add_argument("-c", "--competition", type=str, help="Single competition ID to evaluate")
    parser.add_argument(
        "--lite", action="store_true", help="Run all 22 MLE-bench Lite competitions"
    )
    parser.add_argument("--small", action="store_true", help="Run only small competitions (<1GB)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./mlebench_results",
        help="Output directory for results",
    )
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum workflow iterations")
    parser.add_argument(
        "--timeout", type=int, default=3000, help="Timeout per component in seconds"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed (sets RUN_SEED). Recorded on every result row.",
    )
    parser.add_argument(
        "--wall-clock-budget",
        type=int,
        default=None,
        help=(
            "Cooperative agent budget per competition in seconds "
            "(0 disables the deadline)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed attempts in results.json and rerun everything",
    )

    args = parser.parse_args()

    # Must be set before kaggle_agents reads it to derive folds and seeding.
    if args.seed is not None:
        os.environ["RUN_SEED"] = str(args.seed)

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
        wall_clock_budget_s=args.wall_clock_budget,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
