#!/usr/bin/env python3
"""
Kaggle Competition Evaluation Script.

This script runs kaggle-agents on real Kaggle competitions,
auto-submits predictions via the Kaggle API, and retrieves
the public leaderboard score.

Usage:
    python kaggle_eval.py --competition titanic
    python kaggle_eval.py --competition titanic --max-iterations 5 --auto-submit
"""

import argparse
import json
import os
import time
import traceback as tb
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force flush for Colab/Jupyter compatibility
console = Console(force_terminal=True)


def _log(msg: str, level: str = "INFO") -> None:
    """Log message to both Rich console and stdout for Colab compatibility."""
    print(f"[{level}] {msg}", flush=True)
    try:
        style = {"ERROR": "red", "WARN": "yellow", "INFO": "cyan"}.get(level, "white")
        console.print(f"[{style}]{msg}[/{style}]")
    except Exception:
        pass


@dataclass
class KaggleResult:
    """Result from a real Kaggle competition evaluation."""

    competition_id: str
    success: bool
    submission_path: str | None = None

    # Kaggle submission results
    submitted: bool = False
    public_score: float | None = None
    private_score: float | None = None
    submission_status: str | None = None
    submission_message: str | None = None

    # Workflow metrics
    iterations: int = 0
    components_implemented: int = 0
    execution_time: float = 0.0

    # Error info
    error: str | None = None
    traceback: str | None = None

    # Raw submission data
    submission_response: dict | None = None


class KaggleCompetitionRunner:
    """
    Runner for real Kaggle competition evaluation.

    This class handles:
    - Running the kaggle-agents workflow (with Kaggle API data download)
    - Auto-submitting predictions via Kaggle API
    - Retrieving public leaderboard scores
    - Collecting metrics and results
    """

    def __init__(self, workspace_base: Path | None = None):
        """
        Initialize Kaggle competition runner.

        Args:
            workspace_base: Base path for workspaces (default: /content/kaggle_competitions)
        """
        self.workspace_base = workspace_base or Path("/content/kaggle_competitions")

    def _display_header(self, competition_id: str):
        """Display runner header."""
        header = f"""
[bold green]KAGGLE COMPETITION MODE[/bold green]

[bold]Competition:[/bold] {competition_id}
[bold]Goal:[/bold] Download data, solve, and submit to Kaggle
"""
        console.print(Panel(header, border_style="green"))

    def _find_submission(self, workspace: Path) -> Path | None:
        """Find submission file in workspace."""
        candidates = [
            workspace / "submission.csv",
            workspace / "sample_submission.csv",
        ]

        for f in workspace.glob("submission_*.csv"):
            candidates.append(f)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _submit_and_poll(
        self,
        competition_id: str,
        submission_path: Path,
        message: str = "kaggle-agents auto-submission",
        poll_interval: int = 10,
        max_wait: int = 300,
    ) -> dict[str, Any]:
        """
        Submit prediction to Kaggle and poll for the score.

        Args:
            competition_id: Competition slug
            submission_path: Path to submission CSV
            message: Submission description
            poll_interval: Seconds between polling attempts
            max_wait: Maximum seconds to wait for scoring

        Returns:
            Dict with submission result and score
        """
        from kaggle_agents.tools.kaggle_api import KaggleAPIClient

        client = KaggleAPIClient()

        # Submit
        _log(f"Submitting {submission_path.name} to '{competition_id}'...")
        submit_result = client.submit_prediction(
            competition=competition_id,
            file_path=str(submission_path),
            message=message,
            quiet=False,
        )
        _log(f"Submission sent: {submit_result.get('status', 'unknown')}")

        # Poll for score
        _log("Waiting for Kaggle to score the submission...")
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                submissions = client.get_my_submissions(competition_id)
            except Exception as e:
                _log(f"Error polling submissions: {e}", "WARN")
                continue

            if not submissions:
                continue

            latest = submissions[0]
            status = latest.get("status", "unknown")

            if status == "complete":
                _log(f"Scoring complete! Public score: {latest.get('publicScore')}")
                return {
                    "submitted": True,
                    "status": status,
                    "public_score": latest.get("publicScore"),
                    "private_score": latest.get("privateScore"),
                    "date": latest.get("date"),
                    "description": latest.get("description"),
                }
            elif status == "error":
                _log(f"Submission error: {latest}", "ERROR")
                return {
                    "submitted": True,
                    "status": "error",
                    "error": f"Kaggle scoring error: {latest}",
                }
            else:
                _log(f"  Status: {status} (waiting... {elapsed}s/{max_wait}s)")

        _log("Timed out waiting for score", "WARN")
        return {
            "submitted": True,
            "status": "timeout",
            "error": f"Score not available after {max_wait}s",
        }

    def run(
        self,
        competition_id: str,
        max_iterations: int = 3,
        timeout_per_component: int = 3000,
        auto_submit: bool = True,
        enable_checkpoint_recovery: bool = True,
    ) -> KaggleResult:
        """
        Run kaggle-agents workflow on a real Kaggle competition.

        Args:
            competition_id: Kaggle competition slug (e.g. 'titanic')
            max_iterations: Maximum workflow iterations
            timeout_per_component: Timeout per component in seconds
            auto_submit: Whether to auto-submit to Kaggle
            enable_checkpoint_recovery: Enable checkpoint recovery on timeout

        Returns:
            KaggleResult with all results and metrics
        """
        start_time = time.time()

        result = KaggleResult(
            competition_id=competition_id,
            success=False,
        )

        try:
            # Lazy imports to support path injection from notebooks
            try:
                from kaggle_agents.core.config import get_config
                from kaggle_agents.core.state import CompetitionInfo, create_initial_state
                from kaggle_agents.workflow import compile_workflow
            except ModuleNotFoundError:
                import sys

                repo_root = Path(__file__).resolve().parents[1]
                sys.path.insert(0, str(repo_root))
                from kaggle_agents.core.config import get_config
                from kaggle_agents.core.state import CompetitionInfo, create_initial_state
                from kaggle_agents.workflow import compile_workflow

            self._display_header(competition_id)

            workspace = self.workspace_base / "competitions" / competition_id
            workspace.mkdir(parents=True, exist_ok=True)

            # Step 1: Initialize state
            _log("Step 1: Initializing workflow state")

            state = create_initial_state(
                competition_name=competition_id,
                working_dir=str(workspace),
            )

            # Set competition info (the workflow's data_download_node will
            # fetch real metadata from Kaggle API)
            state["competition_info"] = CompetitionInfo(
                name=competition_id,
                description="",
                evaluation_metric="unknown",
                problem_type="unknown",
            )

            state["max_iterations"] = max_iterations
            state["run_mode"] = "kaggle"
            state["objective"] = "top20"
            state["timeout_per_component"] = timeout_per_component
            state["enable_checkpoint_recovery"] = enable_checkpoint_recovery

            # Training configuration from environment
            state["cv_folds"] = int(os.getenv("KAGGLE_AGENTS_CV_FOLDS", "5"))
            state["fast_mode"] = False
            state["epoch_budget"] = int(os.getenv("KAGGLE_AGENTS_MAX_EPOCHS", "600"))
            state["early_stopping_patience"] = int(
                os.getenv("KAGGLE_AGENTS_PATIENCE", "60")
            )
            state["timeout_history"] = []

            target_score_env = os.getenv("KAGGLE_AGENTS_TARGET_SCORE") or os.getenv(
                "TARGET_SCORE"
            )
            if target_score_env:
                try:
                    state["target_score"] = float(target_score_env)
                except ValueError:
                    state["target_score"] = None
            else:
                state["target_score"] = None

            # Step 2: Run the main workflow (includes data_download_node)
            _log("Step 2: Running workflow (with data download)")
            _log(f"  Max iterations: {max_iterations}")
            _log(f"  Timeout per component: {timeout_per_component}s")

            _log("  Creating workflow graph...")
            workflow = compile_workflow()

            agent_cfg = get_config()
            recursion_limit = getattr(
                getattr(agent_cfg, "iteration", None),
                "langgraph_recursion_limit",
                300,
            )
            config = {
                "recursion_limit": recursion_limit,
                "metadata": {
                    "competition": competition_id,
                    "mode": "kaggle",
                    "timeout_per_component": timeout_per_component,
                    "enable_checkpoint_recovery": enable_checkpoint_recovery,
                },
            }

            _log("  Invoking workflow... (this may take a while)")
            final_state = workflow.invoke(state, config)
            _log("  Workflow completed!")

            # Collect workflow metrics
            dev_results = final_state.get("development_results", [])
            result.iterations = final_state.get("current_iteration", 0)
            result.components_implemented = len(dev_results)
            _log(
                f"  Iterations: {result.iterations}, "
                f"Components: {result.components_implemented}"
            )

            # Step 3: Find submission
            _log("Step 3: Locating submission file")
            submission_path = self._find_submission(workspace)

            if submission_path:
                result.submission_path = str(submission_path)
                result.success = True
                _log(f"  Found submission: {submission_path.name}")

                # Step 4: Auto-submit to Kaggle
                if auto_submit:
                    _log("Step 4: Auto-submitting to Kaggle")
                    try:
                        submit_result = self._submit_and_poll(
                            competition_id=competition_id,
                            submission_path=submission_path,
                        )
                        result.submitted = submit_result.get("submitted", False)
                        result.submission_status = submit_result.get("status")
                        result.public_score = submit_result.get("public_score")
                        result.private_score = submit_result.get("private_score")
                        result.submission_response = submit_result

                        if result.public_score is not None:
                            _log(f"  Public score: {result.public_score}")
                        if submit_result.get("error"):
                            _log(
                                f"  Submit issue: {submit_result['error']}", "WARN"
                            )
                    except Exception as e:
                        _log(f"  Auto-submit failed: {e}", "ERROR")
                        result.submission_message = f"Submit error: {e}"
                else:
                    _log("Step 4: Auto-submit disabled (use --auto-submit to enable)")
            else:
                result.error = "No submission file generated"
                _log("  No submission file found!", "ERROR")

        except Exception as e:
            result.error = str(e)
            result.traceback = tb.format_exc()
            _log(f"EXCEPTION: {e}", "ERROR")
            _log(f"Traceback:\n{result.traceback}", "ERROR")

        result.execution_time = time.time() - start_time

        self._display_results(result)

        return result

    def _display_results(self, result: KaggleResult):
        """Display evaluation results."""
        table = Table(
            title="Kaggle Competition Results",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", max_width=50)

        table.add_row("Competition", result.competition_id)
        table.add_row("Success", "Yes" if result.success else "No")
        table.add_row("Submitted", "Yes" if result.submitted else "No")

        if result.public_score is not None:
            table.add_row("Public Score", f"{result.public_score}")
        if result.private_score is not None:
            table.add_row("Private Score", f"{result.private_score}")

        table.add_row(
            "Submission Status", result.submission_status or "N/A"
        )
        table.add_row("Execution Time", f"{result.execution_time:.1f}s")
        table.add_row("Iterations", str(result.iterations))
        table.add_row("Components", str(result.components_implemented))

        if result.error:
            error_display = (
                result.error[:200] if len(result.error) > 200 else result.error
            )
            table.add_row("Error", error_display)

        console.print("\n")
        console.print(table)


def solve_kaggle(
    competition_id: str,
    max_iterations: int = 3,
    timeout_per_component: int = 3000,
    auto_submit: bool = True,
    enable_checkpoint_recovery: bool = True,
    workspace_base: str | None = None,
) -> KaggleResult:
    """
    Solve a real Kaggle competition.

    This is the main entry point. It:
    1. Runs the kaggle-agents workflow (with Kaggle API data download)
    2. Finds the generated submission.csv
    3. Auto-submits to Kaggle via API
    4. Polls for the public leaderboard score
    5. Returns comprehensive results

    Args:
        competition_id: Kaggle competition slug (e.g. 'titanic')
        max_iterations: Maximum workflow iterations
        timeout_per_component: Timeout per component in seconds
        auto_submit: Whether to auto-submit to Kaggle
        enable_checkpoint_recovery: Enable checkpoint recovery on timeout
        workspace_base: Base path for workspaces

    Returns:
        KaggleResult with evaluation results

    Example:
        >>> from notebooks.kaggle_eval import solve_kaggle
        >>> result = solve_kaggle("titanic", max_iterations=3, auto_submit=True)
        >>> print(f"Score: {result.public_score}")
    """
    runner = KaggleCompetitionRunner(
        workspace_base=Path(workspace_base) if workspace_base else None,
    )

    return runner.run(
        competition_id=competition_id,
        max_iterations=max_iterations,
        timeout_per_component=timeout_per_component,
        auto_submit=auto_submit,
        enable_checkpoint_recovery=enable_checkpoint_recovery,
    )


def run_evaluation(
    competition_ids: list[str],
    output_dir: str = "./kaggle_results",
    max_iterations: int = 3,
    timeout_per_component: int = 3000,
    auto_submit: bool = True,
):
    """
    Run kaggle-agents evaluation on one or more real Kaggle competitions.

    Args:
        competition_ids: List of competition slugs to evaluate
        output_dir: Directory to save results
        max_iterations: Maximum workflow iterations
        timeout_per_component: Timeout per component in seconds
        auto_submit: Whether to auto-submit to Kaggle
    """
    print(f"[kaggle_eval] Starting evaluation at {datetime.now()}", flush=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    start_time = datetime.now()

    print("=" * 70, flush=True)
    print("KAGGLE COMPETITION EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Competitions: {len(competition_ids)}", flush=True)
    print(f"Max iterations: {max_iterations}", flush=True)
    print(f"Timeout per component: {timeout_per_component}s", flush=True)
    print(f"Auto-submit: {auto_submit}", flush=True)
    print("=" * 70, flush=True)

    for idx, comp_id in enumerate(competition_ids, 1):
        print(f"\n{'#' * 70}", flush=True)
        print(f"# [{idx}/{len(competition_ids)}] {comp_id}", flush=True)
        print(f"{'#' * 70}", flush=True)

        try:
            result = solve_kaggle(
                competition_id=comp_id,
                max_iterations=max_iterations,
                timeout_per_component=timeout_per_component,
                auto_submit=auto_submit,
            )

            print("  solve_kaggle() returned!", flush=True)
            print(f"  Success: {result.success}", flush=True)

            result_dict = {
                "competition_id": comp_id,
                "success": result.success,
                "submitted": result.submitted,
                "public_score": result.public_score,
                "private_score": result.private_score,
                "submission_status": result.submission_status,
                "execution_time": result.execution_time,
                "iterations": result.iterations,
                "components_implemented": result.components_implemented,
                "error": result.error,
            }

            if result.traceback:
                result_dict["traceback"] = result.traceback
                print(f"  Traceback:\n{result.traceback}", flush=True)

        except Exception as e:
            error_tb = tb.format_exc()
            print(f"  EXCEPTION in solve_kaggle: {e}", flush=True)
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
        "submitted": sum(1 for r in all_results if r.get("submitted")),
        "scored": sum(
            1 for r in all_results if r.get("public_score") is not None
        ),
        "total_time_seconds": total_time,
    }
    total = summary["total_competitions"] or 1
    summary["success_rate"] = summary["successful"] / total

    # Save summary
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total competitions: {summary['total_competitions']}")
    print(f"Successful: {summary['successful']}")
    print(f"Submitted: {summary['submitted']}")
    print(f"Scored: {summary['scored']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"\nResults saved to: {output_path}")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle Competition Evaluation for Kaggle Agents"
    )
    parser.add_argument(
        "-c",
        "--competition",
        type=str,
        required=True,
        help="Kaggle competition slug (e.g. 'titanic')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./kaggle_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum workflow iterations",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3000,
        help="Timeout per component in seconds",
    )
    parser.add_argument(
        "--auto-submit",
        action="store_true",
        default=False,
        help="Auto-submit predictions to Kaggle",
    )

    args = parser.parse_args()

    run_evaluation(
        competition_ids=[args.competition],
        output_dir=args.output,
        max_iterations=args.max_iterations,
        timeout_per_component=args.timeout,
        auto_submit=args.auto_submit,
    )


if __name__ == "__main__":
    main()
