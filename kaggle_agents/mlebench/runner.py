"""
MLE-bench Runner.

This module provides the main entry point for running kaggle-agents
on MLE-bench competitions with proper data handling and grading.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback as tb
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.config import get_config
from ..core.state import CompetitionInfo, create_initial_state
from .data_adapter import MLEBenchDataAdapter


# Force flush for Colab/Jupyter compatibility
console = Console(force_terminal=True)


def _log(msg: str, level: str = "INFO") -> None:
    """Log message to both Rich console and stdout for Colab compatibility."""
    # Always print to stdout for Colab
    print(f"[{level}] {msg}", flush=True)
    # Also try Rich console
    try:
        style = {"ERROR": "red", "WARN": "yellow", "INFO": "cyan"}.get(level, "white")
        console.print(f"[{style}]{msg}[/{style}]")
    except Exception:
        pass


@dataclass
class MLEBenchResult:
    """Result from MLE-bench evaluation."""

    competition_id: str
    success: bool
    submission_path: str | None = None

    # MLE-bench grading results
    valid_submission: bool = False
    score: float | None = None
    gold_medal: bool = False
    silver_medal: bool = False
    bronze_medal: bool = False
    above_median: bool = False

    # Workflow metrics
    iterations: int = 0
    components_implemented: int = 0
    execution_time: float = 0.0

    # Error info
    error: str | None = None
    traceback: str | None = None

    # Raw grading output
    grading_output: dict | None = None


class MLEBenchRunner:
    """
    Runner for MLE-bench competition evaluation.

    This class handles:
    - Loading MLE-bench prepared data
    - Running the kaggle-agents workflow (without Kaggle API download)
    - Grading submissions with mlebench grade-sample
    - Collecting metrics and results
    """

    def __init__(
        self,
        mle_cache_path: Path | None = None,
        workspace_base: Path | None = None,
    ):
        """
        Initialize MLE-bench runner.

        Args:
            mle_cache_path: Path to MLE-bench cache (default: /root/.cache/mle-bench/data)
            workspace_base: Base path for workspaces (default: /content/kaggle_competitions)
        """
        self.config = get_config()
        self.data_adapter = MLEBenchDataAdapter(mle_cache_path)
        self.workspace_base = workspace_base or Path("/content/kaggle_competitions")
        self.console = Console()

    def _display_header(
        self,
        competition_id: str,
        problem_type: str,
        evaluation_metric: str,
    ):
        """Display runner header."""
        header = f"""
[bold cyan]MLE-BENCH MODE[/bold cyan]

[bold]Competition:[/bold] {competition_id}
[bold]Problem Type:[/bold] {problem_type}
[bold]Metric:[/bold] {evaluation_metric}
[bold]Goal:[/bold] Generate valid submission for MLE-bench grading
"""
        console.print(Panel(header, border_style="cyan"))

    def _grade_submission(
        self,
        competition_id: str,
        submission_path: Path,
    ) -> dict[str, Any]:
        """
        Grade submission using MLE-bench.

        Args:
            competition_id: Competition ID
            submission_path: Path to submission CSV

        Returns:
            Grading results dictionary
        """
        console.print("\n[bold]Grading submission with MLE-bench...[/bold]")

        try:
            result = subprocess.run(
                ["mlebench", "grade-sample", str(submission_path), competition_id],
                check=False, capture_output=True,
                text=True,
                timeout=60,
            )

            output = result.stdout + result.stderr

            # Parse JSON from output
            try:
                json_start = output.find("{")
                json_end = output.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = output[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            return {
                "valid_submission": False,
                "error": f"Could not parse grading output: {output[:500]}",
            }

        except subprocess.TimeoutExpired:
            return {
                "valid_submission": False,
                "error": "Grading timeout (60s)",
            }
        except FileNotFoundError:
            return {
                "valid_submission": False,
                "error": "mlebench command not found. Install with: pip install -e /path/to/mle-bench",
            }
        except Exception as e:
            return {
                "valid_submission": False,
                "error": str(e),
            }

    def _find_submission(self, workspace: Path) -> Path | None:
        """Find submission file in workspace."""
        candidates = [
            workspace / "submission.csv",
            workspace / "sample_submission.csv",
        ]

        # Also check for backup submissions
        for f in workspace.glob("submission_*.csv"):
            candidates.append(f)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def run(
        self,
        competition_id: str,
        problem_type: str = "unknown",
        evaluation_metric: str = "unknown",
        max_iterations: int = 3,
        timeout_per_component: int = 1800,  # 30 min default for fast MLE-bench iteration
        enable_checkpoint_recovery: bool = True,
    ) -> MLEBenchResult:
        """
        Run kaggle-agents workflow on MLE-bench competition.

        Args:
            competition_id: MLE-bench competition ID
            problem_type: Problem type (classification, regression, etc.)
            evaluation_metric: Evaluation metric
            max_iterations: Maximum workflow iterations
            timeout_per_component: Timeout per component in seconds
            enable_checkpoint_recovery: Enable checkpoint recovery on timeout

        Returns:
            MLEBenchResult with all results and metrics
        """
        start_time = time.time()

        result = MLEBenchResult(
            competition_id=competition_id,
            success=False,
        )

        try:
            # Display header
            self._display_header(competition_id, problem_type, evaluation_metric)

            # Step 1: Prepare data from MLE-bench
            _log("Step 1: Preparing MLE-bench data")
            _log(f"  MLE-bench cache path: {self.data_adapter.mle_cache}")
            _log(f"  Competition: {competition_id}")

            # Check if competition is prepared
            comp_path = self.data_adapter.get_competition_path(competition_id)
            _log(f"  Checking path: {comp_path}")

            if not self.data_adapter.is_competition_prepared(competition_id):
                _log(f"Competition '{competition_id}' not prepared!", "ERROR")
                _log(f"Expected path: {comp_path / 'public'}", "ERROR")
                _log(f"Run: mlebench prepare -c {competition_id}", "ERROR")
                raise FileNotFoundError(
                    f"Competition '{competition_id}' not prepared. "
                    f"Run: mlebench prepare -c {competition_id}"
                )

            _log("  Data is prepared!")

            workspace = self.workspace_base / "competitions" / competition_id
            _log(f"  Workspace: {workspace}")

            data_info = self.data_adapter.prepare_workspace(
                competition_id=competition_id,
                workspace_path=workspace,
            )

            # Step 2: Create initial state with MLE-bench data
            _log("Step 2: Initializing workflow state")

            state = create_initial_state(
                competition_name=competition_id,
                working_dir=str(workspace),
            )

            # Update state with MLE-bench data paths
            state_paths = self.data_adapter.get_state_paths(data_info)
            for key, value in state_paths.items():
                if key in state:
                    state[key] = value

            # Set competition info
            description = self.data_adapter.read_description(data_info)
            state["competition_info"] = CompetitionInfo(
                name=competition_id,
                description=description[:2000] if description else "",
                evaluation_metric=evaluation_metric,
                problem_type=problem_type,
            )

            # Set iteration config
            state["max_iterations"] = max_iterations
            # Mark run mode so agents can switch objective/reward behavior
            state["run_mode"] = "mlebench"
            state["objective"] = "mlebench_medal"
            state["timeout_per_component"] = timeout_per_component
            state["enable_checkpoint_recovery"] = enable_checkpoint_recovery

            # MLE-bench training configuration - start aggressive like SOTA (600 epochs, patience=30)
            state["cv_folds"] = int(os.getenv("KAGGLE_AGENTS_CV_FOLDS", "5"))
            state["fast_mode"] = False  # Disabled - use adaptive epoch budget instead
            state["epoch_budget"] = int(
                os.getenv("KAGGLE_AGENTS_MAX_EPOCHS", "600")
            )  # SOTA uses 600
            state["early_stopping_patience"] = int(
                os.getenv("KAGGLE_AGENTS_PATIENCE", "60")
            )  # SOTA uses 30
            state["timeout_history"] = []  # Track timeouts for adaptive reduction
            # Use explicit target score only if provided via environment.
            target_score_env = os.getenv("KAGGLE_AGENTS_TARGET_SCORE") or os.getenv("TARGET_SCORE")
            if target_score_env:
                try:
                    state["target_score"] = float(target_score_env)
                except ValueError:
                    state["target_score"] = None
            else:
                state["target_score"] = None

            # Step 3: Run MLE-bench workflow
            _log("Step 3: Running workflow")
            _log(f"  Max iterations: {max_iterations}")
            _log(f"  Timeout per component: {timeout_per_component}s")

            from ..workflow import create_mlebench_workflow

            _log("  Creating workflow graph...")
            workflow = create_mlebench_workflow()
            config = {
                "recursion_limit": 150,
                "metadata": {
                    "competition": competition_id,
                    "mode": "mlebench",
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
            _log(f"  Iterations: {result.iterations}, Components: {result.components_implemented}")

            # Step 4: Find and grade submission
            _log("Step 4: Grading submission")

            submission_path = self._find_submission(workspace)

            if submission_path:
                result.submission_path = str(submission_path)
                _log(f"  Found submission: {submission_path.name}")

                grading = self._grade_submission(competition_id, submission_path)
                result.grading_output = grading

                result.valid_submission = grading.get("valid_submission", False)
                result.score = grading.get("score")
                result.gold_medal = grading.get("gold_medal", False)
                result.silver_medal = grading.get("silver_medal", False)
                result.bronze_medal = grading.get("bronze_medal", False)
                result.above_median = grading.get("above_median", False)

                if result.valid_submission:
                    result.success = True
                    _log(f"  Valid submission! Score: {result.score}")
                else:
                    result.error = grading.get("error", "Invalid submission")
                    _log(f"  Invalid submission: {result.error}", "WARN")
            else:
                result.error = "No submission file generated"
                _log("  No submission file found!", "ERROR")

        except Exception as e:
            result.error = str(e)
            result.traceback = tb.format_exc()
            _log(f"EXCEPTION: {e}", "ERROR")
            _log(f"Traceback:\n{result.traceback}", "ERROR")

        # Record execution time
        result.execution_time = time.time() - start_time

        # Display results
        self._display_results(result)

        return result

    def _display_results(self, result: MLEBenchResult):
        """Display evaluation results."""
        table = Table(title="MLE-bench Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green")

        table.add_row("Competition", result.competition_id)
        table.add_row("Success", "Yes" if result.success else "No")
        table.add_row("Valid Submission", "Yes" if result.valid_submission else "No")
        table.add_row("Score", f"{result.score:.4f}" if result.score else "N/A")

        medals = []
        if result.gold_medal:
            medals.append("Gold")
        if result.silver_medal:
            medals.append("Silver")
        if result.bronze_medal:
            medals.append("Bronze")
        table.add_row("Medals", ", ".join(medals) if medals else "None")

        table.add_row("Above Median", "Yes" if result.above_median else "No")
        table.add_row("Execution Time", f"{result.execution_time:.1f}s")
        table.add_row("Components", str(result.components_implemented))

        if result.error:
            table.add_row("Error", result.error[:100])

        console.print("\n")
        console.print(table)


def solve_mlebench(
    competition_id: str,
    mle_cache_path: str | None = None,
    problem_type: str = "unknown",
    evaluation_metric: str = "unknown",
    max_iterations: int = 3,
    timeout_per_component: int = 3000,
    enable_checkpoint_recovery: bool = True,
    workspace_base: str | None = None,
) -> MLEBenchResult:
    """
    Solve an MLE-bench competition.

    This is the main entry point for MLE-bench evaluation. It:
    1. Loads prepared data from MLE-bench cache
    2. Runs the kaggle-agents workflow (without Kaggle API download)
    3. Grades the submission with mlebench grade-sample
    4. Returns comprehensive results

    Args:
        competition_id: MLE-bench competition ID (e.g., 'aerial-cactus-identification')
        mle_cache_path: Path to MLE-bench cache (default: /root/.cache/mle-bench/data)
        problem_type: Problem type for the competition
        evaluation_metric: Evaluation metric used
        max_iterations: Maximum workflow iterations
        timeout_per_component: Timeout per component in seconds
        enable_checkpoint_recovery: Enable checkpoint recovery on timeout
        workspace_base: Base path for workspaces

    Returns:
        MLEBenchResult with evaluation results

    Example:
        >>> from kaggle_agents.mlebench import solve_mlebench
        >>> result = solve_mlebench(
        ...     competition_id="aerial-cactus-identification",
        ...     problem_type="binary_classification",
        ...     evaluation_metric="auc",
        ... )
        >>> print(f"Score: {result.score}, Medal: {result.gold_medal}")
    """
    runner = MLEBenchRunner(
        mle_cache_path=Path(mle_cache_path) if mle_cache_path else None,
        workspace_base=Path(workspace_base) if workspace_base else None,
    )

    return runner.run(
        competition_id=competition_id,
        problem_type=problem_type,
        evaluation_metric=evaluation_metric,
        max_iterations=max_iterations,
        timeout_per_component=timeout_per_component,
        enable_checkpoint_recovery=enable_checkpoint_recovery,
    )
