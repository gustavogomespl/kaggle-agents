"""
MLE-bench Runner.

This module provides the main entry point for running kaggle-agents
on MLE-bench competitions with proper data handling and grading.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import traceback as tb
import uuid
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.config import get_config, get_run_seed
from ..core.state import CompetitionInfo, create_initial_state
from ..utils.contamination import derive_competition_identity_aliases
from ..utils.submission_artifacts import verified_accepted_submission
from ..utils.telemetry import collect_run_provenance, summarize_run_telemetry
from .data_adapter import MLEBenchDataAdapter


# Force flush for Colab/Jupyter compatibility
console = Console(force_terminal=True)

_UNKNOWN_METRIC_NAMES = frozenset({"", "auto", "unknown", "none", "null", "n/a"})
_SUPPORTED_METRIC_ALIASES = {
    # Canonical aliases and public MLE-bench grader names used by the Lite split.
    "auc": "auc",
    "auc roc": "auc",
    "roc auc": "auc",
    "column wise auc": "auc",
    "column wise roc auc": "auc",
    "mean column wise auc": "auc",
    "mean column wise roc auc": "auc",
    "log loss": "log_loss",
    "logloss": "log_loss",
    "multi class log loss": "log_loss",
    "rmse": "rmse",
    "root mean squared error": "rmse",
    "rmsle": "rmsle",
    "root mean squared logarithmic error": "rmsle",
    "mean column wise rmsle": "rmsle",
    "accuracy": "accuracy",
    "classification accuracy": "accuracy",
    "multi class classification accuracy": "accuracy",
    "qwk": "quadratic_weighted_kappa",
    "quadratic weighted kappa": "quadratic_weighted_kappa",
}


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

    # Run telemetry (guardrail interventions, recovery routes, search audit)
    telemetry: dict | None = None


@dataclass(frozen=True)
class MetricResolution:
    """Auditable mapping from public metric metadata to a host scorer."""

    canonical_name: str
    raw_name: str
    source: str


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

    @staticmethod
    def _normalize_evaluation_metric(raw_metric: str) -> str:
        """Map an exact supported public alias to the canonical host contract."""
        raw = str(raw_metric or "").strip()
        key = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
        canonical = _SUPPORTED_METRIC_ALIASES.get(key)
        if canonical is None:
            raise RuntimeError(
                "MLE-bench metric preflight failed before workflow execution: "
                f"metric {raw!r} is not supported by the host-side canonical "
                "OOF scorer. Pass one of: auc, log_loss, rmse, rmsle, accuracy, "
                "quadratic_weighted_kappa."
            )
        return canonical

    @staticmethod
    def _resolve_evaluation_metric(
        competition_id: str,
        evaluation_metric: str | None,
    ) -> MetricResolution:
        """Resolve the public metric contract before any workflow work starts.

        The installed MLE-bench registry is the benchmark's public, versioned
        source of truth: each competition config declares ``grader.name``.
        Candidate stdout, labels, sample values, and private grading output are
        deliberately excluded as metric sources.
        """
        requested = str(evaluation_metric or "").strip()
        if requested.lower() not in _UNKNOWN_METRIC_NAMES:
            return MetricResolution(
                canonical_name=MLEBenchRunner._normalize_evaluation_metric(requested),
                raw_name=requested,
                source="explicit_argument",
            )

        try:
            registry_module = import_module("mlebench.registry")
            registry = registry_module.registry
            competition = registry.get_competition(competition_id)
            grader = getattr(competition, "grader", None)
            resolved = str(getattr(grader, "name", "") or "").strip()
        except (Exception, SystemExit) as exc:
            raise RuntimeError(
                "MLE-bench metric preflight failed before workflow execution: "
                f"could not read the public registry entry for {competition_id!r} "
                f"({exc}). Pass evaluation_metric explicitly or install the "
                "matching MLE-bench package."
            ) from exc

        if resolved.lower() in _UNKNOWN_METRIC_NAMES:
            raise RuntimeError(
                "MLE-bench metric preflight failed before workflow execution: "
                f"the public registry entry for {competition_id!r} has no "
                "declared grader name. Pass evaluation_metric explicitly."
            )

        canonical = MLEBenchRunner._normalize_evaluation_metric(resolved)
        _log(
            "  Metric resolved from public MLE-bench registry: "
            f"{resolved} -> {canonical}"
        )
        return MetricResolution(
            canonical_name=canonical,
            raw_name=resolved,
            source="mlebench_public_registry",
        )

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
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = result.stdout + result.stderr

            # Check for common MLE-bench infrastructure errors
            if "Private directory is empty" in output:
                return {
                    "valid_submission": False,
                    "error": (
                        "MLE-bench private directory is empty - ground truth data not available for grading. "
                        f"Run 'mlebench prepare {competition_id}' to download full dataset."
                    ),
                    "grading_unavailable": True,
                }

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
                "error": f"Could not parse mlebench output (exit={result.returncode}): {output[:500]}",
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

    def _ablation_label(self) -> str:
        """Return a stable path component for the active system ablation."""
        toggles = getattr(self.config, "ablation_toggles", None)
        disabled = toggles.disabled_components() if toggles is not None else []
        return "full" if not disabled else "without-" + "-".join(sorted(disabled))

    def _create_run_workspace(
        self,
        competition_id: str,
        random_seed: int,
    ) -> tuple[str, Path]:
        """Create an empty workspace unique to this run, seed, and ablation."""
        competition_dir = re.sub(r"[^A-Za-z0-9._-]+", "-", competition_id).strip(".-")
        if not competition_dir:
            raise ValueError("Competition ID does not contain a safe workspace name")

        run_id = uuid.uuid4().hex
        workspace = (
            self.workspace_base
            / "runs"
            / competition_dir
            / f"seed-{random_seed}"
            / self._ablation_label()
            / run_id
        )
        workspace.mkdir(parents=True, exist_ok=False)
        return run_id, workspace

    def _find_submission(
        self,
        workspace: Path,
        final_state: dict[str, Any] | None = None,
    ) -> Path | None:
        """Return only the hash-verified artifact explicitly accepted in this run."""
        if final_state is None:
            return None
        return verified_accepted_submission(final_state, workspace)

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
            forbidden_domain_overrides = {
                name: os.getenv(name)
                for name in (
                    "KAGGLE_AGENTS_FORCE_DATA_TYPE",
                    "KAGGLE_AGENTS_DATA_TYPE",
                    "KAGGLE_AGENTS_FORCE_DOMAIN",
                )
                if os.getenv(name)
            }
            if forbidden_domain_overrides:
                configured = ", ".join(
                    f"{name}={value!r}" for name, value in forbidden_domain_overrides.items()
                )
                raise RuntimeError(
                    "MLE-bench forbids manual domain overrides because they "
                    f"create a task-specific hint channel: {configured}. "
                    "Unset them and rerun."
                )

            target_score_hints = {
                name: os.getenv(name)
                for name in (
                    "KAGGLE_AGENTS_TARGET_SCORE",
                    "TARGET_SCORE",
                )
                if os.getenv(name)
            }
            if target_score_hints:
                configured = ", ".join(
                    f"{name}={value!r}" for name, value in target_score_hints.items()
                )
                raise RuntimeError(
                    "MLE-bench forbids manual target-score hints because they "
                    f"can encode task-specific leaderboard knowledge: {configured}. "
                    "Unset them and let canonical CV drive iteration."
                )

            metric_resolution = self._resolve_evaluation_metric(
                competition_id,
                evaluation_metric,
            )
            evaluation_metric = metric_resolution.canonical_name

            # Display header
            self._display_header(competition_id, problem_type, evaluation_metric)

            # Step 1: Prepare data from MLE-bench
            _log("Step 1: Preparing MLE-bench data")
            _log(f"  MLE-bench cache path: {self.data_adapter.mle_cache}")
            _log(f"  Competition: {competition_id}")

            # Check if competition is prepared
            comp_path = self.data_adapter.get_competition_path(competition_id)
            _log(f"  Checking path: {comp_path}")
            private_path_present = (comp_path / "private").exists()
            if private_path_present:
                _log(
                    "  The in-process runner can see the MLE-bench private-data "
                    "directory. Generated code is guarded and receives only the "
                    "staged public workspace, but this is not OS-level isolation. "
                    "For publication runs, execute the agent in a container or "
                    "mount namespace without private labels and grade externally.",
                    "WARN",
                )

            if not self.data_adapter.is_competition_prepared(competition_id):
                _log(f"Competition '{competition_id}' not in MLE-bench cache", "WARN")
                _log("Attempting auto-download from Kaggle API...", "INFO")

                # Try to auto-prepare via Kaggle API
                if self.data_adapter._auto_prepare_via_kaggle_api(competition_id):
                    _log("Auto-download successful!", "INFO")
                else:
                    _log(f"Competition '{competition_id}' could not be prepared!", "ERROR")
                    _log(f"Expected path: {comp_path / 'public'}", "ERROR")
                    _log(f"Run: mlebench prepare -c {competition_id}", "ERROR")
                    _log(
                        "Or ensure Kaggle credentials are configured (~/.kaggle/kaggle.json)",
                        "ERROR",
                    )
                    raise FileNotFoundError(
                        f"Competition '{competition_id}' not prepared and auto-download failed.\n"
                        f"Run: mlebench prepare -c {competition_id}\n"
                        f"Or configure Kaggle API credentials."
                    )

            _log("  Data is prepared!")

            random_seed = get_run_seed()
            run_id, workspace = self._create_run_workspace(competition_id, random_seed)
            _log(f"  Run ID: {run_id}")
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
            identity_aliases, identity_alias_evidence = derive_competition_identity_aliases(
                competition_id,
                description,
            )
            # Configurable description limit (default 14000 to preserve format details)
            try:
                max_desc_len = int(os.getenv("KAGGLE_AGENTS_MAX_DESC_LENGTH", "14000"))
            except ValueError:
                max_desc_len = 14000
            state["competition_info"] = CompetitionInfo(
                name=competition_id,
                description=description[:max_desc_len] if description else "",
                evaluation_metric=evaluation_metric,
                problem_type=problem_type,
                identity_aliases=identity_aliases,
                identity_alias_evidence=identity_alias_evidence,
            )

            # Set iteration config
            state["max_iterations"] = max_iterations
            # Mark run mode so agents can apply the formal isolation protocol.
            # Keep the model-facing objective target-agnostic: medal thresholds
            # and benchmark branding are evaluation metadata, not planning
            # evidence.
            state["run_mode"] = "mlebench"
            state["mlebench_cache_path"] = str(
                self.data_adapter.mle_cache.resolve()
            )
            state["objective"] = "fixed_budget_public_cv"
            state["run_id"] = run_id
            state["timeout_per_component"] = timeout_per_component
            state["enable_checkpoint_recovery"] = enable_checkpoint_recovery

            # MLE-bench training configuration. Epochs are an upper bound;
            # measured throughput, early stopping, and the component deadline
            # determine how much training is actually feasible.
            state["cv_folds"] = int(os.getenv("KAGGLE_AGENTS_CV_FOLDS", "5"))
            state["random_seed"] = random_seed
            fast_mode_env = str(os.getenv("KAGGLE_AGENTS_FAST_MODE", "")).strip().lower()
            state["fast_mode"] = (
                fast_mode_env in {"1", "true", "yes"} or timeout_per_component <= 1200
            )
            state["max_components"] = int(os.getenv("KAGGLE_AGENTS_MAX_COMPONENTS", "3"))
            state["epoch_budget"] = int(os.getenv("KAGGLE_AGENTS_MAX_EPOCHS", "600"))
            state["early_stopping_patience"] = int(os.getenv("KAGGLE_AGENTS_PATIENCE", "60"))
            state["timeout_history"] = []  # Track timeouts for adaptive reduction
            # Benchmark iteration is driven only by public canonical CV. A
            # leaderboard-derived target would be a task-specific hint channel.
            state["target_score"] = None

            # Step 3: Run MLE-bench workflow
            _log("Step 3: Running workflow")
            _log(f"  Max iterations: {max_iterations}")
            _log(f"  Timeout per component: {timeout_per_component}s")
            toggles = getattr(get_config(), "ablation_toggles", None)
            if toggles and toggles.disabled_components():
                _log(f"  ABLATION active - disabled: {toggles.disabled_components()}", "WARN")

            from ..workflow import create_mlebench_workflow

            _log("  Creating workflow graph...")
            workflow = create_mlebench_workflow()
            # Use centralized recursion_limit from config (default 300)
            agent_cfg = get_config()
            recursion_limit = getattr(
                getattr(agent_cfg, "iteration", None), "langgraph_recursion_limit", 300
            )
            config = {
                "recursion_limit": recursion_limit,
                "metadata": {
                    "competition": competition_id,
                    "mode": "mlebench",
                    "run_id": run_id,
                    "workspace": str(workspace),
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

            # Telemetry: per-run measurements for ablation/guardrail analysis
            try:
                telemetry = summarize_run_telemetry(final_state)
                telemetry["provenance"] = collect_run_provenance(
                    self.config,
                    Path(__file__).resolve().parents[2],
                    competition=competition_id,
                    problem_type=problem_type,
                    evaluation_metric=evaluation_metric,
                    evaluation_metric_raw=metric_resolution.raw_name,
                    evaluation_metric_source=metric_resolution.source,
                    max_iterations=max_iterations,
                    timeout_per_component=timeout_per_component,
                    checkpoint_recovery=enable_checkpoint_recovery,
                    cv_folds=state["cv_folds"],
                    random_seed=state["random_seed"],
                    epoch_budget=state["epoch_budget"],
                    early_stopping_patience=state["early_stopping_patience"],
                    fast_mode=state["fast_mode"],
                    max_components=state["max_components"],
                    generated_code_boundary={
                        "public_inputs_staged_in_run_workspace": True,
                        "credentials_removed_from_child_environment": True,
                        "python_audit_hook_enabled": True,
                        "private_path_present_in_runner_namespace": private_path_present,
                        "os_private_label_isolation": ("not_enforced_by_in_process_runner"),
                        "publication_protocol": (
                            "run_agent_without_private_labels_then_grade_externally"
                        ),
                    },
                )
                telemetry["provenance"]["run_id"] = run_id
                telemetry["provenance"]["workspace"] = str(workspace)
                # Config-based view of the toggles (event-based detection alone
                # misses a toggle whose env var was set but never took effect)
                toggles = getattr(get_config(), "ablation_toggles", None)
                if toggles is not None:
                    telemetry.setdefault("ablation", {})["disabled_components_config"] = (
                        toggles.disabled_components()
                    )
                result.telemetry = telemetry
                telemetry_path = workspace / "telemetry.json"
                with telemetry_path.open("w", encoding="utf-8") as f:
                    json.dump(telemetry, f, indent=2, default=str)
                _log(f"  Telemetry written: {telemetry_path}")
                search_status = telemetry.get("search", {})
                if not search_status.get("eligible_retrieved", False):
                    reason = search_status.get("eligibility_reason") or "not_attempted"
                    _log(
                        "  External search retrieved no eligible source "
                        f"({reason}). Do not aggregate this run as a full-search "
                        "treatment without reporting or rerunning it.",
                        "WARN",
                    )
            except Exception as telemetry_err:
                _log(f"  Telemetry write failed (non-fatal): {telemetry_err}", "WARN")

            # Step 4: Find and grade submission
            _log("Step 4: Grading submission")

            if not final_state.get("workflow_valid", True):
                result.error = final_state.get("submission_validation_error") or final_state.get(
                    "termination_reason", "Workflow ended with an invalid candidate"
                )
                _log(f"  Grading blocked (fail-closed): {result.error}", "ERROR")
                submission_path = None
            else:
                submission_path = self._find_submission(workspace, final_state)

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
            elif final_state.get("workflow_valid", True):
                result.error = "No hash-verified accepted submission generated in this run"
                _log(f"  {result.error}", "ERROR")

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
        table.add_column("Value", style="green", max_width=50)

        table.add_row("Competition", result.competition_id)
        table.add_row("Success", "Yes" if result.success else "No")

        # Check if grading was unavailable (e.g., private directory empty)
        grading_unavailable = result.grading_output and result.grading_output.get(
            "grading_unavailable", False
        )
        if grading_unavailable:
            table.add_row("Grading", "Unavailable (no ground truth)")
            table.add_row("Valid Submission", "Unknown")
            table.add_row("Score", "N/A (grading unavailable)")
        else:
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

        if result.telemetry is not None:
            search_status = result.telemetry.get("search", {})
            if search_status.get("eligible_retrieved", False):
                gain_status = search_status.get(
                    "downstream_gain_status",
                    "unknown_not_measured",
                )
                search_display = f"Eligible sources retrieved (gain: {gain_status})"
            elif search_status.get("attempted", False):
                reason = search_status.get("eligibility_reason") or "no eligible sources"
                search_display = f"No eligible sources ({reason})"
            else:
                reason = search_status.get("eligibility_reason") or "not attempted"
                search_display = f"Not attempted ({reason})"
            table.add_row("External Search", search_display)

        if result.error:
            # Show more of the error, wrapped properly
            error_display = result.error[:200] if len(result.error) > 200 else result.error
            table.add_row("Error", error_display)

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
        competition_id: Opaque MLE-bench competition slug
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
        ...     competition_id="competition-slug",
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
