"""
Submission Agent for Kaggle Competition Upload and Monitoring.

This agent handles submission creation, Kaggle upload, leaderboard monitoring,
and score-based iteration decisions.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import (
    calculate_score_improvement,
    compare_scores,
    get_config,
    metric_reads_rows_as_distribution,
)
from ..core.state import KaggleState, SubmissionResult
from ..utils.csv_utils import read_csv_auto
from ..utils.submission_artifacts import (
    restore_accepted_submission,
    sha256_file,
    snapshot_accepted_submission,
    verified_accepted_submission,
    verified_best_candidate_submission,
)


# Provenance sources meaning "the host recomputed this exact artifact's score
# from canonical OOF". Kept distinct so an audit can tell a model combination
# apart from a tuned decision rule over a single model.
_HOST_OOF_SCORE_SOURCES = frozenset({"host_oof_ensemble", "host_oof_postprocessing"})


class SubmissionAgent:
    """
    Agent responsible for Kaggle submission and monitoring.

    Features:
    - Submission file validation
    - Kaggle API upload
    - Leaderboard score fetching
    - Percentile calculation
    - Score-based iteration decisions
    """

    _TERMINAL_FAILURE_STATUSES = frozenset(
        {"error", "failed", "failure", "cancelled", "canceled", "invalid", "rejected"}
    )

    def __init__(self):
        """Initialize the submission agent."""
        self.config = get_config()
        self.kaggle_api: Any | None = None
        self.authenticated = False
        self._current_metric_name = ""

    def _ensure_kaggle_api(self) -> bool:
        """Initialize and authenticate Kaggle lazily when an upload needs it."""
        if self.kaggle_api is not None:
            return self.authenticated

        try:
            # Kaggle's package can raise SystemExit while importing when credentials
            # are absent, so both import and authentication must stay inside this guard.
            from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            self.authenticated = True
        except (Exception, SystemExit):
            self.kaggle_api = None
            self.authenticated = False
        return self.authenticated

    @staticmethod
    def _finite_score(value: Any) -> float | None:
        """Return a finite numeric score without accepting booleans."""
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if np.isfinite(score) else None

    @classmethod
    def _resolve_mlebench_cv_provenance(
        cls,
        state: KaggleState,
        working_dir: Path,
        submission_path: Path,
    ) -> tuple[float | None, str | None, str | None]:
        """Resolve host-side CV evidence for the exact submitted artifact.

        Generic progress fields such as ``current_performance_score`` and
        ``baseline_cv_score`` are deliberately excluded: they do not prove
        which model produced ``submission.csv``. A score is reportable only
        when an exact artifact hash binds it to either:

        * a host-scored ensemble; or
        * a robustness-approved component with a host-recomputed trusted score;
        * a prior accepted snapshot carrying one of those provenances.
        """
        try:
            current_digest = sha256_file(submission_path)
        except OSError:
            return None, None, None

        ensemble_score = cls._finite_score(state.get("ensemble_oof_score"))
        ensemble_source = str(state.get("ensemble_score_source") or "")
        if (
            ensemble_score is not None
            and str(state.get("ensemble_submission_sha256") or "").lower()
            == current_digest
            and state.get("ensemble_submission_owner") == "ensemble"
            # Both sources mean the same thing: the host recomputed this exact
            # artifact's score from canonical OOF. They are kept distinct so the
            # audit can tell a combination apart from a tuned decision rule.
            and ensemble_source in _HOST_OOF_SCORE_SOURCES
        ):
            return ensemble_score, ensemble_source, "ensemble"

        best_snapshot = verified_best_candidate_submission(state, working_dir)
        owner = str(state.get("best_candidate_submission_component_name") or "")
        approvals = state.get("robustness_approved_components") or {}
        trusted_scores = state.get("trusted_component_scores") or {}
        if (
            best_snapshot is not None
            and current_digest
            == str(state.get("best_candidate_submission_sha256") or "").lower()
            and owner
            and isinstance(approvals, dict)
            and approvals.get(owner) is True
            and isinstance(trusted_scores, dict)
        ):
            raw_score = trusted_scores.get(owner)
            if isinstance(raw_score, dict):
                raw_score = raw_score.get("score", raw_score.get("cv_score"))
            component_score = cls._finite_score(raw_score)
            if component_score is not None:
                return component_score, "trusted_component_scores", owner

        accepted_snapshot = verified_accepted_submission(state, working_dir)
        accepted_score = cls._finite_score(
            state.get("accepted_submission_cv_score")
        )
        accepted_source = str(
            state.get("accepted_submission_score_source") or ""
        )
        accepted_owner = str(
            state.get("accepted_submission_score_owner") or ""
        )
        accepted_provenance_valid = (
            accepted_source in _HOST_OOF_SCORE_SOURCES
            and accepted_owner == "ensemble"
        )
        if accepted_source == "trusted_component_scores" and accepted_owner:
            accepted_provenance_valid = (
                isinstance(approvals, dict)
                and approvals.get(accepted_owner) is True
            )
        if (
            accepted_snapshot is not None
            and accepted_score is not None
            and accepted_provenance_valid
            and current_digest
            == str(state.get("accepted_submission_sha256") or "").lower()
        ):
            return accepted_score, accepted_source, accepted_owner

        return None, None, None

    def _keep_better_accepted_submission(
        self,
        state: KaggleState,
        working_dir: Path,
        submission_path: Path,
        cv_score: float | None,
    ) -> dict[str, Any] | None:
        """Keep the best accepted artifact of the run, not the most recent one.

        Every iteration used to snapshot unconditionally, so a refinement that
        made things worse still became the graded artifact: iteration 1 at CV
        0.84 was replaced by iteration 3 at 0.79. The comparison uses only the
        hash-bound CV provenance already resolved for each artifact -- no
        leaderboard, medal, or test-set signal is involved.

        Returns state updates when the previous artifact is kept, or ``None``
        to let the caller accept the current one.
        """
        previous_score = self._finite_score(state.get("accepted_submission_cv_score"))
        if previous_score is None:
            # Nothing comparable was accepted yet (including runs whose domain
            # has no canonical labels and therefore no scored lane).
            return None

        previous_snapshot = verified_accepted_submission(state, working_dir)
        if previous_snapshot is None:
            return None

        try:
            current_digest = sha256_file(submission_path)
        except OSError:
            return None
        if current_digest == str(state.get("accepted_submission_sha256") or "").lower():
            # Same bytes: nothing to choose between.
            return None

        try:
            metric_name = state["competition_info"].evaluation_metric
        except (KeyError, AttributeError, TypeError):
            metric_name = ""

        if cv_score is not None:
            improved = (
                calculate_score_improvement(cv_score, previous_score, metric_name) > 0
            )
            if improved:
                return None
            reason = (
                f"CV {cv_score:.6f} does not improve on accepted "
                f"{previous_score:.6f} ({metric_name})"
            )
        else:
            # An artifact without hash-bound CV evidence cannot displace one
            # that has it.
            reason = (
                f"current artifact has no hash-bound CV provenance; accepted "
                f"{previous_score:.6f} is retained"
            )

        restored = restore_accepted_submission(state, working_dir)
        if restored is None:
            # Restoration is the only safe way to keep the better artifact. If
            # it fails, fall through and accept the current one rather than
            # leaving submission.csv disagreeing with the recorded state.
            print("⚠️  Could not restore the previously accepted submission")
            return None

        print(f"↩️  Keeping the previously accepted submission: {reason}")
        return {
            "submissions": [
                SubmissionResult(
                    submission_id=None,
                    public_score=None,
                    private_score=None,
                    percentile=None,
                    cv_score=previous_score,
                    file_path=str(previous_snapshot),
                    valid=True,
                    error=None,
                    submitted_at=datetime.now(),
                )
            ],
            "submission_validation_error": None,
            "retry_submission_count": 0,
            "last_updated": datetime.now(),
        }

    @staticmethod
    def _retry_updates(
        state: KaggleState,
        error: str,
        *,
        failure_kind: str = "invalid",
    ) -> dict[str, Any]:
        """Persist bounded retry control as a node update.

        LangGraph route functions choose an edge; mutating state inside a route
        is not a durable state update. Submission failures therefore advance
        the counter and reset the developer cursor in this node.
        """
        try:
            failure_count = max(0, int(state.get("retry_submission_count", 0))) + 1
        except (TypeError, ValueError):
            failure_count = 1

        plan = state.get("ablation_plan", []) or []

        def component_type(item: Any) -> str | None:
            if isinstance(item, dict):
                return item.get("component_type")
            return getattr(item, "component_type", None)

        retry_index = next(
            (
                index
                for index in range(len(plan) - 1, -1, -1)
                if component_type(plan[index]) in {"model", "ensemble"}
            ),
            max(0, len(plan) - 1),
        )
        updates: dict[str, Any] = {
            "submission_validation_error": error,
            "retry_submission_count": failure_count,
            "current_component_index": retry_index,
            "skip_remaining_components": False,
        }
        if failure_count > 3:
            updates.update(
                {
                    "should_continue": False,
                    "termination_reason": (
                        f"submission_{failure_kind}_after_retries"
                    ),
                }
            )
            # A hash-verified artifact accepted in an earlier iteration can
            # still be graded; only invalidate the whole run when no such
            # artifact exists. Otherwise retry exhaustion in a later iteration
            # would convert a real partial success into "no submission".
            has_verified_accepted = bool(
                state.get("accepted_submission_snapshot_path")
                and state.get("accepted_submission_sha256")
            )
            if not has_verified_accepted:
                updates["workflow_valid"] = False
        return updates

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute submission upload and monitoring.

        Args:
            state: Current workflow state

        Returns:
            State updates with submission results
        """
        print("\n" + "=" * 60)
        print("📤 SUBMISSION AGENT: Uploading to Kaggle")
        print("=" * 60)

        working_dir = Path(state["working_directory"])
        competition_name = state["competition_info"].name
        metric_name = state["competition_info"].evaluation_metric
        self._current_metric_name = metric_name
        sample_submission_path = (
            state.get("sample_submission_path") or working_dir / "sample_submission.csv"
        )
        mlebench_mode = str(state.get("run_mode", "")).lower() == "mlebench" or os.getenv(
            "MLEBENCH_MODE", ""
        ).lower() in {"1", "true", "yes"}

        # Upstream MLE-bench gates use ``workflow_valid=False`` when a required
        # immutable artifact cannot be verified. Never let a shape-valid
        # mutable file left in the workspace bypass that decision.
        if mlebench_mode and state.get("workflow_valid") is False:
            message = str(
                state.get("submission_validation_error")
                or "Workflow invalidated before submission"
            )
            print(f"❌ Submission blocked: {message}")
            submission_path = working_dir / "submission.csv"
            submission_result = SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=None,
                file_path=str(submission_path) if submission_path.is_file() else None,
                valid=False,
                error=message,
            )
            return {
                "submissions": [submission_result],
                **self._retry_updates(state, message),
                "workflow_valid": False,
                "last_updated": datetime.now(),
            }

        # Benchmark runs accept only the current canonical output. Recursive
        # discovery could accidentally promote a prior/rejected hidden snapshot.
        submission_path = (
            working_dir / "submission.csv"
            if mlebench_mode and (working_dir / "submission.csv").is_file()
            else None
        )
        if not mlebench_mode:
            submission_path = self._find_submission_file(working_dir)

        if not submission_path:
            print("❌ No submission file found")
            return {
                "last_updated": datetime.now(),
                **self._retry_updates(
                    state,
                    "No submission file found",
                    failure_kind="missing",
                ),
            }

        print(f"\n📄 Submission file: {submission_path.name}")

        # Validate submission
        # Determine problem type for validation heuristics
        problem_type = None
        metric_name = None
        try:
            problem_type = state["competition_info"].problem_type
            metric_name = state["competition_info"].evaluation_metric
        except Exception:
            problem_type = None
            metric_name = None

        is_valid, message = self._validate_submission(
            submission_path,
            sample_submission_path,
            problem_type=problem_type,
            metric_name=metric_name,
            target_cols=[
                str(column)
                for column in (
                    (state.get("submission_contract") or {}).get("target_cols")
                    or []
                )
            ],
        )
        if is_valid and mlebench_mode and Path(sample_submission_path).is_file():
            try:
                is_template = Path(sample_submission_path).samefile(submission_path) or (
                    sha256_file(submission_path) == sha256_file(Path(sample_submission_path))
                )
            except OSError:
                is_template = False
            if is_template:
                is_valid = False
                message = "Generated submission is identical to the sample submission template"

        if not is_valid:
            print(f"❌ Validation failed: {message}")
            submission_result = SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=None,
                file_path=str(submission_path),
                valid=False,
                error=message,
            )
            return {
                "last_updated": datetime.now(),
                "submissions": [submission_result],
                **self._retry_updates(state, message),
            }

        print("✅ Validation passed")

        # In MLE-bench mode, validate and return the artifact without uploading it.
        # The runner performs the sole test-set grading pass after this workflow ends.
        if mlebench_mode:
            cv_score, score_source, score_owner = (
                self._resolve_mlebench_cv_provenance(
                    state,
                    working_dir,
                    submission_path,
                )
            )

            kept = self._keep_better_accepted_submission(
                state,
                working_dir,
                submission_path,
                cv_score,
            )
            if kept is not None:
                return kept

            try:
                snapshot_path, snapshot_sha256 = snapshot_accepted_submission(
                    working_dir,
                    submission_path,
                    run_id=str(state.get("run_id") or ""),
                    iteration=int(state.get("current_iteration", 0) or 0),
                )
            except (OSError, TypeError, ValueError) as exc:
                message = f"Failed to preserve accepted submission: {exc}"
                print(f"❌ {message}")
                submission_result = SubmissionResult(
                    submission_id=None,
                    public_score=None,
                    private_score=None,
                    percentile=None,
                    cv_score=None,
                    file_path=str(submission_path),
                    valid=False,
                    error=message,
                )
                return {
                    "submissions": [submission_result],
                    **self._retry_updates(state, message),
                    "last_updated": datetime.now(),
                }

            print(
                "✅ MLE-bench artifact snapshotted; final grading deferred to the runner"
            )
            submission_result = SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=float(cv_score) if isinstance(cv_score, (int, float)) else None,
                file_path=str(snapshot_path),
                valid=True,
                error=None,
                submitted_at=datetime.now(),
            )
            return {
                "submissions": [submission_result],
                "accepted_submission_path": str(snapshot_path),
                "accepted_submission_snapshot_path": str(snapshot_path),
                "accepted_submission_sha256": snapshot_sha256,
                "accepted_submission_cv_score": cv_score,
                "accepted_submission_score_owner": score_owner,
                "accepted_submission_score_source": score_source,
                "submission_validation_error": None,
                "retry_submission_count": 0,
                "last_updated": datetime.now(),
            }

        # Upload to Kaggle
        submission_result = self._upload_to_kaggle(
            competition_name=competition_name,
            submission_path=submission_path,
            state=state,
        )

        # Check score and percentile
        if submission_result.public_score is not None:
            self._check_goal_achievement(submission_result, state)

        # Update best_score considering metric direction
        # IMPORTANT: best_score must ALWAYS be numeric (never None) to avoid
        # TypeError in workflow.py when formatting with :.4f
        current_best = state.get("best_score", 0.0)
        if current_best is None:
            current_best = 0.0
        new_score = submission_result.public_score
        # Only update if we have a valid new score
        if new_score is not None:
            # First valid score OR comparison with existing best
            if current_best == 0.0 and len(state.get("submissions", [])) == 0:
                updated_best = new_score
            else:
                updated_best = compare_scores(current_best, new_score, metric_name)
        else:
            # No score available (hidden score competition), keep previous best
            updated_best = current_best

        updates: dict[str, Any] = {
            "submissions": [submission_result],
            "best_score": updated_best,  # Guaranteed to be float
            "last_updated": datetime.now(),
        }
        if submission_result.valid:
            updates["submission_validation_error"] = None
            updates["retry_submission_count"] = 0
        else:
            # A terminal upload failure must advance the bounded retry
            # counter; resetting it here made the route retry forever.
            updates.update(
                self._retry_updates(
                    state,
                    submission_result.error or "Kaggle submission failed",
                    failure_kind="upload",
                )
            )
        return updates

    def _find_submission_file(self, working_dir: Path) -> Path | None:
        """Find submission file in working directory."""
        # Check standard location
        submission_path = working_dir / "submission.csv"

        if submission_path.exists():
            return submission_path

        # Search for a generated alternative, never the template itself.
        for file in working_dir.rglob("*submission*.csv"):
            if any(part.startswith(".") for part in file.relative_to(working_dir).parts):
                continue
            if file.name.lower() in {"sample_submission.csv", "sample-submission.csv"}:
                continue
            return file

        return None

    def _validate_submission(
        self,
        submission_path: Path,
        sample_submission_path: Path | None,
        problem_type: str | None = None,
        metric_name: str | None = None,
        target_cols: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Validate submission file format.

        Args:
            submission_path: Path to submission CSV
            sample_submission_path: Path to sample_submission.csv for comparison
            problem_type: Competition problem type (classification, regression, etc.)

        Returns:
            Tuple of (is_valid, message)
        """
        if sample_submission_path and Path(sample_submission_path).is_file():
            # Keep the terminal workflow gate on the same role-aware,
            # chunked contract used immediately after code execution. Loading
            # both CSVs here used to reintroduce OOM risk for pixel-level
            # submissions and counted blank echoed inputs as missing
            # predictions.
            from ..tools.code_executor.submission import SubmissionValidationMixin

            normalized_problem = str(problem_type or "").lower()
            if "multiclass" in normalized_problem:
                normalized_problem = "multiclass"
            # Same metric-aware row-sum rule as the developer gate. Leaving the
            # default (True) here rejected, at the very end of the run, bytes
            # the developer had validated and snapshotted under a ranking
            # metric — the terminal gate must not hold the artifact to a
            # stricter contract than the one it was accepted under.
            return SubmissionValidationMixin().validate_submission_format(
                Path(submission_path),
                Path(sample_submission_path),
                component_type="model",
                problem_type=normalized_problem or None,
                target_cols=target_cols,
                require_normalized_rows=metric_reads_rows_as_distribution(
                    metric_name or ""
                ),
            )

        try:
            df = pd.read_csv(submission_path)

            # Basic checks
            if len(df) == 0:
                return False, "Submission is empty"

            if len(df.columns) < 2:
                return False, "Submission must have at least 2 columns (ID + prediction)"

            # Check for nulls
            if df.isnull().any().any():
                null_count = df.isnull().sum().sum()
                return False, f"Submission contains {null_count} null values"

            # Validate against sample_submission if available
            if sample_submission_path and Path(sample_submission_path).exists():
                try:
                    sample_sub = read_csv_auto(sample_submission_path)

                    if df.shape[0] != sample_sub.shape[0]:
                        return (
                            False,
                            "Row-count mismatch versus the supplied sample template "
                            f"(got {df.shape[0]:,}, expected {sample_sub.shape[0]:,}). "
                            "Generate one prediction for every observed template ID "
                            "without assuming an ID encoding or coordinate base.",
                        )

                    if df.columns.tolist() != sample_sub.columns.tolist():
                        if set(df.columns) == set(sample_sub.columns):
                            # Auto-fix column order to match sample_submission
                            df = df[sample_sub.columns]
                            df.to_csv(submission_path, index=False)
                            print(
                                "⚠️ Column order mismatch fixed: reordered columns to match sample_submission"
                            )
                        else:
                            # CRITICAL: Check for target column count mismatch (common error)
                            expected_target_cols = sample_sub.columns[1:].tolist()
                            actual_target_cols = df.columns[1:].tolist()
                            n_expected = len(expected_target_cols)
                            n_actual = len(actual_target_cols)

                            if n_actual != n_expected:
                                return (
                                    False,
                                    f"""
CRITICAL TARGET COLUMN COUNT MISMATCH!

Expected {n_expected} target columns: {expected_target_cols[:5]}{'...' if n_expected > 5 else ''}
Got {n_actual} target columns: {actual_target_cols[:5]}{'...' if n_actual > 5 else ''}

This is a CRITICAL error - your model is not predicting all required classes!

REQUIRED FIX in your model code:
```python
# FIRST: Count target columns from sample_submission
sample_sub = pd.read_csv(sample_submission_path)
target_cols = sample_sub.columns[1:].tolist()
N_CLASSES = len(target_cols)  # Must be {n_expected}!

# Model MUST output exactly N_CLASSES predictions
# PyTorch: nn.Linear(..., N_CLASSES)  # N_CLASSES={n_expected}
# Keras: Dense(N_CLASSES, activation=...)
```

Common causes:
1. Hard-coded number of classes instead of reading from sample_submission
2. Using wrong target columns from train.csv
3. LabelEncoder dropping infrequent classes
""",
                                )
                            return (
                                False,
                                f"Column mismatch vs sample_submission: {df.columns.tolist()} != {sample_sub.columns.tolist()}",
                            )

                    # Check ID column match if sample_submission includes an ID column
                    id_col = None
                    if sample_sub.shape[1] >= 2:
                        # Prefer explicit ID-like column names
                        for col in sample_sub.columns:
                            col_lower = col.lower()
                            if (
                                col_lower == "id"
                                or col_lower.endswith("_id")
                                or col_lower.endswith("id")
                            ):
                                id_col = col
                                break
                        # Fallback to first column only when multi-column sample looks like it has IDs
                        if id_col is None:
                            first_col = sample_sub.columns[0]
                            if sample_sub[first_col].nunique(dropna=False) == len(sample_sub):
                                id_col = first_col

                    if id_col and id_col not in df.columns:
                        return False, f"ID column '{id_col}' missing from submission"

                    if id_col and not df[id_col].astype(str).equals(sample_sub[id_col].astype(str)):
                        # Check if it's just an ordering issue vs completely wrong IDs
                        sub_ids = set(df[id_col].astype(str))
                        sample_ids = set(sample_sub[id_col].astype(str))
                        if sub_ids != sample_ids:
                            missing = sample_ids - sub_ids
                            extra = sub_ids - sample_ids
                            return (
                                False,
                                f"ID values don't match sample_submission. Missing {len(missing)} IDs, {len(extra)} unexpected IDs.",
                            )
                        # Auto-fix ID order to match sample_submission
                        sample_ids_order = sample_sub[id_col].astype(str).to_list()
                        df_indexed = df.set_index(df[id_col].astype(str))
                        try:
                            df = df_indexed.loc[sample_ids_order].reset_index()
                        except KeyError as exc:
                            return False, f"Failed to reorder submission IDs: {exc!s}"
                        # Ensure column order matches sample_submission (ID first)
                        df = df[sample_sub.columns]
                        df.to_csv(submission_path, index=False)
                        print(
                            "⚠️ ID order mismatch fixed: reordered rows to match sample_submission"
                        )

                    # Warn if multi-class probabilities do not sum to 1
                    if problem_type and "class" in problem_type.lower() and sample_sub.shape[1] > 2:
                        target_cols = sample_sub.columns[1:]
                        try:
                            vals = df[target_cols].astype(float).to_numpy()
                            if (vals >= 0).all() and (vals <= 1).all():
                                row_sums = vals.sum(axis=1)
                                if not np.allclose(row_sums, 1.0, atol=1e-2):
                                    print(
                                        "⚠️ Warning: row probabilities do not sum to 1.0. If multi-class, apply softmax; if multi-label, this is expected."
                                    )
                        except Exception:
                            pass

                except Exception as e:
                    return False, f"Failed to compare with sample_submission: {e!s}"

            # Prediction sanity checks
            problem_lower = (problem_type or "").lower()
            is_classification = (
                "class" in problem_lower
            )  # covers binary_classification, classification, multiclass

            pred_col = df.columns[1]
            preds = df[pred_col]
            if not pd.api.types.is_numeric_dtype(preds):
                return False, f"Prediction column {pred_col} must be numeric"

            # For classification/probabilities, enforce [0,1]; for regression, allow any numeric range
            if is_classification:
                vals = preds.astype(float).to_numpy()
                if (vals < 0).any():
                    return False, "Predictions must be >= 0"

                metric_lower = (metric_name or "").lower()
                prob_metrics = (
                    "logloss",
                    "log_loss",
                    "log loss",
                    "cross_entropy",
                    "brier",
                    "auc",
                    "roc",
                    "prc",
                    "average_precision",
                )
                label_metrics = (
                    "accuracy",
                    "f1",
                    "precision",
                    "recall",
                    "kappa",
                    "qwk",
                    "quadratic_weighted_kappa",
                    "mcc",
                )
                expects_prob = any(m in metric_lower for m in prob_metrics)
                expects_label = any(m in metric_lower for m in label_metrics)

                sample_suggests_prob = False
                sample_suggests_label = False
                if sample_submission_path and Path(sample_submission_path).exists():
                    try:
                        sample_sub = read_csv_auto(sample_submission_path)
                        if sample_sub.shape[1] > 2:
                            sample_suggests_prob = True
                        elif sample_sub.shape[1] >= 2:
                            sample_vals = sample_sub.iloc[:, 1]
                            if pd.api.types.is_numeric_dtype(sample_vals):
                                svals = sample_vals.to_numpy()
                                if svals.size:
                                    if (svals < 0).any() or (svals > 1).any():
                                        sample_suggests_label = True
                                    elif not np.allclose(svals, np.round(svals)):
                                        sample_suggests_prob = True
                    except Exception:
                        pass

                # Final decision: prefer metric signal; fall back to sample hints.
                if expects_prob or (not expects_label and sample_suggests_prob):
                    if (vals > 1).any():
                        return (
                            False,
                            f"Predictions outside [0,1] range (min={preds.min():.4f}, max={preds.max():.4f})",
                        )
                elif expects_label or sample_suggests_label:
                    # Accept label-style outputs without coercion.
                    pass
                else:
                    # Ambiguous: allow both label-style (values > 1) and probability-style outputs.
                    pass

            if not preds.replace([float("inf"), float("-inf")], pd.NA).notna().all():
                return False, "Predictions contain inf or NaN values"

            return True, "Valid"

        except Exception as e:
            return False, f"Error reading submission: {e!s}"

    def _upload_to_kaggle(
        self,
        competition_name: str,
        submission_path: Path,
        state: KaggleState,
    ) -> SubmissionResult:
        """
        Upload submission to Kaggle.

        Args:
            competition_name: Competition name
            submission_path: Path to submission file
            state: Current state

        Returns:
            SubmissionResult
        """
        working_dir = Path(state["working_directory"])

        # Authentication is intentionally lazy so local/MLE-bench validation does
        # not require Kaggle credentials or trigger package-level exits.
        if not self._ensure_kaggle_api():
            print("⚠️  Kaggle API not authenticated")
            print("   Set KAGGLE_USERNAME and KAGGLE_KEY to enable uploads")

            return SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=None,
                submitted_at=datetime.now(),
            )

        # Check if auto-submit is enabled
        if not self.config.kaggle.auto_submit:
            print("⚠️  Auto-submit is disabled (set KAGGLE_AUTO_SUBMIT=true)")

            return SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=None,
                submitted_at=datetime.now(),
            )

        # Create submission message
        iteration = state.get("current_iteration", 0)
        cv_score = state.get("best_score", 0.0)

        message = self.config.kaggle.submission_message_template.format(
            iteration=iteration,
            cv_score=cv_score,
        )

        try:
            print("\n📤 Uploading to Kaggle...")
            print(f"   Competition: {competition_name}")
            print(f"   Message: {message}")

            # Try using Kaggle CLI first (more reliable in some environments)
            try:
                import subprocess

                cmd = [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    competition_name,
                    "-f",
                    str(submission_path),
                    "-m",
                    message,
                ]
                result_cli = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=30
                )

                if result_cli.returncode == 0:
                    print("✅ Uploaded successfully via CLI!")
                    print(f"   {result_cli.stdout}")
                    submission_id = None  # CLI doesn't return ID easily
                else:
                    # Fall back to API
                    raise Exception("CLI failed, using API")

            except Exception:
                # Fall back to Python API
                print("   ℹ️  CLI upload failed, using Python API...")
                result = self.kaggle_api.competition_submit(
                    file_name=str(submission_path),
                    message=message,
                    competition=competition_name,
                )
                submission_id = result.get("id")
                print("✅ Uploaded successfully via API!")

            # Poll for score with retries
            print("\n⏳ Waiting for score...")
            public_score, percentile, terminal_error = self._poll_for_score(
                competition_name
            )

            if public_score is not None:
                print(f"\n📊 Public Score: {public_score:.4f}")
                print(f"   Percentile: {percentile:.1f}%")
            else:
                print("\n⏳ Score not yet available (check leaderboard later)")

            # Save temporal version (Success Memory)
            versioned_path = (
                working_dir
                / f"submission_iter_{state.get('current_iteration', 0)}_score_{public_score if public_score is not None else 0.0:.4f}.csv"
            )
            try:
                import shutil

                shutil.copy2(submission_path, versioned_path)
                print(f"✅ Saved temporal backup: {versioned_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to save temporal backup: {e}")
                versioned_path = None

            return SubmissionResult(
                submission_id=submission_id,
                public_score=public_score,
                private_score=None,
                percentile=percentile,
                cv_score=cv_score,
                file_path=str(versioned_path) if versioned_path else None,
                valid=terminal_error is None,
                error=terminal_error,
                submitted_at=datetime.now(),
            )

        except Exception as e:
            print(f"❌ Upload failed: {e!s}")

            return SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=cv_score,
                submitted_at=datetime.now(),
            )

    def _poll_for_score(
        self,
        competition_name: str,
        *,
        poll_timeout: int = 600,
        poll_interval: int = 20,
    ) -> tuple[float | None, float | None, str | None]:
        """Poll until a score, timeout, or terminal submission failure."""
        elapsed = 0
        while elapsed < poll_timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            public_score, percentile, status = self._fetch_score(competition_name)
            if public_score is not None:
                return public_score, percentile, None
            if str(status).strip().lower() in self._TERMINAL_FAILURE_STATUSES:
                error = f"Submission ended with terminal status: {status}"
                print(f"   ❌ {error}")
                return None, None, error
            print(
                f"   Score not ready yet (status: {status}) "
                f"({elapsed}s/{poll_timeout}s)..."
            )
        return None, None, None

    def _fetch_score(self, competition_name: str) -> tuple[float | None, float | None, str]:
        """
        Fetch latest submission score from leaderboard.

        Args:
            competition_name: Competition name

        Returns:
            Tuple of (public_score, percentile, status)
        """
        try:
            # Get recent submissions
            submissions = self.kaggle_api.competition_submissions(competition_name)

            if not submissions:
                return None, None, "Unknown"

            # Get latest submission - handle both dict and raw Kaggle objects
            latest = submissions[0]

            if isinstance(latest, dict):
                raw_score = latest.get("publicScore")
                status = latest.get("status", "Unknown")
            elif hasattr(latest, "publicScore"):
                raw_score = latest.publicScore
                status = getattr(latest, "status", "Unknown")
            else:
                return None, None, "Unknown"

            # Normalize score: skip None, empty, "None" strings
            if raw_score is None or raw_score in ("", "None"):
                return None, None, status

            try:
                public_score = float(raw_score)
            except (ValueError, TypeError):
                return None, None, status

            percentile = self._calculate_percentile(
                competition_name, public_score, metric_name=self._current_metric_name
            )

            return public_score, percentile, status

        except Exception as e:
            print(f"⚠️  Could not fetch score: {e!s}")
            return None, None, "FetchError"

    def _calculate_percentile(
        self, competition_name: str, score: float, metric_name: str = ""
    ) -> float | None:
        """
        Calculate percentile rank on leaderboard.

        Args:
            competition_name: Competition name
            score: Public score
            metric_name: Evaluation metric name (used to determine direction)

        Returns:
            Percentile (0-100, lower is better — 1% = top 1%)
        """
        from ..core.config import is_metric_minimization

        try:
            # Get leaderboard
            leaderboard = self.kaggle_api.competition_leaderboard_view(competition_name)

            if not leaderboard:
                return None

            minimize = is_metric_minimization(metric_name) if metric_name else False

            # Count submissions better than ours
            if minimize:
                # Lower is better: entries with score < ours are better
                better_count = sum(
                    1 for entry in leaderboard
                    if float(entry.score if hasattr(entry, "score") else entry["score"]) < score
                )
            else:
                # Higher is better: entries with score > ours are better
                better_count = sum(
                    1 for entry in leaderboard
                    if float(entry.score if hasattr(entry, "score") else entry["score"]) > score
                )

            total_count = len(leaderboard)

            return (better_count / total_count) * 100

        except Exception:
            # Fallback: estimate based on submissions
            # Assume we're in the middle if we can't get leaderboard
            return 50.0

    def _check_goal_achievement(self, submission_result: SubmissionResult, state: KaggleState):
        """Check if we achieved the goal (top 20%)."""
        target_percentile = state.get("target_percentile", 20.0)

        if submission_result.percentile is None:
            return

        if submission_result.percentile <= target_percentile:
            print(f"\n🎉 GOAL ACHIEVED! Top {target_percentile}%")
            print(f"   Your percentile: {submission_result.percentile:.1f}%")
            print(f"   Public score: {submission_result.public_score:.4f}")

            # Update state to stop iterations
            state["should_continue"] = False
            state["termination_reason"] = "goal_achieved"
        else:
            print(
                f"\n📈 Progress: {submission_result.percentile:.1f}% (target: {target_percentile}%)"
            )
            remaining = submission_result.percentile - target_percentile
            print(f"   Need to improve by {remaining:.1f} percentile points")


# ==================== LangGraph Node Function ====================


def submission_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for the submission agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = SubmissionAgent()
    return agent(state)
