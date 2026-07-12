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

from ..core.config import compare_scores, get_config
from ..core.state import KaggleState, SubmissionResult
from ..utils.csv_utils import read_csv_auto


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

        # Find submission file
        submission_path = self._find_submission_file(working_dir)

        if not submission_path:
            print("❌ No submission file found")
            return {
                "last_updated": datetime.now(),
                "submission_validation_error": "No submission file found",
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
        )

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
                "submission_validation_error": message,
            }

        print("✅ Validation passed")

        # In MLE-bench mode, validate and return the artifact without uploading it.
        # The runner performs the sole test-set grading pass after this workflow ends.
        mlebench_mode = str(state.get("run_mode", "")).lower() == "mlebench" or os.getenv(
            "MLEBENCH_MODE", ""
        ).lower() in {"1", "true", "yes"}
        if mlebench_mode:
            cv_score = state.get("current_performance_score")
            if not isinstance(cv_score, (int, float)):
                cv_score = state.get("best_single_model_score")
            if not isinstance(cv_score, (int, float)):
                cv_score = state.get("baseline_cv_score")

            print("✅ MLE-bench artifact ready; final grading deferred to the runner")
            submission_result = SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=float(cv_score) if isinstance(cv_score, (int, float)) else None,
                file_path=str(submission_path),
                valid=True,
                error=None,
                submitted_at=datetime.now(),
            )
            return {
                "submissions": [submission_result],
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

        return {
            "submissions": [submission_result],
            "best_score": updated_best,  # Guaranteed to be float
            "submission_validation_error": None,
            "retry_submission_count": 0,
            "last_updated": datetime.now(),
        }

    def _find_submission_file(self, working_dir: Path) -> Path | None:
        """Find submission file in working directory."""
        # Check standard location
        submission_path = working_dir / "submission.csv"

        if submission_path.exists():
            return submission_path

        # Search for a generated alternative, never the template itself.
        for file in working_dir.rglob("*submission*.csv"):
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

                    # Enhanced shape mismatch detection for pixel-level format
                    if df.shape[0] != sample_sub.shape[0]:
                        expected_rows = sample_sub.shape[0]
                        actual_rows = df.shape[0]
                        ratio = expected_rows / max(actual_rows, 1)

                        # Detect pixel-level format mismatch (expected >> actual)
                        if ratio > 100:
                            return (
                                False,
                                f"""
PIXEL-LEVEL FORMAT MISMATCH DETECTED!

Expected: {expected_rows:,} rows (one per pixel)
Got: {actual_rows:,} rows (looks like one per image)

This appears to be an image-to-image task (denoising, segmentation, super-resolution)
that requires PIXEL-LEVEL predictions.

YOUR MODEL ARCHITECTURE IS WRONG. You likely used a classifier (e.g., EfficientNet,
ResNet with FC head) instead of an encoder-decoder (e.g., U-Net, autoencoder).

REQUIREMENTS:
1. Model must output a FULL IMAGE (same H x W as input), not a single value
2. Use encoder-decoder architecture (U-Net, autoencoder, FCN)
3. Flatten output to pixel-level format for submission

CORRECT CODE PATTERN:
```python
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)  # {expected_rows:,} rows

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem  # e.g., "1" from "1.png"
    pred = model(preprocess(img))  # OUTPUT: (H, W) image, NOT single value
    H, W = pred.shape
    for row in range(H):
        for col in range(W):
            pixel_id = f"{{img_id}}_{{row+1}}_{{col+1}}"
            submission_rows.append({{"id": pixel_id, "value": pred[row, col]}})

assert len(submission_rows) == expected_rows
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
```

DO NOT USE:
- Image classifiers (EfficientNet, ResNet, VGG with FC head)
- Models that output a single value per image
- Global average pooling followed by dense layers
""",
                            )
                        return (
                            False,
                            f"Shape mismatch vs sample_submission (got {df.shape}, expected {sample_sub.shape})",
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
