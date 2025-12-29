"""
Submission Agent for Kaggle Competition Upload and Monitoring.

This agent handles submission creation, Kaggle upload, leaderboard monitoring,
and score-based iteration decisions.
"""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from ..core.config import compare_scores, get_config
from ..core.state import KaggleState, SubmissionResult


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

    def __init__(self):
        """Initialize the submission agent."""
        self.config = get_config()
        self.kaggle_api = KaggleApi()

        # Try to authenticate
        try:
            self.kaggle_api.authenticate()
            self.authenticated = True
        except Exception:
            self.authenticated = False

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute submission upload and monitoring.

        Args:
            state: Current workflow state

        Returns:
            State updates with submission results
        """
        print("\n" + "="*60)
        print("📤 SUBMISSION AGENT: Uploading to Kaggle")
        print("="*60)

        working_dir = Path(state["working_directory"])
        competition_name = state["competition_info"].name
        metric_name = state["competition_info"].evaluation_metric
        sample_submission_path = state.get("sample_submission_path") or working_dir / "sample_submission.csv"

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

        mlebench_grade: dict[str, Any] | None = None

        # In MLE-bench mode (or when explicitly enabled), grade locally so downstream
        # feedback/rewards can optimize for medals without needing Kaggle API.
        mlebench_mode = (
            str(state.get("run_mode", "")).lower() == "mlebench"
            or os.getenv("MLEBENCH_MODE", "").lower() in {"1", "true", "yes"}
        )
        if mlebench_mode:
            grading = self._grade_with_mlebench(
                competition_name=competition_name,
                submission_path=submission_path,
            )
            mlebench_grade = grading

            # Surface grading in state for meta-evaluator feedback/reward signals
            score = grading.get("score")
            valid = bool(grading.get("valid_submission", False))

            if valid and isinstance(score, (int, float)):
                print(
                    f"✅ MLE-bench grade: score={float(score):.5f} "
                    f"medal={'gold' if grading.get('gold_medal') else 'silver' if grading.get('silver_medal') else 'bronze' if grading.get('bronze_medal') else 'none'}"
                )
                # Save temporal version (Success Memory)
                versioned_path = working_dir / f"submission_iter_{state.get('current_iteration', 0)}_score_{float(score):.4f}.csv"
                try:
                    import shutil
                    shutil.copy2(submission_path, versioned_path)
                    print(f"✅ Saved temporal backup: {versioned_path.name}")
                except Exception as e:
                    print(f"⚠️ Failed to save temporal backup: {e}")
                    versioned_path = None

                submission_result = SubmissionResult(
                    submission_id=None,
                    public_score=float(score),
                    private_score=None,
                    percentile=None,
                    cv_score=None,
                    file_path=str(versioned_path) if versioned_path else None,
                    valid=True,
                    error=None,
                    submitted_at=datetime.now(),
                )
                updated_best = compare_scores(
                    state.get("best_score", 0.0) or 0.0,
                    float(score),
                    metric_name,
                )
                return {
                    "submissions": [submission_result],
                    "best_score": updated_best,
                    "current_performance_score": float(score),
                    "mlebench_grade": grading,
                    "submission_validation_error": None,
                    "retry_submission_count": 0,
                    "last_updated": datetime.now(),
                }

            print(f"⚠️  MLE-bench grading failed: {grading.get('error', 'unknown error')}")
            # Fall back to the usual Kaggle upload path if possible.

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
            **({"mlebench_grade": mlebench_grade} if mlebench_grade is not None else {}),
        }

    def _grade_with_mlebench(
        self,
        competition_name: str,
        submission_path: Path,
    ) -> dict[str, Any]:
        """
        Grade a submission with the local `mlebench grade-sample` CLI.

        Returns a dict compatible with MLE-bench output, e.g.:
        {valid_submission, score, gold_medal, silver_medal, bronze_medal, above_median, error}
        """
        if shutil.which("mlebench") is None:
            return {
                "valid_submission": False,
                "error": "mlebench CLI not found in PATH",
            }

        try:
            result = subprocess.run(
                ["mlebench", "grade-sample", str(submission_path), competition_name],
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = (result.stdout or "") + (result.stderr or "")

            # Extract the first JSON object from output (mlebench prints extra lines)
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    parsed = json.loads(output[json_start:json_end])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            return {
                "valid_submission": False,
                "error": f"Could not parse mlebench output (exit={result.returncode}): {output[:500]}",
            }

        except subprocess.TimeoutExpired:
            return {
                "valid_submission": False,
                "error": "MLE-bench grading timeout (60s)",
            }
        except Exception as e:
            return {
                "valid_submission": False,
                "error": str(e),
            }

    def _find_submission_file(self, working_dir: Path) -> Path | None:
        """Find submission file in working directory."""
        # Check standard location
        submission_path = working_dir / "submission.csv"

        if submission_path.exists():
            return submission_path

        # Search for any file with "submission" in name
        for file in working_dir.rglob("*submission*.csv"):
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
                    sample_sub = pd.read_csv(sample_submission_path)

                    # Enhanced shape mismatch detection for pixel-level format
                    if df.shape[0] != sample_sub.shape[0]:
                        expected_rows = sample_sub.shape[0]
                        actual_rows = df.shape[0]
                        ratio = expected_rows / max(actual_rows, 1)

                        # Detect pixel-level format mismatch (expected >> actual)
                        if ratio > 100:
                            return False, f"""
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
"""
                        else:
                            return False, f"Shape mismatch vs sample_submission (got {df.shape}, expected {sample_sub.shape})"

                    if df.columns.tolist() != sample_sub.columns.tolist():
                        if set(df.columns) == set(sample_sub.columns):
                            # Auto-fix column order to match sample_submission
                            df = df[sample_sub.columns]
                            df.to_csv(submission_path, index=False)
                            print("⚠️ Column order mismatch fixed: reordered columns to match sample_submission")
                        else:
                            return False, f"Column mismatch vs sample_submission: {df.columns.tolist()} != {sample_sub.columns.tolist()}"

                    # Check ID column match if sample_submission includes an ID column
                    id_col = None
                    if sample_sub.shape[1] >= 2:
                        # Prefer explicit ID-like column names
                        for col in sample_sub.columns:
                            col_lower = col.lower()
                            if col_lower == "id" or col_lower.endswith("_id") or col_lower.endswith("id"):
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
                            return False, f"ID values don't match sample_submission. Missing {len(missing)} IDs, {len(extra)} unexpected IDs."
                        else:
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
                            print("⚠️ ID order mismatch fixed: reordered rows to match sample_submission")

                    # Warn if multi-class probabilities do not sum to 1
                    if problem_type and "class" in problem_type.lower() and sample_sub.shape[1] > 2:
                        target_cols = sample_sub.columns[1:]
                        try:
                            vals = df[target_cols].astype(float).to_numpy()
                            if (vals >= 0).all() and (vals <= 1).all():
                                row_sums = vals.sum(axis=1)
                                if not np.allclose(row_sums, 1.0, atol=1e-2):
                                    print("⚠️ Warning: row probabilities do not sum to 1.0. If multi-class, apply softmax; if multi-label, this is expected.")
                        except Exception:
                            pass

                except Exception as e:
                    return False, f"Failed to compare with sample_submission: {e!s}"

            # Prediction sanity checks
            problem_lower = (problem_type or "").lower()
            is_classification = "class" in problem_lower  # covers binary_classification, classification, multiclass

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
                        sample_sub = pd.read_csv(sample_submission_path)
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
                        return False, f"Predictions outside [0,1] range (min={preds.min():.4f}, max={preds.max():.4f})"
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
        # Check if authenticated
        if not self.authenticated:
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
                    "kaggle", "competitions", "submit",
                    "-c", competition_name,
                    "-f", str(submission_path),
                    "-m", message
                ]
                result_cli = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)

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

            # Wait a bit for processing
            print("\n⏳ Waiting for score (30s)...")
            time.sleep(30)

            # Fetch score
            public_score, percentile = self._fetch_score(competition_name)

            if public_score is not None:
                print(f"\n📊 Public Score: {public_score:.4f}")
                print(f"   Percentile: {percentile:.1f}%")
            else:
                print("\n⏳ Score not yet available (check leaderboard later)")

            # Save temporal version (Success Memory)
            versioned_path = working_dir / f"submission_iter_{state.get('current_iteration', 0)}_score_{public_score if public_score is not None else 0.0:.4f}.csv"
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

    def _fetch_score(self, competition_name: str) -> tuple[float | None, float | None]:
        """
        Fetch latest submission score from leaderboard.

        Args:
            competition_name: Competition name

        Returns:
            Tuple of (public_score, percentile)
        """
        try:
            # Get recent submissions
            submissions = self.kaggle_api.competition_submissions(competition_name)

            if not submissions:
                return None, None

            # Get latest submission
            latest = submissions[0]

            public_score = latest.get("publicScore")
            percentile = self._calculate_percentile(competition_name, public_score)

            return public_score, percentile

        except Exception as e:
            print(f"⚠️  Could not fetch score: {e!s}")
            return None, None

    def _calculate_percentile(self, competition_name: str, score: float) -> float | None:
        """
        Calculate percentile rank on leaderboard.

        Args:
            competition_name: Competition name
            score: Public score

        Returns:
            Percentile (0-100)
        """
        try:
            # Get leaderboard
            leaderboard = self.kaggle_api.competition_leaderboard_view(competition_name)

            if not leaderboard:
                return None

            # Count submissions better than ours
            better_count = sum(1 for entry in leaderboard if entry["score"] > score)
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
            print(f"\n📈 Progress: {submission_result.percentile:.1f}% (target: {target_percentile}%)")
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
