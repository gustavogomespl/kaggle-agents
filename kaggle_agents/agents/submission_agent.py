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
            }

        print(f"\n📄 Submission file: {submission_path.name}")

        # Validate submission
        # Determine problem type for validation heuristics
        problem_type = None
        try:
            problem_type = state["competition_info"].problem_type
        except Exception:
            problem_type = None

        is_valid, message = self._validate_submission(
            submission_path,
            sample_submission_path,
            problem_type=problem_type,
        )

        if not is_valid:
            print(f"❌ Validation failed: {message}")
            return {
                "last_updated": datetime.now(),
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
    ) -> tuple[bool, str]:
        """
        Validate submission file format.

        Args:
            submission_path: Path to submission CSV

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
                    if df.shape != sample_sub.shape:
                        return False, f"Shape mismatch vs sample_submission (got {df.shape}, expected {sample_sub.shape})"
                    if df.columns.tolist() != sample_sub.columns.tolist():
                        return False, f"Column mismatch vs sample_submission: {df.columns.tolist()} != {sample_sub.columns.tolist()}"
                    if "id" in df.columns and not df["id"].equals(sample_sub["id"]):
                        return False, "ID column does not match sample_submission"
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
                integer_like = np.all(np.isclose(vals, np.round(vals)))
                if integer_like:
                    if (vals < 0).any():
                        return False, "Class labels must be >= 0"
                else:
                    if (vals < 0).any() or (vals > 1).any():
                        return False, f"Predictions outside [0,1] range (min={preds.min():.4f}, max={preds.max():.4f})"

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
