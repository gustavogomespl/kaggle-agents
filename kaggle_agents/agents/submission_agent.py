"""
Submission Agent for Kaggle Competition Upload and Monitoring.

This agent handles submission creation, Kaggle upload, leaderboard monitoring,
and score-based iteration decisions.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from ..core.state import KaggleState, SubmissionResult
from ..core.config import get_config


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

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
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

        # Find submission file
        submission_path = self._find_submission_file(working_dir)

        if not submission_path:
            print("❌ No submission file found")
            return {
                "last_updated": datetime.now(),
            }

        print(f"\n📄 Submission file: {submission_path.name}")

        # Validate submission
        is_valid, message = self._validate_submission(submission_path)

        if not is_valid:
            print(f"❌ Validation failed: {message}")
            return {
                "last_updated": datetime.now(),
            }

        print(f"✅ Validation passed")

        # Upload to Kaggle
        submission_result = self._upload_to_kaggle(
            competition_name=competition_name,
            submission_path=submission_path,
            state=state,
        )

        # Check score and percentile
        if submission_result.public_score is not None:
            self._check_goal_achievement(submission_result, state)

        return {
            "submissions": [submission_result],
            "best_score": max(state.get("best_score", 0.0), submission_result.public_score or 0.0),
            "last_updated": datetime.now(),
        }

    def _find_submission_file(self, working_dir: Path) -> Optional[Path]:
        """Find submission file in working directory."""
        # Check standard location
        submission_path = working_dir / "submission.csv"

        if submission_path.exists():
            return submission_path

        # Search for any file with "submission" in name
        for file in working_dir.rglob("*submission*.csv"):
            return file

        return None

    def _validate_submission(self, submission_path: Path) -> tuple[bool, str]:
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

            return True, "Valid"

        except Exception as e:
            return False, f"Error reading submission: {str(e)}"

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
            print(f"\n📤 Uploading to Kaggle...")
            print(f"   Competition: {competition_name}")
            print(f"   Message: {message}")

            # Submit
            result = self.kaggle_api.competition_submit(
                file_name=str(submission_path),
                message=message,
                competition=competition_name,
            )

            print(f"✅ Uploaded successfully!")

            # Wait a bit for processing
            print(f"\n⏳ Waiting for score (30s)...")
            time.sleep(30)

            # Fetch score
            public_score, percentile = self._fetch_score(competition_name)

            if public_score is not None:
                print(f"\n📊 Public Score: {public_score:.4f}")
                print(f"   Percentile: {percentile:.1f}%")
            else:
                print(f"\n⏳ Score not yet available (check leaderboard later)")

            return SubmissionResult(
                submission_id=result.get("id"),
                public_score=public_score,
                private_score=None,
                percentile=percentile,
                cv_score=cv_score,
                submitted_at=datetime.now(),
            )

        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")

            return SubmissionResult(
                submission_id=None,
                public_score=None,
                private_score=None,
                percentile=None,
                cv_score=cv_score,
                submitted_at=datetime.now(),
            )

    def _fetch_score(self, competition_name: str) -> tuple[Optional[float], Optional[float]]:
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
            print(f"⚠️  Could not fetch score: {str(e)}")
            return None, None

    def _calculate_percentile(self, competition_name: str, score: float) -> Optional[float]:
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

            percentile = (better_count / total_count) * 100

            return percentile

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

def submission_agent_node(state: KaggleState) -> Dict[str, Any]:
    """
    LangGraph node function for the submission agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = SubmissionAgent()
    return agent(state)
