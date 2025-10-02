"""Kaggle API tools for downloading data and submitting predictions."""

import os
from pathlib import Path
from typing import List, Dict, Any
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleAPIClient:
    """Client for interacting with Kaggle API."""

    def __init__(self):
        """Initialize Kaggle API client."""
        self.api = KaggleApi()
        self.api.authenticate()

    def download_competition_data(
        self, competition: str, path: str = "./data"
    ) -> Dict[str, str]:
        """Download competition data.

        Args:
            competition: Competition name
            path: Path to download data to

        Returns:
            Dictionary with paths to downloaded files
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.api.competition_download_files(competition, path=path)

        # Unzip if needed
        import zipfile

        for file in Path(path).glob("*.zip"):
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(path)
            file.unlink()  # Remove zip file

        # Return paths to key files
        files = {}
        for file in Path(path).glob("*"):
            if file.is_file():
                if "train" in file.name.lower():
                    files["train"] = str(file)
                elif "test" in file.name.lower():
                    files["test"] = str(file)
                elif "sample_submission" in file.name.lower():
                    files["sample_submission"] = str(file)

        return files

    def get_competition_info(self, competition: str) -> Dict[str, Any]:
        """Get competition metadata.

        Args:
            competition: Competition name

        Returns:
            Competition information
        """
        comp = self.api.competition_view(competition)
        return {
            "name": comp.ref,
            "title": comp.title,
            "description": comp.description,
            "evaluation": comp.evaluationMetric,
            "deadline": str(comp.deadline),
            "category": comp.category,
        }

    def submit_prediction(
        self, competition: str, file_path: str, message: str
    ) -> Dict[str, Any]:
        """Submit predictions to competition.

        Args:
            competition: Competition name
            file_path: Path to submission file
            message: Submission message

        Returns:
            Submission result
        """
        result = self.api.competition_submit(file_path, message, competition)
        return {"message": message, "status": "submitted"}

    def get_leaderboard(
        self, competition: str, top_n: int = 100
    ) -> List[Dict[str, Any]]:
        """Get competition leaderboard.

        Args:
            competition: Competition name
            top_n: Number of top entries to return

        Returns:
            Leaderboard entries
        """
        leaderboard = self.api.competition_leaderboard_view(competition)
        entries = []

        for i, entry in enumerate(leaderboard[:top_n]):
            entries.append(
                {
                    "rank": i + 1,
                    "teamName": entry.teamName,
                    "score": entry.score,
                    "submissionDate": str(entry.submissionDate),
                }
            )

        return entries

    def get_my_submissions(self, competition: str) -> List[Dict[str, Any]]:
        """Get user's submissions for a competition.

        Args:
            competition: Competition name

        Returns:
            List of submissions
        """
        submissions = self.api.competition_submissions(competition)
        return [
            {
                "date": str(sub.date),
                "description": sub.description,
                "status": sub.status,
                "publicScore": sub.publicScore,
                "privateScore": sub.privateScore,
            }
            for sub in submissions
        ]
