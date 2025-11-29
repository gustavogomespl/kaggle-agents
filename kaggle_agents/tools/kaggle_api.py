"""Kaggle API tools for downloading data and submitting predictions.

Follows official Kaggle API documentation:
https://github.com/Kaggle/kaggle-api
"""

from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleAPIClient:
    """Client for interacting with Kaggle API.

    Implements best practices from official Kaggle API:
    - Proper authentication handling
    - Error handling for API calls
    - Correct parameter usage
    - Data file management
    """

    def __init__(self):
        """Initialize Kaggle API client with authentication."""
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except Exception as e:
            raise RuntimeError(
                f"Kaggle API authentication failed: {e!s}\n"
                "Ensure KAGGLE_USERNAME and KAGGLE_KEY are set, "
                "or ~/.kaggle/kaggle.json exists with credentials."
            )

    def download_competition_data(
        self, competition: str, path: str = "./data", quiet: bool = False
    ) -> dict[str, str]:
        """Download competition data files.

        Uses: api.competition_download_files(competition, path, force, quiet)

        Args:
            competition: Competition URL suffix (e.g., 'titanic')
            path: Download destination directory
            quiet: Suppress progress output

        Returns:
            Dictionary with paths to key files (train, test, sample_submission)

        Raises:
            Exception: If competition not found or download fails
        """
        download_path = Path(path)
        download_path.mkdir(parents=True, exist_ok=True)

        try:
            # Download all competition files
            # API signature: competition_download_files(competition, path=None, force=False, quiet=False)
            self.api.competition_download_files(
                competition,
                path=str(download_path),
                force=False,  # Don't re-download existing files
                quiet=quiet
            )
        except Exception as e:
            raise Exception(
                f"Failed to download competition '{competition}': {e!s}\n"
                f"Ensure competition exists and you have accepted the rules."
            )

        # Unzip downloaded files
        import zipfile

        for zip_file in download_path.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(download_path)
                zip_file.unlink()  # Remove zip after extraction
            except zipfile.BadZipFile:
                print(f"Warning: {zip_file.name} is not a valid zip file, skipping")

        # Identify key files
        files = {}
        for file_path in download_path.glob("*"):
            if file_path.is_file():
                filename_lower = file_path.name.lower()
                if "train" in filename_lower and filename_lower.endswith((".csv", ".parquet")):
                    files["train"] = str(file_path)
                elif "test" in filename_lower and filename_lower.endswith((".csv", ".parquet")):
                    files["test"] = str(file_path)
                elif "sample_submission" in filename_lower or "samplesubmission" in filename_lower:
                    files["sample_submission"] = str(file_path)

        if not files.get("train") or not files.get("test"):
            raise Exception(
                f"Could not find train/test files in {download_path}. "
                f"Available files: {[f.name for f in download_path.glob('*')]}"
            )

        return files

    def get_competition_info(self, competition: str) -> dict[str, Any]:
        """Get competition metadata.

        Uses: api.competitions_list(search=competition) to find competition details.
        Note: The Kaggle API doesn't have a direct competition_view method.

        Args:
            competition: Competition URL suffix (e.g., 'titanic')

        Returns:
            Dictionary with competition information:
                - name: Competition reference
                - title: Competition title
                - description: Competition description
                - evaluation: Evaluation metric
                - deadline: Competition deadline
                - category: Competition category

        Raises:
            Exception: If competition not found
        """
        try:
            # Search for the specific competition
            # The Kaggle API doesn't have competition_view, so we use competitions_list with search
            competitions = self.api.competitions_list(search=competition)

            # Find exact match
            comp = None
            for c in competitions:
                if c.ref == competition or c.ref.endswith(f"/{competition}"):
                    comp = c
                    break

            if not comp:
                # If no exact match, try the first result
                if competitions:
                    comp = competitions[0]
                else:
                    raise Exception(f"Competition '{competition}' not found")

            return {
                "name": comp.ref,
                "title": comp.title if hasattr(comp, 'title') else competition,
                "description": comp.description if hasattr(comp, 'description') else "",
                "evaluation": comp.evaluationMetric if hasattr(comp, 'evaluationMetric') else "unknown",
                "deadline": str(comp.deadline) if hasattr(comp, 'deadline') and comp.deadline else "N/A",
                "category": comp.category if hasattr(comp, 'category') else "unknown",
                "reward": comp.reward if hasattr(comp, 'reward') else "N/A",
                "team_count": comp.teamCount if hasattr(comp, 'teamCount') else 0,
            }
        except Exception as e:
            raise Exception(
                f"Failed to get info for competition '{competition}': {e!s}"
            )

    def submit_prediction(
        self, competition: str, file_path: str, message: str, quiet: bool = False
    ) -> dict[str, Any]:
        """Submit predictions to competition.

        Uses: api.competition_submit(file_name, message, competition, quiet)

        Args:
            competition: Competition URL suffix (e.g., 'titanic')
            file_path: Path to submission CSV file
            message: Submission description/message
            quiet: Suppress progress output

        Returns:
            Dictionary with submission result:
                - message: Submission message
                - status: Submission status
                - file: Submitted file path

        Raises:
            Exception: If submission fails or file not found
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")

        try:
            # API signature: competition_submit(file_name, message, competition, quiet=False)
            # Note: API returns submission result object
            self.api.competition_submit(
                file_path,
                message,
                competition,
                quiet=quiet
            )

            return {
                "message": message,
                "status": "submitted",
                "file": file_path,
                "competition": competition
            }
        except Exception as e:
            raise Exception(
                f"Failed to submit to competition '{competition}': {e!s}\n"
                f"Ensure you have accepted competition rules and file format is correct."
            )

    def get_leaderboard(
        self, competition: str, top_n: int = 100
    ) -> list[dict[str, Any]]:
        """Get competition leaderboard.

        Uses: api.competition_leaderboard_view(id)

        Args:
            competition: Competition URL suffix (e.g., 'titanic')
            top_n: Number of top entries to return (max available entries)

        Returns:
            List of leaderboard entries with:
                - rank: Team rank
                - teamName: Team name
                - score: Team score
                - submissionDate: Last submission date
                - entries: Number of submissions

        Raises:
            Exception: If leaderboard access fails
        """
        try:
            # API signature: competition_leaderboard_view(id)
            # Returns list of leaderboard entries
            leaderboard = self.api.competition_leaderboard_view(competition)

            entries = []
            # Leaderboard may return fewer than requested
            for i, entry in enumerate(leaderboard[:top_n]):
                entries.append({
                    "rank": i + 1,
                    "teamName": entry.teamName if hasattr(entry, 'teamName') else "Unknown",
                    "score": float(entry.score) if hasattr(entry, 'score') else 0.0,
                    "submissionDate": str(entry.submissionDate) if hasattr(entry, 'submissionDate') else "N/A",
                    "entries": entry.entries if hasattr(entry, 'entries') else 0,
                })

            return entries
        except Exception as e:
            raise Exception(
                f"Failed to get leaderboard for '{competition}': {e!s}\n"
                f"Leaderboard may be private or competition may not exist."
            )

    def get_my_submissions(self, competition: str) -> list[dict[str, Any]]:
        """Get user's submissions for a competition.

        Uses: api.competition_submissions(id)

        Args:
            competition: Competition URL suffix (e.g., 'titanic')

        Returns:
            List of user's submissions with:
                - date: Submission date
                - description: Submission message
                - status: Submission status (complete/pending/error)
                - publicScore: Public leaderboard score
                - privateScore: Private leaderboard score (if available)
                - fileName: Submitted file name

        Raises:
            Exception: If unable to fetch submissions
        """
        try:
            # API signature: competition_submissions(id)
            # Returns list of user's submission objects
            submissions = self.api.competition_submissions(competition)

            result = []
            for sub in submissions:
                result.append({
                    "date": str(sub.date) if hasattr(sub, 'date') else "N/A",
                    "description": sub.description if hasattr(sub, 'description') else "",
                    "status": sub.status if hasattr(sub, 'status') else "unknown",
                    "publicScore": float(sub.publicScore) if hasattr(sub, 'publicScore') and sub.publicScore else 0.0,
                    "privateScore": float(sub.privateScore) if hasattr(sub, 'privateScore') and sub.privateScore else 0.0,
                    "fileName": sub.fileName if hasattr(sub, 'fileName') else "",
                })

            return result
        except Exception as e:
            raise Exception(
                f"Failed to get submissions for '{competition}': {e!s}\n"
                f"Ensure you have made at least one submission to this competition."
            )

    def list_competitions(
        self,
        group: str = "general",
        category: str = "all",
        sort_by: str = "latestDeadline",
        page: int = 1,
        search: str | None = None
    ) -> list[dict[str, Any]]:
        """List Kaggle competitions.

        Uses: api.competitions_list(group, category, sort_by, page, search)

        Args:
            group: Competition group ('general', 'entered', 'inClass')
            category: Competition category ('all', 'featured', 'research', 'recruitment',
                     'gettingStarted', 'masters', 'playground')
            sort_by: Sort order ('grouped', 'prize', 'earliestDeadline', 'latestDeadline',
                    'numberOfTeams', 'recentlyCreated')
            page: Page number for pagination
            search: Search term to filter competitions

        Returns:
            List of competitions with metadata

        Raises:
            Exception: If listing fails
        """
        try:
            # API signature: competitions_list(group, category, sort_by, page, search)
            competitions = self.api.competitions_list(
                group=group,
                category=category,
                sort_by=sort_by,
                page=page,
                search=search
            )

            result = []
            for comp in competitions:
                result.append({
                    "ref": comp.ref if hasattr(comp, 'ref') else "",
                    "title": comp.title if hasattr(comp, 'title') else "",
                    "deadline": str(comp.deadline) if hasattr(comp, 'deadline') and comp.deadline else "N/A",
                    "category": comp.category if hasattr(comp, 'category') else "",
                    "reward": comp.reward if hasattr(comp, 'reward') else "N/A",
                    "teamCount": comp.teamCount if hasattr(comp, 'teamCount') else 0,
                })

            return result
        except Exception as e:
            raise Exception(f"Failed to list competitions: {e!s}")
