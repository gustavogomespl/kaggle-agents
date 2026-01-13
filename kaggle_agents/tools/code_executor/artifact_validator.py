"""
Artifact validation for code execution.

Contains the ArtifactValidator class for validating generated artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ArtifactValidator:
    """
    Validate generated artifacts (models, data files, submissions).

    This class provides domain-specific validation:
    - Model files (.pkl, .joblib)
    - Data files (train/test CSVs)
    - Submission files
    - Checkpoints
    """

    def __init__(self):
        """Initialize artifact validator."""
        pass

    def validate_model_artifacts(self, working_dir: str) -> dict[str, Any]:
        """
        Validate that model artifacts were created correctly.

        Args:
            working_dir: Working directory

        Returns:
            Validation results dictionary
        """
        working_path = Path(working_dir) if isinstance(working_dir, str) else working_dir

        results = {
            "valid": False,
            "models_found": [],
            "issues": [],
        }

        # Check for model files
        model_extensions = [".pkl", ".joblib", ".h5", ".pth", ".pt"]
        models_dir = working_path / "models"

        model_files = []
        if models_dir.exists():
            for ext in model_extensions:
                model_files.extend(models_dir.glob(f"*{ext}"))

        if not model_files:
            results["issues"].append("No model files found in models/ directory")
        else:
            results["models_found"] = [f.name for f in model_files]
            results["valid"] = True

        return results

    def validate_submission_artifact(
        self,
        working_dir: str,
        expected_columns: list = None,
    ) -> dict[str, Any]:
        """
        Validate submission file.

        Args:
            working_dir: Working directory
            expected_columns: Expected column names

        Returns:
            Validation results
        """
        import pandas as pd

        working_path = Path(working_dir) if isinstance(working_dir, str) else working_dir

        results = {
            "valid": False,
            "submission_path": None,
            "row_count": 0,
            "columns": [],
            "issues": [],
        }

        # Check for submission.csv
        submission_file = working_path / "submission.csv"

        if not submission_file.exists():
            results["issues"].append("submission.csv not found")
            return results

        try:
            # Read and validate
            df = pd.read_csv(submission_file)

            results["submission_path"] = str(submission_file)
            results["row_count"] = len(df)
            results["columns"] = df.columns.tolist()

            # Check for empty
            if len(df) == 0:
                results["issues"].append("Submission file is empty")
                return results

            # Check expected columns
            if expected_columns:
                missing = set(expected_columns) - set(df.columns)
                if missing:
                    results["issues"].append(f"Missing columns: {missing}")
                    return results

            # Check for NaN values
            if df.isnull().any().any():
                results["issues"].append("Submission contains NaN values")
                # Allow warning but not failure for NaNs

            results["valid"] = True

        except Exception as e:
            results["issues"].append(f"Error reading submission: {e!s}")

        return results
