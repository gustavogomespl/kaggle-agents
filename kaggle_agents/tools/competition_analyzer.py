"""
Competition Analyzer for Auto-Detecting Problem Type and Metric.

This module analyzes Kaggle competitions to automatically detect:
- Problem type (binary_classification, multiclass_classification, regression)
- Evaluation metric (accuracy, auc, rmse, etc.)
- Data information (shapes, target column, etc.)
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


class CompetitionAnalyzer:
    """Auto-detect competition configuration from Kaggle API and data."""

    def __init__(self):
        """Initialize analyzer with Kaggle API."""
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except Exception as e:
            raise RuntimeError(
                f"Kaggle API authentication failed: {str(e)}\n"
                "Ensure KAGGLE_USERNAME and KAGGLE_KEY are set."
            )

    def analyze(self, competition_name: str) -> Dict[str, Any]:
        """
        Analyze competition and auto-detect configuration.

        Args:
            competition_name: Kaggle competition name/slug

        Returns:
            Dictionary with:
                - problem_type: str (binary_classification, multiclass_classification, regression)
                - metric: str (accuracy, auc, rmse, etc.)
                - confidence: float (0-1)
                - reasoning: dict with detection rationale
                - competition_info: dict with Kaggle metadata
                - data_info: dict with dataset information

        Raises:
            Exception: If competition not found or analysis fails
        """
        print(f"🔍 Analyzing competition: {competition_name}")

        # 1. Get competition metadata from Kaggle API
        competition_info = self._get_competition_info(competition_name)
        print(f"   ✓ Retrieved competition metadata")

        # 2. Download and analyze sample submission
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            try:
                # Download sample_submission.csv
                print(f"   ⏳ Downloading sample data...")
                self.api.competition_download_file(
                    competition_name, "sample_submission.csv", path=str(tmppath)
                )

                # Unzip if needed
                self._unzip_files(tmppath)

                # Find sample submission file
                sample_sub_path = self._find_file(tmppath, "sample_submission")

                if sample_sub_path:
                    print(f"   ✓ Sample submission found")
                    problem_type, metric, reasoning = self._analyze_sample_submission(
                        sample_sub_path, competition_info
                    )
                else:
                    print(f"   ⚠️  No sample_submission.csv, using metadata only")
                    problem_type, metric, reasoning = self._infer_from_metadata(
                        competition_info
                    )

            except Exception as e:
                print(f"   ⚠️  Sample download failed: {str(e)}")
                print(f"   → Using metadata-only inference")
                problem_type, metric, reasoning = self._infer_from_metadata(
                    competition_info
                )

        # 3. Get additional data info if possible
        data_info = self._get_data_info(competition_name)

        return {
            "problem_type": problem_type,
            "metric": metric,
            "confidence": reasoning.get("confidence", 0.5),
            "reasoning": reasoning,
            "competition_info": competition_info,
            "data_info": data_info,
        }

    def _get_competition_info(self, competition_name: str) -> Dict[str, Any]:
        """Get competition metadata from Kaggle API."""
        try:
            # Search for competition
            competitions = self.api.competitions_list(search=competition_name)

            # Find exact match
            comp = None
            for c in competitions:
                if c.ref == competition_name or c.ref.endswith(f"/{competition_name}"):
                    comp = c
                    break

            if not comp and competitions:
                # Use first result
                comp = competitions[0]

            if not comp:
                raise Exception(f"Competition '{competition_name}' not found")

            return {
                "name": comp.ref,
                "title": getattr(comp, "title", competition_name),
                "description": getattr(comp, "description", ""),
                "evaluationMetric": getattr(comp, "evaluationMetric", "unknown"),
                "category": getattr(comp, "category", "unknown"),
                "deadline": str(getattr(comp, "deadline", "N/A")),
                "reward": getattr(comp, "reward", "N/A"),
            }

        except Exception as e:
            raise Exception(f"Failed to get competition info: {str(e)}")

    def _analyze_sample_submission(
        self, filepath: Path, competition_info: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Analyze sample_submission.csv to detect problem type and metric."""
        df = pd.read_csv(filepath)

        # Get target column (usually last column, skip ID column)
        target_col = df.columns[-1]

        # Get unique values
        unique_values = df[target_col].nunique()
        sample_values = df[target_col].head(100)

        reasoning = {"target_column": target_col, "unique_values": unique_values}

        # Official metric from Kaggle
        official_metric = competition_info.get("evaluationMetric", "").lower()

        # Detect problem type based on target column
        if unique_values == 2:
            # Binary classification
            problem_type = "binary_classification"
            reasoning["problem_type"] = f"Target has 2 unique values: {set(sample_values.unique())}"
            reasoning["confidence"] = 0.95

            # Map metric
            if "auc" in official_metric or "roc" in official_metric:
                metric = "auc"
            elif "log" in official_metric:
                metric = "logloss"
            else:
                metric = "accuracy"

        elif 3 <= unique_values <= 20:
            # Multiclass classification
            problem_type = "multiclass_classification"
            reasoning["problem_type"] = f"Target has {unique_values} classes"
            reasoning["confidence"] = 0.90

            # Map metric
            if "log" in official_metric:
                metric = "logloss"
            else:
                metric = "accuracy"

        else:
            # Regression or many-class classification
            if df[target_col].dtype in ["float64", "float32"]:
                # Regression
                problem_type = "regression"
                reasoning["problem_type"] = f"Target is continuous (dtype: {df[target_col].dtype})"
                reasoning["confidence"] = 0.85

                # Map metric
                if "rmse" in official_metric or "root mean" in official_metric:
                    metric = "rmse"
                elif "mae" in official_metric or "absolute" in official_metric:
                    metric = "mae"
                elif "r2" in official_metric or "r-squared" in official_metric:
                    metric = "r2"
                elif "mape" in official_metric:
                    metric = "mape"
                else:
                    metric = "rmse"  # Default

            else:
                # Many-class classification
                problem_type = "multiclass_classification"
                reasoning["problem_type"] = f"Target has {unique_values} classes (many-class)"
                reasoning["confidence"] = 0.75
                metric = "accuracy"

        reasoning["metric"] = f"Official Kaggle metric: {competition_info.get('evaluationMetric', 'unknown')}"

        return problem_type, metric, reasoning

    def _infer_from_metadata(
        self, competition_info: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Infer problem type and metric from competition metadata only."""
        official_metric = competition_info.get("evaluationMetric", "").lower()
        description = competition_info.get("description", "").lower()
        title = competition_info.get("title", "").lower()

        reasoning = {
            "source": "metadata_only",
            "official_metric": competition_info.get("evaluationMetric"),
        }

        # Map common metrics to problem types
        metric_mappings = {
            "accuracy": ("binary_classification", "accuracy", 0.7),
            "auc": ("binary_classification", "auc", 0.8),
            "roc auc": ("binary_classification", "auc", 0.8),
            "log loss": ("binary_classification", "logloss", 0.8),
            "logloss": ("binary_classification", "logloss", 0.8),
            "rmse": ("regression", "rmse", 0.8),
            "mae": ("regression", "mae", 0.8),
            "mape": ("regression", "mape", 0.8),
            "r2": ("regression", "r2", 0.8),
            "f1": ("binary_classification", "f1", 0.7),
        }

        # Try to match official metric
        for key, (ptype, metric, conf) in metric_mappings.items():
            if key in official_metric:
                reasoning["problem_type"] = f"Inferred from metric: {official_metric}"
                reasoning["confidence"] = conf
                return ptype, metric, reasoning

        # Fallback: look for keywords in description/title
        if any(word in description or word in title for word in ["classify", "classification", "predict class"]):
            problem_type = "binary_classification"
            metric = "accuracy"
            reasoning["problem_type"] = "Keywords in description suggest classification"
            reasoning["confidence"] = 0.5
        elif any(word in description or word in title for word in ["predict", "regression", "price", "value"]):
            problem_type = "regression"
            metric = "rmse"
            reasoning["problem_type"] = "Keywords in description suggest regression"
            reasoning["confidence"] = 0.5
        else:
            # Ultimate fallback
            problem_type = "binary_classification"
            metric = "accuracy"
            reasoning["problem_type"] = "Default fallback (insufficient information)"
            reasoning["confidence"] = 0.3

        return problem_type, metric, reasoning

    def _get_data_info(self, competition_name: str) -> Dict[str, Any]:
        """Get information about competition data files."""
        try:
            # List data files
            files = self.api.competition_list_files(competition_name)

            data_info = {"files": []}

            for f in files.files:
                file_info = {
                    "name": f.name,
                    "size": getattr(f, "totalBytes", 0),
                    "size_mb": getattr(f, "totalBytes", 0) / 1024 / 1024,
                }
                data_info["files"].append(file_info)

            return data_info

        except Exception as e:
            return {"files": [], "error": str(e)}

    def _unzip_files(self, directory: Path):
        """Unzip all .zip files in directory."""
        for zip_path in directory.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(directory)
                zip_path.unlink()  # Remove zip after extraction
            except zipfile.BadZipFile:
                pass  # Not a valid zip file, skip

    def _find_file(self, directory: Path, pattern: str) -> Path:
        """Find file matching pattern in directory."""
        for file_path in directory.rglob("*"):
            if file_path.is_file() and pattern.lower() in file_path.name.lower():
                return file_path
        return None


# Convenience function
def auto_detect_competition_config(competition_name: str) -> Dict[str, Any]:
    """
    Auto-detect competition configuration.

    Args:
        competition_name: Kaggle competition name

    Returns:
        Configuration dictionary

    Example:
        >>> config = auto_detect_competition_config("titanic")
        >>> print(config["problem_type"])  # "binary_classification"
        >>> print(config["metric"])  # "accuracy"
    """
    analyzer = CompetitionAnalyzer()
    return analyzer.analyze(competition_name)
