"""
Domain Detection Module for Kaggle Competitions.

Automatically detects the type of competition (tabular, CV, NLP, time series, multi-modal)
based on data files, competition description, and data analysis.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd

from ..core.state import DomainType, CompetitionInfo


class DomainDetector:
    """
    Detects the domain/type of a Kaggle competition.

    This class analyzes competition metadata, file types, and data samples
    to determine whether the competition is:
    - Tabular (structured data with features)
    - Computer Vision (images)
    - NLP (text/language)
    - Time Series (temporal data)
    - Multi-modal (combination of above)
    """

    # Keywords for domain detection in competition descriptions
    DOMAIN_KEYWORDS = {
        "computer_vision": [
            "image",
            "images",
            "vision",
            "computer vision",
            "cv",
            "cnn",
            "object detection",
            "segmentation",
            "classification of images",
            "photo",
            "photos",
            "visual",
            "pixel",
            "resnet",
            "vgg",
            "yolo",
            "unet",
            "detection",
            "recognition",
        ],
        "nlp": [
            "text",
            "language",
            "nlp",
            "natural language",
            "sentiment",
            "translation",
            "question answering",
            "qa",
            "named entity",
            "ner",
            "topic modeling",
            "document",
            "corpus",
            "tokenization",
            "bert",
            "gpt",
            "transformer",
            "embedding",
            "word",
            "sentence",
        ],
        "time_series": [
            "time series",
            "temporal",
            "forecasting",
            "forecast",
            "prediction",
            "timeseries",
            "sequential",
            "time-dependent",
            "datetime",
            "trend",
            "seasonality",
            "lag",
            "arima",
            "lstm for time",
            "sales forecast",
            "demand forecast",
            "stock price",
        ],
    }

    # File extensions indicating domain
    FILE_EXTENSIONS = {
        "computer_vision": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"},
        "nlp": {".txt", ".json", ".xml", ".html"},
    }

    def __init__(self):
        """Initialize the domain detector."""
        pass

    def detect(
        self,
        competition_info: CompetitionInfo,
        data_directory: Path | str,
    ) -> Tuple[DomainType, float]:
        """
        Detect the domain type of a competition.

        Args:
            competition_info: Competition metadata
            data_directory: Path to competition data files

        Returns:
            Tuple of (detected_domain, confidence_score)
            confidence_score is between 0 and 1
        """
        data_dir = (
            Path(data_directory) if isinstance(data_directory, str) else data_directory
        )

        # Score each domain
        scores = {
            "tabular": 0.0,
            "computer_vision": 0.0,
            "nlp": 0.0,
            "time_series": 0.0,
            "multi_modal": 0.0,
        }

        # 1. Analyze competition description and name
        description_text = (
            f"{competition_info.name} {competition_info.description}".lower()
        )

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            keyword_matches = sum(1 for kw in keywords if kw in description_text)
            scores[domain] += keyword_matches * 10  # High weight for description

        # 2. Analyze file extensions
        if data_dir.exists():
            all_files = list(data_dir.rglob("*"))

            # Count file types
            image_files = sum(
                1
                for f in all_files
                if f.suffix.lower() in self.FILE_EXTENSIONS["computer_vision"]
            )
            text_files = sum(
                1 for f in all_files if f.suffix.lower() in self.FILE_EXTENSIONS["nlp"]
            )
            csv_files = sum(1 for f in all_files if f.suffix.lower() == ".csv")

            if image_files > 0:
                scores["computer_vision"] += image_files * 5

            if text_files > 0:
                scores["nlp"] += text_files * 5

            if csv_files > 0:
                # CSV could be tabular, time series, or even contain file paths
                # Analyze CSV content to determine
                csv_analysis_score = self._analyze_csv_files(data_dir)
                for domain, score in csv_analysis_score.items():
                    scores[domain] += score

        # 3. Detect multi-modal
        # If multiple domains have high scores, it's likely multi-modal
        high_scoring_domains = [
            domain for domain, score in scores.items() if score > 20
        ]
        if len(high_scoring_domains) >= 2:
            scores["multi_modal"] = (
                sum(scores[d] for d in high_scoring_domains if d != "multi_modal") * 0.5
            )

        # 4. Default to tabular if no strong signals
        if max(scores.values()) < 10:
            scores["tabular"] = 30  # Default assumption

        # Calculate confidence and determine winner
        total_score = sum(scores.values())
        if total_score == 0:
            return "tabular", 0.5  # Default with low confidence

        max_domain = max(scores, key=scores.get)
        confidence = scores[max_domain] / total_score

        return max_domain, confidence  # type: ignore

    def _analyze_csv_files(self, data_dir: Path) -> dict[str, float]:
        """
        Analyze CSV files to provide domain scoring hints.

        Args:
            data_dir: Directory containing CSV files

        Returns:
            Dictionary with domain scores based on CSV analysis
        """
        scores = {
            "tabular": 0.0,
            "time_series": 0.0,
            "computer_vision": 0.0,
            "nlp": 0.0,
        }

        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return scores

        # Analyze first CSV (usually train.csv)
        try:
            # Read only first few rows for efficiency
            df = pd.read_csv(csv_files[0], nrows=100)

            # Check for datetime columns (time series indicator)
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns
            if len(datetime_cols) > 0:
                scores["time_series"] += 20

            # Check column names for datetime patterns
            date_patterns = [
                "date",
                "time",
                "timestamp",
                "datetime",
                "year",
                "month",
                "day",
            ]
            date_like_cols = [
                col
                for col in df.columns
                if any(p in col.lower() for p in date_patterns)
            ]
            if date_like_cols:
                scores["time_series"] += 10

            # Check for image path columns (CV indicator)
            path_patterns = ["image", "img", "photo", "file", "path", "filename"]
            path_cols = [
                col
                for col in df.columns
                if any(p in col.lower() for p in path_patterns)
            ]

            if path_cols:
                # Check if values look like file paths
                for col in path_cols:
                    sample_values = df[col].dropna().head(5).astype(str)
                    if any(
                        any(ext in val.lower() for ext in [".jpg", ".png", ".jpeg"])
                        for val in sample_values
                    ):
                        scores["computer_vision"] += 25
                        break

            # Check for text columns (NLP indicator)
            text_patterns = [
                "text",
                "comment",
                "review",
                "description",
                "content",
                "message",
                "title",
            ]
            text_cols = [
                col
                for col in df.columns
                if any(p in col.lower() for p in text_patterns)
            ]

            if text_cols:
                # Check if values are long strings (typical for NLP)
                for col in text_cols:
                    sample_values = df[col].dropna().head(5).astype(str)
                    avg_length = (
                        sum(len(val) for val in sample_values) / len(sample_values)
                        if sample_values.size > 0
                        else 0
                    )

                    if avg_length > 50:  # Long text suggests NLP
                        scores["nlp"] += 20
                        break

            # Default: if nothing specific detected, likely tabular
            if max(scores.values()) < 10:
                num_features = len(df.columns)
                num_numeric = len(df.select_dtypes(include=["number"]).columns)

                # High proportion of numeric columns suggests tabular
                if num_numeric / num_features > 0.5:
                    scores["tabular"] += 20

        except Exception:
            # If we can't read the CSV, assume tabular
            scores["tabular"] += 15

        return scores

    def get_domain_description(self, domain: DomainType) -> str:
        """
        Get a human-readable description of a domain.

        Args:
            domain: The domain type

        Returns:
            Description string
        """
        descriptions = {
            "tabular": "Structured data with features in tabular format (CSV/database-like)",
            "computer_vision": "Image-based tasks (classification, detection, segmentation)",
            "nlp": "Natural language processing tasks (text classification, NER, QA, etc.)",
            "time_series": "Temporal/sequential data requiring time-based modeling",
            "multi_modal": "Combination of multiple data types (e.g., images + text + tabular)",
        }
        return descriptions.get(domain, "Unknown domain type")


# ==================== Convenience Function ====================


def detect_competition_domain(
    competition_info: CompetitionInfo,
    data_directory: Path | str,
) -> Tuple[DomainType, float]:
    """
    Convenience function to detect competition domain.

    Args:
        competition_info: Competition metadata
        data_directory: Path to competition data

    Returns:
        Tuple of (domain_type, confidence)
    """
    detector = DomainDetector()
    return detector.detect(competition_info, data_directory)
