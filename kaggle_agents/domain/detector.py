"""
Domain Detection Module for Kaggle Competitions.

Uses LLM to classify competition domain based on description and file metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from ..core.state import CompetitionInfo, DomainType, SubmissionFormatType
from ..utils.llm_utils import get_text_content


# Image extensions for format detection
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}


class DomainDetector:
    """
    Detects the domain/type of a Kaggle competition using LLM.

    Supports granular domain classification for various competition types.
    """

    DOMAINS = [
        # Image-based
        "image_classification",
        "image_regression",
        "image_to_image",
        "image_segmentation",
        "object_detection",
        # Text-based
        "text_classification",
        "seq_to_seq",
        "text_regression",
        # Audio-based
        "audio_classification",
        "audio_regression",
        # Tabular
        "tabular_classification",
        "tabular_regression",
        # Time series
        "time_series_forecasting",
        # Multi-modal
        "multi_modal",
    ]

    PROMPT = """Classify this Kaggle competition into exactly ONE category.

Categories:
- image_classification: Classify images into categories (dog breeds, cancer detection, plant diseases)
- image_regression: Predict continuous values from images (age estimation, severity scores)
- image_to_image: Transform images (denoising, super-resolution, style transfer) or pixel-level (one row per pixel) if it's an image-to-image task and the sample submission has pixel-level format in csv
- image_segmentation: Pixel-wise classification of images
- object_detection: Locate and classify objects in images
- text_classification: Classify text (sentiment, toxicity, spam, author identification)
- seq_to_seq: Sequence to sequence (translation, text normalization, summarization)
- text_regression: Predict continuous values from text
- audio_classification: Classify audio signals (speaker, music genre, species by sound)
- audio_regression: Predict continuous values from audio
- tabular_classification: Classify rows in structured CSV data
- tabular_regression: Predict continuous values from structured CSV data
- time_series_forecasting: Predict future values from temporal sequences
- multi_modal: Combination of multiple data types (images + text + tabular)

Competition Name: {name}
Description: {description}
Data Files: {files}

IMPORTANT CLUES FOR DETECTION:
- Directories ending with "/" containing .jpg/.png files → image_* domain
- Directories with .wav/.mp3 files → audio_* domain
- Directories with .txt files → text_* domain
- Only .csv/.parquet files with no directories → tabular_* domain

Respond with ONLY the category name, nothing else. Example: image_classification"""

    DESCRIPTIONS = {
        "image_classification": "Classify images into discrete categories",
        "image_regression": "Predict continuous values from images",
        "image_to_image": "Transform images (denoising, super-resolution, style transfer) or pixel-level (one row per pixel)",
        "image_segmentation": "Pixel-wise classification of images",
        "object_detection": "Locate and classify objects in images",
        "text_classification": "Classify text into categories",
        "seq_to_seq": "Sequence to sequence transformation (translation, normalization)",
        "text_regression": "Predict continuous values from text",
        "audio_classification": "Classify audio signals",
        "audio_regression": "Predict continuous values from audio",
        "tabular_classification": "Classify rows in structured tabular data",
        "tabular_regression": "Predict continuous values from tabular data",
        "time_series_forecasting": "Predict future values from temporal data",
        "multi_modal": "Combination of multiple data types",
    }

    def __init__(self, llm: "BaseChatModel | None" = None):
        """
        Initialize the domain detector.

        Args:
            llm: LangChain LLM client. If None, defaults to tabular domain.
        """
        self.llm = llm

    def _detect_from_structure(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float]:
        """Heuristic detection from local files when no LLM is available."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        text_exts = {".txt", ".json"}
        tabular_exts = {".csv", ".parquet"}

        def classify(counts: dict[str, int], total: int) -> tuple[DomainType, float] | None:
            if total == 0:
                return None

            image_ratio = sum(counts.get(ext, 0) for ext in image_exts) / total
            audio_ratio = sum(counts.get(ext, 0) for ext in audio_exts) / total
            text_ratio = sum(counts.get(ext, 0) for ext in text_exts) / total
            tabular_ratio = sum(counts.get(ext, 0) for ext in tabular_exts) / total

            if image_ratio >= 0.3:
                return ("image_classification", 0.90)
            if audio_ratio >= 0.3:
                return ("audio_classification", 0.85)
            if text_ratio >= 0.3:
                # Use regression hint if problem_type mentions it
                if "regression" in (competition_info.problem_type or "").lower():
                    return ("text_regression", 0.80)
                return ("text_classification", 0.80)
            if tabular_ratio >= 0.5:
                if "regression" in (competition_info.problem_type or "").lower():
                    return ("tabular_regression", 0.80)
                return ("tabular_classification", 0.80)

            return None

        def analyze_dir(dir_path: Path) -> tuple[dict[str, int], int]:
            counts: dict[str, int] = {}
            total = 0
            for i, file_path in enumerate(dir_path.rglob("*")):
                if i >= 600:
                    break
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext:
                        counts[ext] = counts.get(ext, 0) + 1
                        total += 1
            return counts, total

        # Prefer train/test folders if present
        candidate_dirs = (
            [
                p
                for p in data_dir.iterdir()
                if p.is_dir() and p.name.lower().startswith(("train", "test"))
            ]
            if data_dir.exists()
            else []
        )

        # Image-to-image heuristic: look for paired train + clean/target directories
        if data_dir.exists():
            dir_map = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
            clean_dir_names = [
                "train_cleaned",
                "train_clean",
                "clean",
                "cleaned",
                "gt",
                "ground_truth",
                "train_gt",
                "target",
                "targets",
                "train_target",
            ]
            train_dir = None
            for name in ("train", "training", "train_images"):
                if name in dir_map:
                    train_dir = dir_map[name]
                    break
            clean_dir = None
            for name in clean_dir_names:
                if name in dir_map:
                    clean_dir = dir_map[name]
                    break

            if train_dir and clean_dir:
                train_counts, train_total = analyze_dir(train_dir)
                clean_counts, clean_total = analyze_dir(clean_dir)
                train_result = classify(train_counts, train_total)
                clean_result = classify(clean_counts, clean_total)
                if train_result and clean_result:
                    if train_result[0].startswith("image_") and clean_result[0].startswith(
                        "image_"
                    ):
                        return ("image_to_image", 0.92)

        for dir_path in candidate_dirs:
            counts, total = analyze_dir(dir_path)
            result = classify(counts, total)
            if result:
                return result

        # Fall back to scanning the whole directory tree
        if data_dir.exists():
            counts, total = analyze_dir(data_dir)
            result = classify(counts, total)
            if result:
                return result

        # Default tabular guess
        if "regression" in (competition_info.problem_type or "").lower():
            return ("tabular_regression", 0.50)
        return ("tabular_classification", 0.50)

    def _detect_data_type(self, data_dir: Path) -> str:
        """
        Detect the primary data type from file structure.

        Returns one of: "image", "audio", "text", "tabular"
        """
        if not data_dir.exists():
            return "tabular"

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        text_exts = {".txt", ".json"}

        counts: dict[str, int] = {"image": 0, "audio": 0, "text": 0, "tabular": 0}

        # Check direct files and subdirectories
        for path in data_dir.iterdir():
            if path.is_file():
                ext = path.suffix.lower()
                if ext in image_exts:
                    counts["image"] += 1
                elif ext in audio_exts:
                    counts["audio"] += 1
                elif ext in text_exts:
                    counts["text"] += 1
                elif ext in {".csv", ".parquet"}:
                    counts["tabular"] += 1
            elif path.is_dir():
                # Sample first 100 files in subdirectory
                for i, subfile in enumerate(path.rglob("*")):
                    if i >= 100:
                        break
                    if subfile.is_file():
                        ext = subfile.suffix.lower()
                        if ext in image_exts:
                            counts["image"] += 10  # Weight directories higher
                        elif ext in audio_exts:
                            counts["audio"] += 10
                        elif ext in text_exts:
                            counts["text"] += 10

        # Return dominant type
        if counts["image"] > max(counts["audio"], counts["text"], counts["tabular"]):
            return "image"
        if counts["audio"] > max(counts["text"], counts["tabular"]):
            return "audio"
        if counts["text"] > counts["tabular"]:
            return "text"
        return "tabular"

    def _find_train_file(self, data_dir: Path) -> Path | None:
        """Find a train.csv/train.parquet file in the data directory or one level below."""
        candidates = [data_dir / "train.csv", data_dir / "train.parquet"]
        for path in candidates:
            if path.exists():
                return path

        if data_dir.exists():
            for child in data_dir.iterdir():
                if not child.is_dir():
                    continue
                for name in ("train.csv", "train.parquet"):
                    path = child / name
                    if path.exists():
                        return path
        return None

    def _detect_tabular_from_csv(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float] | None:
        """
        Heuristic override: if train.csv is wide with many numeric columns,
        treat the competition as tabular even if images exist.
        """
        train_path = self._find_train_file(data_dir)
        if not train_path:
            return None

        try:
            if train_path.suffix.lower() == ".parquet":
                if train_path.stat().st_size > 200 * 1024 * 1024:
                    return None
                df = pd.read_parquet(train_path)
            else:
                df = pd.read_csv(train_path, nrows=50)
        except Exception:
            return None

        if df.empty:
            return None

        n_cols = len(df.columns)
        if n_cols < 8:
            return None

        numeric_cols = df.select_dtypes(include="number").columns
        numeric_ratio = len(numeric_cols) / max(n_cols, 1)

        # Strong signal: many numeric feature columns
        if len(numeric_cols) >= 10 or (n_cols >= 12 and numeric_ratio >= 0.5):
            is_regression = "regression" in (competition_info.problem_type or "").lower()
            return ("tabular_regression", 0.92) if is_regression else ("tabular_classification", 0.92)

        return None

    def detect(
        self,
        competition_info: CompetitionInfo,
        data_directory: Path | str,
    ) -> tuple[DomainType, float]:
        """
        Detect the domain type of a competition using LLM-First approach.

        Strategy:
        1. If no LLM, use structural heuristics (_detect_from_structure)
        2. If LLM available, use it for granular domain classification
        3. Fallback to structural heuristics if LLM fails

        Args:
            competition_info: Competition metadata
            data_directory: Path to competition data files

        Returns:
            Tuple of (detected_domain, confidence_score)
        """
        data_dir = Path(data_directory) if isinstance(data_directory, str) else data_directory

        # Step 1: If no LLM, use sophisticated structural heuristics
        # (detects image_to_image, segmentation, time_series, etc.)
        if self.llm is None:
            tabular_override = self._detect_tabular_from_csv(competition_info, data_dir)
            if tabular_override:
                return tabular_override
            return self._detect_from_structure(competition_info, data_dir)

        # Step 2: Detect data type for context (used in LLM prompt)
        data_type = self._detect_data_type(data_dir)

        # Tabular override for mixed datasets (e.g., images + rich numeric features)
        tabular_override = self._detect_tabular_from_csv(competition_info, data_dir)
        if tabular_override:
            return tabular_override

        # Step 3: Use LLM for granular domain classification (LLM-First)
        # Scan files to provide context to LLM
        files = []
        if data_dir.exists():
            for path in data_dir.glob("*"):
                if path.is_file():
                    files.append(path.name)
                elif path.is_dir():
                    # Analyze directory contents
                    contents = list(path.glob("*"))[:100]
                    if contents:
                        extensions: dict[str, int] = {}
                        for item in contents:
                            ext = item.suffix.lower()
                            extensions[ext] = extensions.get(ext, 0) + 1
                        if extensions:
                            dominant = max(extensions.items(), key=lambda x: x[1])
                            files.append(
                                f"{path.name}/ ({len(contents)} files, mostly {dominant[0]})"
                            )
            files = files[:20]  # Limit to 20 entries

        # Enhanced prompt with data type hint
        prompt = f"""Classify this Kaggle competition into exactly ONE category.

Categories:
- image_classification: Classify images into categories (dog breeds, cancer detection, plant diseases, species identification)
- image_regression: Predict continuous values from images (age estimation, severity scores)
- image_to_image: Transform images (denoising, super-resolution, style transfer) or pixel-level predictions
- image_segmentation: Pixel-wise classification of images (mask prediction)
- object_detection: Locate AND classify objects with bounding boxes in images
- text_classification: Classify text (sentiment, toxicity, spam, author identification)
- seq_to_seq: Sequence to sequence (translation, text normalization, summarization)
- text_regression: Predict continuous values from text
- audio_classification: Classify audio signals (speaker, music genre, species by sound)
- audio_regression: Predict continuous values from audio
- tabular_classification: Classify rows in structured CSV data
- tabular_regression: Predict continuous values from structured CSV data
- time_series_forecasting: Predict future values from temporal sequences
- multi_modal: Combination of multiple data types (images + text + tabular)

Competition Name: {competition_info.name}
Description: {(competition_info.description or "")[:800]}
Data Files: {files if files else ["No files found"]}
Detected Data Type: {data_type}

IMPORTANT CLASSIFICATION RULES:
1. "breed", "species", "identify", "classify", "categorize" → usually image_classification or text_classification
2. object_detection REQUIRES bounding box predictions (x, y, width, height)
3. image_segmentation REQUIRES pixel-wise masks or RLE encoding
4. If task is to identify/classify items in images WITHOUT bounding boxes → image_classification

Respond with ONLY the category name, nothing else. Example: image_classification"""

        try:
            response = self.llm.invoke(prompt)
            content = (
                get_text_content(response.content)
                if hasattr(response, "content")
                else str(response)
            )
            domain = content.strip().lower().replace(" ", "_")

            if domain in self.DOMAINS:
                return domain, 0.95  # type: ignore
            # LLM returned invalid domain, use structural heuristics
            return self._detect_from_structure(competition_info, data_dir)

        except Exception:
            # LLM failed, use structural heuristics
            return self._detect_from_structure(competition_info, data_dir)

    def get_domain_description(self, domain: DomainType) -> str:
        """Get a human-readable description of a domain."""
        return self.DESCRIPTIONS.get(domain, "Unknown domain type")

    def detect_submission_format(
        self,
        sample_submission_path: Path | str,
        test_dir: Path | str | None = None,
        competition_info: CompetitionInfo | None = None,
    ) -> tuple[SubmissionFormatType, dict[str, Any]]:
        """
        Detect the submission format by analyzing sample_submission.csv.

        This is critical for image-to-image tasks where submission format is
        pixel-level (one row per pixel) rather than standard (one row per sample).

        Args:
            sample_submission_path: Path to sample_submission.csv
            test_dir: Optional path to test data directory
            competition_info: Optional competition metadata

        Returns:
            Tuple of (format_type, metadata)
            metadata includes: expected_rows, id_pattern, pixel_format_detected, etc.
        """
        sample_path = Path(sample_submission_path)
        test_path = Path(test_dir) if test_dir else None

        metadata: dict[str, Any] = {
            "expected_rows": 0,
            "n_test_samples": 0,
            "id_column": "",
            "value_columns": [],
            "id_pattern": None,
            "pixel_format_detected": False,
        }

        # Read sample submission
        if not sample_path.exists():
            return "standard", metadata

        try:
            sample_sub = pd.read_csv(sample_path)
        except Exception:
            return "standard", metadata

        n_rows = len(sample_sub)
        metadata["expected_rows"] = n_rows

        if len(sample_sub.columns) == 0:
            return "standard", metadata

        id_col = sample_sub.columns[0]
        metadata["id_column"] = id_col
        metadata["value_columns"] = list(sample_sub.columns[1:])

        # Count test samples (images or files)
        n_test_samples = 0
        if test_path and test_path.exists():
            if test_path.is_dir():
                # Count test images
                test_files = list(test_path.glob("*"))
                n_test_samples = len(
                    [f for f in test_files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                )
                # If no images found, count all files
                if n_test_samples == 0:
                    n_test_samples = len([f for f in test_files if f.is_file()])

        metadata["n_test_samples"] = n_test_samples

        # Heuristic 1: If rows >> test_samples, likely pixel-level
        if n_test_samples > 0 and n_rows > n_test_samples * 100:
            # Check ID pattern for pixel format (e.g., "1_1_1" = image_row_col)
            sample_ids = sample_sub[id_col].astype(str).head(20).tolist()

            # Check if IDs contain underscores (common pixel format: image_row_col)
            if sample_ids and all("_" in str(id_val) for id_val in sample_ids):
                parts = str(sample_ids[0]).split("_")
                if len(parts) >= 3:
                    metadata["id_pattern"] = "image_row_col"
                    metadata["pixel_format_detected"] = True
                    metadata["estimated_pixels_per_image"] = n_rows // n_test_samples
                    return "pixel_level", metadata
                if len(parts) == 2:
                    # Could be image_pixel_index format
                    metadata["id_pattern"] = "image_pixel"
                    metadata["pixel_format_detected"] = True
                    metadata["estimated_pixels_per_image"] = n_rows // n_test_samples
                    return "pixel_level", metadata

            # Even without underscore pattern, high ratio suggests pixel-level
            ratio = n_rows / n_test_samples
            if ratio > 1000:  # More than 1000 rows per test sample
                metadata["pixel_format_detected"] = True
                metadata["estimated_pixels_per_image"] = int(ratio)
                return "pixel_level", metadata

        # Heuristic 2: Check for RLE encoding pattern (segmentation)
        if "rle" in id_col.lower() or "EncodedPixels" in sample_sub.columns:
            metadata["id_pattern"] = "rle_encoded"
            return "rle_encoded", metadata

        # Heuristic 3: Check for multi-label format (multiple rows per sample)
        if n_test_samples > 0 and n_rows > n_test_samples * 2:
            # Could be multi-label, check for repeated IDs
            sample_ids = sample_sub[id_col].head(100)
            if sample_ids.duplicated().any():
                metadata["id_pattern"] = "multi_label"
                return "multi_label", metadata

        # Default: standard format (one row per sample)
        return "standard", metadata


# ==================== Convenience Function ====================


def detect_competition_domain(
    competition_info: CompetitionInfo,
    data_directory: Path | str,
    llm: "BaseChatModel | None" = None,
) -> tuple[DomainType, float]:
    """
    Convenience function to detect competition domain.

    Args:
        competition_info: Competition metadata
        data_directory: Path to competition data
        llm: Optional LLM client

    Returns:
        Tuple of (domain_type, confidence)
    """
    detector = DomainDetector(llm=llm)
    return detector.detect(competition_info, data_directory)


def detect_submission_format(
    sample_submission_path: Path | str,
    test_dir: Path | str | None = None,
    competition_info: CompetitionInfo | None = None,
) -> tuple[SubmissionFormatType, dict[str, Any]]:
    """
    Convenience function to detect submission format.

    Critical for distinguishing between:
    - Standard format (one row per sample) - most competitions
    - Pixel-level format (one row per pixel) - image-to-image, segmentation

    Args:
        sample_submission_path: Path to sample_submission.csv
        test_dir: Optional path to test data directory
        competition_info: Optional competition metadata

    Returns:
        Tuple of (format_type, metadata)
    """
    detector = DomainDetector()
    return detector.detect_submission_format(sample_submission_path, test_dir, competition_info)
