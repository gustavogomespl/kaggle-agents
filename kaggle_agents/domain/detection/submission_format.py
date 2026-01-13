"""
Submission format detection.

Contains methods for detecting submission format from sample_submission.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .constants import IMAGE_EXTS


if TYPE_CHECKING:
    from ...core.state import CompetitionInfo, SubmissionFormatType


class SubmissionFormatMixin:
    """Mixin providing submission format detection methods."""

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
