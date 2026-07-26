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
        has_image_test_files = False
        if test_path and test_path.exists():
            if test_path.is_dir():
                # Count test images
                test_files = list(test_path.glob("*"))
                n_test_images = len(
                    [f for f in test_files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                )
                n_test_samples = n_test_images
                has_image_test_files = n_test_images > 0
                # If no images found, count all files
                if n_test_samples == 0:
                    n_test_samples = len([f for f in test_files if f.is_file()])

        metadata["n_test_samples"] = n_test_samples

        # Detect repeated sample prefixes followed by numeric coordinate/index
        # suffixes. This uses only the observed template structure and never
        # assumes a coordinate base. Structure alone is not enough: ordinary
        # per-sample IDs such as "Test_0" or "ISIC_0052060" also match the
        # prefix+numeric-suffix shape, so pixel-level additionally requires
        # far more template rows than test samples (one row per pixel).
        sample_ids = sample_sub[id_col].astype(str).head(200).tolist()
        parsed_prefixes: list[str] = []
        suffix_widths: set[int] = set()
        for sample_id in sample_ids:
            parts = sample_id.rsplit("_", 2)
            if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
                parsed_prefixes.append(parts[0])
                suffix_widths.add(2)
                continue
            parts = sample_id.rsplit("_", 1)
            if len(parts) == 2 and parts[-1].isdigit():
                parsed_prefixes.append(parts[0])
                suffix_widths.add(1)
                continue
            parsed_prefixes = []
            suffix_widths.clear()
            break

        suffix_width = next(iter(suffix_widths)) if len(suffix_widths) == 1 else None
        if (
            parsed_prefixes
            and suffix_width is not None
            and len(set(parsed_prefixes)) < len(parsed_prefixes)
            and n_test_samples > 0
            and len(set(parsed_prefixes)) <= n_test_samples
            and n_rows > n_test_samples * 100
            and (suffix_width == 2 or has_image_test_files)
        ):
            metadata["id_pattern"] = (
                "prefix_two_numeric_suffixes"
                if suffix_width == 2
                else "prefix_numeric_suffix"
            )
            metadata["pixel_format_detected"] = True
            if n_test_samples > 0:
                metadata["estimated_pixels_per_image"] = n_rows // n_test_samples
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
