"""Tests for submission format detection."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.utils.submission_format import (
    SubmissionFormatInfo,
    detect_audio_submission_format,
    generate_submission_code_hint,
    _detect_multiplier_pattern,
)


class TestDetectAudioSubmissionFormat:
    """Tests for detect_audio_submission_format function."""

    def test_wide_format_detection(self, tmp_path: Path) -> None:
        """Test detection of wide format (BirdCLEF style)."""
        # Create sample submission with wide format
        df = pd.DataFrame({
            "row_id": ["audio_0001", "audio_0002", "audio_0003"],
            "species_0": [0.5, 0.5, 0.5],
            "species_1": [0.5, 0.5, 0.5],
            "species_2": [0.5, 0.5, 0.5],
        })
        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        assert result.format_type == "wide"
        assert result.id_column == "row_id"
        assert result.num_classes == 3
        assert "species_0" in result.target_columns
        assert "species_1" in result.target_columns
        assert "species_2" in result.target_columns

    def test_wide_format_binary_classification(self, tmp_path: Path) -> None:
        """Test detection of wide format with single target (binary classification).

        Two-column submissions (ID + single target) should be wide format
        unless a long-format ID pattern is detected.
        """
        # Create sample submission with 2 columns (binary classification)
        df = pd.DataFrame({
            "id": ["audio_0001", "audio_0002", "audio_0003", "audio_0004"],
            "target": [0.5, 0.5, 0.5, 0.5],
        })
        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        # Should be wide format (one row per sample) not long
        assert result.format_type == "wide", (
            f"Binary classification misclassified as {result.format_type}. "
            "Two-column submissions without long-format ID patterns should be wide."
        )
        assert result.id_column == "id"
        assert result.target_columns == ["target"]
        assert result.num_classes == 1

    def test_wide_format_single_probability_column(self, tmp_path: Path) -> None:
        """Test that standard single-column probability submissions are wide format."""
        # Numeric IDs without long-format pattern (sequential simple IDs)
        df = pd.DataFrame({
            "Id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Probability": [0.5] * 10,
        })
        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        # Sequential IDs (1,2,3...) don't match long-format patterns
        assert result.format_type == "wide"
        assert result.num_classes == 1

    def test_long_format_mlsp_style(self, tmp_path: Path) -> None:
        """Test detection of long format with MLSP ID pattern."""
        # Create sample submission with MLSP style (Id = rec_id * 100 + species_id)
        ids = []
        probs = []
        for rec_id in range(1, 4):  # 3 records
            for species_id in range(19):  # 19 species
                ids.append(rec_id * 100 + species_id)
                probs.append(0.5)

        df = pd.DataFrame({"Id": ids, "Probability": probs})
        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        assert result.format_type == "long"
        assert result.id_column == "Id"
        assert result.target_columns == ["Probability"]
        assert result.id_multiplier == 100
        assert result.num_classes == 19
        assert "rec_id * 100 + class_id" in result.id_pattern

    def test_numeric_dtype_detection_regression(self, tmp_path: Path) -> None:
        """Regression test: ensure numeric IDs are properly detected.

        This tests the fix for the bug where dtype comparison used strings
        instead of pd.api.types.is_numeric_dtype().
        """
        # Create MLSP-style submission with numeric IDs
        # Need at least 10 IDs for pattern detection
        ids = []
        for rec_id in range(1, 5):  # 4 records
            for species_id in range(5):  # 5 species = 20 IDs total
                ids.append(rec_id * 100 + species_id)

        df = pd.DataFrame({
            "Id": ids,
            "Probability": [0.5] * len(ids),
        })
        # Verify the dtype is actually numeric (int64)
        assert pd.api.types.is_numeric_dtype(df["Id"])

        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        # The critical assertion: numeric pattern MUST be detected
        assert result.id_multiplier == 100, (
            f"Numeric ID pattern not detected! Got id_multiplier={result.id_multiplier}, "
            f"id_pattern={result.id_pattern}. This indicates the dtype check is broken."
        )
        assert result.num_classes == 5
        assert result.format_type == "long"

    def test_long_format_underscore_pattern(self, tmp_path: Path) -> None:
        """Test detection of long format with underscore ID pattern.

        Valid long format requires:
        - Multiple unique prefixes (rec_1, rec_2, rec_3)
        - Each prefix appears with multiple small suffixes (0, 1, 2, 3, 4)
        - Forms a complete grid (num_rows = num_prefixes * num_suffixes)
        """
        # Create sample submission with underscore pattern (rec_id_class_id)
        ids = []
        probs = []
        for rec_id in range(1, 4):  # 3 unique prefixes
            for class_id in range(5):  # 5 class indices (0-4)
                ids.append(f"rec_{rec_id}_{class_id}")
                probs.append(0.5)

        df = pd.DataFrame({"Id": ids, "Probability": probs})
        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        assert result.format_type == "long"
        assert result.id_column == "Id"
        assert "underscore" in (result.id_pattern or "").lower()
        assert result.num_classes == 5

    def test_nonexistent_file(self) -> None:
        """Test handling of nonexistent file."""
        result = detect_audio_submission_format(Path("/nonexistent/path.csv"))

        assert result.format_type == "unknown"
        assert len(result.warnings) > 0

    def test_empty_dataframe(self, tmp_path: Path) -> None:
        """Test handling of empty submission file."""
        df = pd.DataFrame()
        sample_path = tmp_path / "sample_submission.csv"
        df.to_csv(sample_path, index=False)

        result = detect_audio_submission_format(sample_path)

        assert result.format_type == "unknown"


class TestDetectMultiplierPattern:
    """Tests for _detect_multiplier_pattern function."""

    def test_mlsp_pattern_100(self) -> None:
        """Test detection of MLSP pattern with multiplier 100."""
        # Generate MLSP-style IDs: rec_id * 100 + species_id
        ids = []
        for rec_id in range(1, 10):
            for species_id in range(19):
                ids.append(rec_id * 100 + species_id)

        multiplier, num_classes = _detect_multiplier_pattern(sorted(ids))

        assert multiplier == 100
        assert num_classes == 19

    def test_pattern_with_multiplier_1000(self) -> None:
        """Test detection of pattern with multiplier 1000 (more than 100 classes)."""
        # Use more than 100 classes so multiplier 100 doesn't work
        ids = []
        for rec_id in range(1, 5):
            for class_id in range(150):  # 150 classes > 100
                ids.append(rec_id * 1000 + class_id)

        multiplier, num_classes = _detect_multiplier_pattern(sorted(ids))

        assert multiplier == 1000
        assert num_classes == 150

    def test_no_pattern_random_ids(self) -> None:
        """Test that random IDs don't match a pattern."""
        ids = [1, 5, 23, 47, 89, 101, 234, 567, 890]

        multiplier, num_classes = _detect_multiplier_pattern(sorted(ids))

        assert multiplier is None
        assert num_classes is None

    def test_insufficient_data(self) -> None:
        """Test handling of insufficient data."""
        ids = [100, 101, 102]

        multiplier, num_classes = _detect_multiplier_pattern(ids)

        assert multiplier is None
        assert num_classes is None


class TestGenerateSubmissionCodeHint:
    """Tests for generate_submission_code_hint function."""

    def test_wide_format_hint(self) -> None:
        """Test code hint generation for wide format."""
        format_info = SubmissionFormatInfo(
            format_type="wide",
            id_column="row_id",
            target_columns=["species_0", "species_1", "species_2"],
            num_classes=3,
        )

        code = generate_submission_code_hint(format_info)

        assert "Wide format" in code
        assert "submission[col] = predictions[:, i]" in code

    def test_long_format_mlsp_hint(self) -> None:
        """Test code hint generation for MLSP long format."""
        format_info = SubmissionFormatInfo(
            format_type="long",
            id_column="Id",
            target_columns=["Probability"],
            num_classes=19,
            id_pattern="rec_id * 100 + class_id",
            id_multiplier=100,
        )

        code = generate_submission_code_hint(format_info)

        assert "Long format" in code
        assert "MLSP" in code
        assert "rec_id * 100 + class_id" in code

    def test_unknown_format_hint(self) -> None:
        """Test code hint generation for unknown format."""
        format_info = SubmissionFormatInfo(
            format_type="unknown",
            id_column="Id",
            target_columns=["target"],
        )

        code = generate_submission_code_hint(format_info)

        # Should still generate some code
        assert "submission" in code.lower()
