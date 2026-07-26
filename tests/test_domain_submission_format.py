"""Data-driven submission-format detection regression tests.

Pixel-level detection must require BOTH the coordinate-suffix ID structure and
far more template rows than test samples. Structure alone also matches
ordinary per-sample IDs ("Test_0", "ISIC_0052060"), which once misrouted
classification competitions into image_to_image plans.
"""

from pathlib import Path

import pandas as pd

from kaggle_agents.domain.detection.submission_format import SubmissionFormatMixin


def _write_images(test_dir: Path, names: list[str]) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (test_dir / name).write_bytes(b"image")


def test_detects_observed_coordinate_structure_without_fixed_dimensions(
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "visual_assets"
    _write_images(test_dir, ["case-alpha.png", "case-beta.png"])

    row_keys = [
        f"case-{case}_{row}_{col}"
        for case in ("alpha", "beta")
        for row in range(11)
        for col in range(10)
    ]  # 110 pixel rows per image, no assumption about a coordinate base
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"row_key": row_keys, "prediction": [0.0] * len(row_keys)}).to_csv(
        sample_path, index=False
    )

    format_type, metadata = SubmissionFormatMixin().detect_submission_format(
        sample_path, test_dir
    )

    assert format_type == "pixel_level"
    assert metadata["id_pattern"] == "prefix_two_numeric_suffixes"
    assert metadata["pixel_format_detected"] is True


def test_does_not_treat_long_class_ids_as_pixel_coordinates(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "row_key": ["record-a_0", "record-a_1", "record-b_0", "record-b_1"],
            "prediction": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(sample_path, index=False)

    format_type, metadata = SubmissionFormatMixin().detect_submission_format(sample_path)

    assert format_type == "standard"
    assert metadata["pixel_format_detected"] is False


def test_one_row_per_image_ids_with_shared_literal_prefix_are_standard(
    tmp_path: Path,
) -> None:
    # "Test_0".."Test_N" with one row per test image: a constant literal
    # prefix repeats without the template being pixel-level.
    test_dir = tmp_path / "images"
    _write_images(test_dir, [f"Test_{i}.jpg" for i in range(6)])

    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "image_id": [f"Test_{i}" for i in range(6)],
            "healthy": [0.25] * 6,
        }
    ).to_csv(sample_path, index=False)

    format_type, metadata = SubmissionFormatMixin().detect_submission_format(
        sample_path, test_dir
    )

    assert format_type == "standard"
    assert metadata["pixel_format_detected"] is False


def test_catalog_style_numeric_suffix_ids_are_standard(tmp_path: Path) -> None:
    # "ISIC_0000000"-style catalog IDs, one row per test image.
    test_dir = tmp_path / "images"
    _write_images(test_dir, [f"ISIC_{i:07d}.jpg" for i in range(5)])

    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "image_name": [f"ISIC_{i:07d}" for i in range(5)],
            "target": [0.0] * 5,
        }
    ).to_csv(sample_path, index=False)

    format_type, metadata = SubmissionFormatMixin().detect_submission_format(
        sample_path, test_dir
    )

    assert format_type == "standard"
    assert metadata["pixel_format_detected"] is False


def test_three_part_ids_without_test_media_are_standard(tmp_path: Path) -> None:
    # Tabular composite keys ("1_2_3") with no countable test samples must not
    # be classified as pixel coordinates on structure alone.
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "id": ["1_2_3", "1_2_4", "2_1_1", "2_1_2"],
            "sales": [0.0] * 4,
        }
    ).to_csv(sample_path, index=False)

    format_type, metadata = SubmissionFormatMixin().detect_submission_format(sample_path)

    assert format_type == "standard"
    assert metadata["pixel_format_detected"] is False
