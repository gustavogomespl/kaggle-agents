"""
Submission format detection for audio competitions.

Detects Wide vs Long submission formats and extracts ID patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SubmissionFormatInfo:
    """Detected submission format information."""

    format_type: str  # 'wide', 'long', 'pixel_level', 'unknown'
    id_column: str
    target_columns: list[str]
    num_classes: int | None = None
    id_pattern: str | None = None  # e.g., 'rec_id * 100 + class_id'
    id_multiplier: int | None = None  # e.g., 100 for MLSP
    sample_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "format_type": self.format_type,
            "id_column": self.id_column,
            "target_columns": self.target_columns,
            "num_classes": self.num_classes,
            "id_pattern": self.id_pattern,
            "id_multiplier": self.id_multiplier,
            "warnings": self.warnings,
        }


def detect_audio_submission_format(
    sample_submission_path: Path | str,
    num_test_samples: int | None = None,
) -> SubmissionFormatInfo:
    """
    Detect submission format for audio competitions.

    Distinguishes between:
    - Wide format: One row per sample, one column per class (BirdCLEF style)
    - Long format: One row per (sample, class) pair (MLSP style)

    Args:
        sample_submission_path: Path to sample_submission.csv
        num_test_samples: Number of test samples (if known), helps detect long format

    Returns:
        SubmissionFormatInfo with detected format details

    Examples:
        Wide format (BirdCLEF):
        ```
        row_id,species_0,species_1,species_2
        audio_0001,0.5,0.5,0.5
        audio_0002,0.5,0.5,0.5
        ```

        Long format (MLSP 2013 Birds):
        ```
        Id,Probability
        100,0.5        # rec_id=1, species=0 → Id=1*100+0=100
        101,0.5        # rec_id=1, species=1 → Id=1*100+1=101
        ```
    """
    sample_submission_path = Path(sample_submission_path)
    warnings: list[str] = []

    if not sample_submission_path.exists():
        return SubmissionFormatInfo(
            format_type="unknown",
            id_column="Id",
            target_columns=["target"],
            warnings=[f"Sample submission not found: {sample_submission_path}"],
        )

    try:
        df = pd.read_csv(sample_submission_path)
    except Exception as e:
        return SubmissionFormatInfo(
            format_type="unknown",
            id_column="Id",
            target_columns=["target"],
            warnings=[f"Failed to read sample submission: {e}"],
        )

    if df.empty or len(df.columns) < 2:
        return SubmissionFormatInfo(
            format_type="unknown",
            id_column=df.columns[0] if len(df.columns) > 0 else "Id",
            target_columns=list(df.columns[1:]) if len(df.columns) > 1 else ["target"],
            warnings=["Sample submission is empty or has insufficient columns"],
        )

    # Identify ID column (usually first column)
    id_column = df.columns[0]
    target_columns = list(df.columns[1:])

    # Check if wide format (multiple target columns)
    if len(target_columns) > 1:
        # Wide format: row_id, class_0, class_1, ..., class_N
        return SubmissionFormatInfo(
            format_type="wide",
            id_column=id_column,
            target_columns=target_columns,
            num_classes=len(target_columns),
            sample_ids=df[id_column].tolist()[:10],
        )

    # Two columns (ID + single target): could be wide (binary/single-class) or long format
    # Try to detect long-format ID patterns first
    format_info = _detect_long_format_pattern(df, id_column, target_columns[0], num_test_samples)

    # If no long-format pattern detected, treat as wide (single-class)
    # Long format requires a detectable ID pattern (multiplier, underscore, dash)
    if format_info.id_pattern is None and format_info.id_multiplier is None:
        # No long-format pattern found - this is wide format with single target
        return SubmissionFormatInfo(
            format_type="wide",
            id_column=id_column,
            target_columns=target_columns,
            num_classes=1,
            sample_ids=df[id_column].tolist()[:10],
        )

    return format_info


def _detect_long_format_pattern(
    df: pd.DataFrame,
    id_column: str,
    target_column: str,
    num_test_samples: int | None = None,
) -> SubmissionFormatInfo:
    """
    Detect pattern in long format submission IDs.

    Common patterns:
    - MLSP: Id = rec_id * 100 + species_id (e.g., 100, 101, 102, ..., 118, 200, 201, ...)
    - Underscore: Id = "rec_id_species_id" (e.g., "1_0", "1_1", ...)
    - Dash: Id = "rec_id-species_id" (e.g., "1-0", "1-1", ...)
    """
    warnings: list[str] = []
    ids = df[id_column].tolist()

    # Check if IDs are numeric (MLSP style)
    # Use pd.api.types.is_numeric_dtype() because dtype objects don't compare to strings
    if pd.api.types.is_numeric_dtype(df[id_column]):
        numeric_ids = sorted([int(x) for x in ids if pd.notna(x)])

        # Try to detect multiplier pattern (e.g., 100 for MLSP)
        multiplier, num_classes = _detect_multiplier_pattern(numeric_ids)

        if multiplier:
            return SubmissionFormatInfo(
                format_type="long",
                id_column=id_column,
                target_columns=[target_column],
                num_classes=num_classes,
                id_pattern=f"rec_id * {multiplier} + class_id",
                id_multiplier=multiplier,
                sample_ids=numeric_ids[:10],
                warnings=warnings,
            )

    # Check for string patterns (underscore, dash)
    # Use pd.api.types.is_string_dtype() or is_object_dtype() for proper dtype comparison
    if pd.api.types.is_string_dtype(df[id_column]) or pd.api.types.is_object_dtype(df[id_column]):
        str_ids = [str(x) for x in ids if pd.notna(x)]

        # Check underscore pattern: "rec_id_class_id"
        # Only valid if:
        # 1. Multiple different prefixes exist (multiple samples), AND
        # 2. Each prefix appears with multiple suffixes (multiple classes per sample)
        # This distinguishes "1_0, 1_1, 2_0, 2_1" (long) from "audio_0001, audio_0002" (wide)
        underscore_pattern = r'^(.+)_(\d+)$'
        if str_ids and re.match(underscore_pattern, str_ids[0]):
            matches = [re.match(underscore_pattern, s) for s in str_ids]
            valid_matches = [m for m in matches if m]
            if valid_matches:
                prefixes = [m.group(1) for m in valid_matches]
                suffixes = [int(m.group(2)) for m in valid_matches]
                unique_prefixes = set(prefixes)
                unique_suffixes = set(suffixes)

                # Long format indicators:
                # - Multiple unique prefixes (different samples)
                # - Suffixes are small class indices (0, 1, 2...) not large unique IDs
                # - Number of rows = num_prefixes * num_suffixes (complete grid)
                is_long_format = (
                    len(unique_prefixes) > 1  # Multiple samples
                    and len(unique_suffixes) > 1  # Multiple classes
                    and max(unique_suffixes) < 100  # Suffixes are class indices, not unique IDs
                    and len(valid_matches) == len(unique_prefixes) * len(unique_suffixes)  # Complete grid
                )

                if is_long_format:
                    num_classes = max(unique_suffixes) + 1 if unique_suffixes else None
                    return SubmissionFormatInfo(
                        format_type="long",
                        id_column=id_column,
                        target_columns=[target_column],
                        num_classes=num_classes,
                        id_pattern="rec_id_class_id (underscore)",
                        sample_ids=str_ids[:10],
                        warnings=warnings,
                    )

        # Check dash pattern: "rec_id-class_id"
        # Same logic as underscore
        dash_pattern = r'^(.+)-(\d+)$'
        if str_ids and re.match(dash_pattern, str_ids[0]):
            matches = [re.match(dash_pattern, s) for s in str_ids]
            valid_matches = [m for m in matches if m]
            if valid_matches:
                prefixes = [m.group(1) for m in valid_matches]
                suffixes = [int(m.group(2)) for m in valid_matches]
                unique_prefixes = set(prefixes)
                unique_suffixes = set(suffixes)

                is_long_format = (
                    len(unique_prefixes) > 1
                    and len(unique_suffixes) > 1
                    and max(unique_suffixes) < 100
                    and len(valid_matches) == len(unique_prefixes) * len(unique_suffixes)
                )

                if is_long_format:
                    num_classes = max(unique_suffixes) + 1 if unique_suffixes else None
                    return SubmissionFormatInfo(
                        format_type="long",
                        id_column=id_column,
                        target_columns=[target_column],
                        num_classes=num_classes,
                        id_pattern="rec_id-class_id (dash)",
                        sample_ids=str_ids[:10],
                        warnings=warnings,
                    )

    # Could not detect pattern - assume simple long format
    warnings.append("Could not detect ID pattern in long format")

    return SubmissionFormatInfo(
        format_type="long",
        id_column=id_column,
        target_columns=[target_column],
        num_classes=None,
        id_pattern=None,
        sample_ids=ids[:10],
        warnings=warnings,
    )


def _detect_multiplier_pattern(sorted_ids: list[int]) -> tuple[int | None, int | None]:
    """
    Detect multiplier pattern in numeric IDs.

    For MLSP: IDs go 100, 101, ..., 118, 200, 201, ..., 218, ...
    Pattern: Id = rec_id * 100 + species_id where species_id ∈ [0, 18]

    Returns:
        (multiplier, num_classes) or (None, None) if no pattern detected
    """
    if len(sorted_ids) < 10:
        return None, None

    # Try common multipliers: 100, 1000, 10
    for multiplier in [100, 1000, 10, 50]:
        # Check if IDs follow the pattern
        rec_ids = set()
        class_ids = set()

        for id_val in sorted_ids:
            rec_id = id_val // multiplier
            class_id = id_val % multiplier
            rec_ids.add(rec_id)
            class_ids.add(class_id)

        # Valid pattern if:
        # 1. Class IDs are consecutive from 0 to N-1
        # 2. Number of class IDs is reasonable (< multiplier)
        # 3. Total rows = num_rec_ids * num_classes
        if class_ids:
            min_class = min(class_ids)
            max_class = max(class_ids)
            num_classes = max_class - min_class + 1

            if (
                min_class == 0
                and num_classes == len(class_ids)
                and num_classes < multiplier
                and len(sorted_ids) == len(rec_ids) * num_classes
            ):
                return multiplier, num_classes

    return None, None


def generate_submission_code_hint(format_info: SubmissionFormatInfo) -> str:
    """
    Generate code hint for creating submission in detected format.

    Args:
        format_info: Detected submission format

    Returns:
        Python code snippet for generating submission
    """
    if format_info.format_type == "wide":
        return f'''# Wide format submission (BirdCLEF style)
# predictions shape: (num_samples, {format_info.num_classes})
submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
for i, col in enumerate({format_info.target_columns}):
    submission[col] = predictions[:, i]
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
'''

    if format_info.format_type == "long" and format_info.id_multiplier:
        return f'''# Long format submission (MLSP style)
# ID pattern: {format_info.id_pattern}
# predictions shape: (num_samples, {format_info.num_classes})
submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
pred_map = {{}}

for i, rec_id in enumerate(test_rec_ids):
    for class_id in range({format_info.num_classes}):
        submission_id = rec_id * {format_info.id_multiplier} + class_id
        pred_map[submission_id] = predictions[i, class_id]

submission['{format_info.target_columns[0]}'] = submission['{format_info.id_column}'].map(pred_map)
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
'''

    return f'''# Long format submission (pattern: {format_info.id_pattern or 'unknown'})
# Adjust mapping based on your specific ID format
submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
# TODO: Map predictions to submission IDs
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
'''


def print_format_info(format_info: SubmissionFormatInfo) -> None:
    """Print formatted submission format information."""
    print("\n" + "=" * 60)
    print("=== SUBMISSION FORMAT DETECTION ===")
    print("=" * 60)
    print(f"Format type: {format_info.format_type}")
    print(f"ID column: {format_info.id_column}")
    print(f"Target columns: {format_info.target_columns}")
    print(f"Num classes: {format_info.num_classes}")

    if format_info.id_pattern:
        print(f"ID pattern: {format_info.id_pattern}")

    if format_info.id_multiplier:
        print(f"ID multiplier: {format_info.id_multiplier}")

    if format_info.sample_ids:
        print(f"Sample IDs: {format_info.sample_ids}")

    if format_info.warnings:
        print("\nWarnings:")
        for w in format_info.warnings:
            print(f"  - {w}")

    print("=" * 60 + "\n")
