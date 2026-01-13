"""Submission validation and alignment functions."""

import shutil
from pathlib import Path

import pandas as pd


def validate_and_align_submission(
    submission_path: Path,
    sample_submission_path: Path,
    output_path: Path | None = None,
) -> tuple[bool, str, Path | None]:
    """Validate submission against sample_submission schema.

    If IDs are same set but different order, reorders to match sample.

    Args:
        submission_path: Path to submission to validate
        sample_submission_path: Path to sample_submission.csv
        output_path: Where to save aligned submission (if None, overwrites in place)

    Returns:
        Tuple of (is_valid, error_message, aligned_path)
    """
    output_path = output_path or submission_path

    try:
        sub_df = pd.read_csv(submission_path)
        sample_df = pd.read_csv(sample_submission_path)
    except Exception as e:
        return False, f"Failed to read CSV: {e}", None

    # Check columns match
    if list(sub_df.columns) != list(sample_df.columns):
        return False, f"Column mismatch: {sub_df.columns.tolist()} vs {sample_df.columns.tolist()}", None

    # Check row count
    if len(sub_df) != len(sample_df):
        return False, f"Row count mismatch: {len(sub_df)} vs {len(sample_df)}", None

    # Check ID column - same SET but possibly different order
    id_col = sub_df.columns[0]
    sub_ids = set(sub_df[id_col])
    sample_ids = set(sample_df[id_col])

    if sub_ids != sample_ids:
        missing = sample_ids - sub_ids
        extra = sub_ids - sample_ids
        return False, f"ID mismatch: missing={len(missing)}, extra={len(extra)}", None

    # If order differs, reorder to match sample
    if not sub_df[id_col].equals(sample_df[id_col]):
        print("      [LOG:INFO] Reordering submission to match sample_submission ID order")
        # Reorder using merge
        sub_df = sample_df[[id_col]].merge(sub_df, on=id_col, how='left')

    # Check for NaN in predictions (after potential reorder)
    pred_cols = sub_df.columns[1:]
    nan_count = sub_df[pred_cols].isna().sum().sum()
    if nan_count > 0:
        return False, f"Submission contains {nan_count} NaN values", None

    # Save aligned submission
    sub_df.to_csv(output_path, index=False)
    return True, "", output_path


def safe_restore_submission(
    source_path: Path,
    dest_path: Path,
    sample_submission_path: Path | None,
) -> bool:
    """Safely restore submission with validation.

    Args:
        source_path: Path to source submission (e.g., submission_best.csv)
        dest_path: Path to destination (e.g., submission.csv)
        sample_submission_path: Path to sample_submission.csv for validation

    Returns:
        True if restoration succeeded, False otherwise
    """
    if not source_path.exists():
        print(f"      Warning: Source submission not found: {source_path}")
        return False

    if sample_submission_path and Path(sample_submission_path).exists():
        is_valid, error_msg, _ = validate_and_align_submission(
            source_path,
            sample_submission_path,
            dest_path
        )
        if is_valid:
            print(f"      OK: Validated and restored submission to {dest_path}")
            return True
        print(f"      Warning: Submission validation failed: {error_msg}")
        print("      Copying without validation as fallback...")
        shutil.copy(source_path, dest_path)
        return True
    # No sample_submission available, just copy
    shutil.copy(source_path, dest_path)
    print(f"      OK: Restored submission to {dest_path} (no validation)")
    return True
