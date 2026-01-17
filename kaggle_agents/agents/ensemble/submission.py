"""Submission validation and alignment functions."""

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def validate_test_id_alignment(
    models_dir: Path,
    sample_submission_path: Path,
    model_names: Optional[list[str]] = None,
) -> tuple[bool, str]:
    """
    Validate that saved test predictions align with sample_submission ID order.

    This prevents the score=0.50 bug caused by loading test files via glob()
    in filesystem order instead of sample_submission order.

    Args:
        models_dir: Directory containing test_ids_{name}.npy files
        sample_submission_path: Path to sample_submission.csv
        model_names: Optional list of model names to check (if None, checks all)

    Returns:
        Tuple of (is_aligned, warning_message)
    """
    warnings = []

    # Load canonical test IDs from sample_submission
    try:
        sample_sub = pd.read_csv(sample_submission_path)
        canonical_ids = sample_sub.iloc[:, 0].astype(str).values
    except Exception as e:
        return False, f"Failed to read sample_submission: {e}"

    # Find all test_ids files
    if model_names:
        id_files = [models_dir / f"test_ids_{name}.npy" for name in model_names]
        id_files = [f for f in id_files if f.exists()]
    else:
        id_files = list(models_dir.glob("test_ids_*.npy"))

    if not id_files:
        warnings.append(
            "🚨 NO test_ids_*.npy files found! "
            "If test files were loaded via glob(), predictions may be MISALIGNED (score=0.50 bug)!"
        )
        return False, "\n".join(warnings)

    # Check each model's test IDs
    all_aligned = True
    for id_file in id_files:
        model_name = id_file.stem.replace("test_ids_", "")
        try:
            saved_ids = np.load(id_file, allow_pickle=True).astype(str)

            # Check if same set of IDs
            saved_set = set(saved_ids)
            canonical_set = set(canonical_ids)

            missing = canonical_set - saved_set
            extra = saved_set - canonical_set

            if missing:
                warnings.append(
                    f"   ❌ {model_name}: Missing {len(missing)} IDs from predictions"
                )
                all_aligned = False
            if extra:
                warnings.append(
                    f"   ⚠️  {model_name}: {len(extra)} extra IDs in predictions"
                )

            # Check if same order (critical for alignment)
            if len(saved_ids) == len(canonical_ids):
                if not np.array_equal(saved_ids[:10], canonical_ids[:10]):
                    warnings.append(
                        f"   🚨 {model_name}: ID ORDER MISMATCH (first 10 differ)! "
                        f"Predictions: {saved_ids[:3]}... vs Canonical: {canonical_ids[:3]}..."
                    )
                    all_aligned = False
                elif not np.array_equal(saved_ids, canonical_ids):
                    warnings.append(
                        f"   ⚠️  {model_name}: ID order differs from sample_submission"
                    )
                    all_aligned = False
                else:
                    print(f"   ✅ {model_name}: Test IDs aligned with sample_submission")
            else:
                warnings.append(
                    f"   ❌ {model_name}: ID count mismatch - {len(saved_ids)} vs {len(canonical_ids)}"
                )
                all_aligned = False

        except Exception as e:
            warnings.append(f"   ❌ {model_name}: Failed to load test_ids: {e}")
            all_aligned = False

    warning_msg = "\n".join(warnings) if warnings else ""
    return all_aligned, warning_msg


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
