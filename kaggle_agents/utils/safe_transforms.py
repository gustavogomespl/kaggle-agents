"""
Safe DataFrame Transformations for Kaggle Agents.

Provides merge and groupby operations with row count validation
to prevent silent data corruption from cartesian products.

Research: Merge operations creating cartesian products cause 25M vs 55M row misalignment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    expected_ratio: float = 1.0,
    tolerance: float = 0.1,
    validate: str | None = None,
) -> pd.DataFrame:
    """Merge with row count validation.

    Prevents silent data corruption from cartesian products by validating
    that the result row count is within expected bounds.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to merge on
        how: Merge type ('left', 'right', 'inner', 'outer')
        expected_ratio: Expected ratio of result rows to left rows (default 1.0)
        tolerance: Allowed deviation from expected ratio (default 0.1 = 10%)
        validate: Pandas merge validation ('one_to_one', 'one_to_many', etc.)

    Returns:
        Merged DataFrame

    Raises:
        ValueError: If row count deviates beyond tolerance

    Examples:
        >>> # Safe left join (expecting same row count)
        >>> result = safe_merge(train_df, features_df, on='id', how='left')

        >>> # Merge expecting 2x rows (e.g., cross-join on 2 categories)
        >>> result = safe_merge(df, lookup, on='category', expected_ratio=2.0)
    """
    original_len = len(left)

    merge_kwargs: dict[str, Any] = {"on": on, "how": how}
    if validate:
        merge_kwargs["validate"] = validate

    result = left.merge(right, **merge_kwargs)

    actual_ratio = len(result) / original_len if original_len > 0 else 0

    if abs(actual_ratio - expected_ratio) > tolerance:
        raise ValueError(
            f"Merge row count violation: {original_len:,} → {len(result):,} "
            f"(ratio {actual_ratio:.2f}, expected {expected_ratio:.2f} ± {tolerance})"
        )

    return result


def safe_groupby_agg(
    df: pd.DataFrame,
    group_cols: str | list[str],
    agg_dict: dict[str, Any],
    expected_rows: int | None = None,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    """GroupBy aggregation with optional row count validation.

    Args:
        df: Input DataFrame
        group_cols: Column(s) to group by
        agg_dict: Aggregation specification (e.g., {'col': 'mean'})
        expected_rows: Expected number of result rows (optional)
        tolerance: Allowed deviation as fraction of expected (default 0.1 = 10%)

    Returns:
        Aggregated DataFrame

    Examples:
        >>> # Group and aggregate with validation
        >>> result = safe_groupby_agg(
        ...     df, 'user_id', {'amount': 'sum'},
        ...     expected_rows=1000
        ... )
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    result = df.groupby(group_cols, as_index=False).agg(agg_dict)

    if expected_rows is not None and expected_rows > 0:
        deviation = abs(len(result) - expected_rows) / expected_rows
        if deviation > tolerance:
            print(
                f"[WARNING] GroupBy produced {len(result):,} rows, "
                f"expected ~{expected_rows:,} (deviation: {deviation:.1%})"
            )

    return result


def validate_merge_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
) -> dict[str, Any]:
    """Analyze merge keys before merging to predict row count.

    Use this to check if a merge is safe before executing it.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to merge on

    Returns:
        Dict with analysis results:
        - left_unique_keys: Number of unique keys in left
        - right_unique_keys: Number of unique keys in right
        - common_keys: Number of keys present in both
        - right_max_duplicates: Maximum duplicates per key in right
        - predicted_ratio: Predicted row count ratio
        - safe_to_merge: True if merge won't cause row explosion

    Examples:
        >>> analysis = validate_merge_keys(train_df, features_df, on='id')
        >>> if not analysis['safe_to_merge']:
        ...     print(f"Warning: merge may create {analysis['predicted_ratio']:.1f}x rows")
    """
    if isinstance(on, str):
        on = [on]

    left_keys = left[on].drop_duplicates()
    right_keys = right[on].drop_duplicates()

    # Check for duplicates in right (causes row multiplication)
    right_dups = right.groupby(on).size()
    max_dup = int(right_dups.max()) if len(right_dups) > 0 else 1

    # Count common keys
    common = pd.merge(left_keys, right_keys, on=on)

    return {
        "left_unique_keys": len(left_keys),
        "right_unique_keys": len(right_keys),
        "common_keys": len(common),
        "right_max_duplicates": max_dup,
        "predicted_ratio": float(max_dup) if max_dup > 1 else 1.0,
        "safe_to_merge": max_dup <= 1,
    }


def safe_concat(
    dfs: list[pd.DataFrame],
    expected_total: int | None = None,
    tolerance: float = 0.1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Concatenate DataFrames with row count validation.

    Args:
        dfs: List of DataFrames to concatenate
        expected_total: Expected total row count (optional)
        tolerance: Allowed deviation as fraction of expected
        **kwargs: Additional arguments passed to pd.concat

    Returns:
        Concatenated DataFrame

    Raises:
        ValueError: If row count deviates significantly from expected
    """
    result = pd.concat(dfs, **kwargs)

    if expected_total is not None and expected_total > 0:
        actual = len(result)
        deviation = abs(actual - expected_total) / expected_total
        if deviation > tolerance:
            raise ValueError(
                f"Concat row count violation: {actual:,} rows, "
                f"expected ~{expected_total:,} (deviation: {deviation:.1%})"
            )

    return result


def check_duplicates_before_merge(
    df: pd.DataFrame,
    on: str | list[str],
    name: str = "DataFrame",
) -> bool:
    """Check if DataFrame has duplicates on merge keys.

    Args:
        df: DataFrame to check
        on: Column(s) to check for duplicates
        name: Name for logging (default 'DataFrame')

    Returns:
        True if duplicates exist, False otherwise
    """
    if isinstance(on, str):
        on = [on]

    dups = df.groupby(on).size()
    has_dups = (dups > 1).any()

    if has_dups:
        max_dups = int(dups.max())
        n_dup_keys = int((dups > 1).sum())
        print(
            f"[WARNING] {name} has duplicates on {on}: "
            f"{n_dup_keys:,} keys with duplicates, max {max_dups}x"
        )

    return bool(has_dups)


def deduplicate_for_merge(
    df: pd.DataFrame,
    on: str | list[str],
    keep: str = "first",
    agg_dict: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Deduplicate DataFrame before merge to prevent row explosion.

    Args:
        df: DataFrame to deduplicate
        on: Column(s) to deduplicate on
        keep: Which duplicate to keep ('first', 'last') if not aggregating
        agg_dict: Aggregation dict - if provided, aggregates instead of dropping

    Returns:
        Deduplicated DataFrame

    Examples:
        >>> # Simple deduplication
        >>> clean_df = deduplicate_for_merge(features_df, on='id')

        >>> # Aggregate duplicates
        >>> clean_df = deduplicate_for_merge(
        ...     features_df, on='id',
        ...     agg_dict={'value': 'mean', 'count': 'sum'}
        ... )
    """
    if isinstance(on, str):
        on = [on]

    if agg_dict:
        # Aggregate duplicates
        return df.groupby(on, as_index=False).agg(agg_dict)
    # Drop duplicates
    return df.drop_duplicates(subset=on, keep=keep)


def validate_row_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "df1",
    name2: str = "df2",
) -> bool:
    """Validate that two DataFrames have the same number of rows.

    Useful for checking OOF predictions alignment with training data.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        name1: Name of first DataFrame for logging
        name2: Name of second DataFrame for logging

    Returns:
        True if row counts match, False otherwise
    """
    if len(df1) != len(df2):
        print(
            f"[ERROR] Row count mismatch: {name1}={len(df1):,}, {name2}={len(df2):,}"
        )
        return False
    return True


def validate_array_alignment(
    arr1: np.ndarray,
    arr2: np.ndarray,
    name1: str = "arr1",
    name2: str = "arr2",
) -> bool:
    """Validate that two arrays have compatible shapes.

    Useful for checking OOF predictions alignment.

    Args:
        arr1: First array
        arr2: Second array
        name1: Name of first array for logging
        name2: Name of second array for logging

    Returns:
        True if shapes are compatible, False otherwise
    """
    if arr1.shape[0] != arr2.shape[0]:
        print(
            f"[ERROR] Shape mismatch: {name1}={arr1.shape}, {name2}={arr2.shape}"
        )
        return False
    return True
