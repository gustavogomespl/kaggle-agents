"""
Cross-validation utilities for consistent evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


# Common group column names in medical imaging and other competitions
GROUP_COLUMN_CANDIDATES = [
    "PatientID", "patient_id", "patient", "subject_id", "subject",
    "StudyInstanceUID", "study_id", "SeriesInstanceUID", "series_id",
    "user_id", "userId", "group_id", "groupId", "session_id",
]

# Temporal column patterns for TimeSeriesSplit detection
TEMPORAL_COLUMN_PATTERNS = [
    "date", "datetime", "timestamp", "time", "day", "month", "year",
    "created_at", "updated_at", "posted_at", "transaction_date",
    "event_time", "event_date", "order_date", "purchase_date",
]


def _detect_group_column(df: pd.DataFrame, group_col: str | None = None) -> str | None:
    """
    Auto-detect group column for GroupKFold to prevent data leakage.

    In medical imaging competitions (like RANZCR), multiple images from the same
    patient must NOT be split between train and validation sets.

    Args:
        df: Training dataframe
        group_col: Explicitly specified group column (optional)

    Returns:
        Name of detected group column, or None if not found
    """
    if group_col and group_col in df.columns:
        return group_col

    for col in GROUP_COLUMN_CANDIDATES:
        if col in df.columns:
            # Verify it's a valid grouping column (not all unique values)
            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_unique < n_rows * 0.9:  # At least 10% rows share groups
                return col

    return None


def _detect_temporal_column(df: pd.DataFrame, temporal_col: str | None = None) -> str | None:
    """
    Auto-detect temporal column for TimeSeriesSplit to prevent future leakage.

    In time series competitions, data must be split chronologically:
    - Training folds contain older data
    - Validation folds contain newer data
    - NO shuffle allowed

    Args:
        df: Training dataframe
        temporal_col: Explicitly specified temporal column (optional)

    Returns:
        Name of detected temporal column, or None if not found
    """
    if temporal_col and temporal_col in df.columns:
        return temporal_col

    for col in df.columns:
        col_lower = col.lower()

        # Pattern matching
        if any(p in col_lower for p in TEMPORAL_COLUMN_PATTERNS):
            # Validate if it's actually a temporal column
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

            # Try to parse as datetime
            try:
                sample = df[col].iloc[:100].dropna()
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    return col
            except (ValueError, TypeError):
                continue

    return None


def _validate_temporal_folds(
    df: pd.DataFrame,
    folds: np.ndarray,
    temporal_col: str,
    n_folds: int,
) -> None:
    """
    Validate that folds respect temporal ordering.

    Args:
        df: DataFrame with temporal column
        folds: Fold assignments array
        temporal_col: Name of temporal column
        n_folds: Number of folds

    Raises:
        ValueError: If temporal ordering is violated
    """
    fold_max_times = []
    for fold_idx in range(n_folds):
        fold_mask = folds == fold_idx
        if fold_mask.sum() > 0:
            max_time = df.loc[fold_mask, temporal_col].max()
            fold_max_times.append((fold_idx, max_time))

    # Verify non-decreasing order (allow equal timestamps)
    for i in range(1, len(fold_max_times)):
        prev_fold, prev_max = fold_max_times[i - 1]
        curr_fold, curr_max = fold_max_times[i]
        if curr_max < prev_max:  # Only fail if curr < prev (not <=)
            raise ValueError(
                f"Temporal ordering violation: fold {curr_fold} max time ({curr_max}) "
                f"< fold {prev_fold} max time ({prev_max}). Data should be chronological."
            )

    print("      ✅ Temporal folds validated: non-decreasing order")


def generate_folds(
    train_path: str,
    target_col: str,
    output_path: str,
    n_folds: int = 5,
    seed: int = 42,
    group_col: str | None = None,
    temporal_col: str | None = None,
    force_strategy: str | None = None,
) -> str:
    """
    Generate fixed cross-validation folds and save to CSV.

    CRITICAL: Auto-detects:
    - Group columns (PatientID, subject_id, etc.) to prevent patient/subject-level leakage
    - Temporal columns (date, timestamp, etc.) to prevent future leakage

    Args:
        train_path: Path to training data CSV
        target_col: Name of target column
        output_path: Path to save folds CSV
        n_folds: Number of folds
        seed: Random seed
        group_col: Optional explicit group column name
        temporal_col: Optional explicit temporal column name
        force_strategy: Force a specific strategy ("kfold", "stratified", "timeseries", "group")

    Returns:
        Path to saved folds file
    """
    print(f"   🔄 Generating {n_folds} fixed folds...")

    df = pd.read_csv(train_path)

    # Create 'fold' column
    df["fold"] = -1

    # CRITICAL: Auto-detect temporal column for TimeSeriesSplit
    detected_temporal = _detect_temporal_column(df, temporal_col)

    # CRITICAL: Auto-detect group column to prevent patient/subject-level leakage
    detected_group = _detect_group_column(df, group_col)

    if detected_temporal:
        print(f"      ⏰ Detected temporal column: '{detected_temporal}'")
        print("      🔒 Using TimeSeriesSplit to prevent future leakage!")
    elif detected_group:
        n_groups = df[detected_group].nunique()
        print(f"      ⚠️  Detected group column: '{detected_group}' ({n_groups} unique groups)")
        print("      🔒 Using GroupKFold to prevent data leakage!")

    # Check if target_col exists - for multi-label competitions, target_col might not exist
    # in train.csv (targets are multiple columns matching sample_submission)
    target_col_exists = target_col in df.columns

    # Determine problem type for splitting strategy
    # Simple heuristic: if target has few unique values -> classification
    if target_col_exists:
        n_unique = df[target_col].nunique()
        is_classification = n_unique < 20
    else:
        # Multi-label or target not in train - can't stratify
        print(f"      ⚠️ Target column '{target_col}' not found in train data")
        print("      This is likely a MULTI-LABEL competition - using GroupKFold/KFold without stratification")
        n_unique = 100  # Force non-stratified
        is_classification = False

    # Determine CV strategy
    if force_strategy:
        strategy = force_strategy
    elif detected_temporal:
        strategy = "timeseries"
    elif detected_group:
        strategy = "group"
    elif is_classification and target_col_exists:
        strategy = "stratified"
    else:
        strategy = "kfold"

    # CRITICAL: TimeSeriesSplit - NO shuffle, NO stratification
    if strategy == "timeseries":
        # Sort by temporal column BEFORE creating splits
        df = df.sort_values(detected_temporal).reset_index(drop=True)

        # TimeSeriesSplit (pure chronological, no shuffle)
        print("      Using TimeSeriesSplit (chronological, no shuffle)")
        kf = TimeSeriesSplit(n_splits=n_folds)
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, "fold"] = fold

        # Validate temporal ordering
        folds_array = df["fold"].values
        _validate_temporal_folds(df, folds_array, detected_temporal, n_folds)

        print(f"      ✅ TimeSeriesSplit: fold 0 = oldest, fold {n_folds - 1} = newest")

    elif strategy == "group":
        # Use GroupKFold or StratifiedGroupKFold to prevent leakage
        # Note: strategy is set to "group" when detected_group is found and no force_strategy is set
        if detected_group is None:
            raise ValueError("strategy='group' requires a group column but none was detected")
        groups = df[detected_group]
        if is_classification and n_unique <= 10 and target_col_exists:
            try:
                print("      Using StratifiedGroupKFold (classification + groups)")
                kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                y = df[target_col]
                for fold, (_train_idx, val_idx) in enumerate(kf.split(df, y, groups)):
                    df.loc[val_idx, "fold"] = fold
            except Exception as e:
                # Fallback to GroupKFold if stratification fails
                print(f"      StratifiedGroupKFold failed ({e}), using GroupKFold")
                kf = GroupKFold(n_splits=n_folds)
                for fold, (_train_idx, val_idx) in enumerate(kf.split(df, groups=groups)):
                    df.loc[val_idx, "fold"] = fold
        else:
            print("      Using GroupKFold (groups detected, no stratification)")
            kf = GroupKFold(n_splits=n_folds)
            for fold, (_train_idx, val_idx) in enumerate(kf.split(df, groups=groups)):
                df.loc[val_idx, "fold"] = fold
    elif strategy == "stratified" and target_col_exists:
        print(
            f"      Detected classification (unique targets: {n_unique}) -> Using StratifiedKFold"
        )
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y = df[target_col]
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df, y)):
            df.loc[val_idx, "fold"] = fold
    else:
        # Default: KFold (regression or force_strategy="kfold")
        print("      Using KFold (shuffle=True)")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, "fold"] = fold

    # Save only index (if needed) or full dataframe?
    # Saving full dataframe with 'fold' column is safest for alignment
    # But to save space/time, maybe just save the fold column?
    # Actually, user requested "folds.csv". Let's save the whole dataframe with the fold column
    # so agents can just load this instead of train.csv if they want, OR join it.
    # BETTER: Save just the 'fold' column aligned with original index, or the full file.
    # Saving full file is easiest for agents to use: "Load folds.csv instead of train.csv"

    df.to_csv(output_path, index=False)
    print(f"      ✅ Saved fixed folds to: {output_path}")

    return output_path


class AdaptiveCrossValidator:
    """Utility helpers for selecting CV and scoring strategies."""

    @staticmethod
    def get_cv_strategy(strategy_name: str, n_splits: int = 5, *, random_state: int = 42):
        """Return a scikit-learn CV splitter for a given strategy name."""
        name = (strategy_name or "").lower()
        if "timeseriessplit" in name or "time" in name:
            return TimeSeriesSplit(n_splits=n_splits)
        if "stratified" in name:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    @staticmethod
    def determine_best_strategy(X: pd.DataFrame, y: pd.Series, data_chars: dict) -> str:
        """Heuristic strategy selection based on dataset characteristics."""
        temporal = bool((data_chars.get("temporal") or {}).get("has_date_columns"))
        if temporal:
            return "TimeSeriesSplit"

        target = data_chars.get("target") or {}
        imbalance_ratio = target.get("imbalance_ratio")
        if isinstance(imbalance_ratio, (int, float)) and float(imbalance_ratio) >= 2.0:
            return "StratifiedKFold"

        # Default
        return "KFold"

    @staticmethod
    def get_scoring_metric(problem_type: str, metric_name: str | None = None) -> str:
        """Map problem type / metric name to scikit-learn scoring identifiers."""
        metric = (metric_name or "").strip().lower()
        ptype = (problem_type or "").strip().lower()

        if metric in {"auc", "roc_auc", "roc-auc"}:
            return "roc_auc"
        if metric in {"rmse", "root_mean_squared_error"}:
            return "neg_root_mean_squared_error"
        if metric in {"mae", "mean_absolute_error"}:
            return "neg_mean_absolute_error"

        if ptype == "regression":
            return "neg_mean_squared_error"
        return "accuracy"
