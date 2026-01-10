"""
Canonical Data Contract for Kaggle Agents.

This module provides a single "prepare once, consume many" data contract
that all model components must obey. It solves the problem of inconsistent
data handling across components (different sampling, filtering, column order).

Key artifacts generated:
- canonical/train_ids.npy - Stable row IDs after all filtering/sampling
- canonical/y.npy - Target aligned with train_ids
- canonical/folds.npy - Fold assignment per row
- canonical/feature_cols.json - Final feature list (intersection of train/test)
- canonical/metadata.json - Sampling info, original row count, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


# Common group column names for preventing data leakage
GROUP_COLUMN_CANDIDATES = [
    "PatientID", "patient_id", "patient", "subject_id", "subject",
    "StudyInstanceUID", "study_id", "SeriesInstanceUID", "series_id",
    "user_id", "userId", "group_id", "groupId", "session_id",
]


def _detect_id_column(df: pd.DataFrame) -> str | None:
    """Detect the ID column in a dataframe."""
    candidates = ["id", "Id", "ID", "key", "Key", "index"]
    for col in candidates:
        if col in df.columns:
            return col
    # Fallback: first column if it looks like an ID
    first_col = df.columns[0]
    if df[first_col].nunique() == len(df):
        return first_col
    return None


def _detect_group_column(df: pd.DataFrame) -> str | None:
    """Auto-detect group column for GroupKFold to prevent data leakage."""
    for col in GROUP_COLUMN_CANDIDATES:
        if col in df.columns:
            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_unique < n_rows * 0.9:  # At least 10% rows share groups
                return col
    return None


def validate_schema_parity(
    train_path: str | Path,
    test_path: str | Path,
    id_col: str | None = None,
    target_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Validate that train and test have compatible schemas.

    Returns:
        Tuple of (common_feature_cols, missing_in_test)
    """
    train_cols = set(pd.read_csv(train_path, nrows=0).columns)
    test_cols = set(pd.read_csv(test_path, nrows=0).columns)

    # Columns to exclude from features
    exclude_cols = set()
    if id_col:
        exclude_cols.add(id_col)
    if target_col:
        exclude_cols.add(target_col)

    # Feature columns = intersection (excluding id/target)
    common = train_cols & test_cols - exclude_cols
    missing_in_test = train_cols - test_cols - exclude_cols

    # Deterministic order
    return sorted(list(common)), sorted(list(missing_in_test))


def select_cv_strategy(
    n_rows: int,
    timeout_s: int | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """
    Select CV strategy based on dataset size and budget.

    Args:
        n_rows: Number of training rows
        timeout_s: Component timeout in seconds
        fast_mode: Whether running in fast mode

    Returns:
        Dict with n_folds and strategy name
    """
    if fast_mode or n_rows > 2_000_000:
        return {"n_folds": 3, "strategy": "kfold"}
    elif n_rows > 500_000:
        return {"n_folds": 3, "strategy": "stratified_kfold"}
    elif n_rows > 200_000:
        return {"n_folds": 4, "strategy": "stratified_kfold"}
    else:
        return {"n_folds": 5, "strategy": "stratified_kfold"}


def _hash_based_sample(
    df: pd.DataFrame,
    id_col: str,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deterministic hash-based sampling.

    Uses hash of ID to select rows - same result every run.

    Returns:
        Tuple of (sampled_df, sampling_metadata)
    """
    original_rows = len(df)
    if original_rows <= max_rows:
        return df, {"sampled": False, "original_rows": original_rows}

    sample_frac = max_rows / original_rows
    threshold = int(10000 * sample_frac)

    def should_include(id_val):
        return (hash(str(id_val)) % 10000) < threshold

    sample_mask = df[id_col].apply(should_include).values
    sampled_df = df[sample_mask].reset_index(drop=True)

    metadata = {
        "sampled": True,
        "original_rows": original_rows,
        "sampled_rows": len(sampled_df),
        "sampling_method": "hash_based",
        "sampling_threshold": threshold,
        "deterministic": True,
    }

    return sampled_df, metadata


def prepare_canonical_data(
    train_path: str | Path,
    test_path: str | Path,
    target_col: str,
    output_dir: str | Path,
    id_col: str | None = None,
    max_rows: int | None = None,
    n_folds: int | None = None,
    fast_mode: bool = False,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """
    Prepare canonical data artifacts that all model components must use.

    This is the single source of truth for:
    - Which rows to use (train_ids)
    - Target values (y)
    - Fold assignments (folds)
    - Feature columns (feature_cols)

    Args:
        train_path: Path to training data
        test_path: Path to test data
        target_col: Name of target column
        output_dir: Working directory for competition
        id_col: ID column name (auto-detected if None)
        max_rows: Maximum rows to use (hash-based sampling if exceeded)
        n_folds: Number of CV folds (auto-selected if None)
        fast_mode: Whether running in fast mode
        timeout_s: Component timeout in seconds

    Returns:
        Dict with paths to all canonical artifacts
    """
    train_path = Path(train_path)
    test_path = Path(test_path)
    output_dir = Path(output_dir)

    # Create canonical directory
    canonical_dir = output_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   Preparing canonical data contract...")

    # Load training data
    train_df = pd.read_csv(train_path)
    original_rows = len(train_df)
    print(f"   Loaded {original_rows:,} training rows")

    # Detect ID column
    if id_col is None:
        id_col = _detect_id_column(train_df)
    if id_col is None:
        # Create synthetic ID
        id_col = "_row_id"
        train_df[id_col] = range(len(train_df))
        print(f"   Created synthetic ID column: {id_col}")
    else:
        print(f"   Using ID column: {id_col}")

    # Validate target exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    # Hash-based sampling if needed
    sampling_metadata = {"sampled": False, "original_rows": original_rows}
    if max_rows and len(train_df) > max_rows:
        train_df, sampling_metadata = _hash_based_sample(train_df, id_col, max_rows)
        print(f"   Sampled {len(train_df):,} rows via hash-based selection")

    # Schema parity check
    feature_cols, missing_in_test = validate_schema_parity(
        train_path, test_path, id_col, target_col
    )
    if missing_in_test:
        print(f"   Warning: {len(missing_in_test)} columns missing in test: {missing_in_test[:5]}...")

    print(f"   Using {len(feature_cols)} feature columns")

    # Select CV strategy
    if n_folds is None:
        cv_config = select_cv_strategy(len(train_df), timeout_s, fast_mode)
        n_folds = cv_config["n_folds"]
    else:
        cv_config = {"n_folds": n_folds, "strategy": "stratified_kfold"}

    print(f"   CV strategy: {n_folds} folds ({cv_config['strategy']})")

    # Detect group column for preventing data leakage
    group_col = _detect_group_column(train_df)
    if group_col:
        print(f"   Detected group column: {group_col} (using GroupKFold)")

    # Generate fold assignments
    y = train_df[target_col].values
    n_unique = len(np.unique(y))
    is_classification = n_unique < 20

    fold_assignments = np.zeros(len(train_df), dtype=int)

    if group_col:
        groups = train_df[group_col].values
        if is_classification and n_unique <= 10:
            try:
                kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
                for fold, (_, val_idx) in enumerate(kf.split(train_df, y, groups)):
                    fold_assignments[val_idx] = fold
            except Exception:
                kf = GroupKFold(n_splits=n_folds)
                for fold, (_, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                    fold_assignments[val_idx] = fold
        else:
            kf = GroupKFold(n_splits=n_folds)
            for fold, (_, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                fold_assignments[val_idx] = fold
    elif is_classification:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(kf.split(train_df, y)):
            fold_assignments[val_idx] = fold
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(kf.split(train_df)):
            fold_assignments[val_idx] = fold

    # Extract canonical data
    train_ids = train_df[id_col].values

    # Save canonical artifacts
    np.save(canonical_dir / "train_ids.npy", train_ids)
    np.save(canonical_dir / "y.npy", y)
    np.save(canonical_dir / "folds.npy", fold_assignments)

    with open(canonical_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metadata
    metadata = {
        "original_rows": original_rows,
        "canonical_rows": len(train_df),
        "n_folds": n_folds,
        "cv_strategy": cv_config["strategy"],
        "id_col": id_col,
        "target_col": target_col,
        "n_features": len(feature_cols),
        "group_col": group_col,
        "is_classification": is_classification,
        "n_classes": n_unique if is_classification else None,
        **sampling_metadata,
    }

    with open(canonical_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Saved canonical artifacts to {canonical_dir}")

    return {
        "canonical_dir": str(canonical_dir),
        "train_ids_path": str(canonical_dir / "train_ids.npy"),
        "y_path": str(canonical_dir / "y.npy"),
        "folds_path": str(canonical_dir / "folds.npy"),
        "feature_cols_path": str(canonical_dir / "feature_cols.json"),
        "metadata_path": str(canonical_dir / "metadata.json"),
        "metadata": metadata,
    }


def load_canonical_data(working_dir: str | Path) -> dict[str, Any]:
    """
    Load all canonical data artifacts.

    Args:
        working_dir: Competition working directory

    Returns:
        Dict with all canonical data loaded:
        - train_ids: np.ndarray of row IDs
        - y: np.ndarray of target values
        - folds: np.ndarray of fold assignments
        - feature_cols: list of feature column names
        - metadata: dict with sampling/CV info
    """
    canonical_dir = Path(working_dir) / "canonical"

    if not canonical_dir.exists():
        raise FileNotFoundError(
            f"Canonical data not found at {canonical_dir}. "
            "Run prepare_canonical_data() first."
        )

    train_ids = np.load(canonical_dir / "train_ids.npy", allow_pickle=True)
    y = np.load(canonical_dir / "y.npy", allow_pickle=True)
    folds = np.load(canonical_dir / "folds.npy")

    with open(canonical_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(canonical_dir / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "train_ids": train_ids,
        "y": y,
        "folds": folds,
        "feature_cols": feature_cols,
        "metadata": metadata,
        "canonical_dir": str(canonical_dir),
    }


def validate_oof_alignment(
    oof: np.ndarray,
    working_dir: str | Path,
    model_train_ids: np.ndarray | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate that OOF predictions align with canonical data.

    Args:
        oof: OOF predictions array
        working_dir: Competition working directory
        model_train_ids: Train IDs used by the model (optional)

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    canonical = load_canonical_data(working_dir)
    canonical_ids = canonical["train_ids"]
    n_canonical = len(canonical_ids)

    # Check shape
    if oof.shape[0] != n_canonical:
        issues.append(
            f"OOF shape mismatch: {oof.shape[0]} rows vs {n_canonical} canonical rows"
        )

    # Check ID alignment if provided
    if model_train_ids is not None:
        if not np.array_equal(model_train_ids, canonical_ids):
            # Check overlap
            common = np.intersect1d(model_train_ids, canonical_ids)
            overlap_pct = len(common) / n_canonical * 100
            issues.append(
                f"Train ID mismatch: {overlap_pct:.1f}% overlap with canonical IDs"
            )

    # Check for NaN/Inf
    if not np.isfinite(oof).all():
        n_invalid = (~np.isfinite(oof)).sum()
        issues.append(f"OOF contains {n_invalid} NaN/Inf values")

    # Check for empty rows
    if oof.ndim > 1:
        empty_mask = oof.sum(axis=1) == 0
    else:
        empty_mask = np.abs(oof) < 1e-10
    n_empty = empty_mask.sum()
    if n_empty > 0:
        issues.append(f"OOF has {n_empty} empty/zero rows")

    return len(issues) == 0, issues


def align_oof_by_id(
    oof: np.ndarray,
    model_ids: np.ndarray,
    canonical_ids: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Align OOF predictions to canonical ID order.

    Useful when model was trained on a subset or different order.

    Args:
        oof: OOF predictions from model
        model_ids: IDs corresponding to oof rows
        canonical_ids: Target canonical ID order
        fill_value: Value to use for missing predictions

    Returns:
        OOF aligned to canonical ID order
    """
    # Create ID to index mapping for model predictions
    model_id_to_idx = {id_val: idx for idx, id_val in enumerate(model_ids)}

    # Initialize aligned OOF
    if oof.ndim > 1:
        aligned_oof = np.full((len(canonical_ids), oof.shape[1]), fill_value)
    else:
        aligned_oof = np.full(len(canonical_ids), fill_value)

    # Map model predictions to canonical order
    for canonical_idx, canonical_id in enumerate(canonical_ids):
        if canonical_id in model_id_to_idx:
            model_idx = model_id_to_idx[canonical_id]
            aligned_oof[canonical_idx] = oof[model_idx]

    return aligned_oof
