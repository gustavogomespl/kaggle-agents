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

import hashlib
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

    # Deterministic order - convert to str to handle mixed types (str/float in column names)
    # This can happen with CSVs that have numeric column headers
    common_str = [str(c) for c in common]
    missing_str = [str(c) for c in missing_in_test]
    return sorted(common_str), sorted(missing_str)


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


def _deterministic_hash(value: str, seed: int = 42) -> int:
    """
    Deterministic hash using MD5 + seed.

    Unlike Python's built-in hash(), this produces the same result
    across different Python processes (PYTHONHASHSEED independent).

    Args:
        value: String value to hash
        seed: Random seed for reproducibility

    Returns:
        Integer hash value (0 to 2^32-1)
    """
    combined = f"{seed}_{value}"
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


def _ensure_id_column(
    df: pd.DataFrame,
    id_col: str | None,
) -> tuple[pd.DataFrame, str, bool]:
    """
    Ensure a valid ID column exists for deterministic sampling.

    If no ID column is found, creates a synthetic '_row_id' based on
    the original row index. This MUST be done BEFORE any transformations.

    Args:
        df: DataFrame to check
        id_col: Detected or specified ID column name

    Returns:
        Tuple of (df_with_id, id_col_name, is_synthetic)
    """
    is_synthetic = False

    if id_col is None or id_col not in df.columns:
        # Create synthetic ID based on original index (preserves order)
        # IMPORTANT: Must be done BEFORE any transformation/shuffle
        df = df.copy()
        df["_row_id"] = df.index.astype(str)
        id_col = "_row_id"
        is_synthetic = True
        print(f"[LOG:WARN] No ID column found, using synthetic '_row_id' for sampling")

    return df, id_col, is_synthetic


def _remove_synthetic_id_from_features(
    df: pd.DataFrame,
    is_synthetic: bool,
) -> pd.DataFrame:
    """
    Remove synthetic _row_id from features AFTER artifacts are generated.

    Args:
        df: DataFrame potentially containing _row_id
        is_synthetic: Whether the ID was synthetically created

    Returns:
        DataFrame without _row_id column (if synthetic)
    """
    if is_synthetic and "_row_id" in df.columns:
        return df.drop(columns=["_row_id"])
    return df


def _hash_based_sample(
    df: pd.DataFrame,
    id_col: str | None,
    max_rows: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deterministic hash-based sampling using MD5.

    Uses MD5 hash of ID (not Python's built-in hash) to ensure
    the same rows are selected across different Python processes.

    Args:
        df: DataFrame to sample
        id_col: ID column name (can be None, will be created)
        max_rows: Maximum rows to keep
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_df, sampling_metadata)
    """
    # Step 1: Ensure ID column exists
    df, id_col, is_synthetic = _ensure_id_column(df, id_col)

    original_rows = len(df)
    if original_rows <= max_rows:
        return df, {
            "sampled": False,
            "original_rows": original_rows,
            "id_column": id_col,
            "id_is_synthetic": is_synthetic,
        }

    # Step 2: Calculate threshold for sampling
    sample_frac = max_rows / original_rows
    threshold = int(10000 * sample_frac)

    # Step 3: Apply deterministic MD5 hash
    def should_include(id_val):
        return (_deterministic_hash(str(id_val), seed) % 10000) < threshold

    sample_mask = df[id_col].apply(should_include).values
    sampled_df = df[sample_mask].reset_index(drop=True)

    # Step 4: Keep _row_id for now (needed for alignment)
    # Will be removed AFTER generating artifacts (folds, train_ids)

    metadata = {
        "sampled": True,
        "original_rows": original_rows,
        "sampled_rows": len(sampled_df),
        "sampling_method": "hash_based_md5",
        "sampling_threshold": threshold,
        "sampling_seed": seed,
        "hash_method": "md5",
        "id_column": id_col,
        "id_is_synthetic": is_synthetic,
        "deterministic": True,
        "canonical_version": "1.2",
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

    # Step 1: Load training data RAW (no transformations yet)
    train_df = pd.read_csv(train_path)
    original_rows = len(train_df)
    print(f"   Loaded {original_rows:,} training rows")

    # Step 2: Detect ID column BEFORE any operations
    if id_col is None:
        id_col = _detect_id_column(train_df)

    # Validate target exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    # Step 3: Ensure ID exists and apply sampling (proper order)
    # The _hash_based_sample function handles _ensure_id_column internally
    is_synthetic_id = False
    sampling_metadata = {"sampled": False, "original_rows": original_rows}

    if max_rows and len(train_df) > max_rows:
        train_df, sampling_metadata = _hash_based_sample(train_df, id_col, max_rows, seed=42)
        id_col = sampling_metadata.get("id_column", id_col)
        is_synthetic_id = sampling_metadata.get("id_is_synthetic", False)
        print(f"   Sampled {len(train_df):,} rows via deterministic MD5 hash (seed=42)")
    else:
        # Even without sampling, ensure ID column exists
        train_df, id_col, is_synthetic_id = _ensure_id_column(train_df, id_col)
        sampling_metadata["id_column"] = id_col
        sampling_metadata["id_is_synthetic"] = is_synthetic_id

    if is_synthetic_id:
        print(f"   Using synthetic ID column: {id_col}")
    else:
        print(f"   Using ID column: {id_col}")

    # Schema parity check
    feature_cols, missing_in_test = validate_schema_parity(
        train_path, test_path, id_col, target_col
    )

    # Remove synthetic _row_id from feature columns if present
    if is_synthetic_id and "_row_id" in feature_cols:
        feature_cols.remove("_row_id")

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
        "id_is_synthetic": is_synthetic_id,
        "target_col": target_col,
        "n_features": len(feature_cols),
        "group_col": group_col,
        "is_classification": is_classification,
        "n_classes": n_unique if is_classification else None,
        "canonical_version": "1.2",
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


# ==================== Code Validation ====================


def validate_canonical_data_usage(
    generated_code: str,
    working_dir: str | Path,
    component_type: str = "model",
) -> tuple[bool, str, list[str]]:
    """
    Validate that generated code uses canonical data correctly.

    Checks for:
    1. Use of canonical data loading (load_canonical_data or npy files)
    2. Proper fold usage for CV
    3. No independent sampling or fold creation

    Args:
        generated_code: The code to validate
        working_dir: Working directory path
        component_type: Type of component (model, feature_engineering, etc.)

    Returns:
        Tuple of (is_valid, error_message, warnings)
    """
    import re

    warnings = []
    code_lower = generated_code.lower()

    # Check if canonical directory exists
    canonical_dir = Path(working_dir) / "canonical"
    canonical_exists = canonical_dir.exists()

    if not canonical_exists:
        # Canonical data not yet prepared - this is OK for early components
        return True, "", ["Canonical data not yet prepared - will be created"]

    # Patterns indicating proper canonical data usage
    canonical_patterns = [
        r"load_canonical_data\s*\(",
        r"canonical/train_ids\.npy",
        r"canonical/folds\.npy",
        r"canonical/y\.npy",
        r"np\.load\s*\([^)]*canonical[^)]*\)",
    ]

    # Check if any canonical pattern is present
    uses_canonical = any(re.search(p, generated_code) for p in canonical_patterns)

    # Anti-patterns: things that suggest independent data handling
    anti_patterns = [
        (r"train_test_split\s*\(", "Using train_test_split - should use canonical folds"),
        (r"StratifiedKFold\s*\(", "Creating new folds - should use canonical folds"),
        (r"KFold\s*\(", "Creating new folds - should use canonical folds"),
        (r"GroupKFold\s*\(", "Creating new folds - should use canonical folds"),
        (r"\.sample\s*\(", "Sampling data - may cause alignment issues with canonical"),
        (r"shuffle\s*=\s*True", "Shuffling data - may cause alignment issues"),
    ]

    violations = []
    for pattern, message in anti_patterns:
        if re.search(pattern, generated_code):
            # Exception: If it's used to create canonical data, that's OK
            if "prepare_canonical_data" in generated_code:
                continue
            violations.append(message)

    # Model components MUST use canonical data (STRICT ENFORCEMENT)
    if component_type == "model":
        # Check for required canonical patterns (MUST have these)
        required_patterns = [
            (r"canonical.*folds\.npy|folds\.npy.*canonical", "Must load canonical/folds.npy"),
            (r"canonical.*train_ids\.npy|train_ids\.npy.*canonical", "Must load canonical/train_ids.npy"),
        ]

        missing_required = []
        for pattern, message in required_patterns:
            if not re.search(pattern, generated_code, re.IGNORECASE):
                missing_required.append(message)

        # Fail if violations exist (creating independent folds)
        if violations:
            error_msg = (
                "Model code violates canonical data contract. "
                f"Violations: {'; '.join(violations)}. "
                "MUST use canonical folds from canonical/folds.npy - do NOT create KFold/StratifiedKFold."
            )
            return False, error_msg, warnings

        # Fail if not using canonical data patterns
        if missing_required and not uses_canonical:
            error_msg = (
                "Model code does not use canonical data contract. "
                f"Missing: {'; '.join(missing_required)}. "
                "Use load_canonical_data() or load canonical/*.npy files for consistent OOF alignment."
            )
            return False, error_msg, warnings

        # Warn if missing required but has some canonical usage
        if missing_required:
            warnings.extend(missing_required)

    # Feature engineering components should be more flexible
    elif component_type == "feature_engineering":
        if violations:
            warnings.append(
                "Feature engineering code may modify data alignment. "
                "Ensure train_ids are preserved."
            )

    # Ensemble MUST use aligned predictions
    elif component_type == "ensemble":
        if not uses_canonical and not re.search(r"oof_.*\.npy|test_.*\.npy", generated_code):
            warnings.append(
                "Ensemble should verify OOF alignment with canonical train_ids"
            )

    return True, "", warnings


def get_canonical_data_instructions(working_dir: str | Path) -> str:
    """
    Generate instructions for using canonical data in generated code.

    Args:
        working_dir: Working directory path

    Returns:
        Instruction string to inject into developer prompt
    """
    canonical_dir = Path(working_dir) / "canonical"

    if canonical_dir.exists():
        # Load metadata for context
        try:
            with open(canonical_dir / "metadata.json") as f:
                metadata = json.load(f)
            n_rows = metadata.get("canonical_rows", "unknown")
            n_folds = metadata.get("n_folds", 5)
            id_col = metadata.get("id_col", "id")
        except Exception:
            n_rows = "unknown"
            n_folds = 5
            id_col = "id"

        return f'''
## MANDATORY: Canonical Data Contract

The canonical data has been prepared with {n_rows} rows and {n_folds} folds.
You MUST use the canonical data to ensure consistency across all models.

### How to Load Canonical Data:

```python
import numpy as np
import json
from pathlib import Path

# Load canonical data
canonical_dir = Path("{working_dir}/canonical")
train_ids = np.load(canonical_dir / "train_ids.npy", allow_pickle=True)
y = np.load(canonical_dir / "y.npy", allow_pickle=True)
folds = np.load(canonical_dir / "folds.npy")

with open(canonical_dir / "feature_cols.json") as f:
    feature_cols = json.load(f)

# Use folds for CV (DO NOT create your own folds!)
n_folds = {n_folds}
for fold_idx in range(n_folds):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    # Train model...
    model.fit(X_train, y_train)

    # Store OOF predictions in order
    oof[val_mask] = model.predict_proba(X_val)
```

### CRITICAL RULES:
1. NEVER use train_test_split() - use canonical folds
2. NEVER create your own KFold/StratifiedKFold - folds are pre-defined
3. NEVER sample or shuffle the data independently
4. ALWAYS save OOF predictions in canonical order (aligned with train_ids)
5. ID column is: "{id_col}"

### Saving Predictions:
```python
# Save OOF aligned with canonical train_ids
np.save("models/oof_{{model_name}}.npy", oof)

# Verify alignment before saving
assert len(oof) == len(train_ids), "OOF must match canonical row count"
```
'''
    else:
        return '''
## Note: Canonical Data Will Be Prepared

The canonical data contract will be prepared before your component runs.
When it's ready, use load_canonical_data() to get train_ids, folds, and y.
Do NOT create your own folds or sampling strategy.
'''
