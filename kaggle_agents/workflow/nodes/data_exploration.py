"""EDA node to populate DataInsights for planning and development.

This node runs after canonical_data_preparation and populates the DataInsights
in state with actual EDA insights about the competition data. This knowledge
is then used by the planner and developer for informed decisions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.state import KaggleState
from ...core.state.memory import DataInsights


def data_exploration_node(state: KaggleState) -> dict[str, Any]:
    """
    Compute EDA insights and store in state.

    Args:
        state: Current state with working_directory and data file paths

    Returns:
        State updates with data_insights populated
    """
    print("\n" + "=" * 60)
    print("= DATA EXPLORATION (EDA)")
    print("=" * 60)

    working_dir = Path(state.get("working_directory", "."))
    data_files = state.get("data_files", {})

    # Try to find train data
    train_path = None
    candidates = [
        Path(data_files.get("train", "")) if data_files.get("train") else None,
        working_dir / "train.csv",
        working_dir / "data" / "train.csv",
    ]
    for cand in candidates:
        if cand and cand.exists():
            train_path = cand
            break

    if not train_path or not train_path.exists():
        print("   ⚠️ No train.csv found. Skipping EDA.")
        return {"last_updated": datetime.now()}

    print(f"   Reading: {train_path}")

    try:
        # Sample for speed (max 50k rows for EDA)
        df = pd.read_csv(train_path, nrows=50000)
    except Exception as e:
        print(f"   ⚠️ Failed to read train data: {e}")
        return {"last_updated": datetime.now()}

    n_train_samples = len(df)
    n_features = len(df.columns)

    # Detect target column
    target_col = state.get("target_column")
    if not target_col:
        # Try common target names
        possible_targets = ["target", "label", "class", "y", "outcome"]
        for pt in possible_targets:
            if pt in df.columns:
                target_col = pt
                break
        if not target_col and df.columns[-1] not in ["id", "Id", "ID"]:
            target_col = df.columns[-1]

    # Feature type detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = []

    # Remove target and ID from feature lists
    id_cols = [c for c in df.columns if c.lower() in ("id", "row_id", "index")]
    exclude_cols = set(id_cols + ([target_col] if target_col else []))

    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    # Try to detect datetime columns from object columns
    for col in categorical_cols[:]:
        try:
            sample = df[col].dropna().iloc[:100]
            if len(sample) > 0:
                pd.to_datetime(sample.iloc[0])
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except (ValueError, TypeError, IndexError):
            pass

    # Missing values analysis
    missing_value_cols = {}
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        if missing_ratio > 0:
            missing_value_cols[col] = float(missing_ratio)

    # High cardinality detection (>50 unique values)
    high_cardinality_cols = []
    for col in categorical_cols:
        nunique = df[col].nunique()
        if nunique > 50:
            high_cardinality_cols.append(col)

    # Constant columns detection
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    # Target distribution analysis
    target_distribution = {}
    is_imbalanced = False
    imbalance_ratio = None
    n_classes = None

    if target_col and target_col in df.columns:
        value_counts = df[target_col].value_counts(normalize=True)
        target_distribution = value_counts.to_dict()
        n_classes = len(value_counts)

        # Guard against division by zero (all same class)
        min_count = value_counts.min()
        if min_count > 0 and n_classes >= 2:
            if n_classes == 2:
                # Binary classification
                imbalance_ratio = float(value_counts.max() / min_count)
                is_imbalanced = imbalance_ratio > 3.0
            elif n_classes <= 100:
                # Multiclass
                imbalance_ratio = float(value_counts.max() / min_count)
                is_imbalanced = imbalance_ratio > 5.0

    # Highly correlated pairs (only for numeric, sample for speed)
    highly_correlated_pairs = []
    if len(numeric_cols) > 1 and len(numeric_cols) <= 50:
        try:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.9:
                            highly_correlated_pairs.append((col1, col2, float(corr)))
        except Exception:
            pass

    # Test data size
    n_test_samples = 0
    test_path = None
    test_candidates = [
        Path(data_files.get("test", "")) if data_files.get("test") else None,
        working_dir / "test.csv",
        working_dir / "data" / "test.csv",
    ]
    for cand in test_candidates:
        if cand and cand.exists():
            test_path = cand
            break

    if test_path and test_path.exists():
        try:
            # Just count rows
            with open(test_path, "r") as f:
                n_test_samples = sum(1 for _ in f) - 1  # -1 for header
        except Exception:
            pass

    # Create DataInsights object
    insights = DataInsights(
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        n_features=n_features,
        n_classes=n_classes,
        target_distribution=target_distribution,
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        datetime_features=datetime_cols,
        text_features=[],  # TODO: detect text features
        missing_value_cols=missing_value_cols,
        high_cardinality_cols=high_cardinality_cols,
        constant_cols=constant_cols,
        highly_correlated_pairs=highly_correlated_pairs,
    )

    # Print summary
    print(f"\n   📊 EDA Summary:")
    print(f"      Train samples: {n_train_samples:,}")
    print(f"      Test samples: {n_test_samples:,}")
    print(f"      Features: {n_features} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    if n_classes:
        print(f"      Classes: {n_classes}")
    if is_imbalanced:
        print(f"      ⚠️ IMBALANCED: ratio={imbalance_ratio:.2f}")
    if high_cardinality_cols:
        print(f"      ⚠️ High cardinality columns: {high_cardinality_cols}")
    if missing_value_cols:
        high_missing = {k: v for k, v in missing_value_cols.items() if v > 0.1}
        if high_missing:
            print(f"      ⚠️ High missing (>10%): {list(high_missing.keys())}")
    if highly_correlated_pairs:
        print(f"      ⚠️ Highly correlated pairs: {len(highly_correlated_pairs)}")
    if constant_cols:
        print(f"      ⚠️ Constant columns (can drop): {constant_cols}")

    return {
        "data_insights": insights,
        "last_updated": datetime.now(),
    }
