"""
Feature engineering utilities (legacy layer).

These helpers are intentionally lightweight and deterministic to support unit
tests and simple baselines without requiring network access or heavy pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AdvancedFeatureEngineer:
    """A small, test-friendly feature engineering helper."""

    created_features: list[str] = field(default_factory=list)

    def handle_missing_values_advanced(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        strategy: str = "basic",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fill missing values in train/test consistently.

        - Numeric: median (from train)
        - Categorical: "missing"
        - strategy="advanced": add *_missing indicator columns
        """
        train = train_df.copy()
        test = test_df.copy()

        all_cols = list(dict.fromkeys([*train.columns, *test.columns]))
        for col in all_cols:
            if col not in train.columns:
                train[col] = np.nan
            if col not in test.columns:
                test[col] = np.nan

            missing_train = train[col].isna()
            missing_test = test[col].isna()
            has_missing = bool(missing_train.any() or missing_test.any())

            if strategy.lower() == "advanced" and has_missing:
                ind_col = f"{col}_missing"
                train[ind_col] = missing_train.astype(int)
                test[ind_col] = missing_test.astype(int)

            if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_numeric_dtype(test[col]):
                median = pd.to_numeric(train[col], errors="coerce").median()
                if pd.isna(median):
                    median = 0.0
                train[col] = pd.to_numeric(train[col], errors="coerce").fillna(median)
                test[col] = pd.to_numeric(test[col], errors="coerce").fillna(median)
            else:
                train[col] = train[col].astype("object").where(~missing_train, "missing")
                test[col] = test[col].astype("object").where(~missing_test, "missing")

        return train, test

    def encode_categorical_adaptive(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        encoding_strategy: dict[str, Any],
        *,
        low_cardinality_threshold: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical columns based on cardinality.

        Supported strategies:
        - low_cardinality: "onehot" | "label"
        - high_cardinality: "label"
        """
        train = train_df.copy()
        test = test_df.copy()

        y = None
        if target_col and target_col in train.columns:
            y = train[target_col]
            train = train.drop(columns=[target_col])

        cat_cols = list(train.select_dtypes(include=["object", "category"]).columns)

        low_strategy = str(encoding_strategy.get("low_cardinality", "onehot")).lower()
        high_strategy = str(encoding_strategy.get("high_cardinality", "label")).lower()

        for col in cat_cols:
            n_unique = int(train[col].nunique(dropna=True))
            is_low = n_unique <= low_cardinality_threshold

            strategy = low_strategy if is_low else high_strategy
            if strategy == "onehot":
                train_d = pd.get_dummies(train[col], prefix=col, dummy_na=False)
                test_d = pd.get_dummies(test[col], prefix=col, dummy_na=False)
                train_d, test_d = train_d.align(test_d, join="outer", axis=1, fill_value=0)
                train = pd.concat([train.drop(columns=[col]), train_d], axis=1)
                test = pd.concat([test.drop(columns=[col]), test_d], axis=1)
            else:  # label encoding
                vals = train[col].astype("object").fillna("missing")
                codes, uniques = pd.factorize(vals, sort=True)
                mapping = {v: int(i) for i, v in enumerate(uniques)}
                train[col] = codes.astype(int)
                test[col] = (
                    test[col]
                    .astype("object")
                    .fillna("missing")
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )

        if y is not None:
            train[target_col] = y

        return train, test

    def create_polynomial_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        degree: int = 2,
        max_features: int = 3,
        target_col: str = "target",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create simple polynomial features from numeric columns (degree 2 only)."""
        if degree != 2:
            raise ValueError("Only degree=2 is supported in the lightweight implementation")

        train = train_df.copy()
        test = test_df.copy()

        numeric_cols = [
            c
            for c in train.select_dtypes(include=[np.number]).columns
            if c != target_col
        ][: max(1, int(max_features))]

        for i, col_a in enumerate(numeric_cols):
            sq_name = f"poly_{col_a}__2"
            train[sq_name] = train[col_a] ** 2
            test[sq_name] = test[col_a] ** 2
            self.created_features.append(sq_name)

            for col_b in numeric_cols[i + 1 :]:
                cross_name = f"poly_{col_a}__x__{col_b}"
                train[cross_name] = train[col_a] * train[col_b]
                test[cross_name] = test[col_a] * test[col_b]
                self.created_features.append(cross_name)

        return train, test

    def create_aggregation_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        target_col: str = "target",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create row-wise aggregations over numeric columns."""
        train = train_df.copy()
        test = test_df.copy()

        numeric_cols = [
            c
            for c in train.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
        if not numeric_cols:
            train["numeric_sum"] = 0.0
            train["numeric_mean"] = 0.0
            train["numeric_std"] = 0.0
            test["numeric_sum"] = 0.0
            test["numeric_mean"] = 0.0
            test["numeric_std"] = 0.0
            return train, test

        train_num = train[numeric_cols]
        test_num = test[numeric_cols] if all(c in test.columns for c in numeric_cols) else test.reindex(columns=numeric_cols, fill_value=0.0)

        train["numeric_sum"] = train_num.sum(axis=1)
        train["numeric_mean"] = train_num.mean(axis=1)
        train["numeric_std"] = train_num.std(axis=1).fillna(0.0)

        test["numeric_sum"] = test_num.sum(axis=1)
        test["numeric_mean"] = test_num.mean(axis=1)
        test["numeric_std"] = test_num.std(axis=1).fillna(0.0)

        return train, test

    def scale_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        method: str = "standard",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numeric features using statistics from train."""
        if method.lower() != "standard":
            raise ValueError("Only method='standard' is supported in the lightweight implementation")

        train = train_df.copy()
        test = test_df.copy()

        means = train.mean(axis=0)
        stds = train.std(axis=0).replace(0, 1.0)

        train_scaled = (train - means) / stds
        test_scaled = (test - means) / stds

        return train_scaled, test_scaled

