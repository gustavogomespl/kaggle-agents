"""Advanced feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.preprocessing import LabelEncoder


class AdvancedFeatureEngineer:
    """Advanced feature engineering techniques."""

    def __init__(self):
        """Initialize feature engineer."""
        self.encoders = {}
        self.scalers = {}
        self.created_features = []

    def handle_missing_values_advanced(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, strategy: str = "advanced"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Advanced missing value imputation.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            strategy: Imputation strategy ('basic', 'advanced')

        Returns:
            Tuple of (train_df, test_df) with missing values handled
        """
        if strategy == "basic":
            # Basic strategy: median/mode
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if train_df[col].isnull().any():
                    median_val = train_df[col].median()
                    train_df[col].fillna(median_val, inplace=True)
                    test_df[col].fillna(median_val, inplace=True)

            categorical_cols = train_df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if train_df[col].isnull().any():
                    mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else "Unknown"
                    train_df[col].fillna(mode_val, inplace=True)
                    test_df[col].fillna(mode_val, inplace=True)
        else:
            # Advanced: create missing indicators and use median/mode
            for col in train_df.columns:
                if train_df[col].isnull().any():
                    # Create missing indicator feature
                    train_df[f"{col}_missing"] = train_df[col].isnull().astype(int)
                    test_df[f"{col}_missing"] = test_df[col].isnull().astype(int)
                    self.created_features.append(f"{col}_missing")

                    # Fill values
                    if pd.api.types.is_numeric_dtype(train_df[col]):
                        fill_value = train_df[col].median()
                    else:
                        fill_value = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else "Unknown"

                    train_df[col].fillna(fill_value, inplace=True)
                    test_df[col].fillna(fill_value, inplace=True)

        return train_df, test_df

    def encode_categorical_adaptive(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        encoding_strategy: Dict[str, str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Adaptive categorical encoding based on cardinality.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            target_col: Target column name
            encoding_strategy: Strategy dict with 'low_cardinality' and 'high_cardinality'

        Returns:
            Tuple of (train_df, test_df) with encoded features
        """
        categorical_cols = train_df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            cardinality = train_df[col].nunique()

            if cardinality < 10:
                # Low cardinality: use specified strategy or label encoding
                strategy = encoding_strategy.get("low_cardinality", "label")

                if strategy == "onehot":
                    # One-hot encoding
                    dummies_train = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
                    dummies_test = pd.get_dummies(test_df[col], prefix=col, drop_first=True)

                    # Align columns - ensure test has same columns as train
                    for dummy_col in dummies_train.columns:
                        if dummy_col not in dummies_test.columns:
                            dummies_test[dummy_col] = 0

                    # Remove any columns in test that aren't in train
                    for dummy_col in dummies_test.columns:
                        if dummy_col not in dummies_train.columns:
                            dummies_test = dummies_test.drop(columns=[dummy_col])

                    # Ensure column order matches
                    dummies_test = dummies_test[dummies_train.columns]

                    train_df = pd.concat([train_df.drop(columns=[col]), dummies_train], axis=1)
                    test_df = pd.concat([test_df.drop(columns=[col]), dummies_test], axis=1)
                    self.created_features.extend(dummies_train.columns.tolist())
                else:
                    # Label encoding
                    le = LabelEncoder()
                    combined = pd.concat([train_df[col], test_df[col]], axis=0)
                    le.fit(combined.astype(str))
                    train_df[col] = le.transform(train_df[col].astype(str))
                    test_df[col] = le.transform(test_df[col].astype(str))
                    self.encoders[col] = le

            else:
                # High cardinality: use target encoding or catboost encoding
                strategy = encoding_strategy.get("high_cardinality", "target")

                if strategy == "target" and target_col in train_df.columns:
                    # Target encoding
                    encoder = TargetEncoder(cols=[col])
                    train_df[col] = encoder.fit_transform(train_df[col], train_df[target_col])
                    test_df[col] = encoder.transform(test_df[col])
                    self.encoders[col] = encoder
                elif strategy == "catboost" and target_col in train_df.columns:
                    # CatBoost encoding
                    encoder = CatBoostEncoder(cols=[col])
                    train_df[col] = encoder.fit_transform(train_df[col], train_df[target_col])
                    test_df[col] = encoder.transform(test_df[col])
                    self.encoders[col] = encoder
                else:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    combined = pd.concat([train_df[col], test_df[col]], axis=0)
                    le.fit(combined.astype(str))
                    train_df[col] = le.transform(train_df[col].astype(str))
                    test_df[col] = le.transform(test_df[col].astype(str))
                    self.encoders[col] = le

        return train_df, test_df

    def create_polynomial_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, degree: int = 2, max_features: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create polynomial features for numeric columns.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            degree: Polynomial degree
            max_features: Maximum number of features to create polynomials for

        Returns:
            Tuple of (train_df, test_df) with polynomial features
        """
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns[:max_features]

        if len(numeric_cols) >= 2:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            train_poly = poly.fit_transform(train_df[numeric_cols])
            test_poly = poly.transform(test_df[numeric_cols])

            # Get feature names
            feature_names = poly.get_feature_names_out(numeric_cols)

            # Add only interaction and polynomial terms (exclude original features)
            new_feature_indices = [i for i, name in enumerate(feature_names) if "^" in name or " " in name]

            for idx in new_feature_indices[:10]:  # Limit to 10 new features
                feature_name = f"poly_{idx}"
                train_df[feature_name] = train_poly[:, idx]
                test_df[feature_name] = test_poly[:, idx]
                self.created_features.append(feature_name)

        return train_df, test_df

    def extract_date_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, date_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features from date columns.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            date_columns: List of date column names

        Returns:
            Tuple of (train_df, test_df) with date features
        """
        for col in date_columns:
            if col in train_df.columns:
                # Convert to datetime
                train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
                test_df[col] = pd.to_datetime(test_df[col], errors="coerce")

                # Extract features
                train_df[f"{col}_year"] = train_df[col].dt.year
                train_df[f"{col}_month"] = train_df[col].dt.month
                train_df[f"{col}_day"] = train_df[col].dt.day
                train_df[f"{col}_dayofweek"] = train_df[col].dt.dayofweek
                train_df[f"{col}_quarter"] = train_df[col].dt.quarter

                test_df[f"{col}_year"] = test_df[col].dt.year
                test_df[f"{col}_month"] = test_df[col].dt.month
                test_df[f"{col}_day"] = test_df[col].dt.day
                test_df[f"{col}_dayofweek"] = test_df[col].dt.dayofweek
                test_df[f"{col}_quarter"] = test_df[col].dt.quarter

                self.created_features.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_dayofweek", f"{col}_quarter"
                ])

                # Drop original date column
                train_df = train_df.drop(columns=[col])
                test_df = test_df.drop(columns=[col])

        return train_df, test_df

    def create_aggregation_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create aggregation features from numeric columns.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe

        Returns:
            Tuple of (train_df, test_df) with aggregation features
        """
        # Get numeric columns that exist in both train and test
        train_numeric = set(train_df.select_dtypes(include=[np.number]).columns)
        test_numeric = set(test_df.select_dtypes(include=[np.number]).columns)
        numeric_cols = list(train_numeric & test_numeric)  # Intersection

        if len(numeric_cols) >= 3:
            train_df["numeric_sum"] = train_df[numeric_cols].sum(axis=1)
            train_df["numeric_mean"] = train_df[numeric_cols].mean(axis=1)
            train_df["numeric_std"] = train_df[numeric_cols].std(axis=1)
            train_df["numeric_min"] = train_df[numeric_cols].min(axis=1)
            train_df["numeric_max"] = train_df[numeric_cols].max(axis=1)

            test_df["numeric_sum"] = test_df[numeric_cols].sum(axis=1)
            test_df["numeric_mean"] = test_df[numeric_cols].mean(axis=1)
            test_df["numeric_std"] = test_df[numeric_cols].std(axis=1)
            test_df["numeric_min"] = test_df[numeric_cols].min(axis=1)
            test_df["numeric_max"] = test_df[numeric_cols].max(axis=1)

            self.created_features.extend(["numeric_sum", "numeric_mean", "numeric_std", "numeric_min", "numeric_max"])

        return train_df, test_df

    def scale_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, method: str = "standard"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numeric features.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            method: Scaling method ('standard' or 'minmax')

        Returns:
            Tuple of (train_df, test_df) with scaled features
        """
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns

        if method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

        self.scalers["main"] = scaler

        return train_df, test_df
