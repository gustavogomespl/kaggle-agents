"""Tests for feature engineering utilities."""

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.utils.feature_engineering import AdvancedFeatureEngineer


class TestAdvancedFeatureEngineer:
    """Tests for AdvancedFeatureEngineer."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return AdvancedFeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data with various types."""
        np.random.seed(42)
        n = 100

        train = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n),
                "cat_low": np.random.choice(["A", "B", "C"], n),
                "cat_high": np.random.choice([f"cat_{i}" for i in range(50)], n),
                "target": np.random.choice([0, 1], n),
            }
        )

        test = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n),
                "cat_low": np.random.choice(["A", "B", "C"], n),
                "cat_high": np.random.choice([f"cat_{i}" for i in range(50)], n),
            }
        )

        # Add some missing values
        train.loc[0:10, "num1"] = np.nan
        train.loc[5:15, "cat_low"] = np.nan

        return train, test

    def test_handle_missing_values_basic(self, engineer, sample_data):
        """Test basic missing value handling."""
        train, test = sample_data
        train_clean, test_clean = engineer.handle_missing_values_advanced(
            train.copy(), test.copy(), strategy="basic"
        )

        assert train_clean["num1"].isnull().sum() == 0
        assert train_clean["cat_low"].isnull().sum() == 0

    def test_handle_missing_values_advanced(self, engineer, sample_data):
        """Test advanced missing value handling with indicators."""
        train, test = sample_data
        train_clean, test_clean = engineer.handle_missing_values_advanced(
            train.copy(), test.copy(), strategy="advanced"
        )

        # Check missing indicators were created
        assert "num1_missing" in train_clean.columns
        assert "cat_low_missing" in train_clean.columns
        assert train_clean["num1"].isnull().sum() == 0

    def test_encode_categorical_adaptive(self, engineer, sample_data):
        """Test adaptive categorical encoding."""
        train, test = sample_data

        # Fill missing values first
        train_clean, test_clean = engineer.handle_missing_values_advanced(
            train.copy(), test.copy(), strategy="basic"
        )

        encoding_strategy = {
            "low_cardinality": "onehot",
            "high_cardinality": "label",
        }

        train_encoded, test_encoded = engineer.encode_categorical_adaptive(
            train_clean, test_clean, "target", encoding_strategy
        )

        # Low cardinality should be one-hot encoded
        assert any("cat_low_" in col for col in train_encoded.columns)

        # High cardinality should be label encoded
        assert train_encoded["cat_high"].dtype in [np.int64, np.float64]

    def test_create_polynomial_features(self, engineer, sample_data):
        """Test polynomial feature creation."""
        train, test = sample_data
        train_clean, test_clean = engineer.handle_missing_values_advanced(
            train.copy(), test.copy(), strategy="basic"
        )

        train_poly, test_poly = engineer.create_polynomial_features(
            train_clean, test_clean, degree=2, max_features=2
        )

        # Check that polynomial features were created
        poly_features = [f for f in engineer.created_features if "poly_" in f]
        assert len(poly_features) > 0

    def test_create_aggregation_features(self, engineer, sample_data):
        """Test aggregation feature creation."""
        train, test = sample_data
        train_clean, test_clean = engineer.handle_missing_values_advanced(
            train.copy(), test.copy(), strategy="basic"
        )

        train_agg, test_agg = engineer.create_aggregation_features(train_clean, test_clean)

        # Check aggregation features
        assert "numeric_sum" in train_agg.columns
        assert "numeric_mean" in train_agg.columns
        assert "numeric_std" in train_agg.columns

    def test_scale_features(self, engineer, sample_data):
        """Test feature scaling."""
        train, test = sample_data
        train_clean, test_clean = engineer.handle_missing_values_advanced(
            train.copy(), test.copy(), strategy="basic"
        )

        # Drop categoricals for scaling test
        train_numeric = train_clean.select_dtypes(include=[np.number])
        test_numeric = test_clean.select_dtypes(include=[np.number])

        train_scaled, test_scaled = engineer.scale_features(
            train_numeric.copy(), test_numeric.copy(), method="standard"
        )

        # Check that features are scaled (mean ~0, std ~1)
        assert abs(train_scaled["num1"].mean()) < 0.1
        assert abs(train_scaled["num1"].std() - 1.0) < 0.1
