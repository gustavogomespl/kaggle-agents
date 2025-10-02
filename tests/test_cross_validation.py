"""Tests for cross-validation utilities."""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from kaggle_agents.utils.cross_validation import AdaptiveCrossValidator


class TestAdaptiveCrossValidator:
    """Tests for AdaptiveCrossValidator."""

    def test_get_stratified_kfold(self):
        """Test StratifiedKFold selection."""
        cv = AdaptiveCrossValidator.get_cv_strategy("StratifiedKFold", n_splits=5)
        assert isinstance(cv, StratifiedKFold)
        assert cv.n_splits == 5

    def test_get_time_series_split(self):
        """Test TimeSeriesSplit selection."""
        cv = AdaptiveCrossValidator.get_cv_strategy("TimeSeriesSplit", n_splits=5)
        assert isinstance(cv, TimeSeriesSplit)
        assert cv.n_splits == 5

    def test_get_kfold_default(self):
        """Test default KFold selection."""
        cv = AdaptiveCrossValidator.get_cv_strategy("KFold", n_splits=5)
        assert isinstance(cv, KFold)
        assert cv.n_splits == 5

    def test_determine_best_strategy_temporal(self):
        """Test strategy determination for temporal data."""
        X = pd.DataFrame({"feature": range(100)})
        y = pd.Series(np.random.choice([0, 1], 100))

        data_chars = {
            "temporal": {"has_date_columns": True},
            "target": {"type": "categorical"},
        }

        strategy = AdaptiveCrossValidator.determine_best_strategy(X, y, data_chars)
        assert strategy == "TimeSeriesSplit"

    def test_determine_best_strategy_imbalanced(self):
        """Test strategy determination for imbalanced data."""
        X = pd.DataFrame({"feature": range(100)})
        y = pd.Series(np.random.choice([0, 1], 100))

        data_chars = {
            "temporal": {"has_date_columns": False},
            "target": {"type": "categorical", "imbalance_ratio": 5.0},
        }

        strategy = AdaptiveCrossValidator.determine_best_strategy(X, y, data_chars)
        assert strategy == "StratifiedKFold"

    def test_get_scoring_metric_classification(self):
        """Test scoring metric for classification."""
        metric = AdaptiveCrossValidator.get_scoring_metric("classification")
        assert metric == "accuracy"

    def test_get_scoring_metric_regression(self):
        """Test scoring metric for regression."""
        metric = AdaptiveCrossValidator.get_scoring_metric("regression")
        assert metric == "neg_mean_squared_error"

    def test_get_scoring_metric_custom(self):
        """Test custom metric mapping."""
        metric = AdaptiveCrossValidator.get_scoring_metric("classification", "AUC")
        assert metric == "roc_auc"

        metric = AdaptiveCrossValidator.get_scoring_metric("regression", "RMSE")
        assert metric == "neg_root_mean_squared_error"
