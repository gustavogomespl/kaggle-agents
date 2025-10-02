"""Adaptive cross-validation strategy selection."""

from typing import Any, Union
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
)
import pandas as pd
import numpy as np


class AdaptiveCrossValidator:
    """Adaptive cross-validation strategy selector."""

    @staticmethod
    def get_cv_strategy(
        strategy_name: str,
        n_splits: int = 5,
        y: pd.Series = None,
        groups: pd.Series = None,
    ) -> Any:
        """Get cross-validation strategy based on name.

        Args:
            strategy_name: Name of CV strategy
            n_splits: Number of splits
            y: Target variable (for stratified)
            groups: Group labels (for grouped)

        Returns:
            Cross-validation splitter instance
        """
        strategy_name_lower = strategy_name.lower()

        # Stratified K-Fold (for classification with imbalanced classes)
        if "stratified" in strategy_name_lower:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Time Series Split (for temporal data)
        elif "time" in strategy_name_lower or "temporal" in strategy_name_lower:
            return TimeSeriesSplit(n_splits=n_splits)

        # Group K-Fold (for grouped data)
        elif "group" in strategy_name_lower:
            if groups is None:
                print("  WARNING: GroupKFold requested but no groups provided, using KFold")
                return KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return GroupKFold(n_splits=n_splits)

        # Default K-Fold
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    @staticmethod
    def determine_best_strategy(
        X: pd.DataFrame,
        y: pd.Series,
        data_characteristics: dict,
    ) -> str:
        """Automatically determine best CV strategy based on data.

        Args:
            X: Feature matrix
            y: Target variable
            data_characteristics: Dict with data analysis results

        Returns:
            Recommended CV strategy name
        """
        # Check for temporal data
        temporal_info = data_characteristics.get("temporal", {})
        if temporal_info.get("has_date_columns", False):
            return "TimeSeriesSplit"

        # Check for grouped data (would need additional logic to detect)
        # For now, we'll rely on the strategy agent's recommendation

        # Check for classification with class imbalance
        target_info = data_characteristics.get("target", {})
        if target_info.get("type") == "categorical":
            imbalance_ratio = target_info.get("imbalance_ratio", 1.0)
            if imbalance_ratio > 2.0:  # Significant imbalance
                return "StratifiedKFold"

        # Check if it's classification
        if y.nunique() < 20 and y.dtype in ['object', 'category', 'int64']:
            return "StratifiedKFold"

        # Default to standard KFold for regression
        return "KFold"

    @staticmethod
    def get_scoring_metric(problem_type: str, metric: str = None) -> str:
        """Get appropriate scoring metric for cross-validation.

        Args:
            problem_type: 'classification' or 'regression'
            metric: Optional specific metric name

        Returns:
            Sklearn scoring metric name
        """
        if metric:
            # Map common Kaggle metrics to sklearn metrics
            metric_mapping = {
                "auc": "roc_auc",
                "accuracy": "accuracy",
                "logloss": "neg_log_loss",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            }
            for key, value in metric_mapping.items():
                if key in metric.lower():
                    return value

        # Default metrics
        if problem_type == "classification":
            return "accuracy"
        else:
            return "neg_mean_squared_error"
