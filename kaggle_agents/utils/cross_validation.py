"""
Cross-validation utilities for consistent evaluation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit


def generate_folds(
    train_path: str,
    target_col: str,
    output_path: str,
    n_folds: int = 5,
    seed: int = 42
) -> str:
    """
    Generate fixed cross-validation folds and save to CSV.

    Args:
        train_path: Path to training data CSV
        target_col: Name of target column
        output_path: Path to save folds CSV
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Path to saved folds file
    """
    print(f"   🔄 Generating {n_folds} fixed folds...")

    df = pd.read_csv(train_path)

    # Create 'fold' column
    df['fold'] = -1

    # Determine problem type for splitting strategy
    # Simple heuristic: if target has few unique values -> classification
    n_unique = df[target_col].nunique()
    is_classification = n_unique < 20

    if is_classification:
        print(f"      Detected classification (unique targets: {n_unique}) -> Using StratifiedKFold")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y = df[target_col]
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df, y)):
            df.loc[val_idx, 'fold'] = fold
    else:
        print("      Detected regression -> Using KFold")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold

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
