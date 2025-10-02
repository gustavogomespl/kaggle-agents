"""Hyperparameter optimization using Optuna."""

import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Callable, Union
import numpy as np
from sklearn.model_selection import cross_val_score


class HyperparameterOptimizer:
    """Hyperparameter optimization with Optuna."""

    def __init__(self, n_trials: int = 50, timeout: int = 300):
        """Initialize optimizer.

        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
        """
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize_xgboost(
        self, X, y, problem_type: str, scoring: str, cv: Union[int, Any] = 5
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'
            scoring: Scoring metric
            cv: CV strategy (int for n_splits or CV object)

        Returns:
            Best hyperparameters
        """
        from xgboost import XGBClassifier, XGBRegressor

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "n_jobs": -1,
            }

            if problem_type == "classification":
                model = XGBClassifier(**params)
            else:
                model = XGBRegressor(**params)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )

        return study.best_params

    def optimize_lightgbm(
        self, X, y, problem_type: str, scoring: str, cv: int = 5
    ) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'
            scoring: Scoring metric
            cv: Number of CV folds

        Returns:
            Best hyperparameters
        """
        from lightgbm import LGBMClassifier, LGBMRegressor

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            if problem_type == "classification":
                model = LGBMClassifier(**params)
            else:
                model = LGBMRegressor(**params)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )

        return study.best_params

    def optimize_catboost(
        self, X, y, problem_type: str, scoring: str, cv: int = 5
    ) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'
            scoring: Scoring metric
            cv: Number of CV folds

        Returns:
            Best hyperparameters
        """
        from catboost import CatBoostClassifier, CatBoostRegressor

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "random_seed": 42,
                "verbose": False,
            }

            if problem_type == "classification":
                model = CatBoostClassifier(**params)
            else:
                model = CatBoostRegressor(**params)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )

        return study.best_params

    def optimize_random_forest(
        self, X, y, problem_type: str, scoring: str, cv: int = 5
    ) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'
            scoring: Scoring metric
            cv: Number of CV folds

        Returns:
            Best hyperparameters
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": 42,
                "n_jobs": -1,
            }

            if problem_type == "classification":
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            objective,
            n_trials=min(self.n_trials, 30),  # RF is slower, use fewer trials
            timeout=self.timeout,
            show_progress_bar=False
        )

        return study.best_params
