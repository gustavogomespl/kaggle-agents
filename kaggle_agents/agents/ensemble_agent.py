"""Ensemble agent for model stacking and blending."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge, LogisticRegression
from ..utils.config import Config
from ..utils.state import KaggleState


class EnsembleAgent:
    """Agent responsible for creating model ensembles."""

    def __init__(self):
        """Initialize ensemble agent."""
        pass

    def create_stacking_ensemble(
        self,
        models: List[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> Any:
        """Create stacking ensemble from best models.

        Args:
            models: List of trained models
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Trained meta-model
        """
        # Generate out-of-fold predictions from base models
        print(f"  Creating stacking ensemble with {len(models)} base models...")

        meta_features = []
        for i, model in enumerate(models):
            print(f"    Generating meta-features from model {i+1}/{len(models)}")
            if problem_type == "classification":
                oof_preds = cross_val_predict(
                    model, X, y, cv=5, method="predict_proba", n_jobs=-1
                )
                # Take probabilities for positive class
                if oof_preds.ndim > 1:
                    meta_features.append(oof_preds[:, 1])
                else:
                    meta_features.append(oof_preds)
            else:
                oof_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                meta_features.append(oof_preds)

        # Create meta-feature matrix
        meta_X = np.column_stack(meta_features)

        # Train meta-model
        if problem_type == "classification":
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            meta_model = Ridge(random_state=42)

        meta_model.fit(meta_X, y)

        # Retrain base models on full data
        for model in models:
            model.fit(X, y)

        return {"meta_model": meta_model, "base_models": models}

    def create_blending_ensemble(
        self,
        models: List[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> Dict[str, Any]:
        """Create blending ensemble using simple averaging.

        Args:
            models: List of trained models
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Dictionary with base models and weights
        """
        print(f"  Creating blending ensemble with {len(models)} models...")

        # Simple averaging - could be improved with weighted average
        # based on individual model performance
        weights = [1.0 / len(models)] * len(models)

        return {"base_models": models, "weights": weights}

    def predict_stacking(
        self, ensemble: Dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using stacking ensemble.

        Args:
            ensemble: Ensemble dictionary with meta_model and base_models
            X: Feature matrix
            problem_type: 'classification' or 'regression'

        Returns:
            Predictions
        """
        base_models = ensemble["base_models"]
        meta_model = ensemble["meta_model"]

        # Generate predictions from base models
        meta_features = []
        for model in base_models:
            if problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        meta_features.append(preds[:, 1])
                    else:
                        meta_features.append(preds)
                else:
                    meta_features.append(model.predict(X))
            else:
                meta_features.append(model.predict(X))

        # Create meta-features
        meta_X = np.column_stack(meta_features)

        # Predict with meta-model
        if problem_type == "classification" and hasattr(meta_model, "predict_proba"):
            return meta_model.predict_proba(meta_X)[:, 1]
        else:
            return meta_model.predict(meta_X)

    def predict_blending(
        self, ensemble: Dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using blending ensemble.

        Args:
            ensemble: Ensemble dictionary with base_models and weights
            X: Feature matrix
            problem_type: 'classification' or 'regression'

        Returns:
            Predictions
        """
        base_models = ensemble["base_models"]
        weights = ensemble["weights"]

        predictions = []
        for model in base_models:
            if problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        predictions.append(preds[:, 1])
                    else:
                        predictions.append(preds)
                else:
                    predictions.append(model.predict(X))
            else:
                predictions.append(model.predict(X))

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred

    def __call__(self, state: KaggleState) -> KaggleState:
        """Create ensemble from trained models.

        Args:
            state: Current workflow state

        Returns:
            Updated state with ensemble model
        """
        print("Ensemble Agent: Creating model ensemble...")

        try:
            # Handle both dict and dataclass state access
            models_trained = state.get("models_trained", []) if isinstance(state, dict) else state.models_trained
            train_data_path = state.get("train_data_path", "") if isinstance(state, dict) else state.train_data_path
            competition_name = state.get("competition_name", "") if isinstance(state, dict) else state.competition_name
            eda_summary = state.get("eda_summary", {}) if isinstance(state, dict) else state.eda_summary
            best_model = state.get("best_model", {}) if isinstance(state, dict) else state.best_model

            # Check if we have multiple models
            if len(models_trained) < 2:
                print("  Only one model trained, skipping ensemble")
                return state

            # Load processed data
            train_df = pd.read_csv(train_data_path)

            # Identify target
            potential_targets = ["target", "label", train_df.columns[-1]]
            target_col = None
            for col in potential_targets:
                if col in train_df.columns:
                    target_col = col
                    break

            if not target_col:
                print("  Could not identify target column, skipping ensemble")
                return state

            # Separate features and target
            X = train_df.drop(columns=[target_col])
            y = train_df[target_col]

            # Determine problem type
            problem_type = "classification" if y.nunique() < 20 else "regression"

            # Load top models (top 3)
            top_models = []
            for model_info in sorted(models_trained, key=lambda x: x["mean_cv_score"], reverse=True)[:3]:
                model_path = f"{Config.MODELS_DIR}/{model_info['name']}_{competition_name}.joblib"
                if Path(model_path).exists():
                    model = joblib.load(model_path)
                    top_models.append(model)

            if len(top_models) < 2:
                print("  Not enough trained models for ensemble")
                return state

            # Get ensemble strategy from state
            strategy = eda_summary.get("strategy", {})
            ensemble_strategy = strategy.get("ensemble_strategy", "stacking")

            # Create ensemble
            if "stack" in ensemble_strategy.lower():
                ensemble = self.create_stacking_ensemble(top_models, X, y, problem_type)
            else:
                ensemble = self.create_blending_ensemble(top_models, X, y, problem_type)

            # Save ensemble
            ensemble_path = f"{Config.MODELS_DIR}/ensemble_{competition_name}.joblib"
            joblib.dump({
                "ensemble": ensemble,
                "problem_type": problem_type,
                "strategy": ensemble_strategy
            }, ensemble_path)

            print(f"Ensemble Agent: Created {ensemble_strategy} ensemble with {len(top_models)} models")

            return {
                "best_model": {
                    "name": f"ensemble_{ensemble_strategy}",
                    "path": ensemble_path,
                    "mean_cv_score": best_model.get("mean_cv_score", 0.0) if best_model else 0.0,  # Use best individual score
                }
            }

        except Exception as e:
            error_msg = f"Ensemble creation failed: {str(e)}"
            print(f"Ensemble Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = state.get("errors", []) if isinstance(state, dict) else state.errors
            return {"errors": errors + [error_msg]}
