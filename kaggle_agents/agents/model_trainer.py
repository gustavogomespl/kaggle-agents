"""Model training agent with multiple algorithms."""

import os
import sys

# Fix matplotlib backend for Google Colab compatibility
# Must be set before importing lightgbm
if 'MPLBACKEND' in os.environ:
    if os.environ['MPLBACKEND'] == 'module://matplotlib_inline.backend_inline':
        # Colab sets this inline backend which causes issues with some versions
        os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.config import Config
from ..utils.state import KaggleState
from ..utils.cross_validation import AdaptiveCrossValidator
from ..utils.hyperparameter_tuning import HyperparameterOptimizer


class ModelTrainingAgent:
    """Agent responsible for training and evaluating models with optimization."""

    def __init__(self):
        """Initialize model training agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )
        self.cv_adapter = AdaptiveCrossValidator()
        self.optimizer = HyperparameterOptimizer(n_trials=30, timeout=180)

    def determine_problem_type(self, y: pd.Series) -> str:
        """Determine if problem is classification or regression.

        Args:
            y: Target variable

        Returns:
            Problem type: 'classification' or 'regression'
        """
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            # Check if values are floating point (not integers)
            if pd.api.types.is_float_dtype(y):
                # If we have continuous values (not just .0), it's regression
                non_integer_mask = (y % 1) != 0
                if non_integer_mask.any():
                    return "regression"

            # Check number of unique values
            n_unique = y.nunique()

            # If very few unique values relative to size, likely classification
            if n_unique < 20:
                return "classification"

            # If many unique values, likely regression
            unique_ratio = n_unique / len(y)
            if unique_ratio > 0.05:  # More than 5% unique values
                return "regression"

        return "classification"

    def get_models(self, problem_type: str) -> Dict[str, Any]:
        """Get appropriate models for problem type.

        Args:
            problem_type: 'classification' or 'regression'

        Returns:
            Dictionary of model name to model instance
        """
        if problem_type == "classification":
            return {
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                "xgboost": XGBClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                "lightgbm": LGBMClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1, verbose=-1
                ),
                "logistic": LogisticRegression(random_state=42, max_iter=1000),
            }
        else:
            return {
                "random_forest": RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                "xgboost": XGBRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                "lightgbm": LGBMRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1, verbose=-1
                ),
                "ridge": Ridge(random_state=42),
            }

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str
    ) -> List[Dict[str, Any]]:
        """Train multiple models and evaluate with cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            List of model results
        """
        models = self.get_models(problem_type)
        results = []

        # Determine scoring metric
        scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"

        for name, model in models.items():
            print(f"  Training {name}...")
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=5, scoring=scoring, n_jobs=-1
                )

                # Train on full dataset
                model.fit(X, y)

                # Feature importance (if available)
                feature_importance = {}
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(
                        zip(X.columns, model.feature_importances_)
                    )

                results.append(
                    {
                        "name": name,
                        "model": model,
                        "cv_scores": cv_scores.tolist(),
                        "mean_cv_score": cv_scores.mean(),
                        "std_cv_score": cv_scores.std(),
                        "feature_importance": feature_importance,
                    }
                )

                print(f"    {name}: CV Score = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            except Exception as e:
                print(f"    Failed to train {name}: {str(e)}")
                continue

        return results

    def __call__(self, state: KaggleState) -> KaggleState:
        """Execute model training.

        Args:
            state: Current workflow state

        Returns:
            Updated state with trained models
        """
        print("🤖 Model Training Agent: Training models...")

        try:
            # Load processed data
            train_df = pd.read_csv(state["train_data_path"])

            # Identify target column
            potential_targets = ["target", "label", train_df.columns[-1]]
            target_col = None
            for col in potential_targets:
                if col in train_df.columns:
                    target_col = col
                    break

            if not target_col:
                raise ValueError("Could not identify target column")

            # Separate features and target
            X = train_df.drop(columns=[target_col])
            y = train_df[target_col]

            # Determine problem type
            problem_type = self.determine_problem_type(y)
            print(f"  Detected problem type: {problem_type}")

            # Train models
            results = self.train_and_evaluate(X, y, problem_type)

            if not results:
                raise ValueError("No models were successfully trained")

            # Sort by CV score (higher is better for classification, less negative for regression)
            results.sort(key=lambda x: x["mean_cv_score"], reverse=True)
            best_result = results[0]

            # Save best model
            Path(Config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
            model_path = f"{Config.MODELS_DIR}/best_model_{state['competition_name']}.joblib"
            joblib.dump(best_result["model"], model_path)

            # Update state
            state["models_trained"] = [
                {
                    "name": r["name"],
                    "mean_cv_score": r["mean_cv_score"],
                    "std_cv_score": r["std_cv_score"],
                }
                for r in results
            ]
            state["best_model"] = {
                "name": best_result["name"],
                "mean_cv_score": best_result["mean_cv_score"],
                "path": model_path,
            }
            state["cv_scores"] = best_result["cv_scores"]
            state["feature_importance"] = best_result["feature_importance"]

            # Get LLM insights on model performance
            system_msg = SystemMessage(
                content="""You are a machine learning expert analyzing model performance.
                Provide insights and recommendations based on the results."""
            )

            human_msg = HumanMessage(
                content=f"""Model Training Results:

Best Model: {best_result['name']}
CV Score: {best_result['mean_cv_score']:.4f} (+/- {best_result['std_cv_score']:.4f})

All Models:
{chr(10).join(f"- {r['name']}: {r['mean_cv_score']:.4f} (+/- {r['std_cv_score']:.4f})" for r in results)}

Problem Type: {problem_type}
Metric: {state.get('metric', 'unknown')}

Provide analysis and next steps recommendations."""
            )

            response = self.llm.invoke([system_msg, human_msg])

            state["messages"].append(
                HumanMessage(
                    content=f"Model training completed. Best model: {best_result['name']} with CV score {best_result['mean_cv_score']:.4f}. Analysis: {response.content}"
                )
            )

            print(f"Model Training Agent: Best model is {best_result['name']} with CV score {best_result['mean_cv_score']:.4f}")

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            print(f"Model Training Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = state.get("errors", []) if isinstance(state, dict) else state.errors
            return {"errors": errors + [error_msg]}

        return state
