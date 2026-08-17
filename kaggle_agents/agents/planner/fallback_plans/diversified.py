"""
Diversified fallback plans focused on a specific improvement axis.

Used by the planner's plan-refinement loop to force diversity when
consecutive plans hash to the same components.
"""

from __future__ import annotations

from typing import Any


def create_diversified_fallback_plan(
    state: dict[str, Any],
    sota_analysis: dict[str, Any],
    focus: str,
) -> list[dict[str, Any]]:
    """
    Create a diversified fallback plan with a specific focus.

    Args:
        state: Current state
        sota_analysis: SOTA analysis
        focus: Focus area ('deep_learning', 'feature_engineering', 'ensemble')

    Returns:
        Diversified plan as list of dicts
    """
    if focus == "deep_learning":
        return [
            {
                "name": "nn_tabular",
                "component_type": "model",
                "description": "Neural network for tabular data (TabNet or MLP)",
                "estimated_impact": 0.18,
                "rationale": "Deep learning alternative to tree models",
                "code_outline": "TabNet/MLP with entity embeddings, batch norm, dropout",
            },
            {
                "name": "gradient_blend",
                "component_type": "ensemble",
                "description": "Gradient-based blending of NN and tree models",
                "estimated_impact": 0.12,
                "rationale": "Combine NN and tree strengths",
                "code_outline": "Weighted average with learned weights via gradient descent",
            },
        ]
    if focus == "feature_engineering":
        return [
            {
                "name": "target_encoding_cv",
                "component_type": "feature_engineering",
                "description": "Target encoding with proper CV to avoid leakage",
                "estimated_impact": 0.15,
                "rationale": "Powerful encoding for categorical features",
                "code_outline": "Fit category_encoders.TargetEncoder inside each injected canonical training fold and transform only its held-out fold",
            },
            {
                "name": "feature_selection",
                "component_type": "feature_engineering",
                "description": "Feature selection using importance + RFE",
                "estimated_impact": 0.10,
                "rationale": "Remove noise features",
                "code_outline": "RFECV or SelectFromModel with LightGBM importances",
            },
            {
                "name": "lightgbm_tuned",
                "component_type": "model",
                "description": "LightGBM with Optuna hyperparameter tuning",
                "estimated_impact": 0.20,
                "rationale": "Better hyperparameters",
                "code_outline": "Optuna study with n_trials=50 for LGBM params",
            },
        ]
    # ensemble focus
    return [
        {
            "name": "stacking_meta",
            "component_type": "ensemble",
            "description": "Stacking ensemble with ridge meta-learner",
            "estimated_impact": 0.15,
            "rationale": "Combine diverse model predictions",
            "code_outline": "StackingClassifier/Regressor with Ridge meta",
        },
        {
            "name": "voting_diverse",
            "component_type": "ensemble",
            "description": "Voting ensemble with diverse base models",
            "estimated_impact": 0.10,
            "rationale": "Simple but effective ensemble",
            "code_outline": "VotingClassifier with LGBM, XGB, CatBoost",
        },
    ]
