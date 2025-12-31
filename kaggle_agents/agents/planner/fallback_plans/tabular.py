"""
Tabular competition fallback plan.

Uses tree-based models (LightGBM, XGBoost, CatBoost) with ensemble.
"""

from typing import Any


def create_tabular_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    curriculum_insights: str = "",
    *,
    fast_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Create fallback plan for tabular competitions (classification/regression).

    Uses tree-based models (LightGBM, XGBoost, CatBoost) with ensemble.

    Args:
        domain: Competition domain
        sota_analysis: SOTA analysis results
        curriculum_insights: Insights from previous iterations (optional)
        fast_mode: If True, return minimal plan for speed

    Returns:
        List of component dictionaries (5 components: 1 FE + 4 models + 1 ensemble)
    """
    if fast_mode:
        return [
            {
                "name": "lightgbm_fast_cv",
                "component_type": "model",
                "description": "LightGBM baseline tuned for speed (no Optuna). Use fewer estimators + early stopping/callbacks. Respect KAGGLE_AGENTS_CV_FOLDS for faster iteration.",
                "estimated_impact": 0.18,
                "rationale": "High ROI baseline for tabular tasks; fast enough to iterate under tight time budgets (MLE-bench).",
                "code_outline": "LGBMClassifier/Regressor with sane defaults, 3-fold CV when FAST_MODE, save OOF/test preds",
            },
            {
                "name": "xgboost_fast_cv",
                "component_type": "model",
                "description": "XGBoost baseline tuned for speed (no Optuna). Use hist/gpu_hist where available. Respect time budget and fold count env vars.",
                "estimated_impact": 0.16,
                "rationale": "Provides diversity vs LightGBM with similar compute budget; useful for a quick ensemble.",
                "code_outline": "XGBClassifier/Regressor with fixed params, 3-fold CV when FAST_MODE, save OOF/test preds",
            },
            {
                "name": "stacking_ensemble",
                "component_type": "ensemble",
                "description": "Stack OOF predictions from LightGBM + XGBoost with LogisticRegression/Ridge meta-learner. Fallback to weighted average if needed.",
                "estimated_impact": 0.10,
                "rationale": "Cheap ensemble step that often improves generalization without additional heavy training.",
                "code_outline": "Load models/oof_*.npy + models/test_*.npy, fit meta-model on OOF, predict test, write submission",
            },
        ]

    plan = []

    # ALWAYS add feature engineering first (high impact)
    plan.append(
        {
            "name": "advanced_feature_engineering",
            "component_type": "feature_engineering",
            "description": "Create polynomial features (degree 2), feature interactions (ratio, diff, product), statistical transformations (log, sqrt), and target encoding for categorical features",
            "estimated_impact": 0.15,
            "rationale": "Comprehensive feature engineering improves scores by 10-20% in tabular competitions",
            "code_outline": "Use PolynomialFeatures(degree=2), create ratio/diff/product features, apply log/sqrt transforms, use TargetEncoder",
        }
    )

    # ALWAYS add 3 diverse models for ensemble diversity
    plan.extend(
        [
            {
                "name": "lightgbm_optuna_tuned",
                "component_type": "model",
                "description": "LightGBM with Optuna hyperparameter optimization: 15 trials, tuning learning_rate, num_leaves, max_depth, min_child_samples",
                "estimated_impact": 0.22,
                "rationale": "LightGBM consistently wins tabular competitions. Optuna finds better parameters than manual tuning.",
                "code_outline": "LGBMRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=100",
            },
            {
                "name": "xgboost_optuna_tuned",
                "component_type": "model",
                "description": "XGBoost with Optuna hyperparameter optimization: 15 trials, tuning max_depth, learning_rate, subsample, colsample_bytree",
                "estimated_impact": 0.20,
                "rationale": "XGBoost provides different regularization than LightGBM. Optuna ensures optimal capacity.",
                "code_outline": "XGBRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=50",
            },
            {
                "name": "catboost_optuna_tuned",
                "component_type": "model",
                "description": "CatBoost with Optuna hyperparameter optimization: 15 trials, tuning depth, learning_rate, l2_leaf_reg",
                "estimated_impact": 0.19,
                "rationale": "CatBoost handles categorical features natively. Tuning depth is critical for performance.",
                "code_outline": "CatBoostRegressor/Classifier with OptunaSearchCV, cat_features parameter, 5-fold CV",
            },
            {
                "name": "neural_network_mlp",
                "component_type": "model",
                "description": "Simple MLP Neural Network using Scikit-Learn or PyTorch (if available). Standard scaling is CRITICAL.",
                "estimated_impact": 0.15,
                "rationale": "Neural Networks capture different patterns than tree-based models, adding valuable diversity to the ensemble.",
                "code_outline": "MLPClassifier/Regressor or PyTorch simple net. Must use StandardScaler/MinMaxScaler on inputs. Early stopping.",
            },
        ]
    )

    # Add diverse models for better ensemble (different from tree-based GBMs)
    plan.extend(
        [
            {
                "name": "extratrees_tuned",
                "component_type": "model",
                "description": "ExtraTrees (Extremely Randomized Trees) with tuned n_estimators=500, max_depth tuned via simple grid.",
                "estimated_impact": 0.16,
                "rationale": "ExtraTrees uses random splits, decorrelated from GBMs. Great for ensemble diversity.",
                "code_outline": "ExtraTreesClassifier/Regressor with n_estimators=500, max_depth tuned, 5-fold CV, save OOF/test preds",
            },
            {
                "name": "ridge_classifier_tuned",
                "component_type": "model",
                "description": "Ridge Classifier with StandardScaler and alpha tuning. Linear model for diversity.",
                "estimated_impact": 0.12,
                "rationale": "Linear models capture different patterns than trees. Fast to train, adds diversity.",
                "code_outline": "Pipeline([StandardScaler, RidgeClassifier(alpha tuned)]), 5-fold CV, save OOF/test preds",
            },
            {
                "name": "linearsvc_calibrated",
                "component_type": "model",
                "description": "Linear SVM with CalibratedClassifierCV for probability outputs. StandardScaler required.",
                "estimated_impact": 0.11,
                "rationale": "SVM with linear kernel captures linear boundaries. Calibration enables predict_proba for ensemble.",
                "code_outline": "Pipeline([StandardScaler, CalibratedClassifierCV(LinearSVC())]), 5-fold CV, save OOF/test preds",
            },
            {
                "name": "gradient_boosting_sklearn",
                "component_type": "model",
                "description": "Scikit-learn GradientBoosting (different implementation from LightGBM/XGBoost).",
                "estimated_impact": 0.14,
                "rationale": "Sklearn GB has different regularization behavior, adds diversity to the ensemble.",
                "code_outline": "GradientBoostingClassifier with n_estimators=200, learning_rate=0.1, max_depth=5, 5-fold CV",
            },
        ]
    )

    # ALWAYS add stacking ensemble (combines all models above)
    plan.append(
        {
            "name": "stacking_ensemble",
            "component_type": "ensemble",
            "description": "Stack LightGBM, XGBoost, CatBoost, and NN predictions using Ridge/Logistic regression as meta-learner",
            "estimated_impact": 0.25,
            "rationale": "Stacking combines diverse models (Trees + NN) and typically improves scores by 5-10%",
            "code_outline": "StackingRegressor/Classifier with base_estimators=[lgb, xgb, cat, nn], final_estimator=Ridge/LogisticRegression, cv=5",
        }
    )

    return plan
