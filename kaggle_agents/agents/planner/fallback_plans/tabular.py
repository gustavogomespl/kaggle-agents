"""
Tabular competition fallback plan.

Uses tree-based models (LightGBM, XGBoost, CatBoost) with ensemble.

In fast mode (MLE-bench / tight budgets) the plan ROTATES model combinations
across stagnation iterations so repeated fallbacks explore different model
families instead of regenerating the same components (anti-stagnation).
"""

from typing import Any


def extract_sota_recommendations(sota_guidance: Any) -> list[str]:
    """
    Extract model/technique recommendations from SOTA guidance (dict or text).

    Returns categories like ["catboost", "feature_engineering"] used to steer
    the stagnation rotation toward what the SOTA search actually found.
    """
    if not sota_guidance:
        return []

    recommendations = []
    sota_str = str(sota_guidance).lower()

    if "catboost" in sota_str:
        recommendations.append("catboost")
    if "lightgbm" in sota_str or "lgbm" in sota_str:
        recommendations.append("lightgbm")
    if "xgboost" in sota_str or "xgb" in sota_str:
        recommendations.append("xgboost")
    if "neural" in sota_str or "tabnet" in sota_str or "mlp" in sota_str:
        recommendations.append("neural_network")
    if "random forest" in sota_str or "randomforest" in sota_str:
        recommendations.append("random_forest")
    if any(k in sota_str for k in ("tabfm", "tabpfn", "tabicl", "foundation model")):
        recommendations.append("tabular_foundation")
    if "feature" in sota_str or "encoding" in sota_str:
        recommendations.append("feature_engineering")
    if "optuna" in sota_str or "hyperparameter" in sota_str or "tuning" in sota_str:
        recommendations.append("hyperparameter_tuning")
    if "ensemble" in sota_str or "stacking" in sota_str or "blend" in sota_str:
        recommendations.append("ensemble")

    return recommendations


# Model rotation sets: each stagnation iteration tries a different combination
# (rotation 0 == the historical fast plan, so iteration 0 is behavior-preserving)
_MODEL_ROTATIONS: list[list[str]] = [
    # Iteration 0: Default (LGBM + XGB)
    ["lightgbm_fast_cv", "xgboost_fast_cv"],
    # Iteration 1: CatBoost + LGBM tuned
    ["catboost_fast_cv", "lightgbm_tuned_cv"],
    # Iteration 2: Zero-shot foundation model + Random Forest (radically
    # different model families - strongest plateau breaker)
    ["tabfm_zero_shot", "random_forest_fast"],
    # Iteration 3: Feature Engineering + CatBoost (FE can shift the score)
    ["target_encoding_fe", "catboost_fast_cv"],
    # Iteration 4: Intensive training - more folds/iterations
    ["lightgbm_intensive", "catboost_fast_cv"],
]

_FAST_COMPONENT_DEFS: dict[str, dict[str, Any]] = {
    "lightgbm_fast_cv": {
        "name": "lightgbm_fast_cv",
        "component_type": "model",
        "description": "LightGBM baseline tuned for speed (no Optuna). Use fewer estimators + early stopping/callbacks. Respect KAGGLE_AGENTS_CV_FOLDS for faster iteration.",
        "estimated_impact": 0.18,
        "rationale": "High ROI baseline for tabular tasks; fast enough to iterate under tight time budgets (MLE-bench).",
        "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. If classification: LGBMClassifier with predict_proba for probabilities [0,1]. If regression: LGBMRegressor. Use the injected canonical folds exactly, budget-aware capacity, and save aligned OOF/test predictions.",
    },
    "xgboost_fast_cv": {
        "name": "xgboost_fast_cv",
        "component_type": "model",
        "description": "XGBoost baseline tuned for speed (no Optuna). Use hist/gpu_hist where available. Respect time budget and fold count env vars.",
        "estimated_impact": 0.16,
        "rationale": "Provides diversity vs LightGBM with similar compute budget; useful for a quick ensemble.",
        "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. If classification: XGBClassifier with predict_proba for probabilities [0,1]. If regression: XGBRegressor. Use the injected canonical folds exactly, budget-aware capacity, and save aligned OOF/test predictions.",
    },
    "catboost_fast_cv": {
        "name": "catboost_fast_cv",
        "component_type": "model",
        "description": "CatBoost baseline tuned for speed (no Optuna). Handles categoricals natively.",
        "estimated_impact": 0.17,
        "rationale": "Different regularization than XGBoost/LightGBM; handles categoricals well.",
        "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. If classification: CatBoostClassifier with predict_proba. If regression: CatBoostRegressor. Use the injected canonical folds exactly and save aligned OOF/test predictions.",
    },
    "lightgbm_tuned_cv": {
        "name": "lightgbm_tuned_cv",
        "component_type": "model",
        "description": "LightGBM with light Optuna tuning (5 trials). Better than defaults, still fast.",
        "estimated_impact": 0.19,
        "rationale": "A bounded search is justified only when it improves identical validation folds within budget.",
        "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Run a bounded LightGBM search on the injected canonical folds, tune only within the measured runtime budget, and save aligned OOF/test predictions.",
    },
    "tabfm_zero_shot": {
        "name": "tabfm_zero_shot",
        "component_type": "model",
        "description": "Optional TabFM zero-shot tabular foundation model: in-context prediction with no tuning or feature engineering. Use only when the real TabFM implementation is available and compatible.",
        "estimated_impact": 0.18,
        "rationale": "A genuinely different model family may add ensemble diversity, but its contribution must never be attributed to a substitute estimator.",
        "code_outline": "Follow TABULAR constraints section 9 (TabFM). Check compatibility before training. If the real TabFM package/model is unavailable, incompatible, or raises, fail this component explicitly with RuntimeError and write no TabFM-named artifacts; NEVER substitute LightGBM or another estimator under the tabfm_zero_shot name. On genuine TabFM success, iterate canonical folds, predict every validation/test row, and only then save models/oof_tabfm_zero_shot.npy and models/test_tabfm_zero_shot.npy.",
    },
    "neural_network_mlp": {
        "name": "neural_network_mlp",
        "component_type": "model",
        "description": "MLP Neural Network with MANDATORY StandardScaler. Different pattern capture than trees.",
        "estimated_impact": 0.14,
        "rationale": "Neural Networks capture non-linear patterns differently, adds diversity.",
        "code_outline": "Fit StandardScaler inside each fold for scale-sensitive neural models; transform validation/test with that fold's scaler. Use an MLPClassifier or MLPRegressor selected from the inferred target type, then save aligned OOF/test predictions.",
    },
    "random_forest_fast": {
        "name": "random_forest_fast",
        "component_type": "model",
        "description": "Random Forest baseline with limited trees (n_estimators=200) for speed.",
        "estimated_impact": 0.13,
        "rationale": "Robust tree ensemble that rarely fails; good fallback option.",
        "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Fit a budget-sized RandomForestClassifier/Regressor on every injected canonical fold and save aligned OOF/test predictions.",
    },
    "target_encoding_fe": {
        "name": "target_encoding_fe",
        "component_type": "feature_engineering",
        "description": "Target encoding with K-fold CV to prevent leakage. Creates powerful features from categoricals.",
        "estimated_impact": 0.16,
        "rationale": "Target encoding is useful only when leakage-safe folds show signal beyond ordinary categorical encodings.",
        "code_outline": "category_encoders.TargetEncoder(cols=categorical_cols, smoothing=0.3) fitted WITHIN CV folds (fit on train fold, transform val fold) to prevent leakage; fit on full train for test. Save encoded matrices to MODELS_DIR for downstream models.",
    },
    "lightgbm_intensive": {
        "name": "lightgbm_intensive",
        "component_type": "model",
        "description": "LightGBM with more capacity on the same canonical folds when measured runtime permits it.",
        "estimated_impact": 0.20,
        "rationale": "Additional folds can reduce estimate variance when the measured runtime budget permits them.",
        "code_outline": "Timeout-aware intensive training: read KAGGLE_AGENTS_COMPONENT_TIMEOUT_S, preserve every injected canonical fold, estimate feasible boosting capacity from a pilot, check remaining time before each fold, reserve time for artifacts, and save per-fold OOF/test checkpoints.",
    },
    "simple_ridge_baseline": {
        "name": "simple_ridge_baseline",
        "component_type": "model",
        "description": "Simple Ridge baseline with StandardScaler. Cannot fail, always produces predictions.",
        "estimated_impact": 0.08,
        "rationale": "Failsafe baseline that always works.",
        "code_outline": "Fit StandardScaler + Ridge (or LogisticRegression for classification) inside every injected canonical fold and save aligned OOF/test predictions.",
    },
    "stacking_ensemble": {
        "name": "stacking_ensemble",
        "component_type": "ensemble",
        "description": "Stack OOF predictions from available models with LogisticRegression/Ridge meta-learner. Fallback to weighted average if needed.",
        "estimated_impact": 0.10,
        "rationale": "Cheap ensemble step that often improves generalization without additional heavy training.",
        "code_outline": "Load models/oof_*.npy + models/test_*.npy. Check IS_CLASSIFICATION from canonical_metadata. If classification: use LogisticRegression as meta-model, evaluate with AUC, clip predictions to [0,1]. If regression: use Ridge as meta-model, evaluate with RMSE. Write submission.csv.",
    },
}


def _create_fast_rotation_plan(
    sota_analysis: dict[str, Any],
    state: dict[str, Any] | None,
    stagnation_iteration: int,
) -> list[dict[str, Any]]:
    """Fast-mode plan with stagnation-aware model rotation (2 models + ensemble)."""
    failed_names: set[str] = set()
    sota_guidance: Any = None
    if state:
        failed_names = set(state.get("failed_component_names", []) or [])
        if failed_names:
            print(f"   Filtering out previously failed components: {failed_names}")
        refinement_guidance = state.get("refinement_guidance", {}) or {}
        sota_guidance = refinement_guidance.get("sota_guidance")

    sota_recommended = extract_sota_recommendations(sota_guidance or sota_analysis)
    if sota_recommended:
        print(f"   SOTA recommends: {sota_recommended}")

    print(f"   Stagnation iteration: {stagnation_iteration}")
    model_rotations = [list(rotation) for rotation in _MODEL_ROTATIONS]
    rotation_slot = stagnation_iteration % len(model_rotations)

    # SOTA-guided rotation override: prioritize what the SOTA search found.
    # First match wins (specific model recommendations beat generic FE advice).
    # Each override fires at ONE specific iteration only - real SOTA text almost
    # always contains 'feature'/model names, and overriding every slot would pin
    # the rotation to a single pair, defeating the anti-stagnation diversity.
    if "catboost" in sota_recommended and stagnation_iteration == 1:
        print("   SOTA override: prioritizing CatBoost rotation")
        model_rotations[rotation_slot] = ["catboost_fast_cv", "lightgbm_tuned_cv"]
    elif "tabular_foundation" in sota_recommended and stagnation_iteration == 1:
        print("   SOTA override: prioritizing TabFM (foundation model) rotation")
        model_rotations[rotation_slot] = ["tabfm_zero_shot", "lightgbm_fast_cv"]
    elif "neural_network" in sota_recommended and stagnation_iteration == 2:
        print("   SOTA override: prioritizing Neural Network rotation")
        model_rotations[rotation_slot] = ["neural_network_mlp", "catboost_fast_cv"]
    elif "feature_engineering" in sota_recommended and stagnation_iteration == 1:
        print("   SOTA override: prioritizing Feature Engineering rotation")
        model_rotations[rotation_slot] = ["target_encoding_fe", "lightgbm_fast_cv"]

    selected_names = model_rotations[rotation_slot]
    print(f"   Selected rotation {rotation_slot}: {selected_names}")

    # Build plan from selected components, filtering out failed ones.
    # Copies: the defs are module-level and must never be mutated by consumers
    # (one process runs many competitions back to back)
    components = [
        dict(_FAST_COMPONENT_DEFS[name])
        for name in selected_names
        if name not in failed_names and name in _FAST_COMPONENT_DEFS
    ]

    # If the rotation got filtered down, backfill with reliable models
    if len(components) < 2:
        fallback_order = [
            "catboost_fast_cv",
            "lightgbm_fast_cv",
            "xgboost_fast_cv",
            "random_forest_fast",
            "simple_ridge_baseline",
        ]
        picked = {c["name"] for c in components}
        for name in fallback_order:
            if name not in failed_names and name not in picked:
                components.append(dict(_FAST_COMPONENT_DEFS[name]))
                picked.add(name)
                if len(components) >= 2:
                    break

    if len(components) < 2:
        components.append(dict(_FAST_COMPONENT_DEFS["simple_ridge_baseline"]))

    final_plan = [*components[:2], dict(_FAST_COMPONENT_DEFS["stacking_ensemble"])]
    print(f"   Fast rotation plan: {[c['name'] for c in final_plan]}")
    return final_plan


def create_tabular_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    curriculum_insights: str = "",
    *,
    fast_mode: bool = False,
    state: dict[str, Any] | None = None,
    stagnation_iteration: int = 0,
) -> list[dict[str, Any]]:
    """
    Create fallback plan for tabular competitions (classification/regression).

    Uses tree-based models (LightGBM, XGBoost, CatBoost) with ensemble.

    Args:
        domain: Competition domain
        sota_analysis: SOTA analysis results
        curriculum_insights: Insights from previous iterations (optional)
        fast_mode: If True, return minimal rotation plan for speed (MLE-bench)
        state: Current workflow state (failed components + SOTA guidance)
        stagnation_iteration: How many times fallback was used (rotates models)

    Returns:
        List of component dictionaries
    """
    if fast_mode:
        return _create_fast_rotation_plan(sota_analysis, state, stagnation_iteration)

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
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Select LGBMClassifier or LGBMRegressor accordingly; run bounded tuning and early stopping on the injected canonical folds only, then save aligned OOF/test predictions.",
            },
            {
                "name": "xgboost_optuna_tuned",
                "component_type": "model",
                "description": "XGBoost with Optuna hyperparameter optimization: 15 trials, tuning max_depth, learning_rate, subsample, colsample_bytree",
                "estimated_impact": 0.20,
                "rationale": "XGBoost provides different regularization than LightGBM. Optuna ensures optimal capacity.",
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Select XGBClassifier or XGBRegressor accordingly; run bounded tuning and early stopping on the injected canonical folds only, then save aligned OOF/test predictions.",
            },
            {
                "name": "catboost_optuna_tuned",
                "component_type": "model",
                "description": "CatBoost with Optuna hyperparameter optimization: 15 trials, tuning depth, learning_rate, l2_leaf_reg",
                "estimated_impact": 0.19,
                "rationale": "CatBoost handles categorical features natively. Tuning depth is critical for performance.",
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Select CatBoostClassifier or CatBoostRegressor accordingly; derive the loss from the target contract and tune on the injected canonical folds only.",
            },
            {
                "name": "neural_network_mlp",
                "component_type": "model",
                "description": "Simple MLP Neural Network using Scikit-Learn or PyTorch (if available). Standard scaling is CRITICAL.",
                "estimated_impact": 0.15,
                "rationale": "Neural Networks capture different patterns than tree-based models, adding valuable diversity to the ensemble.",
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. If classification: use MLPClassifier with predict_proba for probabilities [0,1]. DO NOT use MLPRegressor (produces invalid predictions). If regression: use MLPRegressor. MUST use StandardScaler on inputs. Early stopping with validation_fraction=0.1.",
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
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Fit ExtraTreesClassifier or ExtraTreesRegressor on every injected canonical fold; size and tune the forest within budget and save aligned OOF/test predictions.",
            },
            {
                "name": "ridge_classifier_tuned",
                "component_type": "model",
                "description": "Ridge Classifier with StandardScaler and alpha tuning. Linear model for diversity.",
                "estimated_impact": 0.12,
                "rationale": "Linear models capture different patterns than trees. Fast to train, adds diversity.",
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Fit the scaler and RidgeClassifier/Ridge pipeline independently inside every injected canonical fold, then save aligned OOF/test predictions.",
            },
            {
                "name": "linearsvc_calibrated",
                "component_type": "model",
                "description": "Linear SVM with CalibratedClassifierCV for probability outputs. StandardScaler required.",
                "estimated_impact": 0.11,
                "rationale": "SVM with linear kernel captures linear boundaries. Calibration enables predict_proba for ensemble.",
                "code_outline": "For classification only, fit StandardScaler plus calibrated LinearSVC inside every injected canonical fold and save aligned probability OOF/test predictions; skip for regression.",
            },
            {
                "name": "gradient_boosting_sklearn",
                "component_type": "model",
                "description": "Scikit-learn GradientBoosting (different implementation from LightGBM/XGBoost).",
                "estimated_impact": 0.14,
                "rationale": "Sklearn GB has different regularization behavior, adds diversity to the ensemble.",
                "code_outline": "Check IS_CLASSIFICATION from canonical_metadata. Fit GradientBoostingClassifier or GradientBoostingRegressor on every injected canonical fold; derive capacity from the runtime budget and save aligned OOF/test predictions.",
            },
            {
                "name": "tabfm_zero_shot",
                "component_type": "model",
                "description": "Optional TabFM zero-shot tabular foundation model: use only when the real implementation is installed and compatible.",
                "estimated_impact": 0.17,
                "rationale": "Evaluate a distinct model family without conflating its score or artifacts with a tree-model fallback.",
                "code_outline": "Follow TABULAR constraints section 9 (TabFM). If the real TabFM implementation is unavailable, incompatible, or errors, raise RuntimeError and write no TabFM-named artifacts; NEVER substitute LightGBM or another model under this component name. Save tabfm_zero_shot OOF/test artifacts only after genuine TabFM inference succeeds on all canonical folds.",
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
            "rationale": "Cross-fitted stacking is an ensemble candidate whose benefit must be measured on aligned canonical OOF predictions.",
            "code_outline": "Load only approved, aligned OOF/test prediction pairs. Cross-fit a LogisticRegression meta-learner for classification or Ridge for regression using the injected canonical folds, generate meta-OOF without fitting on its validation rows, refit on all eligible OOF rows for test inference, and retain only after an independently recomputed gain.",
        }
    )

    return plan
