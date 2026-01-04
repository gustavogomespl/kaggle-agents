"""
Optuna hyperparameter tuning instruction builder.
"""


def build_optuna_tuning_instructions(n_trials: int = 5, timeout: int = 540) -> list[str]:
    """Build Optuna hyperparameter tuning instructions."""
    return [
        "\nHYPERPARAMETER OPTIMIZATION (OPTUNA) REQUIRED:",
        "  - MUST use 'optuna' library for hyperparameter search",
        f"  - Run AT MOST {n_trials} trials (n_trials={n_trials}) and timeout={timeout}s to prevent timeouts",
        "  - CRITICAL: Check if 'optuna-integration' is available with try/except:",
        "    try:",
        "        from optuna.integration import OptunaSearchCV",
        "    except ImportError:",
        "        # Use manual Optuna with study.optimize() instead",
        "  - If optuna-integration is missing, use manual Optuna tuning with study.optimize()",
        "  - Use 'TPESampler' for efficient sampling",
        "  - CRITICAL: Do NOT pass early_stopping_rounds to .fit(); use callbacks for LightGBM and XGBoost <2, or constructor params for XGBoost 2.0+",
        "  - Optimize for the competition metric (minimize RMSE/LogLoss or maximize AUC/Accuracy)",
        "  - Print the best parameters found",
        "  - Train final model with best parameters",
        "\n⚡ SPEED OPTIMIZATION (CRITICAL TO AVOID TIMEOUT):",
        "  - **SUBSAMPLE FOR TUNING**: If train dataset > 10,000 rows:",
        "    1. Create tuning subset with train_test_split",
        "    2. For CLASSIFICATION only: pass stratify=y when sampling (y discrete: y.nunique() < 20 or dtype category/object)",
        "    3. For REGRESSION (continuous y): DO NOT use stratify parameter",
        "    4. Run Optuna study on 25% sample (reduce to 15% if memory errors occur)",
        "    5. After finding best_params, retrain on FULL dataset",
        "  - **REDUCE ESTIMATORS DURING TUNING**:",
        "    - Inside objective(): Use n_estimators=150-200 (fast convergence)",
        "    - Final model: Use n_estimators=1000 and apply early stopping via callbacks or constructor (version-aware)",
        "  - **TIMEOUT BUDGET**: Set study.optimize(n_trials=5, timeout=600) for max 10 min tuning",
        "  - **MEMORY SAFETY (PREVENT OOM CRASHES)**:",
        "    - ALWAYS set n_jobs=1 in model __init__ (LGBMClassifier, XGBClassifier, etc.)",
        "    - ALWAYS set n_jobs=1 in cross_val_score (avoid nested parallelism → memory explosion)",
        "    - Add 'import gc; gc.collect()' inside objective() after computing score",
        "    - Delete model object explicitly: 'del model' before gc.collect()",
        "    - If memory errors persist, reduce train_size from 0.25 → 0.15 (15% of data)",
        "  - **ROBUST TRIALS**: Wrap objective logic in try/except; on exception log and return 0.0 so trials finish",
        "  - **NO-COMPLETION GUARD**: After study.optimize, if NO trials completed, fall back to safe default params instead of study.best_params",
    ]
