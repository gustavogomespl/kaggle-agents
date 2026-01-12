"""
Multi-Fidelity Hyperparameter Optimization with Hyperband/ASHA.

This module provides utilities for efficient HPO using early stopping
of unpromising trials, reducing compute waste significantly.

Key concepts:
- ASHA (Asynchronous Successive Halving Algorithm): Stops poorly-performing
  trials early based on intermediate results
- Hyperband: Extension of ASHA with adaptive resource allocation
- Multi-fidelity: Evaluates trials at different resource levels (epochs, iterations)

CRITICAL: For pruning to work, generated code MUST:
1. Call trial.report(score, step) at each iteration
2. Check trial.should_prune() and raise TrialPruned if True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import optuna


def create_study(
    direction: Literal["minimize", "maximize"] = "minimize",
    pruner_type: Literal["hyperband", "median", "none"] = "hyperband",
    n_startup_trials: int = 5,
    min_resource: int = 1,
    max_resource: int = 100,
    reduction_factor: int = 3,
    seed: int = 42,
) -> "optuna.Study":
    """
    Create an Optuna study with multi-fidelity pruning.

    Args:
        direction: Optimization direction ("minimize" or "maximize")
        pruner_type: Type of pruner to use:
            - "hyperband": Best for most cases, adaptive resource allocation
            - "median": Simpler, prunes below median performance
            - "none": No pruning (use for quick experiments)
        n_startup_trials: Trials before pruning starts (for median pruner)
        min_resource: Minimum resource level (e.g., epochs) for hyperband
        max_resource: Maximum resource level for hyperband
        reduction_factor: Reduction factor for successive halving (default 3)
        seed: Random seed for reproducibility

    Returns:
        Configured Optuna study

    Example:
        >>> study = create_study(direction="minimize", pruner_type="hyperband")
        >>> study.optimize(objective, n_trials=50, timeout=600)
    """
    import optuna

    # Select pruner
    if pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )
    elif pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=5,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Use TPE sampler with seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=seed)

    return optuna.create_study(
        direction=direction,
        pruner=pruner,
        sampler=sampler,
    )


def suggest_lgbm_params(
    trial: "optuna.Trial",
    max_depth_range: tuple[int, int] = (3, 12),
    n_estimators_range: tuple[int, int] = (100, 1000),
    learning_rate_range: tuple[float, float] = (0.01, 0.3),
) -> dict[str, Any]:
    """
    Suggest LightGBM hyperparameters for an Optuna trial.

    Args:
        trial: Optuna trial object
        max_depth_range: Range for max_depth
        n_estimators_range: Range for n_estimators
        learning_rate_range: Range for learning_rate (log scale)

    Returns:
        Dictionary of suggested hyperparameters
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", *n_estimators_range),
        "learning_rate": trial.suggest_float("learning_rate", *learning_rate_range, log=True),
        "max_depth": trial.suggest_int("max_depth", *max_depth_range),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def suggest_xgb_params(
    trial: "optuna.Trial",
    max_depth_range: tuple[int, int] = (3, 12),
    n_estimators_range: tuple[int, int] = (100, 1000),
    learning_rate_range: tuple[float, float] = (0.01, 0.3),
) -> dict[str, Any]:
    """
    Suggest XGBoost hyperparameters for an Optuna trial.

    Args:
        trial: Optuna trial object
        max_depth_range: Range for max_depth
        n_estimators_range: Range for n_estimators
        learning_rate_range: Range for learning_rate (log scale)

    Returns:
        Dictionary of suggested hyperparameters
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", *n_estimators_range),
        "learning_rate": trial.suggest_float("learning_rate", *learning_rate_range, log=True),
        "max_depth": trial.suggest_int("max_depth", *max_depth_range),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }


def create_lgbm_pruning_callback(trial: "optuna.Trial", metric: str = "valid_0"):
    """
    Create a LightGBM callback for Optuna pruning.

    Usage in objective function:
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            callbacks=[create_lgbm_pruning_callback(trial)],
        )

    Args:
        trial: Optuna trial object
        metric: Name of the validation set in callbacks

    Returns:
        LightGBM callback function
    """
    import optuna

    def callback(env):
        # Get the metric value from evaluation results
        # Format: [(dataset_name, metric_name, value, is_higher_better), ...]
        if env.evaluation_result_list:
            metric_value = env.evaluation_result_list[0][2]
            trial.report(metric_value, env.iteration)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return callback


def create_xgb_pruning_callback(trial: "optuna.Trial"):
    """
    Create an XGBoost callback for Optuna pruning.

    Usage in objective function:
        model = xgb.train(
            params,
            dtrain,
            evals=[(dvalid, "valid")],
            callbacks=[create_xgb_pruning_callback(trial)],
        )

    Args:
        trial: Optuna trial object

    Returns:
        XGBoost callback object
    """
    import optuna
    from optuna.integration import XGBoostPruningCallback

    return XGBoostPruningCallback(trial, observation_key="valid-logloss")


def validate_pruning_contract(code: str) -> tuple[bool, str]:
    """
    Validate that generated code follows the pruning contract.

    The contract requires:
    1. trial.report(score, step) called at each iteration
    2. trial.should_prune() checked and TrialPruned raised if True

    This validation is CONDITIONAL - only applies when a pruner is active.

    Args:
        code: Generated Python code string

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if Optuna is being used
    uses_optuna = any(pattern in code for pattern in [
        "import optuna",
        "from optuna",
        "optuna.create_study",
        "optuna.Study",
    ])

    if not uses_optuna:
        return True, ""  # No Optuna = no validation needed

    # Check if a pruner is being used (not NopPruner)
    pruner_patterns = [
        ("HyperbandPruner", "Hyperband"),
        ("MedianPruner", "Median"),
        ("SuccessiveHalvingPruner", "SuccessiveHalving"),
        ("ThresholdPruner", "Threshold"),
        ("PercentilePruner", "Percentile"),
    ]

    active_pruner = None
    for pattern, name in pruner_patterns:
        if pattern in code:
            active_pruner = name
            break

    if active_pruner is None:
        return True, ""  # No active pruner = no validation needed

    # Pruner is active - check contract
    has_report = "trial.report" in code
    has_prune_check = "should_prune" in code or "TrialPruned" in code

    if not has_report:
        return False, (
            f"Code uses {active_pruner}Pruner but does not call trial.report(). "
            "The pruner cannot work without intermediate score reporting. "
            "Add: trial.report(score, step) inside your training loop."
        )

    if not has_prune_check:
        return False, (
            f"Code uses {active_pruner}Pruner but does not check trial.should_prune(). "
            "Trials will never be pruned, wasting compute. "
            "Add: if trial.should_prune(): raise optuna.TrialPruned()"
        )

    return True, ""


# Constants for prompt instructions
HPO_MULTI_FIDELITY_INSTRUCTIONS = """
## HPO Multi-Fidelity (ASHA/Hyperband) - OPTIONAL BUT RECOMMENDED

### When to Use Multi-Fidelity HPO
- Large search spaces (>10 hyperparameters)
- Long training times (>30 seconds per trial)
- Limited compute budget

### Contract for Pruning to Work (MANDATORY if using Pruner)

Your code MUST follow this pattern for pruning to work:

```python
import optuna
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction='minimize',
    pruner=HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
    }

    # MANDATORY: Loop with report per step
    for step in range(100):
        score = train_and_eval(params, n_iterations=step+1)

        # MANDATORY: Report score at each step
        trial.report(score, step)

        # MANDATORY: Check for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score

study.optimize(objective, n_trials=50, timeout=600)
```

### For LightGBM (callback-based)
```python
def objective(trial):
    params = {...}

    # Callback to report to Optuna
    def optuna_callback(env):
        score = env.evaluation_result_list[0][2]
        trial.report(score, env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        callbacks=[optuna_callback],  # MANDATORY
    )
    return model.best_score['valid']['logloss']
```

### For XGBoost (use integration)
```python
from optuna.integration import XGBoostPruningCallback

def objective(trial):
    pruning_callback = XGBoostPruningCallback(trial, observation_key="validation-logloss")
    model = xgb.train(
        params,
        dtrain,
        evals=[(dvalid, "validation")],
        callbacks=[pruning_callback],
    )
    return model.best_score
```
"""
