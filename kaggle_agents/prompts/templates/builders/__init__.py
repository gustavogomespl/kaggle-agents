"""
Instruction builders for Developer Agent prompts.

These functions build dynamic, context-aware instruction blocks
for different aspects of model training and evaluation.
"""

from .budget import (
    build_budget_instructions,
    build_mlebench_objective_instructions,
)
from .context import (
    DynamicContext,
    build_context,
)
from .cv import (
    build_calibration_instructions,
    build_cv_instructions,
    build_oof_hygiene_instructions,
    build_stacking_oof_instructions,
)
from .ensemble import build_ensemble_instructions
from .feature_eng import build_feature_engineering_instructions
from .image_model import build_image_model_instructions
from .model import (
    build_dynamic_instructions,
    build_iteration_context,
    build_model_component_instructions,
    build_performance_gap_instructions,
    build_previous_results_context,
    build_standard_requirements,
)
from .optuna import build_optuna_tuning_instructions


__all__ = [
    # Budget
    "build_budget_instructions",
    "build_mlebench_objective_instructions",
    # CV
    "build_cv_instructions",
    "build_calibration_instructions",
    "build_oof_hygiene_instructions",
    "build_stacking_oof_instructions",
    # Optuna
    "build_optuna_tuning_instructions",
    # Feature Engineering
    "build_feature_engineering_instructions",
    # Ensemble
    "build_ensemble_instructions",
    # Model
    "build_iteration_context",
    "build_model_component_instructions",
    "build_performance_gap_instructions",
    "build_previous_results_context",
    "build_standard_requirements",
    "build_dynamic_instructions",
    # Image Model
    "build_image_model_instructions",
    # Context
    "DynamicContext",
    "build_context",
]
