"""
Prompt optimization infrastructure using DSPy and reward models.

Enhanced with:
- ExecutionFeedbackRewardModel: Uses structured logs for rich rewards
- AblationRewardModel: Evaluates ablation study quality
- ImprovementTrackingRewardModel: Tracks score progression across iterations
- PreferenceCollector: DPO-style preference learning from code fixes
- PreferenceRewardModel: Scores code based on learned preferences
- HPO utilities: Multi-fidelity optimization with Hyperband/ASHA
"""

from .hpo import (
    HPO_MULTI_FIDELITY_INSTRUCTIONS,
    create_lgbm_pruning_callback,
    create_study,
    create_xgb_pruning_callback,
    suggest_lgbm_params,
    suggest_xgb_params,
    validate_pruning_contract,
)
from .preference_learning import (
    # DPO-style Preference Learning
    PreferenceCollector,
    PreferenceRewardModel,
    # Factory Functions
    create_preference_collector,
    create_preference_reward_model,
)
from .prompt_optimizer import (
    PromptOptimizer,
    TrainingDataCollector,
    create_optimizer,
    create_training_collector,
)
from .reward_model import (
    # Enhanced Reward Models (NEW)
    AblationRewardModel,
    # Standard Reward Models
    CombinedRewardModel,
    DeveloperRewardModel,
    ExecutionFeedbackRewardModel,
    ImprovementTrackingRewardModel,
    KaggleScoreRewardModel,
    PlannerRewardModel,
    ValidationRewardModel,
    # Factory Functions
    create_ablation_metric,
    create_combined_metric,
    create_developer_metric,
    create_execution_feedback_metric,
    create_improvement_tracker,
    create_kaggle_metric,
    create_planner_metric,
    create_validation_metric,
)


__all__ = [
    # HPO Multi-Fidelity (NEW - Hyperband/ASHA support)
    "HPO_MULTI_FIDELITY_INSTRUCTIONS",
    "create_lgbm_pruning_callback",
    "create_study",
    "create_xgb_pruning_callback",
    "suggest_lgbm_params",
    "suggest_xgb_params",
    "validate_pruning_contract",
    # Enhanced Reward Models (recommended for RL optimization)
    "AblationRewardModel",
    "ExecutionFeedbackRewardModel",
    "ImprovementTrackingRewardModel",
    # DPO-style Preference Learning
    "PreferenceCollector",
    "PreferenceRewardModel",
    # Standard Reward Models
    "CombinedRewardModel",
    "DeveloperRewardModel",
    "KaggleScoreRewardModel",
    "PlannerRewardModel",
    "ValidationRewardModel",
    # Optimizer
    "PromptOptimizer",
    "TrainingDataCollector",
    # Factory Functions
    "create_ablation_metric",
    "create_combined_metric",
    "create_developer_metric",
    "create_execution_feedback_metric",
    "create_improvement_tracker",
    "create_kaggle_metric",
    "create_optimizer",
    "create_planner_metric",
    "create_preference_collector",
    "create_preference_reward_model",
    "create_training_collector",
    "create_validation_metric",
]
