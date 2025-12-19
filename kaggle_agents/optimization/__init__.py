"""
Prompt optimization infrastructure using DSPy and reward models.

Enhanced with:
- ExecutionFeedbackRewardModel: Uses structured logs for rich rewards
- AblationRewardModel: Evaluates ablation study quality
- ImprovementTrackingRewardModel: Tracks score progression across iterations
- PreferenceCollector: DPO-style preference learning from code fixes
- PreferenceRewardModel: Scores code based on learned preferences
"""

from .prompt_optimizer import (
    PromptOptimizer,
    TrainingDataCollector,
    create_optimizer,
    create_training_collector,
)
from .reward_model import (
    # Standard Reward Models
    CombinedRewardModel,
    DeveloperRewardModel,
    KaggleScoreRewardModel,
    PlannerRewardModel,
    ValidationRewardModel,
    # Enhanced Reward Models (NEW)
    AblationRewardModel,
    ExecutionFeedbackRewardModel,
    ImprovementTrackingRewardModel,
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
from .preference_learning import (
    # DPO-style Preference Learning
    PreferenceCollector,
    PreferenceRewardModel,
    # Factory Functions
    create_preference_collector,
    create_preference_reward_model,
)


__all__ = [
    # Enhanced Reward Models (NEW - recommended for RL optimization)
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
