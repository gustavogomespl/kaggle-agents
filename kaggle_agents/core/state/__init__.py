"""
Unified State Management for Kaggle Agents.

This module defines the central state structure that flows through
the LangGraph workflow, containing all data needed for autonomous
Kaggle competition solving.
"""

# Type definitions
from .types import DomainType, SubmissionFormatType

# Competition types
from .competition import (
    AblationComponent,
    CompetitionInfo,
    SOTASolution,
    merge_competition_info,
)

# Result types
from .results import (
    CodeAttempt,
    DevelopmentResult,
    SubmissionResult,
    ValidationResult,
)

# Memory types
from .memory import (
    DataInsights,
    ErrorPatternMemory,
    HyperparameterHistory,
    IterationMemory,
    ModelPerformanceRecord,
    merge_error_pattern_memory,
)

# Learning types (RL)
from .learning import (
    CandidatePlan,
    PreferencePair,
    ReasoningTrace,
    SelfEvaluation,
    SubTask,
)

# Main state
from .base import (
    KaggleState,
    create_initial_state,
    merge_dict,
)

# Memory managers
from .memory_managers import (
    aggregate_feature_importance,
    get_best_hyperparameters,
    get_memory_summary_for_planning,
    update_error_memory,
    update_hyperparameter_history,
    update_model_performance,
    update_strategy_effectiveness,
)

__all__ = [
    # Types
    "DomainType",
    "SubmissionFormatType",
    # Competition
    "AblationComponent",
    "CompetitionInfo",
    "SOTASolution",
    "merge_competition_info",
    # Results
    "CodeAttempt",
    "DevelopmentResult",
    "SubmissionResult",
    "ValidationResult",
    # Memory
    "DataInsights",
    "ErrorPatternMemory",
    "HyperparameterHistory",
    "IterationMemory",
    "ModelPerformanceRecord",
    "merge_error_pattern_memory",
    # Learning
    "CandidatePlan",
    "PreferencePair",
    "ReasoningTrace",
    "SelfEvaluation",
    "SubTask",
    # Main state
    "KaggleState",
    "create_initial_state",
    "merge_dict",
    # Memory managers
    "aggregate_feature_importance",
    "get_best_hyperparameters",
    "get_memory_summary_for_planning",
    "update_error_memory",
    "update_hyperparameter_history",
    "update_model_performance",
    "update_strategy_effectiveness",
]
