"""
Unified State Management for Kaggle Agents.

COMPATIBILITY LAYER: This file re-exports from the new modular structure.
All types and functions are now defined in kaggle_agents.core.state/

For new code, prefer importing from the submodules directly:
    from kaggle_agents.core.state import KaggleState
    from kaggle_agents.core.state.memory import ModelPerformanceRecord
"""

# Re-export everything from the new modular structure
from .state import (
    # Competition
    AblationComponent,
    # Learning
    CandidatePlan,
    # Results
    CodeAttempt,
    CompetitionInfo,
    # Memory
    DataInsights,
    DevelopmentResult,
    # Types
    DomainType,
    ErrorPatternMemory,
    HyperparameterHistory,
    IterationMemory,
    # Main state
    KaggleState,
    ModelPerformanceRecord,
    PreferencePair,
    ReasoningTrace,
    SelfEvaluation,
    SOTASolution,
    SubmissionFormatType,
    SubmissionResult,
    SubTask,
    ValidationResult,
    # Memory managers
    aggregate_feature_importance,
    create_initial_state,
    get_best_hyperparameters,
    get_memory_summary_for_planning,
    merge_competition_info,
    merge_dict,
    merge_error_pattern_memory,
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
