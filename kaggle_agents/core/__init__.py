"""
Core modules for the unified Kaggle agents architecture.

This package contains the fundamental components that power
the autonomous Kaggle competition solving workflow.
"""

from .config import (
    AblationConfig,
    AgentConfig,
    DSPyConfig,
    IterationConfig,
    KaggleConfig,
    LLMConfig,
    LoggingConfig,
    PathConfig,
    SearchConfig,
    ValidationConfig,
    get_competition_dir,
    get_config,
    get_model_save_path,
    get_submission_path,
    reset_config,
    set_config,
)
from .logger import (
    LogContext,
    get_logger,
    log_agent_end,
    log_agent_start,
    log_error_with_context,
    log_metric,
    setup_logging,
)
from .orchestrator import (
    KaggleOrchestrator,
    WorkflowResults,
    solve_competition,
)
from .state import (
    AblationComponent,
    CompetitionInfo,
    DataInsights,
    DevelopmentResult,
    DomainType,
    ErrorPatternMemory,
    HyperparameterHistory,
    IterationMemory,
    KaggleState,
    ModelPerformanceRecord,
    SOTASolution,
    SubmissionResult,
    ValidationResult,
    aggregate_feature_importance,
    create_initial_state,
    get_best_hyperparameters,
    get_memory_summary_for_planning,
    merge_dict,
    update_error_memory,
    update_hyperparameter_history,
    update_model_performance,
    update_strategy_effectiveness,
)


__all__ = [
    "AblationComponent",
    "AblationConfig",
    # Config
    "AgentConfig",
    "CompetitionInfo",
    "DSPyConfig",
    "DataInsights",
    "DevelopmentResult",
    "DomainType",
    "ErrorPatternMemory",
    "HyperparameterHistory",
    "IterationConfig",
    "IterationMemory",
    "KaggleConfig",
    # Orchestrator
    "KaggleOrchestrator",
    # State
    "KaggleState",
    "LLMConfig",
    "LogContext",
    "LoggingConfig",
    "ModelPerformanceRecord",
    "PathConfig",
    "SOTASolution",
    "SearchConfig",
    "SubmissionResult",
    "ValidationConfig",
    "ValidationResult",
    "WorkflowResults",
    # State helper functions
    "aggregate_feature_importance",
    "create_initial_state",
    "get_best_hyperparameters",
    "get_competition_dir",
    "get_config",
    "get_memory_summary_for_planning",
    # Logging
    "get_logger",
    "get_model_save_path",
    "get_submission_path",
    "log_agent_end",
    "log_agent_start",
    "log_error_with_context",
    "log_metric",
    "merge_dict",
    "reset_config",
    "set_config",
    "setup_logging",
    "solve_competition",
    "update_error_memory",
    "update_hyperparameter_history",
    "update_model_performance",
    "update_strategy_effectiveness",
]
