"""
Core modules for the unified Kaggle agents architecture.

This package contains the fundamental components that power
the autonomous Kaggle competition solving workflow.
"""

from .state import (
    KaggleState,
    CompetitionInfo,
    SOTASolution,
    AblationComponent,
    DevelopmentResult,
    ValidationResult,
    SubmissionResult,
    IterationMemory,
    DomainType,
    create_initial_state,
    merge_dict,
)

from .config import (
    AgentConfig,
    LLMConfig,
    SearchConfig,
    AblationConfig,
    ValidationConfig,
    DSPyConfig,
    IterationConfig,
    PathConfig,
    KaggleConfig,
    LoggingConfig,
    get_config,
    set_config,
    reset_config,
    get_competition_dir,
    get_model_save_path,
    get_submission_path,
)

from .orchestrator import (
    KaggleOrchestrator,
    WorkflowResults,
    solve_competition,
)

from .logger import (
    get_logger,
    setup_logging,
    LogContext,
    log_agent_start,
    log_agent_end,
    log_metric,
    log_error_with_context,
)

__all__ = [
    # State
    "KaggleState",
    "CompetitionInfo",
    "SOTASolution",
    "AblationComponent",
    "DevelopmentResult",
    "ValidationResult",
    "SubmissionResult",
    "IterationMemory",
    "DomainType",
    "create_initial_state",
    "merge_dict",
    # Config
    "AgentConfig",
    "LLMConfig",
    "SearchConfig",
    "AblationConfig",
    "ValidationConfig",
    "DSPyConfig",
    "IterationConfig",
    "PathConfig",
    "KaggleConfig",
    "LoggingConfig",
    "get_config",
    "set_config",
    "reset_config",
    "get_competition_dir",
    "get_model_save_path",
    "get_submission_path",
    # Orchestrator
    "KaggleOrchestrator",
    "WorkflowResults",
    "solve_competition",
    # Logging
    "get_logger",
    "setup_logging",
    "LogContext",
    "log_agent_start",
    "log_agent_end",
    "log_metric",
    "log_error_with_context",
]
