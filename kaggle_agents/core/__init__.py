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
    DevelopmentResult,
    DomainType,
    IterationMemory,
    KaggleState,
    SOTASolution,
    SubmissionResult,
    ValidationResult,
    create_initial_state,
    merge_dict,
)


__all__ = [
    "AblationComponent",
    "AblationConfig",
    # Config
    "AgentConfig",
    "CompetitionInfo",
    "DSPyConfig",
    "DevelopmentResult",
    "DomainType",
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
    "PathConfig",
    "SOTASolution",
    "SearchConfig",
    "SubmissionResult",
    "ValidationConfig",
    "ValidationResult",
    "WorkflowResults",
    "create_initial_state",
    "get_competition_dir",
    "get_config",
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
]
