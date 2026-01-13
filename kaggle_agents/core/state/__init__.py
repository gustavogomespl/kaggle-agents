"""
Unified State Management for Kaggle Agents.

This module defines the central state structure that flows through
the LangGraph workflow, containing all data needed for autonomous
Kaggle competition solving.
"""

# Type definitions
# Main state
from .base import (
    KaggleState,
    create_initial_state,
    merge_dict,
)

# Contracts (Source of Truth) - PR1
from .contracts import (
    CanonicalDataContract,
    DataUsageContract,
    EvalFidelityContract,
    MetricContract,
    SubmissionContract,
    create_metric_contract,
    create_submission_contract_from_sample,
)

# MLE-STAR Registries - PR2
from .ablation import AblationExecution, AblationHistory
from .checks import CheckResult, RobustnessChecks
from .model_registry import ModelRegistry, RegisteredModel
from .registry import CodeBlock, CodeBlockRegistry

# Artifact Index - PR3
from .artifacts import ArtifactIndex, ArtifactRef

# Competition types
from .competition import (
    AblationComponent,
    CompetitionInfo,
    SOTASolution,
    merge_competition_info,
)

# Learning types (RL)
from .learning import (
    CandidatePlan,
    PreferencePair,
    ReasoningTrace,
    SelfEvaluation,
    SubTask,
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

# Result types
from .results import (
    CodeAttempt,
    DevelopmentResult,
    SubmissionResult,
    ValidationResult,
)
from .types import DomainType, SubmissionFormatType


__all__ = [
    # Types
    "DomainType",
    "SubmissionFormatType",
    # Contracts (Source of Truth) - PR1
    "MetricContract",
    "CanonicalDataContract",
    "SubmissionContract",
    "EvalFidelityContract",
    "DataUsageContract",
    "create_metric_contract",
    "create_submission_contract_from_sample",
    # MLE-STAR Registries - PR2
    "CodeBlock",
    "CodeBlockRegistry",
    "AblationExecution",
    "AblationHistory",
    "RegisteredModel",
    "ModelRegistry",
    "CheckResult",
    "RobustnessChecks",
    # Artifact Index - PR3
    "ArtifactRef",
    "ArtifactIndex",
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
