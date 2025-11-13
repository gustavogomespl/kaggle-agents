"""
Unified State Management for Kaggle Agents.

This module defines the central state structure that flows through
the LangGraph workflow, containing all data needed for autonomous
Kaggle competition solving.
"""

from typing import TypedDict, Annotated, Literal, Any
from operator import add
from dataclasses import dataclass, field
from datetime import datetime


# ==================== Domain Types ====================

DomainType = Literal["tabular", "computer_vision", "nlp", "time_series", "multi_modal"]


@dataclass
class CompetitionInfo:
    """Competition metadata and configuration."""

    name: str
    description: str
    evaluation_metric: str
    problem_type: str  # classification, regression, ranking, etc.
    domain: DomainType | None = None
    data_files: list[str] = field(default_factory=list)
    submission_format: dict[str, Any] = field(default_factory=dict)
    deadline: datetime | None = None


@dataclass
class SOTASolution:
    """State-of-the-art solution retrieved from search."""

    source: str  # notebook_id or discussion_url
    title: str
    score: float
    votes: int
    code_snippets: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    feature_engineering: list[str] = field(default_factory=list)
    ensemble_approach: str | None = None


@dataclass
class AblationComponent:
    """A code component identified for ablation testing."""

    name: str
    component_type: str  # feature_engineering, model, preprocessing, ensemble
    code: str
    estimated_impact: float = 0.0
    tested: bool = False
    actual_impact: float | None = None


@dataclass
class DevelopmentResult:
    """Result from code development and execution."""

    code: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    artifacts_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from robustness validation."""

    module: str  # debugging, leakage, data_usage, format
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class SubmissionResult:
    """Result from Kaggle submission."""

    submission_id: str | None
    public_score: float | None
    private_score: float | None = None
    percentile: float | None = None
    cv_score: float | None = None
    submitted_at: datetime = field(default_factory=datetime.now)


@dataclass
class IterationMemory:
    """Memory of a single iteration for learning."""

    iteration: int
    phase: str
    actions_taken: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    score_improvement: float = 0.0
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)


# ==================== Main State ====================

class KaggleState(TypedDict):
    """
    Unified state for the entire Kaggle agent workflow.

    This state flows through all nodes in the LangGraph workflow,
    accumulating data and enabling agents to make informed decisions.
    """

    # Competition Context
    competition_info: CompetitionInfo
    working_directory: str

    # Domain Detection
    domain_detected: DomainType | None
    domain_confidence: float

    # Search Phase
    sota_solutions: Annotated[list[SOTASolution], add]
    search_queries_used: Annotated[list[str], add]

    # Planning Phase
    ablation_plan: list[AblationComponent]
    current_component_index: int
    optimization_strategy: str

    # Development Phase
    development_results: Annotated[list[DevelopmentResult], add]
    current_code: str
    code_retry_count: int

    # Validation Phase
    validation_results: Annotated[list[ValidationResult], add]
    overall_validation_score: float
    critical_issues: Annotated[list[str], add]

    # Ensemble Phase
    ensemble_strategy: str | None
    ensemble_weights: dict[str, float]

    # Submission Phase
    submissions: Annotated[list[SubmissionResult], add]
    best_score: float
    target_percentile: float  # goal: 20th percentile (top 20%)

    # Iteration Control
    current_iteration: int
    max_iterations: int
    should_continue: bool
    termination_reason: str | None

    # Memory & Learning
    iteration_memory: Annotated[list[IterationMemory], add]
    learned_patterns: dict[str, Any]

    # Prompt Optimization (DSPy)
    optimized_prompts: dict[str, str]  # agent_name -> optimized_prompt
    prompt_performance: dict[str, float]  # agent_name -> performance_score

    # Metadata
    workflow_start_time: datetime
    last_updated: datetime


# ==================== State Reducers ====================

def merge_dict(existing: dict, new: dict) -> dict:
    """Merge dictionaries, with new values overwriting existing ones."""
    return {**existing, **new}


def merge_competition_info(existing: CompetitionInfo | None, new: CompetitionInfo) -> CompetitionInfo:
    """Merge competition info, preferring new values when provided."""
    if existing is None:
        return new

    # Update existing with new non-None values
    updated = CompetitionInfo(
        name=new.name if new.name else existing.name,
        description=new.description if new.description else existing.description,
        evaluation_metric=new.evaluation_metric if new.evaluation_metric else existing.evaluation_metric,
        problem_type=new.problem_type if new.problem_type else existing.problem_type,
        domain=new.domain if new.domain is not None else existing.domain,
        data_files=new.data_files if new.data_files else existing.data_files,
        submission_format=new.submission_format if new.submission_format else existing.submission_format,
        deadline=new.deadline if new.deadline is not None else existing.deadline,
    )
    return updated


# ==================== State Initialization ====================

def create_initial_state(competition_name: str, working_dir: str) -> KaggleState:
    """
    Create initial state for a new competition.

    Args:
        competition_name: Name of the Kaggle competition
        working_dir: Working directory for artifacts

    Returns:
        Initialized KaggleState
    """
    now = datetime.now()

    return KaggleState(
        # Competition Context
        competition_info=CompetitionInfo(
            name=competition_name,
            description="",
            evaluation_metric="",
            problem_type="",
        ),
        working_directory=working_dir,

        # Domain Detection
        domain_detected=None,
        domain_confidence=0.0,

        # Search Phase
        sota_solutions=[],
        search_queries_used=[],

        # Planning Phase
        ablation_plan=[],
        current_component_index=0,
        optimization_strategy="",

        # Development Phase
        development_results=[],
        current_code="",
        code_retry_count=0,

        # Validation Phase
        validation_results=[],
        overall_validation_score=0.0,
        critical_issues=[],

        # Ensemble Phase
        ensemble_strategy=None,
        ensemble_weights={},

        # Submission Phase
        submissions=[],
        best_score=0.0,
        target_percentile=20.0,  # top 20%

        # Iteration Control
        current_iteration=0,
        max_iterations=10,
        should_continue=True,
        termination_reason=None,

        # Memory & Learning
        iteration_memory=[],
        learned_patterns={},

        # Prompt Optimization
        optimized_prompts={},
        prompt_performance={},

        # Metadata
        workflow_start_time=now,
        last_updated=now,
    )
