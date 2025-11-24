"""
Unified State Management for Kaggle Agents.

This module defines the central state structure that flows through
the LangGraph workflow, containing all data needed for autonomous
Kaggle competition solving.
"""

from typing import TypedDict, Annotated, Literal, Any, Optional
from operator import add
from dataclasses import dataclass, field
from datetime import datetime




DomainType = Literal["tabular", "computer_vision", "nlp", "time_series", "multi_modal"]


@dataclass
class CompetitionInfo:
    """Competition metadata and configuration."""

    name: str
    description: str
    evaluation_metric: str
    problem_type: str
    domain: Optional[DomainType] = None
    data_files: list[str] = field(default_factory=list)
    submission_format: dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None


@dataclass
class SOTASolution:
    """State-of-the-art solution retrieved from search."""

    source: str
    title: str
    score: float
    votes: int
    code_snippets: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    feature_engineering: list[str] = field(default_factory=list)
    ensemble_approach: Optional[str] = None


@dataclass
class AblationComponent:
    """A code component identified for ablation testing."""

    name: str
    component_type: str
    code: str
    estimated_impact: float = 0.0
    tested: bool = False
    actual_impact: Optional[float] = None


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

    module: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class SubmissionResult:
    """Result from Kaggle submission."""

    submission_id: Optional[str]
    public_score: Optional[float]
    private_score: Optional[float] = None
    percentile: Optional[float] = None
    cv_score: Optional[float] = None
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





class KaggleState(TypedDict):
    """
    Unified state for the entire Kaggle agent workflow.

    This state flows through all nodes in the LangGraph workflow,
    accumulating data and enabling agents to make informed decisions.
    """


    competition_info: CompetitionInfo
    working_directory: str
    current_train_path: Optional[str]
    current_test_path: Optional[str]


    domain_detected: Optional[DomainType]
    domain_confidence: float


    sota_solutions: Annotated[list[SOTASolution], add]
    search_queries_used: Annotated[list[str], add]


    ablation_plan: list[AblationComponent]
    current_component_index: int
    optimization_strategy: str


    development_results: Annotated[list[DevelopmentResult], add]
    current_code: str
    code_retry_count: int


    validation_results: Annotated[list[ValidationResult], add]
    overall_validation_score: float
    critical_issues: Annotated[list[str], add]


    ensemble_strategy: Optional[str]
    ensemble_weights: dict[str, float]


    submissions: Annotated[list[SubmissionResult], add]
    best_score: float
    target_percentile: float
    best_single_model_score: Optional[float]
    best_single_model_name: Optional[str]


    current_iteration: int
    max_iterations: int
    should_continue: bool
    needs_refinement: bool
    termination_reason: Optional[str]


    iteration_memory: Annotated[list[IterationMemory], add]
    learned_patterns: dict[str, Any]


    optimized_prompts: dict[str, str]
    prompt_performance: dict[str, float]


    failure_analysis: dict[str, Any]
    refinement_guidance: dict[str, str]
    reward_signals: dict[str, float]


    workflow_start_time: datetime
    last_updated: datetime





def merge_dict(existing: dict, new: dict) -> dict:
    """Merge dictionaries, with new values overwriting existing ones."""
    return {**existing, **new}


def merge_competition_info(
    existing: Optional[CompetitionInfo], new: CompetitionInfo
) -> CompetitionInfo:
    """Merge competition info, preferring new values when provided."""
    if existing is None:
        return new


    updated = CompetitionInfo(
        name=new.name if new.name else existing.name,
        description=new.description if new.description else existing.description,
        evaluation_metric=new.evaluation_metric
        if new.evaluation_metric
        else existing.evaluation_metric,
        problem_type=new.problem_type if new.problem_type else existing.problem_type,
        domain=new.domain if new.domain is not None else existing.domain,
        data_files=new.data_files if new.data_files else existing.data_files,
        submission_format=new.submission_format
        if new.submission_format
        else existing.submission_format,
        deadline=new.deadline if new.deadline is not None else existing.deadline,
    )
    return updated





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

        competition_info=CompetitionInfo(
            name=competition_name,
            description="",
            evaluation_metric="",
            problem_type="",
        ),
        working_directory=working_dir,
        current_train_path=None,
        current_test_path=None,

        domain_detected=None,
        domain_confidence=0.0,

        sota_solutions=[],
        search_queries_used=[],

        ablation_plan=[],
        current_component_index=0,
        optimization_strategy="",

        development_results=[],
        current_code="",
        code_retry_count=0,

        validation_results=[],
        overall_validation_score=0.0,
        critical_issues=[],

        ensemble_strategy=None,
        ensemble_weights={},

        submissions=[],
        best_score=0.0,
        target_percentile=20.0,
        best_single_model_score=None,
        best_single_model_name=None,

        current_iteration=0,
        max_iterations=10,
        should_continue=True,
        needs_refinement=False,
        termination_reason=None,

        iteration_memory=[],
        learned_patterns={},

        optimized_prompts={},
        prompt_performance={},

        failure_analysis={},
        refinement_guidance={},
        reward_signals={},

        workflow_start_time=now,
        last_updated=now,
    )
