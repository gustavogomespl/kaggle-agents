"""
Unified State Management for Kaggle Agents.

This module defines the central state structure that flows through
the LangGraph workflow, containing all data needed for autonomous
Kaggle competition solving.
"""

from dataclasses import dataclass, field
from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal, Optional, TypedDict


# ==================== Domain Types ====================

DomainType = Literal[
    # Image-based
    "image_classification",
    "image_regression",
    "image_to_image",
    "image_segmentation",
    "object_detection",
    # Text-based
    "text_classification",
    "seq_to_seq",
    "text_regression",
    # Audio-based
    "audio_classification",
    "audio_regression",
    # Tabular
    "tabular_classification",
    "tabular_regression",
    # Time series
    "time_series_forecasting",
    # Multi-modal
    "multi_modal",
    # Legacy (for backwards compatibility)
    "tabular",
    "computer_vision",
    "nlp",
    "time_series",
]


# ==================== Submission Format Types ====================

SubmissionFormatType = Literal[
    "standard",      # One row per sample (classification/regression)
    "pixel_level",   # One row per pixel (image-to-image, segmentation)
    "multi_label",   # Multiple rows per sample
    "ranking",       # Ranking format
    "rle_encoded",   # Run-length encoded masks (segmentation)
]


@dataclass
class CompetitionInfo:
    """Competition metadata and configuration."""

    name: str
    description: str
    evaluation_metric: str
    problem_type: str  # classification, regression, ranking, etc.
    domain: Optional[DomainType] = None
    data_files: list[str] = field(default_factory=list)
    submission_format: dict[str, Any] = field(default_factory=dict)
    submission_format_type: Optional[SubmissionFormatType] = None
    submission_format_metadata: dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None


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
    ensemble_approach: Optional[str] = None


@dataclass
class AblationComponent:
    """A code component identified for ablation testing."""

    name: str
    component_type: str  # feature_engineering, model, preprocessing, ensemble
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
    run_fidelity: Literal["full", "debug"] = "full"


@dataclass
class CodeAttempt:
    """A single code execution attempt for learning and prompt feedback."""

    component_name: str
    component_type: str
    stage: Literal["generate", "fix", "debug", "refine"]
    attempt: int
    success: bool
    cv_score: Optional[float] = None
    error: Optional[str] = None
    meta_feedback: Optional[str] = None
    code_excerpt: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    execution_time: float = 0.0
    run_fidelity: Literal["full", "debug"] = "full"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result from robustness validation."""

    module: str  # debugging, leakage, data_usage, format
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)  # Additional structured information


@dataclass
class SubmissionResult:
    """Result from Kaggle submission."""

    submission_id: Optional[str]
    public_score: Optional[float]
    private_score: Optional[float] = None
    percentile: Optional[float] = None
    cv_score: Optional[float] = None
    file_path: Optional[str] = None
    valid: bool = True
    error: Optional[str] = None
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


# ==================== Advanced RL Types (WEBRL, Eureka, GRPO, DPO) ====================

@dataclass
class SubTask:
    """
    WEBRL-style sub-task generated from failure.

    When the agent fails, creates specific sub-tasks to resolve
    the problem before proceeding.
    """

    parent_component: str
    failure_type: str  # "memory", "timeout", "syntax", "validation", etc.
    task_description: str
    priority: int  # 1 (highest) to 5 (lowest)
    status: Literal["pending", "in_progress", "resolved", "skipped"] = "pending"
    resolution_code: Optional[str] = None
    resolution_guidance: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CandidatePlan:
    """
    Eureka-style candidate plan with fitness score.

    Multiple plans are generated with different strategies,
    evaluated, and the best elements are combined.
    """

    components: list[AblationComponent] = field(default_factory=list)
    strategy: str = "balanced"  # "conservative", "aggressive", "balanced"
    fitness_score: float = 0.0
    generation: int = 0  # Evolutionary generation number
    execution_results: dict[str, float] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """
    GRPO-style reasoning trace for code generation.

    Structured chain-of-thought before generating code,
    with process rewards for intermediate steps.
    """

    component_name: str
    requirements_analysis: str = ""
    potential_issues: list[str] = field(default_factory=list)
    solution_approach: str = ""
    implementation_plan: str = ""
    validation_checklist: list[str] = field(default_factory=list)
    step_scores: dict[str, float] = field(default_factory=dict)  # {step_name: score}
    final_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PreferencePair:
    """
    DPO-style preference pair for learning.

    Captures chosen (good) vs rejected (bad) code examples
    for preference-based optimization.
    """

    context: str  # Component/prompt description
    chosen: str  # Better code (succeeded)
    rejected: str  # Worse code (failed)
    margin: float = 0.0  # How much better is chosen (0-1)
    component_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelfEvaluation:
    """
    Quiet-STaR style self-evaluation result.

    Internal reflection before finalizing code generation.
    """

    confidence: float = 0.0  # 0-1
    concerns: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    proceed: bool = True
    reflection_summary: str = ""


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
    run_mode: str  # e.g. "kaggle" | "mlebench"
    objective: str  # e.g. "top20" | "mlebench_medal"
    timeout_per_component: Optional[int]
    enable_checkpoint_recovery: bool
    cv_folds: Optional[int]
    fast_mode: bool
    target_score: Optional[float]
    current_performance_score: float
    mlebench_grade: Optional[dict[str, Any]]
    skip_remaining_components: bool
    errors: Annotated[list[str], add]
    current_train_path: Optional[str]
    current_test_path: Optional[str]
    train_data_path: str
    test_data_path: str
    sample_submission_path: str
    target_col: str
    data_files: dict[str, Any]

    # Domain Detection
    domain_detected: Optional[DomainType]
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
    code_attempts: Annotated[list[CodeAttempt], add]

    # Validation Phase
    validation_results: Annotated[list[ValidationResult], add]
    overall_validation_score: float
    critical_issues: Annotated[list[str], add]

    # Ensemble Phase
    ensemble_strategy: Optional[str]
    ensemble_weights: dict[str, float]

    # Submission Phase
    submissions: Annotated[list[SubmissionResult], add]
    best_score: float
    target_percentile: float  # goal: 20th percentile (top 20%)
    best_single_model_score: Optional[float]
    best_single_model_name: Optional[str]
    baseline_cv_score: Optional[float]
    submission_validation_error: Optional[str]  # Error from last submission validation
    retry_submission_count: int  # Counter for submission retries

    # Iteration Control
    current_iteration: int
    max_iterations: int
    should_continue: bool
    needs_refinement: bool
    termination_reason: Optional[str]

    # Memory & Learning
    iteration_memory: Annotated[list[IterationMemory], add]
    learned_patterns: dict[str, Any]

    # Prompt Optimization (DSPy)
    optimized_prompts: dict[str, str]  # agent_name -> optimized_prompt
    prompt_performance: dict[str, float]  # agent_name -> performance_score

    # Meta-Evaluator & RL (NEW)
    failure_analysis: dict[str, Any]  # Error patterns and component analysis
    refinement_guidance: dict[str, str]  # Guidance for prompt refinement
    reward_signals: dict[str, float]  # RL reward components

    # WEBRL: Curriculum Learning
    curriculum_subtasks: list[dict[str, Any]]  # SubTask dicts
    needs_subtask_resolution: bool

    # Eureka: Multi-candidate Evolutionary Plans
    candidate_plans: list[CandidatePlan]
    current_plan_index: int
    evolutionary_generation: int
    crossover_guidance: dict[str, Any]  # Guidance from evolutionary crossover

    # GRPO: Reasoning Traces
    reasoning_traces: Annotated[list[ReasoningTrace], add]
    current_reasoning: Optional[ReasoningTrace]

    # DPO: Preference Learning
    preference_pairs: Annotated[list[PreferencePair], add]

    # Quiet-STaR: Self-Evaluation
    self_evaluations: Annotated[list[SelfEvaluation], add]
    last_self_evaluation: Optional[SelfEvaluation]

    # Metadata
    workflow_start_time: datetime
    last_updated: datetime


# ==================== State Reducers ====================

def merge_dict(existing: dict, new: dict) -> dict:
    """Merge dictionaries, with new values overwriting existing ones."""
    return {**existing, **new}


def merge_competition_info(existing: Optional[CompetitionInfo], new: CompetitionInfo) -> CompetitionInfo:
    """Merge competition info, preferring new values when provided."""
    if existing is None:
        return new

    # Update existing with new non-None values
    return CompetitionInfo(
        name=new.name if new.name else existing.name,
        description=new.description if new.description else existing.description,
        evaluation_metric=new.evaluation_metric if new.evaluation_metric else existing.evaluation_metric,
        problem_type=new.problem_type if new.problem_type else existing.problem_type,
        domain=new.domain if new.domain is not None else existing.domain,
        data_files=new.data_files if new.data_files else existing.data_files,
        submission_format=new.submission_format if new.submission_format else existing.submission_format,
        submission_format_type=new.submission_format_type if new.submission_format_type is not None else existing.submission_format_type,
        submission_format_metadata=new.submission_format_metadata if new.submission_format_metadata else existing.submission_format_metadata,
        deadline=new.deadline if new.deadline is not None else existing.deadline,
    )


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
        run_mode="kaggle",
        objective="top20",
        timeout_per_component=None,
        enable_checkpoint_recovery=True,
        cv_folds=None,
        fast_mode=False,
        target_score=None,
        current_performance_score=0.0,
        mlebench_grade=None,
        skip_remaining_components=False,
        errors=[],
        current_train_path=None,
        current_test_path=None,
        train_data_path="",
        test_data_path="",
        sample_submission_path="",
        target_col="",
        data_files={},

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
        code_attempts=[],

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
        best_single_model_score=None,
        best_single_model_name=None,
        baseline_cv_score=None,
        submission_validation_error=None,
        retry_submission_count=0,

        # Iteration Control
        current_iteration=0,
        max_iterations=10,
        should_continue=True,
        needs_refinement=False,
        termination_reason=None,

        # Memory & Learning
        iteration_memory=[],
        learned_patterns={},

        # Prompt Optimization
        optimized_prompts={},
        prompt_performance={},

        # Meta-Evaluator & RL
        failure_analysis={},
        refinement_guidance={},
        reward_signals={},

        # WEBRL: Curriculum Learning
        curriculum_subtasks=[],
        needs_subtask_resolution=False,

        # Eureka: Multi-candidate Evolutionary Plans
        candidate_plans=[],
        current_plan_index=0,
        evolutionary_generation=0,
        crossover_guidance={},

        # GRPO: Reasoning Traces
        reasoning_traces=[],
        current_reasoning=None,

        # DPO: Preference Learning
        preference_pairs=[],

        # Quiet-STaR: Self-Evaluation
        self_evaluations=[],
        last_self_evaluation=None,

        # Metadata
        workflow_start_time=now,
        last_updated=now,
    )
