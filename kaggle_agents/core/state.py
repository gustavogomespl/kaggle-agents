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
    """Enhanced iteration memory with structured learning signals."""

    iteration: int
    phase: str
    actions_taken: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    score_improvement: float = 0.0

    # Structured what worked/failed
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)

    # Detailed component performance
    component_scores: dict[str, float] = field(default_factory=dict)  # {component_name: cv_score}

    # Hyperparameter configurations used
    hyperparameters_used: dict[str, dict[str, Any]] = field(default_factory=dict)  # {model_name: {param: value}}

    # Error patterns with attempted solutions
    error_solutions: list[dict[str, Any]] = field(default_factory=list)  # [{error_type, solution_tried, success}]

    # LLM-generated insights from this iteration
    key_insights: list[str] = field(default_factory=list)

    # Time tracking
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# ==================== Structured Memory Types ====================

@dataclass
class ModelPerformanceRecord:
    """Track individual model performance across iterations."""

    model_name: str
    model_type: str  # lightgbm, xgboost, catboost, neural_net, sklearn, etc.
    cv_score: float
    public_lb_score: Optional[float] = None

    # Hyperparameters that achieved this score
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Training characteristics
    training_time_seconds: float = 0.0
    memory_usage_mb: Optional[float] = None

    # Feature information
    features_used: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)  # {feature: importance}

    # Context
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataInsights:
    """Persistent EDA insights about the competition data."""

    # Basic stats
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    n_classes: Optional[int] = None  # For classification

    # Target distribution
    target_distribution: dict[str, float] = field(default_factory=dict)  # {class: proportion} or stats
    is_imbalanced: bool = False
    imbalance_ratio: Optional[float] = None

    # Feature types
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    datetime_features: list[str] = field(default_factory=list)
    text_features: list[str] = field(default_factory=list)

    # Data quality
    missing_value_cols: dict[str, float] = field(default_factory=dict)  # {col: missing_pct}
    high_cardinality_cols: list[str] = field(default_factory=list)
    constant_cols: list[str] = field(default_factory=list)

    # Correlations and insights
    highly_correlated_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    important_features_eda: list[str] = field(default_factory=list)

    # LLM-generated insights (hybrid approach)
    llm_insights: list[str] = field(default_factory=list)


@dataclass
class ErrorPatternMemory:
    """Memory of error patterns and their solutions."""

    error_type: str
    error_pattern: str = ""  # Regex or substring pattern

    occurrences: int = 1

    # Solutions that were tried
    solutions_tried: list[str] = field(default_factory=list)
    successful_solutions: list[str] = field(default_factory=list)

    # Context
    affected_models: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)

    # LLM analysis (hybrid approach)
    root_cause: str = ""
    prevention_strategy: str = ""

    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class HyperparameterHistory:
    """Track hyperparameter configurations and their outcomes."""

    model_type: str  # lightgbm, xgboost, catboost, etc.
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Outcome
    cv_score: float = 0.0
    success: bool = True

    # Issues encountered
    issues: list[str] = field(default_factory=list)

    # Context
    data_size: int = 0
    n_classes: Optional[int] = None
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


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

def merge_error_pattern_memory(
    existing: list["ErrorPatternMemory"],
    new: list["ErrorPatternMemory"],
) -> list["ErrorPatternMemory"]:
    """Merge error pattern memory entries by (error_type, error_pattern)."""
    def _normalize(entry: Any) -> Optional["ErrorPatternMemory"]:
        if isinstance(entry, ErrorPatternMemory):
            return entry
        if isinstance(entry, dict):
            try:
                return ErrorPatternMemory(**entry)
            except TypeError:
                return None
        return None

    def _uniq(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    merged: dict[tuple[str, str], ErrorPatternMemory] = {}
    for entry in list(existing) + list(new):
        em = _normalize(entry)
        if em is None:
            continue
        key = (em.error_type, em.error_pattern)
        if key not in merged:
            merged[key] = em
            continue
        current = merged[key]
        current.occurrences = current.occurrences + em.occurrences
        current.first_seen = min(current.first_seen, em.first_seen)
        current.last_seen = max(current.last_seen, em.last_seen)
        current.solutions_tried = _uniq(current.solutions_tried + em.solutions_tried)
        current.successful_solutions = _uniq(current.successful_solutions + em.successful_solutions)
        current.affected_models = _uniq(current.affected_models + em.affected_models)
        current.affected_components = _uniq(current.affected_components + em.affected_components)
        if em.root_cause:
            current.root_cause = em.root_cause
        if em.prevention_strategy:
            current.prevention_strategy = em.prevention_strategy
        merged[key] = current

    return list(merged.values())

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

    # Structured Memory (NEW)
    data_insights: Optional[DataInsights]
    model_performance_history: Annotated[list[ModelPerformanceRecord], add]
    best_models_by_type: dict[str, Any]  # {model_type: ModelPerformanceRecord dict}
    error_pattern_memory: Annotated[list[ErrorPatternMemory], merge_error_pattern_memory]
    hyperparameter_history: Annotated[list[HyperparameterHistory], add]
    # {model_type: {"hyperparameters": {...}, "cv_score": float} or legacy hyperparams dict}
    best_hyperparameters_by_model: dict[str, dict[str, Any]]
    aggregated_feature_importance: dict[str, float]  # {feature: avg_importance}
    top_features: list[str]  # Top K features across all models
    successful_strategies: list[str]  # Strategies that improved score
    failed_strategies: list[str]  # Strategies that hurt score
    strategy_effectiveness: dict[str, Any]  # {strategy: {"average": float, "count": int}}

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

        # Structured Memory (NEW)
        data_insights=None,
        model_performance_history=[],
        best_models_by_type={},
        error_pattern_memory=[],
        hyperparameter_history=[],
        best_hyperparameters_by_model={},
        aggregated_feature_importance={},
        top_features=[],
        successful_strategies=[],
        failed_strategies=[],
        strategy_effectiveness={},

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


# ==================== Memory Helper Functions ====================

def update_model_performance(state: KaggleState, record: ModelPerformanceRecord) -> dict[str, Any]:
    """
    Update model performance history and track best models by type.

    Args:
        state: Current KaggleState
        record: ModelPerformanceRecord to add

    Returns:
        Dict with updates to apply to state
    """
    updates: dict[str, Any] = {"model_performance_history": [record]}

    # Update best_models_by_type if this is the best for its type
    current_best = dict(state.get("best_models_by_type", {}))
    model_type = record.model_type

    # Compare with existing best (if any)
    existing_best = current_best.get(model_type)
    is_new_best = False

    if existing_best is None:
        is_new_best = True
    elif isinstance(existing_best, dict):
        is_new_best = record.cv_score > existing_best.get("cv_score", float("-inf"))
    elif isinstance(existing_best, ModelPerformanceRecord):
        is_new_best = record.cv_score > existing_best.cv_score

    if is_new_best:
        # Store as dict for JSON serialization
        current_best[model_type] = {
            "model_name": record.model_name,
            "model_type": record.model_type,
            "cv_score": record.cv_score,
            "public_lb_score": record.public_lb_score,
            "hyperparameters": record.hyperparameters,
            "training_time_seconds": record.training_time_seconds,
            "features_used": record.features_used,
            "feature_importance": record.feature_importance,
            "iteration": record.iteration,
        }
        updates["best_models_by_type"] = current_best

        # Also update best hyperparameters (store score for safe comparisons)
        if record.hyperparameters:
            best_hp = dict(state.get("best_hyperparameters_by_model", {}))
            best_hp[model_type] = {
                "hyperparameters": record.hyperparameters,
                "cv_score": record.cv_score,
            }
            updates["best_hyperparameters_by_model"] = best_hp

    return updates


def update_error_memory(
    state: KaggleState,
    error_type: str,
    error_pattern: str,
    solution: str,
    success: bool,
    affected_model: Optional[str] = None,
    affected_component: Optional[str] = None,
    root_cause: str = "",
    prevention_strategy: str = "",
) -> dict[str, Any]:
    """
    Update error pattern memory with a new solution attempt.

    Args:
        state: Current KaggleState
        error_type: Type of error (e.g., "memory_error", "convergence_failure")
        error_pattern: Pattern that identifies this error
        solution: Solution that was tried
        success: Whether the solution resolved the error
        affected_model: Model that encountered this error
        affected_component: Component that had the error
        root_cause: LLM-analyzed root cause
        prevention_strategy: How to prevent this error

    Returns:
        Dict with updates to apply to state
    """
    existing_memory = list(state.get("error_pattern_memory", []))
    now = datetime.now()

    # Find existing error pattern (match by both type AND pattern to avoid merging distinct errors)
    found_idx = None
    for idx, em in enumerate(existing_memory):
        if isinstance(em, ErrorPatternMemory):
            # Match both error_type and error_pattern to distinguish different stack traces
            if em.error_type == error_type and em.error_pattern == error_pattern:
                found_idx = idx
                break
        elif isinstance(em, dict):
            if em.get("error_type") == error_type and em.get("error_pattern", "") == error_pattern:
                found_idx = idx
                break

    if found_idx is not None:
        # Return a delta entry to avoid double-counting during merge.
        delta = ErrorPatternMemory(
            error_type=error_type,
            error_pattern=error_pattern,
            occurrences=1,
            solutions_tried=[solution] if solution else [],
            successful_solutions=[solution] if success and solution else [],
            affected_models=[affected_model] if affected_model else [],
            affected_components=[affected_component] if affected_component else [],
            root_cause=root_cause,
            prevention_strategy=prevention_strategy,
            first_seen=now,
            last_seen=now,
        )
        return {"error_pattern_memory": [delta]}
    else:
        # Create new error pattern
        new_pattern = ErrorPatternMemory(
            error_type=error_type,
            error_pattern=error_pattern,
            occurrences=1,
            solutions_tried=[solution] if solution else [],
            successful_solutions=[solution] if success and solution else [],
            affected_models=[affected_model] if affected_model else [],
            affected_components=[affected_component] if affected_component else [],
            root_cause=root_cause,
            prevention_strategy=prevention_strategy,
            first_seen=now,
            last_seen=now,
        )
        # Use reducer pattern (add to list)
        return {"error_pattern_memory": [new_pattern]}


def aggregate_feature_importance(state: KaggleState, top_k: int = 20) -> dict[str, Any]:
    """
    Aggregate feature importance across all model performance records.

    Args:
        state: Current KaggleState
        top_k: Number of top features to track

    Returns:
        Dict with updates to apply to state
    """
    history = state.get("model_performance_history", [])

    # Aggregate importance scores
    feature_scores: dict[str, list[float]] = {}
    for record in history:
        if isinstance(record, ModelPerformanceRecord):
            importance = record.feature_importance
        elif isinstance(record, dict):
            importance = record.get("feature_importance", {})
        else:
            continue

        for feature, score in importance.items():
            if feature not in feature_scores:
                feature_scores[feature] = []
            feature_scores[feature].append(score)

    # Calculate average importance
    aggregated: dict[str, float] = {}
    for feature, scores in feature_scores.items():
        if scores:
            aggregated[feature] = sum(scores) / len(scores)

    # Get top K features
    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:top_k]]

    return {
        "aggregated_feature_importance": aggregated,
        "top_features": top_features,
    }


def get_best_hyperparameters(state: KaggleState, model_type: str) -> dict[str, Any]:
    """
    Get best hyperparameters for a model type based on history.

    Args:
        state: Current KaggleState
        model_type: Type of model (lightgbm, xgboost, catboost, etc.)

    Returns:
        Best hyperparameters dict or empty dict if none found
    """
    # First check cached best hyperparameters
    best_hp = state.get("best_hyperparameters_by_model", {})
    if model_type in best_hp:
        entry = best_hp[model_type]
        if isinstance(entry, dict) and "hyperparameters" in entry:
            return entry.get("hyperparameters", {})
        return entry

    # Search through hyperparameter history
    history = state.get("hyperparameter_history", [])
    best: Optional[dict[str, Any]] = None
    best_score = float("-inf")

    for record in history:
        if isinstance(record, HyperparameterHistory):
            if record.model_type == model_type and record.success and record.cv_score > best_score:
                best = record.hyperparameters
                best_score = record.cv_score
        elif isinstance(record, dict):
            if (
                record.get("model_type") == model_type
                and record.get("success", True)
                and record.get("cv_score", 0) > best_score
            ):
                best = record.get("hyperparameters", {})
                best_score = record.get("cv_score", 0)

    return best or {}


def update_hyperparameter_history(
    state: KaggleState,
    model_type: str,
    hyperparameters: dict[str, Any],
    cv_score: float,
    success: bool = True,
    issues: Optional[list[str]] = None,
    data_size: int = 0,
    n_classes: Optional[int] = None,
    iteration: int = 0,
) -> dict[str, Any]:
    """
    Record a hyperparameter configuration and its outcome.

    Args:
        state: Current KaggleState
        model_type: Type of model
        hyperparameters: Configuration used
        cv_score: Score achieved
        success: Whether training succeeded
        issues: Any issues encountered
        data_size: Size of training data
        n_classes: Number of classes (for classification)
        iteration: Current iteration number

    Returns:
        Dict with updates to apply to state
    """
    record = HyperparameterHistory(
        model_type=model_type,
        hyperparameters=hyperparameters,
        cv_score=cv_score,
        success=success,
        issues=issues or [],
        data_size=data_size,
        n_classes=n_classes,
        iteration=iteration,
        timestamp=datetime.now(),
    )

    updates: dict[str, Any] = {"hyperparameter_history": [record]}

    # Update best hyperparameters if this is the best
    if success:
        best_hp_dict = dict(state.get("best_hyperparameters_by_model", {}))
        history = state.get("hyperparameter_history", [])

        # Find current best score for this model type from ALL sources:
        # 1. Check hyperparameter_history
        # 2. Check cached best_models_by_type (may exist from restored state)
        current_best_score = float("-inf")

        # Source 1: Search hyperparameter history
        for h in history:
            if isinstance(h, HyperparameterHistory):
                if h.model_type == model_type and h.success:
                    current_best_score = max(current_best_score, h.cv_score)
            elif isinstance(h, dict):
                if h.get("model_type") == model_type and h.get("success", True):
                    current_best_score = max(current_best_score, h.get("cv_score", 0))

        # Source 2: Check cached best_models_by_type (critical for restored state)
        best_models = state.get("best_models_by_type", {})
        if model_type in best_models:
            cached_model = best_models[model_type]
            if isinstance(cached_model, dict):
                cached_score = cached_model.get("cv_score", float("-inf"))
            elif isinstance(cached_model, ModelPerformanceRecord):
                cached_score = cached_model.cv_score
            else:
                cached_score = float("-inf")
            current_best_score = max(current_best_score, cached_score)

        # Source 3: Check cached best hyperparameters (may include score)
        cached_hp_entry = best_hp_dict.get(model_type)
        cached_hp_params = None
        cached_hp_score = None
        if isinstance(cached_hp_entry, dict) and "hyperparameters" in cached_hp_entry:
            cached_hp_params = cached_hp_entry.get("hyperparameters", {})
            cached_hp_score = cached_hp_entry.get("cv_score", float("-inf"))
            current_best_score = max(current_best_score, cached_hp_score)
        elif isinstance(cached_hp_entry, dict):
            cached_hp_params = cached_hp_entry

        should_update = False
        if cached_hp_entry is None:
            should_update = cv_score > current_best_score
        elif cached_hp_score is not None:
            should_update = cv_score > current_best_score
        else:
            # Legacy cached params without score: only overwrite if we can prove improvement
            if cached_hp_params == hyperparameters:
                best_hp_dict[model_type] = {
                    "hyperparameters": hyperparameters,
                    "cv_score": cv_score,
                }
                updates["best_hyperparameters_by_model"] = best_hp_dict
            elif current_best_score != float("-inf") and cv_score > current_best_score:
                should_update = True

        if should_update:
            best_hp_dict[model_type] = {
                "hyperparameters": hyperparameters,
                "cv_score": cv_score,
            }
            updates["best_hyperparameters_by_model"] = best_hp_dict

    return updates


def update_strategy_effectiveness(
    state: KaggleState,
    strategy: str,
    score_improvement: float,
) -> dict[str, Any]:
    """
    Track strategy effectiveness based on score changes.

    Args:
        state: Current KaggleState
        strategy: Strategy name/description
        score_improvement: Score improvement (positive = better, negative = worse)

    Returns:
        Dict with updates to apply to state
    """
    updates: dict[str, Any] = {}

    # Update successful/failed strategies
    successful = list(state.get("successful_strategies", []))
    failed = list(state.get("failed_strategies", []))

    if score_improvement > 0:
        if strategy not in successful:
            successful.append(strategy)
            updates["successful_strategies"] = successful
        # Remove from failed if it was there
        if strategy in failed:
            failed.remove(strategy)
            updates["failed_strategies"] = failed
    elif score_improvement < 0:
        if strategy not in failed:
            failed.append(strategy)
            updates["failed_strategies"] = failed
        # Remove from successful if it was there (strategy regressed)
        if strategy in successful:
            successful.remove(strategy)
            updates["successful_strategies"] = successful

    # Update effectiveness tracking with proper running average
    # Store as {"average": float, "count": int} for accurate mean calculation
    effectiveness = dict(state.get("strategy_effectiveness", {}))

    if strategy in effectiveness:
        existing = effectiveness[strategy]
        # Handle both old (float) and new (dict) formats for backwards compatibility
        if isinstance(existing, dict):
            old_avg = existing.get("average", 0.0)
            count = existing.get("count", 1)
        else:
            # Legacy float format - treat as single observation
            old_avg = float(existing)
            count = 1

        # Incremental mean formula: new_avg = old_avg + (new_value - old_avg) / (count + 1)
        new_count = count + 1
        new_avg = old_avg + (score_improvement - old_avg) / new_count
        effectiveness[strategy] = {"average": new_avg, "count": new_count}
    else:
        effectiveness[strategy] = {"average": score_improvement, "count": 1}

    updates["strategy_effectiveness"] = effectiveness

    return updates


def get_memory_summary_for_planning(state: KaggleState) -> str:
    """
    Generate a summary of structured memory for planning agents.

    Args:
        state: Current KaggleState

    Returns:
        Formatted string with memory insights for planning
    """
    insights: list[str] = []

    # Best models
    best_models = state.get("best_models_by_type", {})
    if best_models:
        insights.append("## Best Models So Far")
        for model_type, record in best_models.items():
            if isinstance(record, dict):
                cv = record.get("cv_score", 0)
                insights.append(f"- {model_type}: CV={cv:.4f}")
            elif isinstance(record, ModelPerformanceRecord):
                insights.append(f"- {model_type}: CV={record.cv_score:.4f}")

    # Top features
    top_features = state.get("top_features", [])
    if top_features:
        insights.append(f"\n## Top Features: {', '.join(top_features[:10])}")

    # Data insights (handle both DataInsights instance and dict from restored state)
    data_insights = state.get("data_insights")
    if data_insights:
        # Extract values handling both instance and dict forms
        if isinstance(data_insights, DataInsights):
            n_train = data_insights.n_train_samples
            n_test = data_insights.n_test_samples
            n_features = data_insights.n_features
            is_imbalanced = data_insights.is_imbalanced
            imbalance_ratio = data_insights.imbalance_ratio
            llm_insights_list = data_insights.llm_insights
        elif isinstance(data_insights, dict):
            n_train = data_insights.get("n_train_samples", 0)
            n_test = data_insights.get("n_test_samples", 0)
            n_features = data_insights.get("n_features", 0)
            is_imbalanced = data_insights.get("is_imbalanced", False)
            imbalance_ratio = data_insights.get("imbalance_ratio")
            llm_insights_list = data_insights.get("llm_insights", [])
        else:
            n_train = n_test = n_features = 0
            is_imbalanced = False
            imbalance_ratio = None
            llm_insights_list = []

        if n_train or n_test or n_features:
            insights.append("\n## Data Insights")
            insights.append(f"- Samples: {n_train} train, {n_test} test")
            insights.append(f"- Features: {n_features}")
            if is_imbalanced:
                insights.append(f"- IMBALANCED (ratio: {imbalance_ratio})")
            if llm_insights_list:
                insights.append(f"- LLM Insights: {'; '.join(llm_insights_list[:3])}")

    # Successful strategies
    successful = state.get("successful_strategies", [])
    if successful:
        insights.append(f"\n## Successful Strategies: {', '.join(successful[:5])}")

    # Failed strategies
    failed = state.get("failed_strategies", [])
    if failed:
        insights.append(f"\n## Failed Strategies (avoid): {', '.join(failed[:5])}")

    # Error patterns to avoid
    error_memory = state.get("error_pattern_memory", [])
    if error_memory:
        insights.append("\n## Known Issues & Solutions")
        for em in error_memory[:5]:
            if isinstance(em, ErrorPatternMemory):
                if em.successful_solutions:
                    insights.append(f"- {em.error_type}: Fix with '{em.successful_solutions[0]}'")
                else:
                    insights.append(f"- {em.error_type}: No solution found yet")
            elif isinstance(em, dict):
                solutions = em.get("successful_solutions", [])
                if solutions:
                    insights.append(f"- {em.get('error_type', 'unknown')}: Fix with '{solutions[0]}'")

    # Best hyperparameters
    best_hp = state.get("best_hyperparameters_by_model", {})
    if best_hp:
        insights.append("\n## Best Hyperparameters")
        for model_type, params in best_hp.items():
            if not params:
                continue
            if isinstance(params, dict) and "hyperparameters" in params:
                param_values = params.get("hyperparameters", {})
            else:
                param_values = params
            if param_values:
                param_str = ", ".join(f"{k}={v}" for k, v in list(param_values.items())[:5])
                insights.append(f"- {model_type}: {param_str}")

    return "\n".join(insights) if insights else "No memory insights available yet."
