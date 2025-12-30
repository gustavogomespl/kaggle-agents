"""
Main KaggleState TypedDict and initialization.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Any, TypedDict

from .competition import AblationComponent, CompetitionInfo, SOTASolution
from .learning import CandidatePlan, PreferencePair, ReasoningTrace, SelfEvaluation
from .memory import (
    DataInsights,
    ErrorPatternMemory,
    HyperparameterHistory,
    IterationMemory,
    ModelPerformanceRecord,
    merge_error_pattern_memory,
)
from .results import CodeAttempt, DevelopmentResult, SubmissionResult, ValidationResult
from .types import DomainType


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
    timeout_per_component: int | None
    enable_checkpoint_recovery: bool
    cv_folds: int | None
    fast_mode: bool
    target_score: float | None
    current_performance_score: float
    mlebench_grade: dict[str, Any] | None
    skip_remaining_components: bool
    errors: Annotated[list[str], add]
    current_train_path: str | None
    current_test_path: str | None
    train_data_path: str
    test_data_path: str
    sample_submission_path: str
    target_col: str
    data_files: dict[str, Any]

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
    code_attempts: Annotated[list[CodeAttempt], add]

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
    target_percentile: float
    best_single_model_score: float | None
    best_single_model_name: str | None
    baseline_cv_score: float | None
    submission_validation_error: str | None
    retry_submission_count: int

    # Iteration Control
    current_iteration: int
    max_iterations: int
    should_continue: bool
    needs_refinement: bool
    termination_reason: str | None

    # Memory & Learning
    iteration_memory: Annotated[list[IterationMemory], add]
    learned_patterns: dict[str, Any]

    # Structured Memory
    data_insights: DataInsights | None
    model_performance_history: Annotated[list[ModelPerformanceRecord], add]
    best_models_by_type: dict[str, Any]
    error_pattern_memory: Annotated[list[ErrorPatternMemory], merge_error_pattern_memory]
    hyperparameter_history: Annotated[list[HyperparameterHistory], add]
    best_hyperparameters_by_model: dict[str, dict[str, Any]]
    aggregated_feature_importance: dict[str, float]
    top_features: list[str]
    successful_strategies: list[str]
    failed_strategies: list[str]
    strategy_effectiveness: dict[str, Any]

    # Prompt Optimization (DSPy)
    optimized_prompts: dict[str, str]
    prompt_performance: dict[str, float]

    # Meta-Evaluator & RL
    failure_analysis: dict[str, Any]
    refinement_guidance: dict[str, str]
    reward_signals: dict[str, float]

    # WEBRL: Curriculum Learning
    curriculum_subtasks: list[dict[str, Any]]
    needs_subtask_resolution: bool

    # Eureka: Multi-candidate Evolutionary Plans
    candidate_plans: list[CandidatePlan]
    current_plan_index: int
    evolutionary_generation: int
    crossover_guidance: dict[str, Any]

    # GRPO: Reasoning Traces
    reasoning_traces: Annotated[list[ReasoningTrace], add]
    current_reasoning: ReasoningTrace | None

    # DPO: Preference Learning
    preference_pairs: Annotated[list[PreferencePair], add]

    # Quiet-STaR: Self-Evaluation
    self_evaluations: Annotated[list[SelfEvaluation], add]
    last_self_evaluation: SelfEvaluation | None

    # Metadata
    workflow_start_time: datetime
    last_updated: datetime


def merge_dict(existing: dict, new: dict) -> dict:
    """Merge dictionaries, with new values overwriting existing ones."""
    return {**existing, **new}


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
        target_percentile=20.0,
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
        # Structured Memory
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
