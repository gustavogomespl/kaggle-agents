"""
Main KaggleState TypedDict and initialization.
"""

from __future__ import annotations

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

    # Expected row counts (for OOF alignment across models)
    expected_train_rows: int | None  # Expected rows in train set
    expected_test_rows: int | None   # Expected rows in test set

    # Data Format Discovery (for non-standard formats)
    data_format_type: str | None  # "traditional", "generated", "custom", or "unknown"
    parsing_info: dict[str, Any] | None  # LLM-generated parsing instructions
    data_loading_code: str | None  # Python code to load non-standard data

    # Domain Detection
    domain_detected: DomainType | None
    domain_confidence: float

    # ========================================================================
    # CONTRACTS (Source of Truth) - PR1
    # All contracts are stored as dict for JSON serialization compatibility
    # Use *Contract.from_dict() to reconstruct objects when needed
    # ========================================================================
    metric_contract: dict[str, Any] | None  # MetricContract.to_dict()
    canonical_contract: dict[str, Any] | None  # CanonicalDataContract.to_dict()
    submission_contract: dict[str, Any] | None  # SubmissionContract.to_dict()
    eval_fidelity: dict[str, Any] | None  # EvalFidelityContract.to_dict()
    data_usage: dict[str, Any] | None  # DataUsageContract.to_dict()

    # ========================================================================
    # MLE-STAR REGISTRIES (PR2)
    # Track code blocks, ablations, models, and robustness checks
    # All stored as dict for JSON serialization compatibility
    # ========================================================================
    code_registry: dict[str, Any] | None  # CodeBlockRegistry.to_dict()
    ablation_history: dict[str, Any] | None  # AblationHistory.to_dict()
    model_registry: dict[str, Any] | None  # ModelRegistry.to_dict()
    robustness_checks: dict[str, Any] | None  # RobustnessChecks.to_dict()

    # ========================================================================
    # ARTIFACT INDEX (PR3)
    # References to artifacts on disk (reduces state bloat)
    # ========================================================================
    artifact_index: dict[str, Any] | None  # ArtifactIndex.to_dict()

    # ========================================================================
    # SEARCH PHASE
    # ========================================================================
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
    best_score: float  # [DERIVABLE] from metric_contract.best_score
    target_percentile: float
    best_single_model_score: float | None  # [DERIVABLE] from model_registry.get_best_overall()
    best_single_model_name: str | None  # [DERIVABLE] from model_registry.get_best_overall()
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

    # ========================================================================
    # MEMORY (accumulated learning)
    # NOTE: Some fields below are DERIVABLE from registries and may be removed
    # in future refactoring. Marked with [DERIVABLE] comment.
    # ========================================================================
    data_insights: DataInsights | None
    model_performance_history: Annotated[list[ModelPerformanceRecord], add]
    best_models_by_type: dict[str, Any]  # [DERIVABLE] from model_registry.get_best_by_type()
    error_pattern_memory: Annotated[list[ErrorPatternMemory], merge_error_pattern_memory]
    hyperparameter_history: Annotated[list[HyperparameterHistory], add]
    best_hyperparameters_by_model: dict[str, dict[str, Any]]  # [DERIVABLE] from model_registry
    aggregated_feature_importance: dict[str, float]  # [DERIVABLE] compute from model artifacts
    top_features: list[str]  # [DERIVABLE] compute from model_registry feature importance
    successful_strategies: list[str]  # [DERIVABLE] from ablation_history.get_effective_ablations()
    failed_strategies: list[str]  # [DERIVABLE] from ablation_history.get_regressions()
    failed_component_names: Annotated[list[str], add]  # Component names that failed (for planner to avoid)
    strategy_effectiveness: dict[str, Any]  # [DERIVABLE] compute from ablation_history

    # ========================================================================
    # OPTIONAL FEATURES
    # These are only initialized when their respective features are enabled.
    # Default to empty dicts/lists. May be removed if unused.
    # ========================================================================

    # Prompt Optimization (DSPy)
    optimized_prompts: dict[str, str]  # [OPTIONAL] Only when DSPy enabled
    prompt_performance: dict[str, float]  # [OPTIONAL] Only when DSPy enabled

    # Meta-Evaluator & RL
    failure_analysis: dict[str, Any]
    refinement_guidance: dict[str, str]
    reward_signals: dict[str, float]

    # WEBRL: Curriculum Learning
    curriculum_subtasks: list[dict[str, Any]]  # [OPTIONAL] Only when WEBRL enabled
    needs_subtask_resolution: bool

    # Eureka: Multi-candidate Evolutionary Plans
    candidate_plans: list[CandidatePlan]  # [OPTIONAL] Only when Eureka enabled
    current_plan_index: int
    evolutionary_generation: int
    crossover_guidance: dict[str, Any]

    # GRPO: Reasoning Traces
    reasoning_traces: Annotated[list[ReasoningTrace], add]  # [OPTIONAL] Only when GRPO enabled
    current_reasoning: ReasoningTrace | None

    # DPO: Preference Learning
    preference_pairs: Annotated[list[PreferencePair], add]  # [OPTIONAL] Only when DPO enabled

    # Quiet-STaR: Self-Evaluation
    self_evaluations: Annotated[list[SelfEvaluation], add]  # [OPTIONAL] Only when Quiet-STaR enabled
    last_self_evaluation: SelfEvaluation | None

    # ========================================================================
    # DEBUG CHAIN (PiML)
    # Specialized debugging with 3-attempt loop before escalation
    # ========================================================================
    debug_attempt: int  # Current debug attempt (0-3)
    debug_escalate: bool  # True when max attempts reached, escalate to planner
    debug_diagnosis: str | None  # Error summary for planner when escalating
    debug_guidance: str | None  # Fix guidance for developer
    debug_history: list[dict[str, Any]]  # History of debug attempts

    # ========================================================================
    # ABLATION VALIDATION (MLE-STAR)
    # A/B testing for change validation
    # ========================================================================
    ablation_baseline_code: str | None  # Baseline code for A/B comparison
    ablation_baseline_score: float | None  # Baseline score for A/B comparison
    ablation_accepted: bool | None  # True if change improved score
    ablation_validation_reason: str | None  # Reason for acceptance/rejection
    ablation_rejection_count: int  # Counter for loop protection (max 3 rejections)

    # ========================================================================
    # METADATA
    # ========================================================================
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
        # Expected row counts
        expected_train_rows=None,
        expected_test_rows=None,
        # Data Format Discovery
        data_format_type=None,
        parsing_info=None,
        data_loading_code=None,
        # Domain Detection
        domain_detected=None,
        domain_confidence=0.0,
        # Contracts (Source of Truth) - PR1
        metric_contract=None,
        canonical_contract=None,
        submission_contract=None,
        eval_fidelity=None,
        data_usage=None,
        # MLE-STAR Registries - PR2
        code_registry=None,
        ablation_history=None,
        model_registry=None,
        robustness_checks=None,
        # Artifact Index - PR3
        artifact_index=None,
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
        failed_component_names=[],
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
        # Debug Chain (PiML)
        debug_attempt=0,
        debug_escalate=False,
        debug_diagnosis=None,
        debug_guidance=None,
        debug_history=[],
        # Ablation Validation (MLE-STAR)
        ablation_baseline_code=None,
        ablation_baseline_score=None,
        ablation_accepted=None,
        ablation_validation_reason=None,
        ablation_rejection_count=0,
        # Metadata
        workflow_start_time=now,
        last_updated=now,
    )
