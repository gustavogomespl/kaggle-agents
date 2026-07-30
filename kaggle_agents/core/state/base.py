"""
Main KaggleState TypedDict and initialization.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from operator import add
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from ..config import get_run_seed
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


def merge_dict(existing: dict, new: dict) -> dict:
    """Merge dictionaries, with new values overwriting existing ones."""
    return {**existing, **new}


def _sota_identity(solution: SOTASolution | dict[str, Any]) -> tuple[str | None, str]:
    """Return stable source and content identities for one retrieved solution."""
    if isinstance(solution, dict):
        source = solution.get("source")
        source_sha256 = solution.get("source_sha256")
        payload = {
            "title": solution.get("title", ""),
            "code_snippets": solution.get("code_snippets", []),
            "strategies": solution.get("strategies", []),
            "models_used": solution.get("models_used", []),
            "feature_engineering": solution.get(
                "feature_engineering",
                [],
            ),
            "ensemble_approach": solution.get("ensemble_approach"),
        }
    else:
        source = getattr(solution, "source", None)
        source_sha256 = getattr(solution, "source_sha256", None)
        payload = {
            "title": getattr(solution, "title", ""),
            "code_snippets": getattr(solution, "code_snippets", []),
            "strategies": getattr(solution, "strategies", []),
            "models_used": getattr(solution, "models_used", []),
            "feature_engineering": getattr(
                solution,
                "feature_engineering",
                [],
            ),
            "ensemble_approach": getattr(solution, "ensemble_approach", None),
        }

    normalized_source = str(source or "").strip().casefold().rstrip("/") or None
    normalized_sha = str(source_sha256 or "").strip().casefold()
    if len(normalized_sha) == 64:
        return normalized_source, normalized_sha

    # Synthetic/domain fallbacks have no downloaded source hash. Hash only
    # stable content fields so object identity and mutable popularity metadata
    # cannot create duplicates.
    serialized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        ensure_ascii=True,
    ).encode("utf-8")
    return normalized_source, hashlib.sha256(serialized).hexdigest()


def merge_sota_solutions(
    existing: list[SOTASolution],
    new: list[SOTASolution],
) -> list[SOTASolution]:
    """
    Merge retrieval attempts with fresh, distinct sources first.

    LangGraph's former append-only reducer left initial top-K results ahead of
    recovery results, so planners that consume a bounded prefix could never see
    the additional sources fetched after stagnation. A source reference or
    complete-source hash identifies duplicates; genuinely distinct earlier
    sources remain available after the fresh batch.
    """
    merged: list[SOTASolution] = []
    seen_sources: set[str] = set()
    seen_hashes: set[str] = set()

    for solution in [*(new or []), *(existing or [])]:
        source_key, hash_key = _sota_identity(solution)
        if (source_key and source_key in seen_sources) or hash_key in seen_hashes:
            continue
        merged.append(solution)
        if source_key:
            seen_sources.add(source_key)
        seen_hashes.add(hash_key)

    return merged


class KaggleState(TypedDict):
    """
    Unified state for the entire Kaggle agent workflow.

    This state flows through all nodes in the LangGraph workflow,
    accumulating data and enabling agents to make informed decisions.
    """

    # Competition Context
    run_id: str
    competition_info: CompetitionInfo
    working_directory: str
    run_mode: str  # e.g. "kaggle" | "mlebench"
    mlebench_cache_path: str | None  # Host-only grader/cache root
    objective: str  # e.g. "top20" | "fixed_budget_public_cv"
    timeout_per_component: int | None
    # Hard wall-clock budget for the agent, and the absolute epoch deadline it
    # implies. Component timeouts alone cannot bound a run's cost.
    run_wall_clock_budget_s: int | None
    run_deadline_ts: float | None
    enable_checkpoint_recovery: bool
    cv_folds: int | None
    random_seed: int
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
    target_cols: list[str]
    target_type: str | None
    data_files: dict[str, Any]

    # Expected row counts (for OOF alignment across models)
    expected_train_rows: int | None  # Expected rows in train set
    expected_test_rows: int | None  # Expected rows in test set
    class_order: list[str] | None
    train_rec_ids: list[Any]
    test_rec_ids: list[Any]
    train_file_paths: list[str]
    test_file_paths: list[str]
    cv_folds_used: bool

    # Data Format Discovery (for non-standard formats)
    data_format_type: str | None  # "traditional", "generated", "custom", or "unknown"
    parsing_info: dict[str, Any] | None  # LLM-generated parsing instructions
    data_loading_code: str | None  # Python code to load non-standard data
    submission_format_info: dict[str, Any] | None
    precomputed_features_info: dict[str, Any] | None
    id_extension_hint: str | None

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
    canonical_data_prepared: bool
    canonical_data_skipped_reason: str | None
    canonical_data_error: str | None
    canonical_dir: str | None
    canonical_train_ids_path: str | None
    canonical_y_path: str | None
    canonical_folds_path: str | None
    canonical_feature_cols_path: str | None
    canonical_test_ids_path: str | None
    canonical_metadata: dict[str, Any] | None
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
    sota_solutions: Annotated[list[SOTASolution], merge_sota_solutions]
    search_queries_used: Annotated[list[str], add]
    sota_retrieval_k: int
    last_sota_update_iteration: int | None
    search_attempted: bool
    search_eligible_retrieved: bool
    search_last_attempt_eligible_retrieved: bool
    search_last_attempt_reason: str | None
    search_eligibility_reason: str | None
    search_downstream_gain: float | None
    search_downstream_gain_status: str
    # Backward-compatible alias for search_eligible_retrieved. This is not
    # evidence that retrieval caused a downstream score improvement.
    search_effective: bool
    search_failure_reason: str | None
    sota_search_triggered: bool
    sota_search_results: dict[str, Any]

    # ========================================================================
    # TELEMETRY & AUDIT (paper instrumentation)
    # telemetry_events: append-only event log (see utils/telemetry.py)
    # search_audit: every retrieved source + contamination-filter decision
    # ========================================================================
    telemetry_events: Annotated[list[dict[str, Any]], add]
    search_audit: Annotated[list[dict[str, Any]], add]

    # Planning Phase
    ablation_plan: list[AblationComponent]
    current_component_index: int
    optimization_strategy: str
    previous_plan_hashes: list[int]
    force_refinement: bool
    force_eureka_planning: bool

    # Development Phase
    development_results: Annotated[list[DevelopmentResult], add]
    # Explicit maps replace dynamic ``oof_available_<name>`` and
    # ``component_result_<name>`` keys, which LangGraph otherwise discards.
    oof_availability: dict[str, bool]
    component_results: dict[str, DevelopmentResult]
    # Independently recomputed from canonical OOF artifacts. Generated stdout
    # is never a score source for this map.
    trusted_component_scores: dict[str, float]
    current_code: str
    code_retry_count: int
    code_attempts: Annotated[list[CodeAttempt], add]

    # Validation Phase
    validation_results: Annotated[list[ValidationResult], add]
    overall_validation_score: float | None
    critical_issues: Annotated[list[str], add]
    robustness_passed: bool | None
    robustness_abstained: bool
    # A scalar robustness result is insufficient when several model artifacts
    # are developed before the single robustness node runs. Ensemble
    # eligibility therefore requires an explicit decision for every component.
    robustness_approved_components: dict[str, bool]
    robustness_failure_details: dict[str, Any]
    robustness_gate_action: str | None
    robustness_recovery_count: int
    max_robustness_recoveries: int
    current_candidate_valid: bool
    workflow_valid: bool

    # Ensemble Phase
    ensemble_strategy: str | None
    ensemble_weights: dict[str, float]
    # Host-recomputed OOF score bound to the exact ensemble submission bytes.
    # These fields are intentionally separate from generic performance fields:
    # MLE-bench submission reporting must be artifact-provenanced.
    ensemble_oof_score: float | None
    ensemble_submission_sha256: str | None
    ensemble_submission_owner: str | None
    ensemble_score_source: str | None

    # Submission Phase
    submissions: Annotated[list[SubmissionResult], add]
    best_score: float  # [DERIVABLE] from metric_contract.best_score
    target_percentile: float
    best_single_model_score: float | None  # [DERIVABLE] from model_registry.get_best_overall()
    best_single_model_name: str | None  # [DERIVABLE] from model_registry.get_best_overall()
    baseline_cv_score: float | None
    accepted_submission_path: str | None
    accepted_submission_sha256: str | None
    accepted_submission_snapshot_path: str | None
    accepted_submission_cv_score: float | None
    accepted_submission_score_owner: str | None
    accepted_submission_score_source: str | None
    best_candidate_submission_snapshot_path: str | None
    best_candidate_submission_sha256: str | None
    best_candidate_submission_component_name: str | None
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
    failed_component_names: Annotated[
        list[str], add
    ]  # Component names that failed (for planner to avoid)
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
    stagnation_detection: dict[str, Any]
    trigger_debug_loop: bool
    debug_target_model: str | None
    debug_hints: list[str]
    performance_gap: float | None

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
    self_evaluations: Annotated[
        list[SelfEvaluation], add
    ]  # [OPTIONAL] Only when Quiet-STaR enabled
    last_self_evaluation: SelfEvaluation | None

    # ========================================================================
    # METADATA
    # ========================================================================
    # Metadata
    workflow_start_time: datetime
    last_updated: datetime


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
        run_id=str(uuid4()),
        competition_info=CompetitionInfo(
            name=competition_name,
            description="",
            evaluation_metric="",
            problem_type="",
            identity_aliases=[competition_name] if competition_name else [],
            identity_alias_evidence=(
                [
                    {
                        "alias": competition_name,
                        "source": "competition_slug",
                    }
                ]
                if competition_name
                else []
            ),
        ),
        working_directory=working_dir,
        run_mode="kaggle",
        mlebench_cache_path=None,
        objective="top20",
        timeout_per_component=None,
        run_wall_clock_budget_s=None,
        run_deadline_ts=None,
        enable_checkpoint_recovery=True,
        cv_folds=None,
        random_seed=get_run_seed(),
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
        target_cols=[],
        target_type=None,
        data_files={},
        # Expected row counts
        expected_train_rows=None,
        expected_test_rows=None,
        class_order=None,
        train_rec_ids=[],
        test_rec_ids=[],
        train_file_paths=[],
        test_file_paths=[],
        cv_folds_used=False,
        # Data Format Discovery
        data_format_type=None,
        parsing_info=None,
        data_loading_code=None,
        submission_format_info=None,
        precomputed_features_info=None,
        id_extension_hint=None,
        # Domain Detection
        domain_detected=None,
        domain_confidence=0.0,
        # Contracts (Source of Truth) - PR1
        metric_contract=None,
        canonical_contract=None,
        canonical_data_prepared=False,
        canonical_data_skipped_reason=None,
        canonical_data_error=None,
        canonical_dir=None,
        canonical_train_ids_path=None,
        canonical_y_path=None,
        canonical_folds_path=None,
        canonical_feature_cols_path=None,
        canonical_test_ids_path=None,
        canonical_metadata=None,
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
        sota_retrieval_k=0,
        last_sota_update_iteration=None,
        search_attempted=False,
        search_eligible_retrieved=False,
        search_last_attempt_eligible_retrieved=False,
        search_last_attempt_reason=None,
        search_eligibility_reason=None,
        search_downstream_gain=None,
        search_downstream_gain_status="not_applicable_not_attempted",
        search_effective=False,
        search_failure_reason=None,
        sota_search_triggered=False,
        sota_search_results={},
        # Telemetry & Audit
        telemetry_events=[],
        search_audit=[],
        # Planning Phase
        ablation_plan=[],
        current_component_index=0,
        optimization_strategy="",
        previous_plan_hashes=[],
        force_refinement=False,
        force_eureka_planning=False,
        # Development Phase
        development_results=[],
        oof_availability={},
        component_results={},
        trusted_component_scores={},
        current_code="",
        code_retry_count=0,
        code_attempts=[],
        # Validation Phase
        validation_results=[],
        overall_validation_score=None,
        critical_issues=[],
        robustness_passed=None,
        robustness_abstained=False,
        robustness_approved_components={},
        robustness_failure_details={},
        robustness_gate_action=None,
        robustness_recovery_count=0,
        max_robustness_recoveries=1,
        current_candidate_valid=True,
        workflow_valid=True,
        # Ensemble Phase
        ensemble_strategy=None,
        ensemble_weights={},
        ensemble_oof_score=None,
        ensemble_submission_sha256=None,
        ensemble_submission_owner=None,
        ensemble_score_source=None,
        # Submission Phase
        submissions=[],
        best_score=0.0,
        target_percentile=20.0,
        best_single_model_score=None,
        best_single_model_name=None,
        baseline_cv_score=None,
        accepted_submission_path=None,
        accepted_submission_sha256=None,
        accepted_submission_snapshot_path=None,
        accepted_submission_cv_score=None,
        accepted_submission_score_owner=None,
        accepted_submission_score_source=None,
        best_candidate_submission_snapshot_path=None,
        best_candidate_submission_sha256=None,
        best_candidate_submission_component_name=None,
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
        stagnation_detection={},
        trigger_debug_loop=False,
        debug_target_model=None,
        debug_hints=[],
        performance_gap=None,
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
