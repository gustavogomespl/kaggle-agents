"""Eureka multi-candidate evolutionary planning for the planner agent."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import KaggleState

from .plan_refinement import (
    _component_evidence,
    _finite_score,
    _metric_direction,
)
from .strategies import (
    EXTENDED_STRATEGIES_CV,
    EXTENDED_STRATEGIES_NLP,
    EXTENDED_STRATEGIES_TABULAR,
)


def generate_multiple_plans(
    state: KaggleState,
    sota_analysis: dict[str, Any],
    n_candidates: int,
    create_fallback_plan_fn,
    coerce_components_fn,
) -> list[tuple[list, str, float]]:
    """
    Eureka-style: Generate multiple candidate plans with different strategies.

    Args:
        state: Current workflow state
        sota_analysis: SOTA analysis results
        n_candidates: Number of candidate plans to generate
        create_fallback_plan_fn: Function to create fallback plans
        coerce_components_fn: Function to coerce components

    Returns:
        List of (plan, strategy, fitness_score) tuples
    """
    print(f"\n   Eureka: Generating {n_candidates} candidate plans...")

    # Domain-aware strategy selection
    domain = state.get("domain_detected", "tabular")

    # Define domain groups for cleaner matching
    IMAGE_CLASSIFICATION_DOMAINS = {
        "image_classification",
        "image_regression",
        "computer_vision",
        "image",
    }
    IMAGE_SEGMENTATION_DOMAINS = {
        "image_segmentation",
        "image_to_image",
        "object_detection",
    }
    NLP_DOMAINS = {
        "nlp",
        "text_classification",
        "text_regression",
        "seq_to_seq",
    }
    AUDIO_DOMAINS = {
        "audio_classification",
        "audio_regression",
    }

    strategies = _get_domain_strategies(
        domain,
        IMAGE_CLASSIFICATION_DOMAINS,
        IMAGE_SEGMENTATION_DOMAINS,
        NLP_DOMAINS,
        AUDIO_DOMAINS,
    )

    print(f"   📊 Domain: {domain}, using domain-specific strategies")

    # Get current iteration to determine if we should use extended strategies
    current_iteration = state.get("current_iteration", 0)

    # After iteration 2, add extended strategies for more diversity
    if current_iteration >= 2:
        # Select domain-appropriate extended strategies
        if domain in IMAGE_CLASSIFICATION_DOMAINS or domain in IMAGE_SEGMENTATION_DOMAINS:
            extended = EXTENDED_STRATEGIES_CV
        elif domain in NLP_DOMAINS:
            extended = EXTENDED_STRATEGIES_NLP
        else:
            extended = EXTENDED_STRATEGIES_TABULAR

        # Add extended strategies to the base strategies
        extended_list = [
            extended.get("feature_engineering_heavy"),
            extended.get("neural_exploration"),
            extended.get("hyperparameter_variant"),
        ]
        strategies.extend([s for s in extended_list if s is not None])
        print(f"   🔄 Iteration {current_iteration}: Added extended strategies for diversity")

    # Dynamically adjust n_candidates based on iteration
    if current_iteration >= 3:
        n_candidates = min(5, len(strategies))
    else:
        n_candidates = min(n_candidates, len(strategies))

    candidate_plans = []

    for strategy in strategies[:n_candidates]:
        print(f"   - Generating {strategy['name']} plan...")

        # Generate plan with strategy-specific modifications
        plan = _generate_plan_with_strategy(
            state, sota_analysis, strategy, create_fallback_plan_fn, coerce_components_fn
        )

        # Apply hyperparameter mutation for variant strategies
        if strategy.get("inherit_from_best") and current_iteration >= 2:
            plan = mutate_plan_hyperparameters(plan, state)

        # Store only a real local measurement. Self-declared impact estimates
        # never enter fitness or selection.
        evidence_kind, fitness, evidence_count = _plan_fitness_evidence(plan, state)

        candidate_plans.append((plan, strategy["name"], fitness))
        if evidence_kind == "actual_impact":
            print(
                "     Fitness evidence: "
                f"{evidence_count} measured actual-impact value(s), "
                f"mean={fitness:.3f}"
            )
        elif evidence_kind == "trusted_oof":
            print(
                "     Fitness evidence: "
                f"{evidence_count} trusted canonical OOF score(s), "
                f"mean={fitness:.3f}, direction={_metric_direction(state)}"
            )
        else:
            print("     Fitness evidence: unmeasured exploration candidate")

    # Stable selection: trusted local evidence first, then deterministic diversity.
    # Python's stable sort preserves strategy generation order for exact ties.
    candidate_plans.sort(
        key=lambda item: _plan_selection_key(item[0], state),
        reverse=True,
    )

    return candidate_plans


def _get_domain_strategies(
    domain: str,
    image_classification_domains: set,
    image_segmentation_domains: set,
    nlp_domains: set,
    audio_domains: set,
) -> list[dict[str, Any]]:
    """Get domain-specific strategies for Eureka planning."""
    if domain in image_classification_domains:
        return [
            {
                "name": "conservative",
                "prompt_modifier": "Use proven CNN architectures: EfficientNet-B0/B3, ResNet50. Focus on stable training with pretrained ImageNet weights.",
                "model_preference": ["efficientnet_b0", "resnet50", "efficientnet_b3"],
            },
            {
                "name": "aggressive",
                "prompt_modifier": "Use SOTA architectures: ResNet200d, EfficientNet-B5/B7, ConvNeXt. Apply heavy augmentation (Cutmix, Mixup). Full fine-tuning.",
                "model_preference": ["resnet200d", "efficientnet_b5", "convnext", "swin"],
            },
            {
                "name": "balanced",
                "prompt_modifier": "Use mid-size models: EfficientNet-B3/B4, ResNet101. Balance speed with accuracy. Use TTA for inference.",
                "model_preference": ["efficientnet_b3", "efficientnet_b4", "resnet101"],
            },
        ]
    if domain in image_segmentation_domains:
        return [
            {
                "name": "conservative",
                "prompt_modifier": "Use proven segmentation architectures: U-Net with ResNet34 encoder, FPN. Focus on stable training with pretrained encoders.",
                "model_preference": ["unet_resnet34", "fpn", "deeplabv3"],
            },
            {
                "name": "aggressive",
                "prompt_modifier": "Use SOTA segmentation: U-Net++ with EfficientNet-B5 encoder, HRNet, Mask R-CNN. Apply heavy augmentation.",
                "model_preference": ["unet_plusplus", "hrnet", "mask_rcnn", "segformer"],
            },
            {
                "name": "balanced",
                "prompt_modifier": "Use mid-size segmentation models: U-Net with EfficientNet-B3 encoder. Balance speed with accuracy.",
                "model_preference": ["unet_effb3", "deeplabv3_plus", "pan"],
            },
        ]
    if domain in nlp_domains:
        return [
            {
                "name": "conservative",
                "prompt_modifier": "Use proven NLP models: DistilBERT, RoBERTa-base. Focus on stable training.",
                "model_preference": ["distilbert", "roberta_base", "bert_base"],
            },
            {
                "name": "aggressive",
                "prompt_modifier": "Use large models: DeBERTa-v3, RoBERTa-large. Apply advanced techniques like MLM pretraining.",
                "model_preference": ["deberta_v3", "roberta_large", "longformer"],
            },
            {
                "name": "balanced",
                "prompt_modifier": "Mix efficient models with strong performance. Use ensemble of BERT variants.",
                "model_preference": ["roberta_base", "deberta", "albert"],
            },
        ]
    if domain in audio_domains:
        return [
            {
                "name": "conservative",
                "prompt_modifier": (
                    "Compare summary features with a compact time-frequency "
                    "model using data-derived preprocessing."
                ),
                "model_preference": [
                    "summary_feature_baseline",
                    "compact_spectrogram_model",
                ],
            },
            {
                "name": "aggressive",
                "prompt_modifier": (
                    "Evaluate an installed pretrained audio candidate only when "
                    "compatible with the observed data and runtime budget."
                ),
                "model_preference": ["pretrained_audio_candidate"],
            },
            {
                "name": "balanced",
                "prompt_modifier": (
                    "Select capacity from observed feature shape, sample count, "
                    "installed resources, and trusted CV."
                ),
                "model_preference": ["budget_matched_audio_model"],
            },
        ]
    # TABULAR domain strategies (default)
    return [
        {
            "name": "conservative",
            "prompt_modifier": "Focus on proven, reliable approaches. Use well-established models like XGBoost, LightGBM. Prioritize stability over novelty.",
            "model_preference": ["xgboost", "lightgbm", "random_forest"],
        },
        {
            "name": "aggressive",
            "prompt_modifier": "Focus on innovative approaches. Prioritize novel feature engineering, creative ensembles, and cutting-edge techniques.",
            "model_preference": ["catboost", "neural_network", "stacking"],
        },
        {
            "name": "balanced",
            "prompt_modifier": "Mix proven models with creative features. Balance stability with innovation.",
            "model_preference": ["xgboost", "lightgbm", "catboost"],
        },
    ]


def _generate_plan_with_strategy(
    state: KaggleState,
    sota_analysis: dict[str, Any],
    strategy: dict[str, Any],
    create_fallback_plan_fn,
    coerce_components_fn,
) -> list:
    """
    Generate a single plan with a specific strategy.

    Args:
        state: Current workflow state
        sota_analysis: SOTA analysis results
        strategy: Strategy configuration
        create_fallback_plan_fn: Function to create fallback plans
        coerce_components_fn: Function to coerce components

    Returns:
        List of ablation components
    """
    domain = state.get("domain_detected", "tabular")

    # Use fallback plan generation with strategy bias
    plan = create_fallback_plan_fn(domain, sota_analysis, state=state)
    plan = coerce_components_fn(plan)

    # Modify plan based on strategy
    if strategy["name"] == "conservative":
        # Filter to keep only well-established models
        plan = [
            c
            for c in plan
            if any(
                m in c.name.lower()
                for m in ["xgboost", "lightgbm", "random", "logistic", "baseline"]
            )
        ] or plan[:2]

    return plan


def mutate_plan_hyperparameters(
    plan: list,
    state: KaggleState,
    mutation_rate: float = 0.3,
) -> list:
    """
    Apply hyperparameter mutations to plan components.

    Eureka-style: Introduce controlled randomness to explore hyperparameter space.

    Args:
        plan: Original plan components
        state: Current workflow state (for accessing best hyperparameters)
        mutation_rate: Probability of mutating each component

    Returns:
        Plan with mutated hyperparameters
    """
    from ...core.state import AblationComponent

    iteration_memory = state.get("iteration_memory", [])

    # Record whether trusted local configurations exist. Mutation hints never
    # inject generic numeric recipes; all variants must survive canonical CV.
    has_local_hyperparameters = False
    if iteration_memory:
        for memory in iteration_memory:
            used = getattr(memory, "hyperparameters_used", None)
            if used:
                has_local_hyperparameters = True

    mutated_plan = []
    for comp in plan:
        # Only mutate model components with some probability
        rng = random.Random(
            f"{state.get('seed', 42)}:{state.get('current_iteration', 0)}:{comp.name}"
        )
        if comp.component_type == "model" and rng.random() < mutation_rate:
            # Create a mutated version
            mutated_name = f"{comp.name}_hp_variant"

            # Define mutation suggestions for common hyperparameters
            mutation_hints = _get_hyperparameter_mutations(
                comp.name,
                has_local_hyperparameters=has_local_hyperparameters,
            )

            mutated_comp = AblationComponent(
                name=mutated_name,
                component_type=comp.component_type,
                code=f"{comp.code}\n# VALIDATION-GATED VARIANT: {mutation_hints}",
                estimated_impact=comp.estimated_impact,
                tested=False,
                actual_impact=None,
                external_source_ids=list(
                    getattr(comp, "external_source_ids", [])
                ),
            )
            mutated_plan.append(mutated_comp)
        else:
            mutated_plan.append(comp)

    return mutated_plan


def _get_hyperparameter_mutations(
    model_name: str,
    *,
    has_local_hyperparameters: bool = False,
) -> str:
    """Describe a validation-gated mutation without prescribing fixed values."""
    local_basis = (
        "the best locally measured configuration"
        if has_local_hyperparameters
        else "a budget-feasible baseline derived from the observed data"
    )
    return (
        f"create one bounded {model_name} variant around {local_basis}; change one "
        "capacity or regularization dimension, measure it on the canonical folds, "
        "and retain it only when trusted CV improves"
    )


def evaluate_plan_fitness(
    plan: list,
    state: KaggleState,
) -> float:
    """
    Return the auditable local evidence value available for a plan.

    Tested finite ``actual_impact`` values take precedence. When they are
    absent, finite independently recomputed canonical OOF scores can provide
    evidence if their structural/artifact/robustness gates are eligible and
    metric direction is known. Unmeasured plans return 0.0 and are tie-broken
    deterministically. ``estimated_impact`` is intentionally ignored.

    Args:
        plan: Candidate plan to evaluate
        state: Current workflow state

    Returns:
        Mean local evidence value, or 0.0 when no measurement exists.
    """
    _kind, value, _count = _plan_fitness_evidence(plan, state)
    return value


def _trusted_actual_impacts(plan: list, state: KaggleState) -> list[float]:
    """Collect finite impacts explicitly marked tested, including prior names."""
    approvals = state.get("robustness_approved_components")
    component_results = state.get("component_results")

    def is_explicitly_rejected(name: str) -> bool:
        return isinstance(approvals, dict) and approvals.get(name) is False

    def structurally_failed(name: str) -> bool:
        if not isinstance(component_results, dict) or name not in component_results:
            return False
        result = component_results[name]
        success = result.get("success") if isinstance(result, dict) else getattr(
            result, "success", None
        )
        return success is False

    prior_by_name = {
        component.name: _finite_score(component.actual_impact)
        for component in (state.get("ablation_plan", []) or [])
        if getattr(component, "tested", False) is True
        and _finite_score(getattr(component, "actual_impact", None)) is not None
        and not is_explicitly_rejected(str(component.name))
        and not structurally_failed(str(component.name))
    }
    impacts: list[float] = []
    for component in plan:
        name = str(getattr(component, "name", ""))
        if is_explicitly_rejected(name) or structurally_failed(name):
            continue
        impact = None
        if getattr(component, "tested", False) is True:
            impact = _finite_score(getattr(component, "actual_impact", None))
        if impact is None:
            impact = prior_by_name.get(name)
        if impact is not None:
            impacts.append(impact)
    return impacts


def _trusted_oof_scores(plan: list, state: KaggleState) -> list[float]:
    """Collect eligible raw OOF scores when their direction is known."""
    if _metric_direction(state) == "unknown":
        return []
    scores = []
    for component in plan:
        evidence = _component_evidence(state, component)
        if evidence["selection_eligible"]:
            scores.append(float(evidence["trusted_oof_score"]))
    return scores


def _plan_fitness_evidence(
    plan: list,
    state: KaggleState,
) -> tuple[str, float, int]:
    """Return evidence kind, raw mean value, and measurement count."""
    impacts = _trusted_actual_impacts(plan, state)
    if impacts:
        return "actual_impact", sum(impacts) / len(impacts), len(impacts)

    scores = _trusted_oof_scores(plan, state)
    if scores:
        return "trusted_oof", sum(scores) / len(scores), len(scores)
    return "unmeasured", 0.0, 0


def _plan_selection_key(
    plan: list,
    state: KaggleState,
) -> tuple[int, float, int, int]:
    """Build an evidence-first, deterministic key for candidate selection."""
    evidence_kind, raw_value, count = _plan_fitness_evidence(plan, state)
    if evidence_kind == "actual_impact":
        evidence_rank = 2
        directed_value = raw_value
    elif evidence_kind == "trusted_oof":
        evidence_rank = 1
        directed_value = (
            -raw_value if _metric_direction(state) == "minimize" else raw_value
        )
    else:
        evidence_rank = 0
        directed_value = 0.0
    unique_types = len({component.component_type for component in plan})
    return evidence_rank, directed_value, count, unique_types


def select_best_plan(
    candidate_plans: list[tuple[list, str, float]],
) -> tuple[list, str]:
    """
    Select the best plan from candidates.

    Args:
        candidate_plans: List of (plan, strategy, fitness) tuples

    Returns:
        Tuple of (best_plan, strategy_name)
    """
    if not candidate_plans:
        return [], "none"

    best_plan, strategy, _fitness = candidate_plans[0]
    print(
        f"\n   Eureka: Selected '{strategy}' plan "
        "using trusted evidence when available, otherwise deterministic exploration"
    )

    return best_plan, strategy


def generate_with_eureka(
    state: KaggleState,
    sota_analysis: dict[str, Any],
    n_candidates: int,
    create_fallback_plan_fn,
    coerce_components_fn,
) -> dict[str, Any]:
    """
    Eureka-style plan generation with multiple candidates.

    Args:
        state: Current workflow state
        sota_analysis: SOTA analysis results
        n_candidates: Number of candidates to generate
        create_fallback_plan_fn: Function to create fallback plans
        coerce_components_fn: Function to coerce components

    Returns:
        State updates with plan and candidate info
    """
    from ...core.state import CandidatePlan

    print("\n   Eureka: Multi-candidate evolutionary planning...")

    # Generate multiple candidate plans
    candidate_plans = generate_multiple_plans(
        state, sota_analysis, n_candidates, create_fallback_plan_fn, coerce_components_fn
    )

    # Select the best plan
    best_plan, strategy = select_best_plan(candidate_plans)

    # Store all candidates for potential crossover in next iteration
    stored_candidates = [
        CandidatePlan(
            components=plan,
            strategy=strat,
            fitness_score=fitness,
            generation=state.get("evolutionary_generation", 0) + 1,
        )
        for plan, strat, fitness in candidate_plans
    ]

    return {
        "ablation_plan": best_plan,
        "candidate_plans": stored_candidates,
        "current_plan_index": 0,
        "evolutionary_generation": state.get("evolutionary_generation", 0) + 1,
        "optimization_strategy": f"eureka_{strategy}",
    }
