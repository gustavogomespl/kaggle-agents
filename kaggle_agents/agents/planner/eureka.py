"""Eureka multi-candidate evolutionary planning for the planner agent."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import KaggleState

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

    for i, strategy in enumerate(strategies[:n_candidates]):
        print(f"   - Generating {strategy['name']} plan...")

        # Generate plan with strategy-specific modifications
        plan = _generate_plan_with_strategy(
            state, sota_analysis, strategy, create_fallback_plan_fn, coerce_components_fn
        )

        # Apply hyperparameter mutation for variant strategies
        if strategy.get("inherit_from_best") and current_iteration >= 2:
            plan = mutate_plan_hyperparameters(plan, state)

        # Evaluate fitness
        fitness = evaluate_plan_fitness(plan, state)

        candidate_plans.append((plan, strategy["name"], fitness))
        print(f"     Fitness: {fitness:.3f}")

    # Sort by fitness (highest first)
    candidate_plans.sort(key=lambda x: x[2], reverse=True)

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
                "prompt_modifier": "Use proven audio models: mel-spectrogram + EfficientNet, simple CNN. Focus on stable preprocessing.",
                "model_preference": ["efficientnet_audio", "resnet_audio", "simple_cnn"],
            },
            {
                "name": "aggressive",
                "prompt_modifier": "Use SOTA audio: AST (Audio Spectrogram Transformer), wav2vec2, PANN. Heavy augmentation (SpecAugment, mixup).",
                "model_preference": ["ast", "wav2vec2", "pann", "whisper"],
            },
            {
                "name": "balanced",
                "prompt_modifier": "Use mid-size audio models: EfficientNet-B2 on mel-specs. Balance preprocessing with model complexity.",
                "model_preference": ["efficientnet_b2_audio", "sed_model", "cnn_transformer"],
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

    elif strategy["name"] == "aggressive":
        # Boost feature engineering and ensemble components
        for comp in plan:
            if comp.component_type in ["feature_engineering", "ensemble"]:
                comp.estimated_impact = min(comp.estimated_impact * 1.3, 1.0)

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

    # Get best hyperparameters from previous iterations
    best_hyperparams = {}
    if iteration_memory:
        for memory in iteration_memory:
            if hasattr(memory, "best_hyperparameters") and memory.best_hyperparameters:
                best_hyperparams.update(memory.best_hyperparameters)

    mutated_plan = []
    for comp in plan:
        # Only mutate model components with some probability
        if comp.component_type == "model" and random.random() < mutation_rate:
            # Create a mutated version
            mutated_name = f"{comp.name}_hp_variant"

            # Define mutation suggestions for common hyperparameters
            mutation_hints = _get_hyperparameter_mutations(comp.name)

            mutated_comp = AblationComponent(
                name=mutated_name,
                component_type=comp.component_type,
                description=f"{getattr(comp, 'description', '')} (hyperparameter variant: {mutation_hints})",
                code=comp.code,
                estimated_impact=comp.estimated_impact * 0.95,  # Slight uncertainty penalty
                dependencies=comp.dependencies,
                ablatable=comp.ablatable,
            )
            mutated_plan.append(mutated_comp)
        else:
            mutated_plan.append(comp)

    return mutated_plan


def _get_hyperparameter_mutations(model_name: str) -> str:
    """Get suggested hyperparameter mutations for a model type."""
    model_lower = model_name.lower()

    if "lightgbm" in model_lower or "lgb" in model_lower:
        mutations = [
            "learning_rate: [0.01, 0.03, 0.05]",
            "num_leaves: [31, 63, 127]",
            "max_depth: [5, 7, 9]",
            "reg_alpha: [0, 0.1, 0.5]",
        ]
        return random.choice(mutations)

    if "xgboost" in model_lower or "xgb" in model_lower:
        mutations = [
            "learning_rate: [0.01, 0.05, 0.1]",
            "max_depth: [4, 6, 8]",
            "subsample: [0.7, 0.8, 0.9]",
            "colsample_bytree: [0.7, 0.8, 0.9]",
        ]
        return random.choice(mutations)

    if "catboost" in model_lower:
        mutations = [
            "learning_rate: [0.01, 0.03, 0.1]",
            "depth: [4, 6, 8]",
            "l2_leaf_reg: [1, 3, 5]",
        ]
        return random.choice(mutations)

    if "neural" in model_lower or "mlp" in model_lower or "tabnet" in model_lower:
        mutations = [
            "learning_rate: [1e-4, 1e-3, 1e-2]",
            "dropout: [0.1, 0.2, 0.3]",
            "hidden_dims: [128, 256, 512]",
        ]
        return random.choice(mutations)

    return "try different hyperparameter values"


def evaluate_plan_fitness(
    plan: list,
    state: KaggleState,
) -> float:
    """
    Eureka-style: Evaluate fitness of a plan based on history.

    Considers:
    - Past success/failure patterns from iteration_memory
    - Component type diversity
    - Estimated impact scores
    - Crossover guidance from meta-evaluator

    Args:
        plan: Candidate plan to evaluate
        state: Current workflow state

    Returns:
        Fitness score (0-1)
    """
    score = 0.0
    iteration_memory = state.get("iteration_memory", [])
    crossover_guidance = state.get("crossover_guidance", {})

    # 1. Historical success/failure (40% weight)
    historical_score = 0.0
    for comp in plan:
        for memory in iteration_memory:
            # Reward components similar to what worked
            if comp.component_type in memory.what_worked:
                historical_score += 0.2
            # Penalize components similar to what failed
            if comp.component_type in memory.what_failed:
                historical_score -= 0.1

    historical_score = max(0, min(historical_score, 1.0))
    score += 0.4 * historical_score

    # 2. Diversity bonus (20% weight)
    unique_types = len(set(c.component_type for c in plan))
    diversity_score = min(unique_types / 4.0, 1.0)
    score += 0.2 * diversity_score

    # 3. Estimated impact (25% weight)
    if plan:
        avg_impact = sum(c.estimated_impact for c in plan) / len(plan)
        score += 0.25 * avg_impact

    # 4. Crossover guidance alignment (15% weight)
    if crossover_guidance:
        preserve_components = crossover_guidance.get("preserve_components", [])
        avoid_components = crossover_guidance.get("avoid_components", [])

        alignment_score = 0.0
        for comp in plan:
            if comp.component_type in preserve_components:
                alignment_score += 0.3
            if comp.component_type in avoid_components:
                alignment_score -= 0.2

        alignment_score = max(0, min(alignment_score, 1.0))
        score += 0.15 * alignment_score

    return min(max(score, 0.0), 1.0)


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

    best_plan, strategy, fitness = candidate_plans[0]
    print(f"\n   Eureka: Selected '{strategy}' plan (fitness: {fitness:.3f})")

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
