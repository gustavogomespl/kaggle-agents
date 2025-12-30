"""
Memory helper functions for updating state.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .memory import (
    DataInsights,
    ErrorPatternMemory,
    HyperparameterHistory,
    ModelPerformanceRecord,
)


if TYPE_CHECKING:
    from .base import KaggleState


def update_model_performance(
    state: "KaggleState", record: ModelPerformanceRecord
) -> dict[str, Any]:
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

        # Also update best hyperparameters
        if record.hyperparameters:
            best_hp = dict(state.get("best_hyperparameters_by_model", {}))
            best_hp[model_type] = {
                "hyperparameters": record.hyperparameters,
                "cv_score": record.cv_score,
            }
            updates["best_hyperparameters_by_model"] = best_hp

    return updates


def update_error_memory(
    state: "KaggleState",
    error_type: str,
    error_pattern: str,
    solution: str,
    success: bool,
    affected_model: str | None = None,
    affected_component: str | None = None,
    root_cause: str = "",
    prevention_strategy: str = "",
) -> dict[str, Any]:
    """
    Update error pattern memory with a new solution attempt.

    Args:
        state: Current KaggleState
        error_type: Type of error
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

    # Find existing error pattern
    found_idx = None
    for idx, em in enumerate(existing_memory):
        if isinstance(em, ErrorPatternMemory):
            if em.error_type == error_type and em.error_pattern == error_pattern:
                found_idx = idx
                break
        elif isinstance(em, dict):
            if em.get("error_type") == error_type and em.get("error_pattern", "") == error_pattern:
                found_idx = idx
                break

    if found_idx is not None:
        # Return a delta entry for merge
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
    return {"error_pattern_memory": [new_pattern]}


def aggregate_feature_importance(state: "KaggleState", top_k: int = 20) -> dict[str, Any]:
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


def get_best_hyperparameters(state: "KaggleState", model_type: str) -> dict[str, Any]:
    """
    Get best hyperparameters for a model type based on history.

    Args:
        state: Current KaggleState
        model_type: Type of model

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
    best: dict[str, Any] | None = None
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
    state: "KaggleState",
    model_type: str,
    hyperparameters: dict[str, Any],
    cv_score: float,
    success: bool = True,
    issues: list[str] | None = None,
    data_size: int = 0,
    n_classes: int | None = None,
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
        n_classes: Number of classes
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

        # Find current best score for this model type
        current_best_score = float("-inf")

        for h in history:
            if isinstance(h, HyperparameterHistory):
                if h.model_type == model_type and h.success:
                    current_best_score = max(current_best_score, h.cv_score)
            elif isinstance(h, dict):
                if h.get("model_type") == model_type and h.get("success", True):
                    current_best_score = max(current_best_score, h.get("cv_score", 0))

        # Check cached best_models_by_type
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

        # Check cached best hyperparameters
        cached_hp_entry = best_hp_dict.get(model_type)
        if isinstance(cached_hp_entry, dict) and "cv_score" in cached_hp_entry:
            cached_hp_score = cached_hp_entry.get("cv_score", float("-inf"))
            current_best_score = max(current_best_score, cached_hp_score)

        if cv_score > current_best_score:
            best_hp_dict[model_type] = {
                "hyperparameters": hyperparameters,
                "cv_score": cv_score,
            }
            updates["best_hyperparameters_by_model"] = best_hp_dict

    return updates


def update_strategy_effectiveness(
    state: "KaggleState",
    strategy: str,
    score_improvement: float,
) -> dict[str, Any]:
    """
    Track strategy effectiveness based on score changes.

    Args:
        state: Current KaggleState
        strategy: Strategy name/description
        score_improvement: Score improvement (positive = better)

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
        if strategy in failed:
            failed.remove(strategy)
            updates["failed_strategies"] = failed
    elif score_improvement < 0:
        if strategy not in failed:
            failed.append(strategy)
            updates["failed_strategies"] = failed
        if strategy in successful:
            successful.remove(strategy)
            updates["successful_strategies"] = successful

    # Update effectiveness tracking
    effectiveness = dict(state.get("strategy_effectiveness", {}))

    if strategy in effectiveness:
        existing = effectiveness[strategy]
        if isinstance(existing, dict):
            old_avg = existing.get("average", 0.0)
            count = existing.get("count", 1)
        else:
            old_avg = float(existing)
            count = 1

        new_count = count + 1
        new_avg = old_avg + (score_improvement - old_avg) / new_count
        effectiveness[strategy] = {"average": new_avg, "count": new_count}
    else:
        effectiveness[strategy] = {"average": score_improvement, "count": 1}

    updates["strategy_effectiveness"] = effectiveness

    return updates


def get_memory_summary_for_planning(state: "KaggleState") -> str:
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

    # Data insights
    data_insights = state.get("data_insights")
    if data_insights:
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

    # Error patterns
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
                    insights.append(
                        f"- {em.get('error_type', 'unknown')}: Fix with '{solutions[0]}'"
                    )

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
