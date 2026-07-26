"""Iteration control and performance evaluation nodes for the Kaggle Agents workflow."""

import math
from datetime import datetime
from typing import Any

from ...core.state import KaggleState


def _finite_score(value: Any) -> float | None:
    """Return a finite score without manufacturing a target or observation."""
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def iteration_control_node(state: KaggleState) -> dict[str, Any]:
    """
    Control iteration and check termination conditions.

    Args:
        state: Current state

    Returns:
        State updates with iteration control
    """
    print("\n" + "=" * 60)
    print("= ITERATION CONTROL")
    print("=" * 60)

    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    best_score = state.get("best_score", 0.0)
    # Fallback: when no Kaggle submission has occurred (best_score == 0),
    # use the best available CV score from component development
    if best_score == 0.0:
        best_score = (
            state.get("best_single_model_score")
            or state.get("baseline_cv_score")
            or 0.0
        )
    target_percentile = state.get("target_percentile", 20.0)

    # Increment iteration
    new_iteration = current_iteration + 1

    print(f"\nIteration: {new_iteration}/{max_iterations}")
    print(f"   Best Score: {best_score:.4f}")
    print(f"   Target: Top {target_percentile}%")

    # Check if we should continue
    should_continue = new_iteration < max_iterations

    # Check if goal achieved
    # Note: In real scenario, would check actual percentile
    # For now, continue until max iterations

    termination_reason = None
    if not should_continue:
        termination_reason = "max_iterations_reached"

    # Every continuation creates a new plan. Reset the component cursor before
    # routing back to the planner so the first refinement plan is not mistaken
    # for an already-completed plan.
    updates = {
        "current_iteration": new_iteration,
        "should_continue": should_continue,
        "termination_reason": termination_reason,
        "last_updated": datetime.now(),
    }

    if should_continue:
        print("   🔄 Starting refinement iteration - resetting component index")
        updates["current_component_index"] = 0
        # Reset skip_remaining_components so new iteration can run all components
        updates["skip_remaining_components"] = False

    return updates


def performance_evaluation_node(state: KaggleState) -> dict[str, Any]:
    """
    Evaluate performance and decide if refinement is needed.

    Args:
        state: Current state

    Returns:
        State updates with refinement decision
    """
    print("\n" + "=" * 60)
    print("= PERFORMANCE EVALUATION")
    print("=" * 60)

    current_score = _finite_score(state.get("best_score"))
    # Fallback: when no Kaggle submission has occurred (best_score == 0),
    # use the best available CV score from component development
    if current_score is None or current_score == 0.0:
        current_score = _finite_score(state.get("best_single_model_score"))
    if current_score is None:
        current_score = _finite_score(state.get("baseline_cv_score"))
    if current_score is None:
        current_score = 0.0
    run_mode = str(state.get("run_mode", "")).lower()
    metric_contract = state.get("metric_contract") or {}
    if not isinstance(metric_contract, dict):
        metric_contract = {}

    # A target is optional in MLE-bench. The held-out test score is unavailable
    # during the run, so absence must not be converted into a synthetic 1.0.
    target_score = _finite_score(state.get("target_score"))
    if target_score is None:
        target_score = _finite_score(metric_contract.get("target_score"))
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # Get submission results if available
    submissions = state.get("submissions", [])
    public_score = None
    if submissions:
        latest_sub = submissions[-1]
        public_score = latest_sub.public_score
        if public_score is not None:
            print(f"\n📊 Public Score: {public_score:.4f}")
            # Use metric direction for score selection
            from ...core.config import compare_scores

            if current_score == 0.0:
                current_score = public_score
            else:
                try:
                    metric_name = state["competition_info"].evaluation_metric
                except Exception:
                    metric_name = ""
                current_score = compare_scores(current_score, public_score, metric_name)

    from ...core.config import is_metric_minimization

    metric_name = ""
    try:
        metric_name = state["competition_info"].evaluation_metric
    except Exception:
        metric_name = ""

    contract_direction = metric_contract.get("is_lower_better")
    minimize = (
        contract_direction
        if isinstance(contract_direction, bool)
        else is_metric_minimization(metric_name)
    )
    gap = None
    if target_score is not None:
        gap = (
            current_score - target_score
            if minimize
            else target_score - current_score
        )

    print(f"\nCurrent Score: {current_score:.4f}")
    if target_score is None:
        print("Target Score:  not configured")
        print(f"Gap:           not available ({'minimize' if minimize else 'maximize'})")
    else:
        print(f"Target Score:  {target_score:.4f}")
        print(f"Gap:           {gap:.4f} ({'minimize' if minimize else 'maximize'})")

    # Analyze component performance
    dev_results = state.get("development_results", [])
    successful_components = [r for r in dev_results if r.success]

    print(
        f"\n📈 Component Success Rate: {len(successful_components)}/{len(dev_results)} ({len(successful_components) / len(dev_results) * 100:.0f}%)"
        if dev_results
        else "\n📈 No components tested yet"
    )

    # Decision: should we refine?
    needs_refinement = False
    refinement_reason = None

    target_achieved = False
    if target_score is not None:
        target_achieved = (
            current_score <= target_score
            if minimize
            else current_score >= target_score
        )

    if target_achieved:
        comparator = "<=" if minimize else ">="
        print(f"\n🎉 Target achieved! ({current_score:.4f} {comparator} {target_score:.4f})")
        needs_refinement = False
    elif current_iteration >= max_iterations:
        print(f"\n⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        needs_refinement = False
    elif target_score is None:
        # With no declared target, keep refining only while the iteration
        # budget remains. CV/OOF improvement is useful guidance, not a
        # fabricated stopping threshold.
        baseline_score = _finite_score(state.get("baseline_cv_score"))
        cv_improvement = (
            baseline_score - current_score
            if minimize and baseline_score is not None
            else current_score - baseline_score
            if baseline_score is not None
            else None
        )
        needs_refinement = True
        reason_prefix = "mlebench_" if run_mode == "mlebench" else ""
        if cv_improvement is not None and cv_improvement > 0:
            refinement_reason = f"{reason_prefix}cv_improved_budget_remaining"
            print(
                "\n🔄 Refinement without a configured target: CV/OOF improved "
                f"by {cv_improvement:.4f}; iteration budget remains"
            )
        else:
            refinement_reason = f"{reason_prefix}cv_budget_remaining"
            print(
                "\n🔄 Refinement: no target configured; "
                "continue within the iteration budget using CV/OOF guidance"
            )
    else:
        # Check if we have room for improvement
        improvement_potential = gap

        if improvement_potential is not None and improvement_potential > 0.001:
            print(f"\n🔄 Refinement needed (gap: {improvement_potential:.4f})")
            needs_refinement = True
            refinement_reason = "score_below_target"
        else:
            print("\n✅ Close enough to target")
            needs_refinement = False

    return {
        "needs_refinement": needs_refinement,
        "refinement_reason": refinement_reason,
        "current_performance_score": current_score,
        "last_updated": datetime.now(),
    }
