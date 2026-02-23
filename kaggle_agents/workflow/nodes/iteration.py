"""Iteration control and performance evaluation nodes for the Kaggle Agents workflow."""

from datetime import datetime
from typing import Any

from ...core.state import KaggleState


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

    # Reset component index for refinement iterations (iteration > 1)
    # This ensures new components from refined plan are implemented
    updates = {
        "current_iteration": new_iteration,
        "should_continue": should_continue,
        "termination_reason": termination_reason,
        "last_updated": datetime.now(),
    }

    # If this is a refinement iteration (> 1), reset component index and skip flag
    if new_iteration > 1:
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

    current_score = state.get("best_score", 0.0)
    # Fallback: when no Kaggle submission has occurred (best_score == 0),
    # use the best available CV score from component development
    if current_score == 0.0:
        current_score = (
            state.get("best_single_model_score")
            or state.get("baseline_cv_score")
            or 0.0
        )
    # Dynamic target_score from state (set by MLE-bench or config), fallback to top 20% threshold
    target_score = state.get("target_score")
    if target_score is None:
        target_score = 1.0
    elif isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0
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

    minimize = is_metric_minimization(metric_name)
    gap = (current_score - target_score) if minimize else (target_score - current_score)

    print(f"\nCurrent Score: {current_score:.4f}")
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

    if minimize:
        target_achieved = current_score <= target_score
    else:
        target_achieved = current_score >= target_score

    if target_achieved:
        comparator = "<=" if minimize else ">="
        print(f"\n🎉 Target achieved! ({current_score:.4f} {comparator} {target_score:.4f})")
        needs_refinement = False
    elif current_iteration >= max_iterations:
        print(f"\n⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        needs_refinement = False
    else:
        # Check if we have room for improvement
        improvement_potential = gap

        if improvement_potential > 0.001:  # 0.1% gap
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
