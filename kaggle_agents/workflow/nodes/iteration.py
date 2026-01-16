"""Iteration control and performance evaluation nodes for the Kaggle Agents workflow."""

from datetime import datetime
from typing import Any

from ...core.state import KaggleState
from ...core.state.persistent_memory import (
    CrossCompetitionMemory,
    DatasetFingerprint,
    WinningStrategy,
)
from ...core.state.persistent_store import get_persistent_store


def _save_to_persistent_memory(state: KaggleState) -> bool:
    """
    Save competition learnings to persistent memory (PiML Cross-Competition Memory).

    Called at the end of a competition run to store successful strategies
    and failed approaches for future reference.

    Args:
        state: Current state

    Returns:
        True if saved successfully
    """
    try:
        # Extract competition info
        comp_info = state.get("competition_info", {})
        if isinstance(comp_info, dict):
            competition_id = comp_info.get("name", "unknown")
            competition_name = comp_info.get("name", "unknown")
            problem_type = comp_info.get("problem_type", "classification")
        else:
            competition_id = getattr(comp_info, "name", "unknown")
            competition_name = getattr(comp_info, "name", "unknown")
            problem_type = getattr(comp_info, "problem_type", "classification")

        # Skip if no competition ID
        if competition_id == "unknown":
            print("   ⚠️  Cannot save to persistent memory: no competition ID")
            return False

        # Determine domain from detected domain
        domain_detected = state.get("domain_detected", "tabular")
        if isinstance(domain_detected, str):
            domain = domain_detected
        else:
            domain = getattr(domain_detected, "value", "tabular") if domain_detected else "tabular"

        # Create dataset fingerprint
        data_insights = state.get("data_insights")
        n_samples = 0
        n_features = 0
        missing_rate = 0.0
        n_classes = None

        if data_insights:
            if isinstance(data_insights, dict):
                n_samples = data_insights.get("n_rows", 0) or data_insights.get("n_samples", 0)
                n_features = data_insights.get("n_columns", 0) or data_insights.get("n_features", 0)
                missing_rate = data_insights.get("missing_rate", 0.0)
                n_classes = data_insights.get("n_classes")
            else:
                n_samples = getattr(data_insights, "n_rows", 0) or getattr(data_insights, "n_samples", 0)
                n_features = getattr(data_insights, "n_columns", 0) or getattr(data_insights, "n_features", 0)
                missing_rate = getattr(data_insights, "missing_rate", 0.0)
                n_classes = getattr(data_insights, "n_classes", None)

        fingerprint = DatasetFingerprint(
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=missing_rate,
            imbalance_ratio=1.0,  # TODO: Calculate from data
            target_type=problem_type,
            n_classes=n_classes,
        )

        # Extract winning strategy from successful strategies
        successful_strategies = state.get("successful_strategies", [])
        ensemble_strategy = state.get("ensemble_strategy")
        ensemble_weights = state.get("ensemble_weights", {})

        model_type = "unknown"
        if successful_strategies:
            model_type = successful_strategies[0] if successful_strategies else "unknown"
        elif ensemble_strategy:
            model_type = f"ensemble_{ensemble_strategy}"

        winning_strategy = WinningStrategy(
            model_type=model_type,
            preprocessing_steps=[],  # TODO: Extract from ablation history
            feature_engineering=[],
            hyperparameters={},
            ensemble_strategy=ensemble_strategy,
            ensemble_weights=ensemble_weights if isinstance(ensemble_weights, dict) else {},
        )

        # Get failed approaches and error patterns
        failed_strategies = state.get("failed_strategies", [])
        failed_component_names = state.get("failed_component_names", [])
        failed_approaches = list(set(failed_strategies + failed_component_names))

        error_patterns = []
        error_pattern_memory = state.get("error_pattern_memory", [])
        for epm in error_pattern_memory:
            if isinstance(epm, dict):
                pattern = epm.get("pattern", epm.get("error_type", ""))
            else:
                pattern = getattr(epm, "pattern", getattr(epm, "error_type", ""))
            if pattern:
                error_patterns.append(pattern)

        # Get final score and medal
        final_score = state.get("best_score", 0.0) or state.get("current_performance_score", 0.0)
        mlebench_grade = state.get("mlebench_grade")
        medal = None
        if isinstance(mlebench_grade, dict):
            if mlebench_grade.get("gold_medal"):
                medal = "gold"
            elif mlebench_grade.get("silver_medal"):
                medal = "silver"
            elif mlebench_grade.get("bronze_medal"):
                medal = "bronze"

        # Get timing info
        workflow_start_time = state.get("workflow_start_time")
        total_time = 0.0
        if workflow_start_time:
            if isinstance(workflow_start_time, datetime):
                total_time = (datetime.now() - workflow_start_time).total_seconds()

        iterations_used = state.get("current_iteration", 0)

        # Create memory record
        memory = CrossCompetitionMemory(
            competition_id=competition_id,
            competition_name=competition_name,
            domain=domain,
            fingerprint=fingerprint,
            winning_strategy=winning_strategy,
            failed_approaches=failed_approaches,
            error_patterns=list(set(error_patterns)),
            final_score=final_score,
            medal=medal,
            iterations_used=iterations_used,
            total_time_seconds=total_time,
            timestamp=datetime.now(),
        )

        # Save to persistent store
        store = get_persistent_store()
        success = store.save(memory)

        if success:
            print(f"   💾 Saved to persistent memory: {competition_name}")
            print(f"      Domain: {domain}, Score: {final_score:.4f}, Medal: {medal or 'none'}")
            print(f"      Memory DB now has {store.count()} competitions")
        else:
            print("   ⚠️  Failed to save to persistent memory")

        return success

    except Exception as e:
        print(f"   ⚠️  Error saving to persistent memory: {e}")
        return False


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

    # PiML: Save to persistent memory when workflow is ending (no more refinement)
    if not needs_refinement:
        print("\n   💾 Saving learnings to persistent memory...")
        _save_to_persistent_memory(state)

    return {
        "needs_refinement": needs_refinement,
        "refinement_reason": refinement_reason,
        "current_performance_score": current_score,
        "last_updated": datetime.now(),
    }
