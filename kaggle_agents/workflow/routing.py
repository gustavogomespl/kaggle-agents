"""Routing functions for the Kaggle Agents workflow."""

from typing import Literal

from ..core.state import KaggleState


def should_continue_workflow(state: KaggleState) -> Literal["continue", "end"]:
    """
    Decide whether to continue or end the workflow.

    Args:
        state: Current state

    Returns:
        "continue" or "end"
    """
    should_continue = state.get("should_continue", True)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # End conditions
    if not should_continue:
        return "end"

    if current_iteration >= max_iterations:
        return "end"

    # Check if we have components to implement
    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    if current_component_index >= len(ablation_plan):
        # All components implemented, could iterate or end
        return "end"

    return "continue"


def should_retry_component(state: KaggleState) -> Literal["retry", "next"]:
    """
    Decide whether to retry current component or move to next.

    Args:
        state: Current state

    Returns:
        "retry" or "next"
    """
    development_results = state.get("development_results", [])

    if not development_results:
        return "next"

    # Check last result
    last_result = development_results[-1]

    if last_result.success:
        return "next"

    # Check retry count
    code_retry_count = state.get("code_retry_count", 0)
    max_retries = 3  # Max retries at workflow level

    if code_retry_count < max_retries:
        return "retry"

    # Max retries reached, move to next component
    return "next"


def route_after_developer(state: KaggleState) -> Literal["iterate", "end"]:
    """
    Route after developer agent completes.

    Simplified routing logic - only stops for:
    1. Explicit skip_remaining_components flag
    2. Medal achievement (MLE-bench success)
    3. Critical errors (data download failed, auth issues)
    4. All components implemented

    Target score checking is delegated to iteration_control to allow
    multiple refinement iterations with meta-evaluator insights.

    Args:
        state: Current state

    Returns:
        "iterate" to continue implementing components, or "end" if done
    """
    # Explicit early-stop flag (e.g., set by DeveloperAgent)
    if state.get("skip_remaining_components"):
        print("\n⏩ skip_remaining_components=True - Moving to validation")
        return "end"

    # Check for medal achievement in MLE-bench mode (immediate success)
    mlebench_grade = state.get("mlebench_grade")
    run_mode = str(state.get("run_mode", "")).lower()

    if run_mode == "mlebench" and isinstance(mlebench_grade, dict):
        if mlebench_grade.get("valid_submission"):
            if any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
                print("\n🏅 MEDAL ACHIEVED - Moving to validation")
                return "end"

    # Check for critical errors (data download failed, auth issues)
    errors = state.get("errors", [])
    if errors:
        for error in errors:
            if "Data download failed" in error or "authentication failed" in error.lower():
                print("\n⚠️ Critical error detected, stopping workflow")
                return "end"

    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    # Check if more components to implement
    if current_component_index < len(ablation_plan):
        # Check if we're stuck on the same component (prevent infinite loop)
        dev_results = state.get("development_results", [])
        if len(dev_results) >= 3:
            # Check if last 3 results all failed on same component
            recent_failures = [r for r in dev_results[-3:] if not r.success]
            if len(recent_failures) == 3:
                # Check if all have same error about data files
                data_errors = [
                    r for r in recent_failures if "Data files not found" in (r.stderr or "")
                ]
                if len(data_errors) == 3:
                    print("\n⚠️ Repeated data file errors, stopping workflow")
                    return "end"

        remaining = len(ablation_plan) - current_component_index
        print(f"\n🔄 {remaining} component(s) remaining - continuing iteration")
        return "iterate"

    # All components done - move to validation
    print(f"\n✅ All {len(ablation_plan)} components implemented - moving to validation")
    return "end"


def route_after_submission(state: KaggleState) -> Literal["retry_developer", "continue"]:
    """
    Route after submission agent - retry if submission is invalid.

    Checks if the submission passed validation. If not, routes back to
    the developer to regenerate with the error context.

    Args:
        state: Current state

    Returns:
        "retry_developer" if submission invalid and retries remaining,
        "continue" otherwise
    """
    submissions = state.get("submissions", [])

    if not submissions:
        # No submission generated at all - retry
        retry_count = state.get("retry_submission_count", 0)
        if retry_count < 3:
            state["retry_submission_count"] = retry_count + 1
            state["submission_validation_error"] = "No submission file generated"
            print(f"⚠️ No submission generated, retrying... ({retry_count + 1}/3)")
            return "retry_developer"
        return "continue"

    last_submission = submissions[-1]

    # Check if submission is valid (handle both dict and object)
    is_valid = True
    error_msg = None

    if isinstance(last_submission, dict):
        is_valid = last_submission.get("valid", True)
        error_msg = last_submission.get("error")
    else:
        # Object with attributes
        is_valid = getattr(last_submission, "valid", True)
        error_msg = getattr(last_submission, "error", None)

    if not is_valid and error_msg:
        retry_count = state.get("retry_submission_count", 0)

        if retry_count < 3:
            state["retry_submission_count"] = retry_count + 1
            state["submission_validation_error"] = error_msg
            print(f"⚠️ Invalid submission: {error_msg[:100]}...")
            print(f"   Retrying with error context... ({retry_count + 1}/3)")
            return "retry_developer"
        print("⚠️ Max submission retries reached, continuing...")

    return "continue"


def route_after_iteration_control(state: KaggleState) -> Literal["refine", "end"]:
    """
    Route after iteration control - decide if we refine or end.

    Uses adaptive iteration logic:
    1. If score gap > threshold, extend iterations
    2. In MLE-bench mode, aggressively refines until medal/max
    3. Respects minimum iterations before early stopping

    Args:
        state: Current state

    Returns:
        "refine" to start refinement iteration, or "end" if done
    """
    from ..core.config import get_config, is_metric_minimization

    config = get_config()
    iter_config = config.iteration

    needs_refinement = state.get("needs_refinement", False)
    current_iteration = state.get("current_iteration", 0)
    base_max_iterations = state.get("max_iterations", iter_config.max_iterations)
    run_mode = str(state.get("run_mode", "")).lower()
    mlebench_grade = state.get("mlebench_grade")

    # Calculate effective max_iterations based on score gap (adaptive)
    max_iterations = base_max_iterations
    if iter_config.adaptive_iterations:
        current_score = state.get("current_performance_score", 0.0)
        target_score = state.get("target_score")
        if target_score and isinstance(target_score, (int, float)) and target_score > 0:
            # Calculate gap percentage
            score_gap = abs(float(target_score) - float(current_score)) / float(target_score)
            if score_gap > iter_config.score_gap_threshold:
                # Extend iterations when gap is large
                max_iterations = min(iter_config.extended_max_iterations, base_max_iterations * 2)
                print(f"   📈 Score gap {score_gap:.1%} > {iter_config.score_gap_threshold:.0%} threshold")
                print(f"      Extended max_iterations: {base_max_iterations} → {max_iterations}")

    print("\n🔀 Routing decision:")
    print(f"   Current iteration: {current_iteration}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Needs refinement: {needs_refinement}")
    print(f"   Run mode: {run_mode}")

    # Check medal status
    has_gold = False
    has_any_medal = False
    if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
        has_gold = mlebench_grade.get("gold_medal", False)
        has_any_medal = any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"])

    # Handle skip flag - in MLE-bench mode, only end if gold or max iterations reached
    if state.get("skip_remaining_components"):
        if run_mode == "mlebench":
            if has_gold:
                print("   🥇 GOLD MEDAL ACHIEVED - Ending")
                return "end"
            if current_iteration >= max_iterations:
                print(f"   ⏱️  Max iterations reached with medal ({current_iteration}/{max_iterations})")
                return "end"
            # Reset skip flag and continue refining for better medal
            print(f"   🔄 Medal achieved but continuing for gold (iteration {current_iteration + 1}/{max_iterations})")
                # Note: State update happens in iteration_control_node, not here
        else:
            print("   ⏩ skip_remaining_components=True - Ending")
            return "end"

    # Check for gold medal achievement (always stop on gold)
    if has_gold:
        print("   🥇 GOLD MEDAL ACHIEVED - Success!")
        return "end"

    # Max iterations reached
    if current_iteration >= max_iterations:
        print(f"   ⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        return "end"

    # MLE-bench mode: aggressively refine until medal or max_iterations
    if run_mode == "mlebench":
        # Log refinement guidance if available
        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance:
            print("   📋 Refinement guidance available from meta-evaluator")
            if refinement_guidance.get("planner_guidance"):
                print(f"      Planner: {refinement_guidance['planner_guidance'][:80]}...")
            if refinement_guidance.get("developer_guidance"):
                print(f"      Developer: {refinement_guidance['developer_guidance'][:80]}...")

        print(f"   🔄 MLE-bench mode: Starting refinement iteration {current_iteration + 1}")
        return "refine"

    # Standard Kaggle mode: check target_score
    current_score = state.get("current_performance_score", 0.0)
    target_score = state.get("target_score")
    if target_score is None:
        target_score = 1.0
    elif isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0

    # Respect metric direction when available
    metric_name = ""
    try:
        metric_name = state["competition_info"].evaluation_metric
    except Exception:
        metric_name = ""

    if isinstance(current_score, str):
        try:
            current_score = float(current_score)
        except ValueError:
            current_score = 0.0

    if isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0

    if isinstance(current_score, (int, float)) and isinstance(target_score, (int, float)):
        goal_achieved = False
        if is_metric_minimization(metric_name):
            goal_achieved = float(current_score) <= float(target_score)
        else:
            goal_achieved = float(current_score) >= float(target_score)

        if goal_achieved:
            # Respect min_iterations before early stopping
            if iter_config.adaptive_iterations and current_iteration < iter_config.min_iterations:
                print(f"   🎯 Goal achieved but below min_iterations ({current_iteration}/{iter_config.min_iterations})")
                print("      Continuing to consolidate improvements...")
                return "refine"
            print(f"   ✅ Goal achieved: {current_score:.4f} vs target {target_score:.4f}")
            return "end"

    # Decide based on refinement flag
    if needs_refinement:
        print(f"   🔄 Starting refinement iteration {current_iteration + 1}")
        return "refine"

    # If below min_iterations, continue even without explicit refinement need
    if iter_config.adaptive_iterations and current_iteration < iter_config.min_iterations:
        print(f"   📊 Below min_iterations ({current_iteration}/{iter_config.min_iterations}) - continuing")
        return "refine"

    print("   ✅ No refinement needed")
    return "end"


def route_after_meta_evaluator(state: KaggleState) -> Literal["sota_search", "curriculum", "continue"]:
    """
    Route after meta-evaluator - check for SOTA search or curriculum learning.

    Priority:
    1. SOTA search if stagnation/score gap detected
    2. Curriculum learning if critical failures
    3. Continue otherwise

    Args:
        state: Current state

    Returns:
        "sota_search", "curriculum", or "continue"
    """
    # Check for SOTA search trigger (stagnation or score gap)
    stagnation = state.get("stagnation_detection", {})
    if stagnation.get("trigger_sota_search"):
        print(f"\n   🔍 SOTA Search triggered: {stagnation.get('reason', 'stagnation detected')}")
        return "sota_search"

    failure_analysis = state.get("failure_analysis", {})
    error_patterns = failure_analysis.get("error_patterns", [])
    failed_components = failure_analysis.get("failed_components", [])

    # Check for critical errors that need curriculum learning
    critical_errors = ["memory_error", "timeout_error", "import_error", "syntax_error", "data_alignment"]
    has_critical = any(e in critical_errors for e in error_patterns)

    # Only trigger curriculum if we have failures and this is a refinement iteration
    current_iteration = state.get("current_iteration", 0)

    if has_critical and current_iteration > 0 and len(failed_components) > 0:
        print("\n   WEBRL: Critical failures detected - triggering curriculum learning")
        return "curriculum"

    return "continue"
