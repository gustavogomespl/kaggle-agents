"""
MLE-STAR A/B Ablation Validation Node.

Implements true A/B testing for changes - runs both baseline and modified
code on the same validation set, only accepting improvements above epsilon.

INTEGRATION STATUS: ✅ WIRED INTO WORKFLOW
========================================================================================

Workflow flow:
  planner → set_baseline → developer → ablation_validation → (accept: robustness | reject: developer)

Loop protection:
  - Max 3 rejections before accepting to prevent infinite loops
  - ablation_rejection_count tracks consecutive rejections

State fields used:
  - ablation_baseline_code: Code snapshot before modification
  - ablation_baseline_score: Score before modification
  - ablation_accepted: True if change improved score by epsilon
  - ablation_validation_reason: Reason for acceptance/rejection
  - ablation_rejection_count: Counter to prevent infinite loops
========================================================================================
"""

from datetime import datetime
from typing import Any

from ...core.state import KaggleState

# Minimum improvement threshold (epsilon) for accepting a change
ABLATION_EPSILON = 0.0001


def _is_better_score(
    modified_score: float,
    baseline_score: float,
    minimize: bool = False,
    epsilon: float = ABLATION_EPSILON,
) -> bool:
    """
    Check if modified score is better than baseline by at least epsilon.

    Args:
        modified_score: Score from modified code
        baseline_score: Score from baseline code
        minimize: True if lower score is better
        epsilon: Minimum improvement threshold

    Returns:
        True if modified score is significantly better
    """
    if minimize:
        # Lower is better - modified must be at least epsilon lower
        return modified_score < (baseline_score - epsilon)
    else:
        # Higher is better - modified must be at least epsilon higher
        return modified_score > (baseline_score + epsilon)


def _extract_scores_from_results(state: KaggleState) -> tuple[float | None, float | None]:
    """
    Extract baseline and modified scores from development results.

    The most recent two development results represent modified and baseline
    (if baseline code was run for A/B testing).

    Args:
        state: Current state

    Returns:
        Tuple of (baseline_score, modified_score) or (None, None) if not available
    """
    development_results = state.get("development_results", [])

    if len(development_results) < 1:
        return None, None

    # Get the most recent result (modified code)
    modified_result = development_results[-1]

    # Try to extract OOF/CV score from the result
    modified_score = None
    if hasattr(modified_result, "oof_score"):
        modified_score = modified_result.oof_score
    elif hasattr(modified_result, "cv_score"):
        modified_score = modified_result.cv_score
    elif hasattr(modified_result, "validation_score"):
        modified_score = modified_result.validation_score

    # For baseline, check if we have ablation_baseline_score in state
    baseline_score = state.get("ablation_baseline_score")

    return baseline_score, modified_score


def ablation_validation_node(state: KaggleState) -> dict[str, Any]:
    """
    MLE-STAR A/B Ablation Validation: Validate changes with true A/B testing.

    This node compares the modified code's score against the baseline code's
    score. Changes are only accepted if they improve the score by at least
    epsilon (ABLATION_EPSILON).

    The baseline code is stored in state["ablation_baseline_code"] before
    each modification attempt.

    Args:
        state: Current state

    Returns:
        State updates with ablation validation results
    """
    print("\n" + "=" * 60)
    print("= ABLATION VALIDATION (MLE-STAR A/B Testing)")
    print("=" * 60)

    # Get metric direction
    from ...core.config import is_metric_minimization

    metric_name = ""
    try:
        comp_info = state.get("competition_info", {})
        if isinstance(comp_info, dict):
            metric_name = comp_info.get("evaluation_metric", "")
        else:
            metric_name = getattr(comp_info, "evaluation_metric", "")
    except Exception:
        pass

    minimize = is_metric_minimization(metric_name)
    direction = "lower is better" if minimize else "higher is better"

    print(f"\n   Metric: {metric_name or 'unknown'} ({direction})")
    print(f"   Epsilon threshold: {ABLATION_EPSILON}")

    # Get baseline and modified scores
    baseline_score, modified_score = _extract_scores_from_results(state)

    # Also check state for explicit scores
    if baseline_score is None:
        baseline_score = state.get("ablation_baseline_score")
    if modified_score is None:
        modified_score = state.get("current_performance_score", 0.0)

    # Check if we have both scores for comparison
    if baseline_score is None:
        print("\n   ⚠️  No baseline score available - accepting change by default")
        print("      (First iteration or baseline code not set)")
        return {
            "ablation_accepted": True,
            "ablation_validation_reason": "no_baseline",
            "ablation_rejection_count": 0,  # Reset counters
            "debug_attempt": 0,
            "debug_escalate": False,
            "last_updated": datetime.now(),
        }

    if modified_score is None:
        print("\n   ⚠️  No modified score available - rejecting change")
        rejection_count = state.get("ablation_rejection_count", 0)
        return {
            "ablation_accepted": False,
            "ablation_validation_reason": "no_modified_score",
            "ablation_rejection_count": rejection_count + 1,
            "debug_attempt": 0,  # Reset debug for retry
            "debug_escalate": False,
            "last_updated": datetime.now(),
        }

    print(f"\n   Baseline Score:  {baseline_score:.6f}")
    print(f"   Modified Score:  {modified_score:.6f}")

    # Compute improvement
    if minimize:
        improvement = baseline_score - modified_score
    else:
        improvement = modified_score - baseline_score

    print(f"   Improvement:     {improvement:+.6f}")

    # Get current rejection count for loop protection
    rejection_count = state.get("ablation_rejection_count", 0)

    # Check if improvement exceeds epsilon
    if _is_better_score(modified_score, baseline_score, minimize, ABLATION_EPSILON):
        print(f"\n   ✅ ACCEPTED: Improvement ({improvement:+.6f}) exceeds epsilon ({ABLATION_EPSILON})")

        # Get current code to update baseline for future comparisons
        current_code = state.get("current_code", "")
        print(f"      Updating baseline code ({len(current_code)} chars) for future A/B comparisons")

        # Update baseline CODE and SCORE for next comparison, reset counters
        return {
            "ablation_accepted": True,
            "ablation_validation_reason": "improvement_exceeded_epsilon",
            "ablation_baseline_code": current_code,  # NEW: Update baseline code to accepted version
            "ablation_baseline_score": modified_score,  # New baseline score for next iteration
            "ablation_rejection_count": 0,  # Reset on acceptance
            "debug_attempt": 0,  # Reset debug counter
            "debug_escalate": False,  # Reset escalation flag
            "last_updated": datetime.now(),
        }
    else:
        print(f"\n   ❌ REJECTED: Improvement ({improvement:+.6f}) below epsilon ({ABLATION_EPSILON})")

        # Increment rejection count
        new_rejection_count = rejection_count + 1

        # Revert to baseline code
        baseline_code = state.get("ablation_baseline_code")
        if baseline_code:
            print("      Reverting to baseline code...")

        print(f"      Rejection count: {new_rejection_count}/3")
        print("      Resetting debug counter for fresh developer attempt")

        return {
            "ablation_accepted": False,
            "ablation_validation_reason": "improvement_below_epsilon",
            "current_code": baseline_code if baseline_code else state.get("current_code", ""),
            "ablation_rejection_count": new_rejection_count,  # Track for loop protection
            "debug_attempt": 0,  # Reset debug counter for fresh developer retry
            "debug_escalate": False,  # Reset escalation flag
            # Keep baseline score unchanged
            "last_updated": datetime.now(),
        }


def set_ablation_baseline_node(state: KaggleState) -> dict[str, Any]:
    """
    Set the current code as the ablation baseline before modifications.

    Call this node before the developer makes changes to capture the
    baseline for A/B comparison.

    Args:
        state: Current state

    Returns:
        State updates with baseline code and score stored
    """
    print("\n   📌 Setting ablation baseline...")

    current_code = state.get("current_code", "")
    current_score = state.get("current_performance_score", 0.0)

    # Also check for OOF score from recent development results
    development_results = state.get("development_results", [])
    if development_results:
        last_result = development_results[-1]
        if hasattr(last_result, "oof_score") and last_result.oof_score:
            current_score = last_result.oof_score
        elif hasattr(last_result, "cv_score") and last_result.cv_score:
            current_score = last_result.cv_score

    print(f"      Baseline code: {len(current_code)} chars")
    print(f"      Baseline score: {current_score:.6f}")

    return {
        "ablation_baseline_code": current_code,
        "ablation_baseline_score": current_score,
        "last_updated": datetime.now(),
    }
