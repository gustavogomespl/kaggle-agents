"""Curriculum learning insights extraction for the planner agent."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...core.state import KaggleState


def extract_curriculum_insights(state: KaggleState) -> str:
    """
    Extract curriculum learning insights from iteration memory.

    Builds on successful patterns and avoids known failures from previous iterations.

    Args:
        state: Current workflow state containing iteration_memory

    Returns:
        Formatted string with learned patterns to inject in prompt
    """
    iteration_memory = state.get("iteration_memory", [])

    if not iteration_memory:
        return "No previous iteration insights available."

    # Aggregate patterns across all iterations
    all_worked = []
    all_failed = []

    for memory in iteration_memory:
        all_worked.extend(memory.what_worked)
        all_failed.extend(memory.what_failed)

    # Build insights string
    insights = ["\n🧠 CURRICULUM LEARNING INSIGHTS (from previous iterations):"]

    if all_worked:
        insights.append("\n✅ What Worked (prioritize these approaches):")
        # Get last 5 unique successful patterns
        unique_worked = list(dict.fromkeys(all_worked))[-5:]
        for pattern in unique_worked:
            insights.append(f"   - {pattern}")

    if all_failed:
        insights.append("\n❌ CRITICAL: What Failed (DO NOT REPEAT these approaches):")
        # Get last 5 unique failure patterns
        unique_failed = list(dict.fromkeys(all_failed))[-5:]
        for pattern in unique_failed:
            insights.append(f"   - {pattern}")

    # Add failure analysis insights from latest iteration
    if iteration_memory:
        latest_memory = iteration_memory[-1]
        if "failure_analysis" in latest_memory.results:
            analysis = latest_memory.results["failure_analysis"]

            if analysis.get("common_errors"):
                insights.append("\n⚠️  Common Errors to Avoid:")
                # Get top 3 most common errors
                common_errors = sorted(
                    analysis["common_errors"].items(), key=lambda x: x[1], reverse=True
                )[:3]
                for error_type, count in common_errors:
                    insights.append(f"   - {error_type}: {count} occurrences")

            # Add component-specific success patterns
            if analysis.get("success_by_component"):
                insights.append("\n📊 Component Success Rates:")
                for comp_type, success_info in list(analysis["success_by_component"].items())[
                    :3
                ]:
                    rate = success_info.get("success_rate", 0.0)
                    insights.append(f"   - {comp_type}: {rate:.0%} success rate")

    # Add score improvement trend
    if len(iteration_memory) >= 2:
        score_improvements = [m.score_improvement for m in iteration_memory[-3:]]
        avg_improvement = sum(score_improvements) / len(score_improvements)
        insights.append(f"\n📈 Recent Score Trend: {avg_improvement:+.4f} avg improvement")

    return "\n".join(insights)
