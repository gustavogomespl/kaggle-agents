"""Auto SOTA search node for the Kaggle Agents workflow."""

from datetime import datetime
from typing import Any

from ...core.state import KaggleState


def auto_sota_search_node(state: KaggleState) -> dict[str, Any]:
    """
    Automatic SOTA search triggered by stagnation or score gap detection.

    Searches for winning solutions and techniques when progress stalls.

    Args:
        state: Current workflow state

    Returns:
        State updates with SOTA search results and guidance
    """
    from ...agents.search_agent import SearchAgent

    print("\n" + "=" * 60)
    print("= AUTO SOTA SEARCH: Finding solutions to break stagnation")
    print("=" * 60)

    stagnation = state.get("stagnation_detection", {})
    if not stagnation.get("trigger_sota_search"):
        print("   Skipping - no SOTA search trigger")
        return {}

    competition_name = state.get("competition_name", "")
    domain = state.get("domain_detected", "tabular")
    current_score = state.get("current_performance_score", 0.0)

    print(f"\n   🔍 Searching SOTA solutions for: {competition_name}")
    print(f"   📊 Current score: {current_score}")
    print(f"   🎯 Trigger reason: {stagnation.get('reason', 'unknown')}")

    try:
        search_agent = SearchAgent()

        # Focus search on areas that could improve the score
        focus_areas = ["feature_engineering", "model_architecture", "ensemble_strategy"]

        # If stagnation is the issue, focus on novel approaches
        if stagnation.get("stagnated"):
            focus_areas.insert(0, "novel_approaches")
            focus_areas.insert(1, "hyperparameter_optimization")

        # Search for solutions
        search_results = search_agent.search_with_focus(
            competition=competition_name,
            domain=domain,
            focus_areas=focus_areas,
            max_results=5,
        ) if hasattr(search_agent, 'search_with_focus') else {}

        # Generate guidance from search results
        sota_guidance = _generate_sota_guidance_from_results(search_results, stagnation)

        print(f"\n   ✅ SOTA search complete - found {len(search_results.get('solutions', []))} relevant solutions")

        return {
            "sota_search_results": search_results,
            "sota_search_triggered": True,
            "refinement_guidance": {
                **state.get("refinement_guidance", {}),
                "sota_guidance": sota_guidance,
                "sota_triggered_by": stagnation.get("reason"),
            },
            "last_updated": datetime.now(),
        }

    except Exception as e:
        print(f"\n   ⚠️ SOTA search failed: {e}")
        # Return minimal guidance even if search fails
        return {
            "sota_search_triggered": True,
            "refinement_guidance": {
                **state.get("refinement_guidance", {}),
                "sota_guidance": _generate_fallback_sota_guidance(domain, stagnation),
            },
        }


def _generate_sota_guidance_from_results(search_results: dict, stagnation: dict) -> str:
    """Generate guidance string from SOTA search results."""
    solutions = search_results.get("solutions", [])

    guidance_parts = [
        "## SOTA Search Results (triggered by stagnation detection)",
        "",
        f"Trigger reason: {stagnation.get('reason', 'unknown')}",
        "",
    ]

    if solutions:
        guidance_parts.append("### Top Solutions Found:")
        for i, sol in enumerate(solutions[:3], 1):
            title = sol.get("title", "Unknown")
            approach = sol.get("approach", "Not specified")
            guidance_parts.append(f"{i}. **{title}**")
            guidance_parts.append(f"   - Approach: {approach}")

        guidance_parts.append("")
        guidance_parts.append("### Recommended Actions:")
        guidance_parts.append("1. Try feature engineering techniques from top solutions")
        guidance_parts.append("2. Consider model architectures used by winners")
        guidance_parts.append("3. Explore ensemble strategies mentioned")
    else:
        guidance_parts.append("### No specific solutions found - general recommendations:")
        guidance_parts.extend(_get_general_improvement_suggestions())

    return "\n".join(guidance_parts)


def _generate_fallback_sota_guidance(domain: str, stagnation: dict) -> str:
    """Generate fallback guidance when SOTA search fails."""
    guidance = [
        "## Stagnation Detected - General Improvement Suggestions",
        "",
        f"Domain: {domain}",
        f"Trigger: {stagnation.get('reason', 'unknown')}",
        "",
    ]
    guidance.extend(_get_general_improvement_suggestions())
    return "\n".join(guidance)


def _get_general_improvement_suggestions() -> list[str]:
    """Get general suggestions for breaking stagnation."""
    return [
        "### General Strategies to Break Stagnation:",
        "1. **Feature Engineering**: Create interaction features, aggregations, or target encoding",
        "2. **Model Diversity**: Try different model families (Neural, Gradient Boosting, Linear)",
        "3. **Hyperparameter Exploration**: Significantly change learning rate, depth, regularization",
        "4. **Ensemble Methods**: Use stacking with diverse base models",
        "5. **Data Augmentation**: For image/audio, add more augmentation strategies",
        "6. **Cross-Validation**: Ensure CV strategy matches competition requirements",
    ]
