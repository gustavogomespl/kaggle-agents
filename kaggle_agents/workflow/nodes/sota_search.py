"""Auto SOTA search node for the Kaggle Agents workflow."""

from datetime import datetime
from typing import Any

from ...core.config import get_config
from ...core.state import KaggleState
from ...utils.telemetry import make_event


def auto_sota_search_node(state: KaggleState) -> dict[str, Any]:
    """
    Automatic SOTA search triggered by stagnation or score gap detection.

    Searches for winning solutions and techniques when progress stalls.
    Retrieval goes through SearchAgent.retrieve(), so the MLE-bench
    contamination guard applies here as well.

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

    try:
        competition_name = state["competition_info"].name
    except Exception:
        competition_name = ""
    domain = state.get("domain_detected", "tabular")
    current_score = state.get("current_performance_score", 0.0)
    current_iteration = state.get("current_iteration", 0)

    print(f"\n   🔍 Searching SOTA solutions for: {competition_name}")
    print(f"   📊 Current score: {current_score}")
    print(f"   🎯 Trigger reason: {stagnation.get('reason', 'unknown')}")

    # Ablation toggle: Search-First disabled -> generic guidance only
    toggles = getattr(get_config(), "ablation_toggles", None)
    if toggles and toggles.disable_search:
        print("   ABLATION: Search disabled - using generic stagnation guidance")
        return {
            "sota_search_triggered": True,
            "refinement_guidance": {
                **state.get("refinement_guidance", {}),
                "sota_guidance": _generate_fallback_sota_guidance(domain, stagnation),
            },
            "telemetry_events": [
                make_event(
                    "ablation",
                    "sota_search_skipped",
                    iteration=current_iteration,
                    component="search",
                )
            ],
            "last_updated": datetime.now(),
        }

    try:
        search_agent = SearchAgent()

        # Retrieve fresh solutions (mode-aware; contamination guard in MLE-bench)
        solutions, _queries, audit_records, events, _k = search_agent.retrieve(
            state, max_results=5
        )

        search_results = {
            "solutions": [
                {
                    "title": sol.title,
                    "approach": ", ".join(sol.models_used) or "; ".join(sol.strategies[:2]),
                }
                for sol in solutions
            ]
        }

        # Generate guidance from search results
        sota_guidance = _generate_sota_guidance_from_results(search_results, stagnation)

        print(f"\n   ✅ SOTA search complete - found {len(solutions)} relevant solutions")

        events.append(
            make_event(
                "recovery",
                "sota_search_executed",
                iteration=current_iteration,
                found=len(solutions),
                reason=stagnation.get("reason", "stagnation"),
            )
        )

        return {
            "sota_solutions": solutions,
            "sota_search_results": search_results,
            "sota_search_triggered": True,
            "search_audit": audit_records,
            "refinement_guidance": {
                **state.get("refinement_guidance", {}),
                "sota_guidance": sota_guidance,
                "sota_triggered_by": stagnation.get("reason"),
            },
            "telemetry_events": events,
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
            "telemetry_events": [
                make_event(
                    "recovery",
                    "sota_search_executed",
                    iteration=current_iteration,
                    found=0,
                    error=str(e)[:300],
                )
            ],
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
