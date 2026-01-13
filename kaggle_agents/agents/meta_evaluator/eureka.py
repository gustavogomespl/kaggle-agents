"""
Eureka evolutionary crossover for Meta-Evaluator.

Contains methods for evolutionary optimization of plans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import IterationMemory


class EurekaMixin:
    """Mixin providing evolutionary crossover methods."""

    def _evolutionary_crossover(self, state) -> dict[str, Any]:
        """
        Eureka-style evolutionary crossover.

        Combines elements from the best-performing iterations to guide
        the next generation of plans.

        Args:
            state: Current workflow state

        Returns:
            Crossover guidance dictionary
        """
        print("\n   🧬 Eureka: Performing evolutionary crossover...")

        iteration_memory = state.get("iteration_memory", [])
        candidate_plans = state.get("candidate_plans", [])

        if not iteration_memory:
            return {}

        # Find top-performing iterations by score improvement
        sorted_memories = sorted(
            iteration_memory,
            key=lambda m: m.score_improvement if hasattr(m, "score_improvement") else 0.0,
            reverse=True,
        )
        top_memories = sorted_memories[:2]  # Top 2 iterations

        # Extract success patterns from top iterations
        success_patterns = set()
        for memory in top_memories:
            what_worked = memory.what_worked if hasattr(memory, "what_worked") else []
            success_patterns.update(what_worked)

        # Extract failure patterns to avoid
        avoid_patterns = set()
        for memory in iteration_memory:
            what_failed = memory.what_failed if hasattr(memory, "what_failed") else []
            avoid_patterns.update(what_failed)

        # Analyze candidate plans if available
        successful_strategies = []
        if candidate_plans:
            # Get strategies from plans with highest fitness
            sorted_candidates = sorted(
                candidate_plans,
                key=lambda p: p.fitness_score if hasattr(p, "fitness_score") else 0.0,
                reverse=True,
            )
            for candidate in sorted_candidates[:2]:
                if hasattr(candidate, "strategy"):
                    successful_strategies.append(candidate.strategy)

        # Generate crossover guidance
        crossover_guidance = {
            "preserve_components": list(success_patterns),
            "avoid_components": list(
                avoid_patterns - success_patterns
            ),  # Don't avoid if also succeeded
            "successful_strategies": successful_strategies,
            "suggested_combinations": self._suggest_combinations(success_patterns),
            "evolutionary_pressure": self._calculate_evolutionary_pressure(iteration_memory),
        }

        print(
            f"   ✓ Crossover: Preserve {len(crossover_guidance['preserve_components'])} patterns, "
            f"Avoid {len(crossover_guidance['avoid_components'])} patterns"
        )

        return crossover_guidance

    def _suggest_combinations(self, success_patterns: set) -> list[str]:
        """
        Suggest promising component combinations based on success patterns.

        Args:
            success_patterns: Set of successful patterns

        Returns:
            List of suggested combinations
        """
        suggestions = []

        pattern_list = list(success_patterns)

        # If we have multiple success patterns, suggest combinations
        if len(pattern_list) >= 2:
            suggestions.append(f"Combine {pattern_list[0]} with {pattern_list[1]}")

        # Standard high-value combinations
        if (
            "model_success" in success_patterns
            and "feature_engineering_success" not in success_patterns
        ):
            suggestions.append("Add advanced feature engineering to successful model")

        if (
            "feature_engineering_success" in success_patterns
            and "ensemble_success" not in success_patterns
        ):
            suggestions.append("Apply ensemble techniques to leverage good features")

        # Suggest based on what's missing
        all_types = {
            "model_success",
            "feature_engineering_success",
            "ensemble_success",
            "preprocessing_success",
        }
        missing = all_types - success_patterns
        if missing:
            missing_type = list(missing)[0].replace("_success", "")
            suggestions.append(f"Explore {missing_type} components for diversity")

        return suggestions[:3]  # Limit to 3 suggestions

    def _calculate_evolutionary_pressure(self, iteration_memory: list[IterationMemory]) -> str:
        """
        Calculate evolutionary pressure based on iteration history.

        Determines if we should explore (low scores, early iterations)
        or exploit (high scores, late iterations).

        Args:
            iteration_memory: List of iteration memories

        Returns:
            "explore", "exploit", or "balanced"
        """
        if not iteration_memory:
            return "explore"

        # Calculate average score improvement
        improvements = [
            m.score_improvement if hasattr(m, "score_improvement") else 0.0
            for m in iteration_memory
        ]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        # Recent trend
        recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
        recent_avg = (
            sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0.0
        )

        # If recent improvements are positive and growing, exploit
        if recent_avg > 0.01 and len(iteration_memory) > 2:
            return "exploit"

        # If recent improvements are stagnating or negative, explore
        if recent_avg <= 0 or (len(iteration_memory) > 3 and avg_improvement < 0.005):
            return "explore"

        return "balanced"
