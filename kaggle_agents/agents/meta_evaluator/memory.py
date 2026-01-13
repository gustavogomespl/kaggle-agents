"""
Iteration memory creation for Meta-Evaluator.

Contains methods for creating iteration memory for learning history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.state import IterationMemory


if TYPE_CHECKING:
    from ...core.state import KaggleState


class MemoryMixin:
    """Mixin providing iteration memory creation methods."""

    def _create_iteration_memory(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> IterationMemory:
        """
        Create iteration memory for learning history.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis
            reward_signals: Reward signals

        Returns:
            IterationMemory object
        """
        current_iteration = state.get("current_iteration", 0)
        current_score = state.get("current_performance_score", 0.0)
        previous_score = state.get("best_score", 0.0)

        return IterationMemory(
            iteration=current_iteration,
            phase="meta_evaluation",
            actions_taken=[
                "analyzed_failures",
                "calculated_rewards",
                "generated_refinement_guidance",
            ],
            results={
                "failure_analysis": failure_analysis,
                "reward_signals": reward_signals,
            },
            score_improvement=current_score - previous_score,
            what_worked=failure_analysis["success_patterns"],
            what_failed=failure_analysis["error_patterns"],
        )
