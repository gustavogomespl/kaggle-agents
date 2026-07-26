"""
Iteration memory creation for Meta-Evaluator.

Contains methods for creating iteration memory for learning history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.config import calculate_score_improvement
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
        previous_score = None
        iteration_memory = state.get("iteration_memory", []) or []
        if iteration_memory:
            previous_results = getattr(
                iteration_memory[-1],
                "results",
                {},
            )
            if isinstance(previous_results, dict):
                previous_score = previous_results.get("current_score")
        if previous_score is None:
            previous_score = state.get("baseline_cv_score")
        if previous_score is None:
            previous_score = state.get("best_score")
        if previous_score is None:
            previous_score = current_score

        metric_name = ""
        try:
            metric_name = state["competition_info"].evaluation_metric or ""
        except Exception:
            pass
        try:
            score_improvement = calculate_score_improvement(
                float(current_score),
                float(previous_score),
                metric_name,
            )
        except (TypeError, ValueError):
            score_improvement = 0.0

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
                "current_score": current_score,
                "previous_score": previous_score,
                "metric_name": metric_name,
            },
            score_improvement=score_improvement,
            what_worked=failure_analysis["success_patterns"],
            what_failed=failure_analysis["error_patterns"],
        )
