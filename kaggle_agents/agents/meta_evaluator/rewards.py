"""
Reward calculation for Meta-Evaluator.

Contains methods for calculating RL reward signals (CodeRL+ pattern).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ...core.config import calculate_score_improvement, is_metric_minimization


if TYPE_CHECKING:
    from ...core.state import KaggleState


class RewardsMixin:
    """Mixin providing reward calculation methods."""

    def _calculate_reward_signals(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
    ) -> dict[str, float]:
        """
        Calculate reward signals for RL optimization (CodeRL+ pattern).

        Implements multi-faceted reward:
        - Functional correctness (execution success)
        - Performance (cross-validation/OOF score)
        - Code quality (execution semantics)

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis results

        Returns:
            Reward signals dictionary
        """
        print("\n   💰 Calculating reward signals...")

        dev_results = state.get("development_results", [])
        competition_info = state.get("competition_info")
        metric_name = competition_info.evaluation_metric if competition_info else ""
        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", "")).lower()

        def _numeric(value: Any) -> float | None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if math.isfinite(numeric) else None

        # MLE-bench test-set scores are intentionally unavailable during the
        # workflow. Rewards are based exclusively on CV/OOF state.
        current_score = _numeric(state.get("current_performance_score"))
        if current_score is None:
            current_score = _numeric(state.get("best_single_model_score"))
        if current_score is None:
            current_score = _numeric(state.get("baseline_cv_score"))
        if current_score is None:
            current_score = 0.0

        best_score = _numeric(state.get("baseline_cv_score"))
        if best_score is None:
            best_score = _numeric(state.get("best_single_model_score"))
        if best_score is None:
            best_score = current_score

        # Reward 1: Functional Correctness (binary)
        total_components = len(dev_results)
        successful_components = len(failure_analysis["success_components"])
        r_functional = successful_components / total_components if total_components > 0 else 0.0

        # Reward 2: Performance (continuous, normalized 0-1)
        target_score = _numeric(state.get("target_score"))
        if target_score is not None:
            if is_metric_minimization(metric_name):
                if current_score <= target_score:
                    r_performance = 1.0
                elif current_score > 0 and target_score >= 0:
                    r_performance = min(target_score / current_score, 1.0)
                else:
                    r_performance = 0.0
            elif target_score > 0:
                r_performance = max(0.0, min(current_score / target_score, 1.0))
            else:
                r_performance = 0.0
        elif is_metric_minimization(metric_name):
            r_performance = 1.0 / (1.0 + max(current_score, 0.0))
        else:
            r_performance = max(0.0, min(current_score, 1.0))

        # Reward 3: Improvement (delta from previous best)
        # Calculate improvement considering metric direction (positive = better)
        score_improvement = calculate_score_improvement(current_score, best_score, metric_name)
        r_improvement = max(0.0, min(score_improvement * 10, 1.0))  # Scale to 0-1

        # Reward 4: Execution Semantics (no errors, fast execution)
        avg_execution_time = (
            sum(r.execution_time for r in dev_results) / total_components
            if total_components > 0
            else 0.0
        )
        r_semantics = 1.0 - min(avg_execution_time / 300.0, 1.0)  # Normalize by 5min timeout

        # Reward 5: Diversity
        # Encourages trying different types of components (e.g. not just 5 XGBoosts)
        unique_types = len(
            {c.get("type", "unknown") for c in failure_analysis["success_components"]}
        )
        r_diversity = min(unique_types / 3.0, 1.0)  # Target: at least 3 different types working

        # Reward 6: independent validation robustness. Public/private
        # leaderboard feedback is unavailable during an MLE-bench workflow.
        validation_score = state.get("overall_validation_score")
        robustness_abstained = bool(state.get("robustness_abstained", False))

        if robustness_abstained:
            # This term is removed and the remaining weights are renormalized
            # below. An ablation must not masquerade as a perfect validation.
            r_robustness = 0.0
        elif isinstance(validation_score, (int, float)):
            r_robustness = max(0.0, min(float(validation_score), 1.0))
        else:
            r_robustness = 1.0

        # Combined reward (weighted)
        # Performance-focused weights: prioritize score improvement for aggressive optimization.
        # MLE-bench mode: prioritize held-out CV/OOF performance.
        if run_mode == "mlebench" or "medal" in objective:
            weights = {
                "functional": 0.15,      # Reduced: working code is baseline
                "performance": 0.50,     # Held-out CV/OOF quality is key
                "improvement": 0.10,     # Increased: reward progress
                "semantics": 0.10,       # Reduced slightly
                "diversity": 0.05,       # Reduced: focus on what works
                "robustness": 0.10,      # Increased: prevent overfitting
            }
        else:
            # Standard Kaggle mode: heavily prioritize performance/score
            weights = {
                "functional": 0.15,      # Reduced from 0.25
                "performance": 0.55,     # Increased from 0.40 - main driver
                "improvement": 0.15,     # Increased from 0.10 - reward progress
                "semantics": 0.05,       # Maintained
                "diversity": 0.05,       # Reduced from 0.10
                "robustness": 0.05,      # Reduced from 0.10
            }

        reward_values = {
            "functional": r_functional,
            "performance": r_performance,
            "improvement": r_improvement,
            "semantics": r_semantics,
            "diversity": r_diversity,
            "robustness": r_robustness,
        }
        active_weights = dict(weights)
        if robustness_abstained:
            active_weights.pop("robustness", None)
        active_weight_total = sum(active_weights.values()) or 1.0
        r_combined = sum(
            weight * reward_values[name] for name, weight in active_weights.items()
        ) / active_weight_total

        rewards = {
            "r_functional": r_functional,
            "r_performance": r_performance,
            "r_improvement": r_improvement,
            "r_semantics": r_semantics,
            "r_diversity": r_diversity,
            "r_robustness": r_robustness,
            "r_combined": r_combined,
        }

        print(
            f"   📊 Rewards: functional={r_functional:.2f}, performance={r_performance:.2f}, "
            f"diversity={r_diversity:.2f}, robustness={r_robustness:.2f}, combined={r_combined:.3f}"
        )

        return rewards
