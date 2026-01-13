"""
Reward calculation for Meta-Evaluator.

Contains methods for calculating RL reward signals (CodeRL+ pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.config import calculate_score_improvement


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
        - Performance (Kaggle score)
        - Code quality (execution semantics)

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis results

        Returns:
            Reward signals dictionary
        """
        print("\n   💰 Calculating reward signals...")

        dev_results = state.get("development_results", [])
        submissions = state.get("submissions", [])
        # Prefer MLE-bench grading when available (enables medal-oriented rewards).
        mlebench_grade = state.get("mlebench_grade")
        current_score = state.get("current_performance_score", 0.0)
        if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
            score = mlebench_grade.get("score")
            if isinstance(score, (int, float)):
                current_score = float(score)
        best_score = state.get("best_score", 0.0)
        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", "")).lower()

        # Reward 1: Functional Correctness (binary)
        total_components = len(dev_results)
        successful_components = len(failure_analysis["success_components"])
        r_functional = successful_components / total_components if total_components > 0 else 0.0

        # Reward 2: Performance (continuous, normalized 0-1)
        # Try to get dynamic target from state (e.g. from leaderboard), else default
        target_score = state.get("target_score")
        if target_score is None:
            target_score = 1.0
        elif isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except (ValueError, TypeError):
                target_score = 1.0
        elif not isinstance(target_score, (int, float)):
            target_score = 1.0

        # Medal-aware shaping (MLE-bench objective)
        r_medal = 0.0
        if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
            if mlebench_grade.get("gold_medal"):
                r_medal = 1.0
            elif mlebench_grade.get("silver_medal"):
                r_medal = 0.8
            elif mlebench_grade.get("bronze_medal"):
                r_medal = 0.6
            elif mlebench_grade.get("above_median"):
                r_medal = 0.4

        # Ensure current_score is numeric
        if isinstance(current_score, str):
            try:
                current_score = float(current_score)
            except (ValueError, TypeError):
                current_score = 0.0

        score_component = (
            min(float(current_score) / float(target_score), 1.0) if float(target_score) > 0 else 0.0
        )
        if run_mode == "mlebench" or "medal" in objective:
            # Blend medal attainment with raw score; medal dominates to keep the objective explicit.
            r_performance = min(0.7 * r_medal + 0.3 * score_component, 1.0)
        else:
            r_performance = score_component

        # Reward 3: Improvement (delta from previous best)
        # Get evaluation metric to handle both minimize and maximize metrics correctly
        competition_info = state.get("competition_info")
        metric_name = competition_info.evaluation_metric if competition_info else ""

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

        # Reward 6: Robustness/Overfitting Penalty
        # Penalize if Public LB score is much lower than Validation score
        validation_score = state.get("overall_validation_score", 0.0)
        public_score = 0.0
        if submissions:
            public_score = submissions[-1].public_score or 0.0

        # If we have both scores, check gap. If gap > 0.1, heavy penalty.
        gap = abs(validation_score - public_score)
        r_robustness = (
            1.0 - min(gap * 5, 1.0) if (validation_score > 0 and public_score > 0) else 1.0
        )

        # Combined reward (weighted)
        # Performance-focused weights: prioritize score improvement for aggressive optimization.
        # MLE-bench mode: speed and medal attainment matter more.
        if run_mode == "mlebench" or "medal" in objective:
            weights = {
                "functional": 0.15,      # Reduced: working code is baseline
                "performance": 0.50,     # Increased: medal achievement is key
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

        r_combined = (
            weights["functional"] * r_functional
            + weights["performance"] * r_performance
            + weights["improvement"] * r_improvement
            + weights["semantics"] * r_semantics
            + weights["diversity"] * r_diversity
            + weights["robustness"] * r_robustness
        )

        rewards = {
            "r_functional": r_functional,
            "r_performance": r_performance,
            "r_improvement": r_improvement,
            "r_semantics": r_semantics,
            "r_diversity": r_diversity,
            "r_robustness": r_robustness,
            "r_medal": r_medal,
            "r_combined": r_combined,
        }

        print(
            f"   📊 Rewards: functional={r_functional:.2f}, performance={r_performance:.2f}, "
            f"diversity={r_diversity:.2f}, robustness={r_robustness:.2f}, combined={r_combined:.3f}"
        )

        return rewards
