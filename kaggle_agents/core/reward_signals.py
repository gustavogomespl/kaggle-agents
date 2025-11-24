"""
Reward Signals for Reinforcement Learning.

Implements CodeRL+ pattern for multi-faceted reward calculation.
Based on execution semantics alignment and performance metrics.
"""

from typing import Dict, List
from dataclasses import dataclass

from .state import KaggleState, DevelopmentResult


@dataclass
class RewardComponents:
    """Individual reward components."""

    functional: float  # Execution success rate
    performance: float  # Kaggle score
    improvement: float  # Delta from previous
    semantics: float  # Execution quality
    combined: float  # Weighted combination


class RewardCalculator:
    """
    Calculate reward signals for RL-based prompt optimization.

    Implements multi-level rewards:
    1. Functional correctness (binary: success/failure)
    2. Performance score (continuous: Kaggle metric)
    3. Score improvement (delta from baseline)
    4. Execution semantics (quality of execution)
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        target_score: float = 0.9238,
    ):
        """
        Initialize reward calculator.

        Args:
            weights: Reward component weights
            target_score: Target performance score
        """
        self.weights = weights or {
            "functional": 0.3,
            "performance": 0.5,
            "improvement": 0.1,
            "semantics": 0.1,
        }
        self.target_score = target_score

    def calculate(self, state: KaggleState) -> RewardComponents:
        """
        Calculate all reward components.

        Args:
            state: Current workflow state

        Returns:
            RewardComponents with all calculated rewards
        """
        dev_results = state.get("development_results", [])
        current_score = state.get("current_performance_score", 0.0)
        best_score = state.get("best_score", 0.0)

        # Calculate individual components
        r_functional = self._calculate_functional(dev_results)
        r_performance = self._calculate_performance(current_score)
        r_improvement = self._calculate_improvement(current_score, best_score)
        r_semantics = self._calculate_semantics(dev_results)

        # Calculate combined reward
        r_combined = (
            self.weights["functional"] * r_functional
            + self.weights["performance"] * r_performance
            + self.weights["improvement"] * r_improvement
            + self.weights["semantics"] * r_semantics
        )

        return RewardComponents(
            functional=r_functional,
            performance=r_performance,
            improvement=r_improvement,
            semantics=r_semantics,
            combined=r_combined,
        )

    def _calculate_functional(self, dev_results: List[DevelopmentResult]) -> float:
        """
        Calculate functional correctness reward.

        Binary reward based on execution success rate.

        Args:
            dev_results: Development results

        Returns:
            Reward in [0, 1]
        """
        if not dev_results:
            return 0.0

        successful = sum(1 for r in dev_results if r.success)
        return successful / len(dev_results)

    def _calculate_performance(self, current_score: float) -> float:
        """
        Calculate performance reward.

        Normalized by target score.

        Args:
            current_score: Current Kaggle score

        Returns:
            Reward in [0, 1]
        """
        if self.target_score <= 0:
            return 0.0

        return min(current_score / self.target_score, 1.0)

    def _calculate_improvement(
        self,
        current_score: float,
        previous_score: float,
    ) -> float:
        """
        Calculate improvement reward.

        Rewards positive delta from previous best.

        Args:
            current_score: Current score
            previous_score: Previous best score

        Returns:
            Reward in [0, 1]
        """
        improvement = current_score - previous_score

        if improvement <= 0:
            return 0.0

        # Scale improvement to [0, 1]
        # Assume 0.01 improvement = full reward
        return min(improvement / 0.01, 1.0)

    def _calculate_semantics(self, dev_results: List[DevelopmentResult]) -> float:
        """
        Calculate execution semantics reward (CodeRL+ pattern).

        Rewards based on execution quality:
        - Fast execution time
        - No warnings in output
        - Clean error-free execution

        Args:
            dev_results: Development results

        Returns:
            Reward in [0, 1]
        """
        if not dev_results:
            return 0.0

        semantics_scores = []

        for result in dev_results:
            score = 0.0

            # Component 1: Execution time (normalize by 5min timeout)
            if result.success:
                time_score = 1.0 - min(result.execution_time / 300.0, 1.0)
                score += 0.4 * time_score

            # Component 2: No errors
            if not result.errors:
                score += 0.3

            # Component 3: Clean stdout (no warnings)
            if result.stdout and "warning" not in result.stdout.lower():
                score += 0.3

            semantics_scores.append(score)

        return sum(semantics_scores) / len(semantics_scores)

    def to_dict(self, components: RewardComponents) -> Dict[str, float]:
        """Convert RewardComponents to dictionary."""
        return {
            "r_functional": components.functional,
            "r_performance": components.performance,
            "r_improvement": components.improvement,
            "r_semantics": components.semantics,
            "r_combined": components.combined,
        }


# RLPrompt Stabilization Techniques


class RewardNormalizer:
    """
    Normalize rewards for stable RL training (RLPrompt pattern).

    Implements:
    - Z-score normalization
    - Moving average baseline
    - Reward clipping
    """

    def __init__(
        self,
        window_size: int = 10,
        clip_range: tuple = (-3.0, 3.0),
    ):
        """
        Initialize normalizer.

        Args:
            window_size: Window for moving average
            clip_range: Min/max values for clipping
        """
        self.window_size = window_size
        self.clip_range = clip_range
        self.reward_history: List[float] = []

    def normalize(self, reward: float) -> float:
        """
        Normalize reward using z-score.

        Args:
            reward: Raw reward

        Returns:
            Normalized reward
        """
        # Add to history
        self.reward_history.append(reward)

        # Keep only recent window
        if len(self.reward_history) > self.window_size:
            self.reward_history = self.reward_history[-self.window_size :]

        # Calculate z-score
        if len(self.reward_history) < 2:
            return reward

        mean = sum(self.reward_history) / len(self.reward_history)
        variance = sum((r - mean) ** 2 for r in self.reward_history) / len(
            self.reward_history
        )
        std = variance**0.5

        if std < 1e-6:
            return 0.0

        z_score = (reward - mean) / std

        # Clip to range
        return max(self.clip_range[0], min(z_score, self.clip_range[1]))

    def reset(self):
        """Reset history."""
        self.reward_history.clear()


class PiecewiseReward:
    """
    Piecewise reward function (RLPrompt pattern).

    Provides different rewards for different performance ranges.
    """

    def __init__(
        self,
        thresholds: List[float] = None,
        rewards: List[float] = None,
    ):
        """
        Initialize piecewise reward.

        Args:
            thresholds: Performance thresholds
            rewards: Corresponding rewards
        """
        self.thresholds = thresholds or [0.0, 0.5, 0.75, 0.9]
        self.rewards = rewards or [0.0, 0.25, 0.5, 1.0]

    def calculate(self, performance: float) -> float:
        """
        Calculate piecewise reward.

        Args:
            performance: Performance score [0, 1]

        Returns:
            Piecewise reward
        """
        for i in range(len(self.thresholds) - 1, -1, -1):
            if performance >= self.thresholds[i]:
                return self.rewards[i]

        return 0.0
