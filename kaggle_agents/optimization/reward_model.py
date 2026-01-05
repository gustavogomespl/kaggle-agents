"""
Reward Model for Prompt Optimization.

This module defines metrics for evaluating agent performance,
used by DSPy to optimize prompts automatically.

Enhanced with:
- Execution-based rewards (CodeRL+ pattern)
- Structured log feedback integration
- Ablation study signals
"""

from __future__ import annotations

import json
import re
from typing import Any

import dspy

from ..utils.log_parser import TrainingFeedback, parse_training_logs


class PlannerRewardModel:
    """
    Reward model for the Planner Agent.

    Evaluates ablation plans based on:
    - Component diversity (different types)
    - Impact estimation quality
    - Completeness (all critical components covered)
    - Feasibility (components are implementable)
    """

    def __init__(
        self,
        weight_diversity: float = 0.3,
        weight_impact: float = 0.4,
        weight_completeness: float = 0.3,
    ):
        """
        Initialize reward model.

        Args:
            weight_diversity: Weight for diversity score
            weight_impact: Weight for impact estimation
            weight_completeness: Weight for completeness
        """
        self.weight_diversity = weight_diversity
        self.weight_impact = weight_impact
        self.weight_completeness = weight_completeness

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        Evaluate a planner prediction.

        Args:
            example: DSPy example with expected outputs
            prediction: DSPy prediction from the model
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Reward score (0-1)
        """
        # Extract ablation plan from prediction (may be JSON string)
        plan = prediction.ablation_plan if hasattr(prediction, "ablation_plan") else []
        if isinstance(plan, str):
            try:
                parsed = json.loads(plan)
                plan = parsed if isinstance(parsed, list) else []
            except Exception:
                plan = []

        if not plan:
            return 0.0

        # Calculate sub-scores
        diversity_score = self._evaluate_diversity(plan)
        impact_score = self._evaluate_impact_estimates(plan)
        completeness_score = self._evaluate_completeness(plan, example)

        # Weighted average
        return (
            self.weight_diversity * diversity_score
            + self.weight_impact * impact_score
            + self.weight_completeness * completeness_score
        )

    def _evaluate_diversity(self, plan: list) -> float:
        """
        Evaluate diversity of component types.

        Args:
            plan: List of ablation components

        Returns:
            Diversity score (0-1)
        """
        if not plan:
            return 0.0

        # Expected component types
        expected_types = {"feature_engineering", "model", "preprocessing", "ensemble"}

        # Count unique types in plan
        plan_types = set()
        for component in plan:
            if isinstance(component, dict):
                comp_type = component.get("component_type", "")
            elif hasattr(component, "component_type"):
                comp_type = component.component_type
            else:
                continue

            plan_types.add(comp_type)

        # Diversity = proportion of expected types covered
        return len(plan_types & expected_types) / len(expected_types)

    def _evaluate_impact_estimates(self, plan: list) -> float:
        """
        Evaluate quality of impact estimates.

        Args:
            plan: List of ablation components

        Returns:
            Impact estimation score (0-1)
        """
        if not plan:
            return 0.0

        valid_estimates = 0
        total = 0

        for component in plan:
            if isinstance(component, dict):
                impact = component.get("estimated_impact", 0)
            elif hasattr(component, "estimated_impact"):
                impact = component.estimated_impact
            else:
                continue

            total += 1

            # Valid impact is between 0 and 1 (0-100%)
            if 0 <= impact <= 1:
                valid_estimates += 1

        if total == 0:
            return 0.0

        return valid_estimates / total

    def _evaluate_completeness(self, plan: list, example: dspy.Example) -> float:
        """
        Evaluate completeness of the plan.

        Args:
            plan: List of ablation components
            example: Example with expected characteristics

        Returns:
            Completeness score (0-1)
        """
        if not plan:
            return 0.0

        # Minimum expected components
        min_components = 3

        if len(plan) >= min_components:
            return 1.0
        return len(plan) / min_components


class DeveloperRewardModel:
    """
    Reward model for the Developer Agent.

    Evaluates generated code based on:
    - Syntactic correctness (no syntax errors)
    - Execution success
    - Output artifacts created
    - Code quality (imports, structure)
    """

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        Evaluate developer code generation.

        Args:
            example: DSPy example
            prediction: Prediction with generated code
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Reward score (0-1)
        """
        code = prediction.code if hasattr(prediction, "code") else ""

        if not code:
            return 0.0

        # Sub-scores
        syntax_score = self._check_syntax(code)
        structure_score = self._evaluate_structure(code)

        # Average
        return (syntax_score + structure_score) / 2

    def _check_syntax(self, code: str) -> float:
        """
        Check if code is syntactically correct.

        Args:
            code: Python code string

        Returns:
            1.0 if valid syntax, 0.0 otherwise
        """
        try:
            compile(code, "<string>", "exec")
            return 1.0
        except SyntaxError:
            return 0.0

    def _evaluate_structure(self, code: str) -> float:
        """
        Evaluate code structure quality.

        Args:
            code: Python code string

        Returns:
            Structure score (0-1)
        """
        score = 0.0

        # Has imports
        if re.search(r"^import\s+\w+", code, re.MULTILINE):
            score += 0.3

        # Has function definitions
        if re.search(r"^def\s+\w+", code, re.MULTILINE):
            score += 0.3

        # Has main execution logic
        if len(code.strip()) > 100:  # Minimal length check
            score += 0.2

        # No obvious errors (no "Error", "Exception" in code comments/strings)
        if "Error:" not in code and "Exception:" not in code:
            score += 0.2

        return min(score, 1.0)


class ValidationRewardModel:
    """
    Reward model for the Robustness Agent.

    Evaluates validation results based on:
    - Number of checks passed
    - Severity of issues found
    - Quality of suggestions
    """

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        Evaluate validation results.

        Args:
            example: DSPy example
            prediction: Prediction with validation results
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Reward score (0-1)
        """
        results = prediction.validation_results if hasattr(prediction, "validation_results") else []

        if not results:
            return 0.0

        passed_count = sum(1 for r in results if self._get_passed(r))
        total_count = len(results)

        if total_count == 0:
            return 0.0

        # Pass rate
        return passed_count / total_count

    def _get_passed(self, result) -> bool:
        """Extract 'passed' field from result."""
        if isinstance(result, dict):
            return result.get("passed", False)
        if hasattr(result, "passed"):
            return result.passed
        return False


class KaggleScoreRewardModel:
    """
    Reward model based on actual Kaggle competition scores.

    This is the ultimate metric - evaluates based on:
    - Public leaderboard score
    - Cross-validation score
    - Score improvement over baseline
    - Percentile ranking
    """

    def __init__(self, target_percentile: float = 20.0):
        """
        Initialize Kaggle score reward model.

        Args:
            target_percentile: Target percentile (e.g., 20 for top 20%)
        """
        self.target_percentile = target_percentile

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        Evaluate based on Kaggle score.

        Args:
            example: DSPy example with competition info
            prediction: Prediction with submission results
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Reward score (0-1)
        """
        # Get submission result
        submission = prediction.submission if hasattr(prediction, "submission") else None

        if not submission:
            return 0.0

        # Extract percentile
        if isinstance(submission, dict):
            percentile = submission.get("percentile")
        elif hasattr(submission, "percentile"):
            percentile = submission.percentile
        else:
            return 0.0

        if percentile is None:
            return 0.0

        # Reward based on percentile
        # Top 20% = full score
        # Linear scaling below that
        if percentile <= self.target_percentile:
            return 1.0
        # Scale from 1.0 at target to 0.0 at 100%
        return max(
            0.0, 1.0 - (percentile - self.target_percentile) / (100 - self.target_percentile)
        )


class CombinedRewardModel:
    """
    Combined reward model using multiple metrics.

    Combines:
    - CV score (validation)
    - Kaggle public score
    - Code quality
    - Validation checks
    """

    def __init__(
        self,
        weight_cv: float = 0.3,
        weight_kaggle: float = 0.5,
        weight_quality: float = 0.2,
    ):
        """
        Initialize combined reward model.

        Args:
            weight_cv: Weight for CV score
            weight_kaggle: Weight for Kaggle score
            weight_quality: Weight for code quality
        """
        self.weight_cv = weight_cv
        self.weight_kaggle = weight_kaggle
        self.weight_quality = weight_quality

        self.kaggle_model = KaggleScoreRewardModel()
        self.developer_model = DeveloperRewardModel()

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        Evaluate using combined metrics.

        Args:
            example: DSPy example
            prediction: Prediction with multiple outputs
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Combined reward score (0-1)
        """
        # Kaggle score component
        kaggle_score = self.kaggle_model(example, prediction)

        # Code quality component
        quality_score = self.developer_model(example, prediction)

        # CV score component (if available)
        cv_score = 0.0
        if hasattr(prediction, "cv_score") and prediction.cv_score is not None:
            # Normalize CV score (assume higher is better)
            # This would need competition-specific normalization
            cv_score = min(prediction.cv_score, 1.0)

        # Weighted combination
        return (
            self.weight_kaggle * kaggle_score
            + self.weight_quality * quality_score
            + self.weight_cv * cv_score
        )


class ExecutionFeedbackRewardModel:
    """
    Reward model based on structured execution feedback.

    Uses TrainingFeedback from log_parser to calculate rich rewards:
    - CV performance and stability
    - Execution efficiency (time, memory)
    - Feature quality
    - Improvement over previous iteration

    This implements the CodeRL+ pattern with execution semantics.
    """

    def __init__(
        self,
        weight_cv_score: float = 0.35,
        weight_cv_stability: float = 0.15,
        weight_improvement: float = 0.20,
        weight_efficiency: float = 0.15,
        weight_feature_quality: float = 0.15,
        target_score: float = 0.90,
        time_budget_seconds: float = 600.0,
        memory_budget_mb: float = 8000.0,
    ):
        """
        Initialize execution feedback reward model.

        Args:
            weight_cv_score: Weight for CV score component
            weight_cv_stability: Weight for CV stability (low std)
            weight_improvement: Weight for improvement over baseline
            weight_efficiency: Weight for time/memory efficiency
            weight_feature_quality: Weight for feature usage quality
            target_score: Target CV score for normalization
            time_budget_seconds: Maximum expected training time
            memory_budget_mb: Maximum expected memory usage
        """
        self.weight_cv_score = weight_cv_score
        self.weight_cv_stability = weight_cv_stability
        self.weight_improvement = weight_improvement
        self.weight_efficiency = weight_efficiency
        self.weight_feature_quality = weight_feature_quality

        self.target_score = target_score
        self.time_budget = time_budget_seconds
        self.memory_budget = memory_budget_mb

        # Track history for improvement calculation
        self.score_history: list[float] = []

    def calculate_from_feedback(
        self,
        feedback: TrainingFeedback,
        previous_score: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate reward components from TrainingFeedback.

        Args:
            feedback: Parsed training feedback
            previous_score: Previous best CV score for improvement calc

        Returns:
            Dictionary with reward components and combined score
        """
        rewards = {}

        # 1. CV Score Reward (normalized by target)
        if feedback.cv_mean > 0:
            rewards["cv_score"] = min(feedback.cv_mean / self.target_score, 1.0)
        else:
            rewards["cv_score"] = 0.0

        # 2. CV Stability Reward (inverse of std, capped)
        if feedback.cv_std >= 0:
            # Lower std = higher reward
            # std=0 -> reward=1.0, std>=0.05 -> reward=0.0
            stability = max(0.0, 1.0 - (feedback.cv_std / 0.05))
            rewards["cv_stability"] = stability
        else:
            rewards["cv_stability"] = 0.0

        # 3. Improvement Reward
        if previous_score is not None and feedback.cv_mean > 0:
            improvement = feedback.cv_mean - previous_score
            if improvement > 0:
                # Scale: 0.01 improvement = full reward
                rewards["improvement"] = min(improvement / 0.01, 1.0)
            else:
                # Penalize regression slightly
                rewards["improvement"] = max(-0.5, improvement / 0.02)
        else:
            rewards["improvement"] = 0.5  # Neutral for first run

        # 4. Efficiency Reward (time and memory)
        time_score = 0.0
        memory_score = 0.0

        if feedback.total_time > 0:
            # Faster = better, but don't penalize too much
            time_ratio = feedback.total_time / self.time_budget
            time_score = max(0.0, 1.0 - time_ratio) if time_ratio <= 1.5 else 0.0

        if feedback.memory_peak_mb > 0:
            memory_ratio = feedback.memory_peak_mb / self.memory_budget
            memory_score = max(0.0, 1.0 - memory_ratio) if memory_ratio <= 1.5 else 0.0
        else:
            memory_score = 1.0  # No memory info = assume good

        rewards["efficiency"] = (time_score + memory_score) / 2

        # 5. Feature Quality Reward
        feature_score = 0.0
        if feedback.top_features and feedback.feature_importances:
            # Reward for having diverse important features
            nonzero_features = sum(1 for imp in feedback.feature_importances if imp > 0)
            if len(feedback.feature_importances) > 0:
                feature_diversity = nonzero_features / len(feedback.feature_importances)
                feature_score = feature_diversity

            # Penalize for too many zero-importance features
            if feedback.zero_importance_features:
                penalty = min(len(feedback.zero_importance_features) * 0.05, 0.3)
                feature_score = max(0, feature_score - penalty)

        rewards["feature_quality"] = feature_score

        # 6. Bonus for Optuna convergence
        if feedback.optuna_trials and len(feedback.optuna_trials) >= 3:
            optuna_scores = [t.get("score", 0) for t in feedback.optuna_trials]
            if max(optuna_scores) - min(optuna_scores) < 0.01:
                rewards["optuna_bonus"] = 0.1  # Small bonus for converged search
            else:
                rewards["optuna_bonus"] = 0.0
        else:
            rewards["optuna_bonus"] = 0.0

        # Calculate combined reward
        combined = (
            self.weight_cv_score * rewards["cv_score"]
            + self.weight_cv_stability * rewards["cv_stability"]
            + self.weight_improvement * rewards["improvement"]
            + self.weight_efficiency * rewards["efficiency"]
            + self.weight_feature_quality * rewards["feature_quality"]
            + rewards["optuna_bonus"]
        )

        rewards["combined"] = min(max(combined, 0.0), 1.0)

        return rewards

    def calculate_from_stdout(
        self,
        stdout: str,
        previous_score: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate reward directly from stdout string.

        Args:
            stdout: Training code stdout
            previous_score: Previous best score

        Returns:
            Reward components dictionary
        """
        feedback = parse_training_logs(stdout)
        return self.calculate_from_feedback(feedback, previous_score)

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        DSPy-compatible interface for reward calculation.

        Args:
            example: DSPy example (may contain previous_score)
            prediction: Prediction with stdout attribute
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Combined reward score
        """
        stdout = ""
        if hasattr(prediction, "stdout"):
            stdout = prediction.stdout
        elif hasattr(prediction, "execution_result"):
            stdout = getattr(prediction.execution_result, "stdout", "")

        previous_score = None
        if hasattr(example, "previous_score"):
            previous_score = example.previous_score

        rewards = self.calculate_from_stdout(stdout, previous_score)
        return rewards.get("combined", 0.0)

    def update_history(self, score: float) -> None:
        """Add score to history for tracking."""
        self.score_history.append(score)

    def get_best_score(self) -> float | None:
        """Get best score from history."""
        return max(self.score_history) if self.score_history else None


class AblationRewardModel:
    """
    Reward model for ablation study quality.

    Evaluates ablation studies based on:
    - Coverage of code components
    - Quality of insights generated
    - Actionability of suggestions
    """

    def __init__(self):
        """Initialize ablation reward model."""
        self.expected_components = {
            "feature_engineering",
            "preprocessing",
            "model_params",
            "cv_strategy",
            "ensemble",
        }

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, *args, **kwargs) -> float:
        """
        Evaluate ablation study quality.

        Args:
            example: DSPy example
            prediction: Prediction with ablation_results
            *args: Additional positional arguments (for DSPy compatibility)
            **kwargs: Additional keyword arguments (for DSPy compatibility)

        Returns:
            Reward score (0-1)
        """
        ablation_results = ""
        if hasattr(prediction, "ablation_results"):
            ablation_results = prediction.ablation_results
        elif hasattr(prediction, "stdout"):
            ablation_results = prediction.stdout

        if not ablation_results:
            return 0.0

        score = 0.0

        # Check for component coverage
        components_found = 0
        for component in self.expected_components:
            if component.lower() in ablation_results.lower():
                components_found += 1
        coverage_score = components_found / len(self.expected_components)
        score += 0.3 * coverage_score

        # Check for quantitative results
        if re.search(r"score.*[0-9]+\.[0-9]+", ablation_results, re.IGNORECASE):
            score += 0.2

        # Check for comparative analysis
        if any(
            word in ablation_results.lower()
            for word in ["better", "worse", "improvement", "decrease"]
        ):
            score += 0.2

        # Check for actionable suggestions
        if any(
            word in ablation_results.lower() for word in ["should", "recommend", "try", "consider"]
        ):
            score += 0.2

        # Check for code modifications identified
        if re.search(r"```|def |class ", ablation_results):
            score += 0.1

        return min(score, 1.0)


class ImprovementTrackingRewardModel:
    """
    Reward model that tracks improvements across iterations.

    Implements the ADK pattern of:
    - Tracking score progression
    - Rewarding consistent improvements
    - Identifying best solutions
    """

    def __init__(
        self,
        improvement_threshold: float = 0.001,
        regression_penalty: float = 0.5,
    ):
        """
        Initialize improvement tracking model.

        Args:
            improvement_threshold: Minimum improvement to reward
            regression_penalty: Penalty multiplier for score regression
        """
        self.improvement_threshold = improvement_threshold
        self.regression_penalty = regression_penalty

        # Iteration tracking
        self.iteration_scores: list[dict[str, Any]] = []
        self.best_score: float = 0.0
        self.best_solution_idx: int = -1

    def add_iteration(
        self,
        score: float,
        solution_code: str,
        feedback: TrainingFeedback | None = None,
    ) -> dict[str, float]:
        """
        Add an iteration and calculate rewards.

        Args:
            score: CV score for this iteration
            solution_code: The code that produced this score
            feedback: Optional parsed training feedback

        Returns:
            Reward components for this iteration
        """
        iteration = {
            "score": score,
            "code": solution_code,
            "feedback": feedback,
            "iteration": len(self.iteration_scores),
        }

        rewards = {}

        # Calculate improvement
        if self.iteration_scores:
            previous_score = self.iteration_scores[-1]["score"]
            improvement = score - previous_score

            if improvement >= self.improvement_threshold:
                rewards["improvement"] = min(improvement / 0.01, 1.0)
            elif improvement < 0:
                rewards["improvement"] = max(improvement / 0.01 * self.regression_penalty, -0.5)
            else:
                rewards["improvement"] = 0.1  # Small reward for no regression
        else:
            rewards["improvement"] = 0.5  # Neutral for first iteration

        # Is this the best solution?
        if score > self.best_score:
            self.best_score = score
            self.best_solution_idx = len(self.iteration_scores)
            rewards["is_best"] = 1.0
        else:
            rewards["is_best"] = 0.0

        # Progress toward target (assumes higher is better)
        rewards["absolute_score"] = min(score, 1.0)

        # Combined
        rewards["combined"] = (
            0.5 * rewards["improvement"]
            + 0.3 * rewards["absolute_score"]
            + 0.2 * rewards["is_best"]
        )

        iteration["rewards"] = rewards
        self.iteration_scores.append(iteration)

        return rewards

    def get_best_solution(self) -> dict[str, Any] | None:
        """Get the best solution found so far."""
        if self.best_solution_idx >= 0 and self.best_solution_idx < len(self.iteration_scores):
            return self.iteration_scores[self.best_solution_idx]
        return None

    def get_score_progression(self) -> list[float]:
        """Get list of scores across iterations."""
        return [it["score"] for it in self.iteration_scores]

    def get_improvement_trajectory(self) -> list[float]:
        """Get cumulative improvements over baseline."""
        if not self.iteration_scores:
            return []

        baseline = self.iteration_scores[0]["score"]
        return [it["score"] - baseline for it in self.iteration_scores]

    def reset(self) -> None:
        """Reset tracking state."""
        self.iteration_scores = []
        self.best_score = 0.0
        self.best_solution_idx = -1


# ==================== Convenience Functions ====================


def create_planner_metric() -> PlannerRewardModel:
    """Create a planner reward model."""
    return PlannerRewardModel()


def create_developer_metric() -> DeveloperRewardModel:
    """Create a developer reward model."""
    return DeveloperRewardModel()


def create_validation_metric() -> ValidationRewardModel:
    """Create a validation reward model."""
    return ValidationRewardModel()


def create_kaggle_metric(target_percentile: float = 20.0) -> KaggleScoreRewardModel:
    """Create a Kaggle score reward model."""
    return KaggleScoreRewardModel(target_percentile)


def create_combined_metric() -> CombinedRewardModel:
    """Create a combined reward model."""
    return CombinedRewardModel()


def create_execution_feedback_metric(
    target_score: float = 0.90,
    time_budget: float = 600.0,
    memory_budget: float = 8000.0,
) -> ExecutionFeedbackRewardModel:
    """
    Create an execution feedback reward model.

    This is the recommended model for RL-based prompt optimization
    as it uses rich signals from actual training execution.

    Args:
        target_score: Target CV score for normalization
        time_budget: Maximum expected training time in seconds
        memory_budget: Maximum expected memory usage in MB

    Returns:
        ExecutionFeedbackRewardModel instance
    """
    return ExecutionFeedbackRewardModel(
        target_score=target_score,
        time_budget_seconds=time_budget,
        memory_budget_mb=memory_budget,
    )


def create_ablation_metric() -> AblationRewardModel:
    """Create an ablation study reward model."""
    return AblationRewardModel()


def create_improvement_tracker(
    improvement_threshold: float = 0.001,
    regression_penalty: float = 0.5,
) -> ImprovementTrackingRewardModel:
    """
    Create an improvement tracking reward model.

    This model tracks score progression across iterations
    and rewards consistent improvements.

    Args:
        improvement_threshold: Minimum improvement to reward
        regression_penalty: Penalty multiplier for regression

    Returns:
        ImprovementTrackingRewardModel instance
    """
    return ImprovementTrackingRewardModel(
        improvement_threshold=improvement_threshold,
        regression_penalty=regression_penalty,
    )
