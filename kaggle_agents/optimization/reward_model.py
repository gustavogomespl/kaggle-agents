"""
Reward Model for Prompt Optimization.

This module defines metrics for evaluating agent performance,
used by DSPy to optimize prompts automatically.
"""

import re

import dspy



class PlannerRewardModel:
    """
    Reward model for the Planner Agent.

    Evaluates ablation plans based on:
    - Component diversity (different types)
    - Impact estimation quality
    - Completeness (all critical components covered)
    - Feasibility (components are implementable)
    """

    def __init__(self, weight_diversity: float = 0.3, weight_impact: float = 0.4, weight_completeness: float = 0.3):
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

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """
        Evaluate a planner prediction.

        Args:
            example: DSPy example with expected outputs
            prediction: DSPy prediction from the model

        Returns:
            Reward score (0-1)
        """
        # Extract ablation plan from prediction
        plan = prediction.ablation_plan if hasattr(prediction, 'ablation_plan') else []

        if not plan:
            return 0.0

        # Calculate sub-scores
        diversity_score = self._evaluate_diversity(plan)
        impact_score = self._evaluate_impact_estimates(plan)
        completeness_score = self._evaluate_completeness(plan, example)

        # Weighted average
        total_score = (
            self.weight_diversity * diversity_score +
            self.weight_impact * impact_score +
            self.weight_completeness * completeness_score
        )

        return total_score

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
        diversity = len(plan_types & expected_types) / len(expected_types)

        return diversity

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
        else:
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

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """
        Evaluate developer code generation.

        Args:
            example: DSPy example
            prediction: Prediction with generated code

        Returns:
            Reward score (0-1)
        """
        code = prediction.code if hasattr(prediction, 'code') else ""

        if not code:
            return 0.0

        # Sub-scores
        syntax_score = self._check_syntax(code)
        structure_score = self._evaluate_structure(code)

        # Average
        total_score = (syntax_score + structure_score) / 2

        return total_score

    def _check_syntax(self, code: str) -> float:
        """
        Check if code is syntactically correct.

        Args:
            code: Python code string

        Returns:
            1.0 if valid syntax, 0.0 otherwise
        """
        try:
            compile(code, '<string>', 'exec')
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
        if re.search(r'^import\s+\w+', code, re.MULTILINE):
            score += 0.3

        # Has function definitions
        if re.search(r'^def\s+\w+', code, re.MULTILINE):
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

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """
        Evaluate validation results.

        Args:
            example: DSPy example
            prediction: Prediction with validation results

        Returns:
            Reward score (0-1)
        """
        results = prediction.validation_results if hasattr(prediction, 'validation_results') else []

        if not results:
            return 0.0

        passed_count = sum(1 for r in results if self._get_passed(r))
        total_count = len(results)

        if total_count == 0:
            return 0.0

        # Pass rate
        pass_rate = passed_count / total_count

        return pass_rate

    def _get_passed(self, result) -> bool:
        """Extract 'passed' field from result."""
        if isinstance(result, dict):
            return result.get("passed", False)
        elif hasattr(result, "passed"):
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

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """
        Evaluate based on Kaggle score.

        Args:
            example: DSPy example with competition info
            prediction: Prediction with submission results

        Returns:
            Reward score (0-1)
        """
        # Get submission result
        submission = prediction.submission if hasattr(prediction, 'submission') else None

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
        else:
            # Scale from 1.0 at target to 0.0 at 100%
            return max(0.0, 1.0 - (percentile - self.target_percentile) / (100 - self.target_percentile))


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

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
        """
        Evaluate using combined metrics.

        Args:
            example: DSPy example
            prediction: Prediction with multiple outputs

        Returns:
            Combined reward score (0-1)
        """
        # Kaggle score component
        kaggle_score = self.kaggle_model(example, prediction)

        # Code quality component
        quality_score = self.developer_model(example, prediction)

        # CV score component (if available)
        cv_score = 0.0
        if hasattr(prediction, 'cv_score') and prediction.cv_score is not None:
            # Normalize CV score (assume higher is better)
            # This would need competition-specific normalization
            cv_score = min(prediction.cv_score, 1.0)

        # Weighted combination
        total_score = (
            self.weight_kaggle * kaggle_score +
            self.weight_quality * quality_score +
            self.weight_cv * cv_score
        )

        return total_score


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
