"""
Score extraction and component validation.

Provides capabilities for extracting CV scores from stdout and
validating component improvements using hill climbing strategy.
"""

import math
import re
from typing import TYPE_CHECKING

from ...core.config import calculate_score_improvement, is_metric_minimization
from ...core.state import AblationComponent, KaggleState


if TYPE_CHECKING:
    from ...tools.code_executor import ExecutionResult


class ValidationMixin:
    """Mixin providing validation capabilities."""

    def _extract_cv_score(self, stdout: str) -> float | None:
        """
        Extract cross-validation score from stdout using regex patterns.

        Args:
            stdout: Standard output from code execution

        Returns:
            Extracted CV score, or None if not found
        """
        # Try multiple patterns to extract CV score
        number = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf)"
        patterns = [
            rf"CV Score.*?{number}",
            rf"Final Validation Performance:\s*{number}",
            rf"ROC-AUC.*?{number}",
            rf"Accuracy.*?{number}",
            rf"RMSE.*?{number}",
            rf"Mean.*?{number}\s*\(",  # Mean score with std
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if math.isnan(value) or math.isinf(value):
                        return None
                    return value
                except ValueError:
                    continue

        return None

    def _validate_component_improvement(
        self,
        component: AblationComponent,
        exec_result: "ExecutionResult",
        state: KaggleState,
    ) -> tuple[bool, float | None]:
        """
        Validate if component improves score using Hill Climbing strategy.

        Implements ablation studies by comparing CV score before and after component.

        Args:
            component: Component being tested
            exec_result: Execution result containing stdout
            state: Current workflow state

        Returns:
            (should_keep, new_score) - Whether to keep component and its CV score
        """
        competition_info = state.get("competition_info")
        metric_name = competition_info.evaluation_metric if competition_info else ""

        is_minimize = is_metric_minimization(metric_name)

        cv_score = self._extract_cv_score(exec_result.stdout)

        if cv_score is None:
            print("\n   📊 Ablation Study (Hill Climbing):")
            print(
                f"      Metric:         {metric_name} ({'↓' if is_minimize else '↑'} {'minimize' if is_minimize else 'maximize'})"
            )
            print("      ⚠️  No CV score found in stdout; skipping rollback and keeping component.")
            return True, None

        baseline_score = state.get("baseline_cv_score")
        if baseline_score is None:
            baseline_score = float("inf") if is_minimize else float("-inf")
        improvement = calculate_score_improvement(cv_score, baseline_score, metric_name)
        direction_symbol = "↓" if is_minimize else "↑"
        direction_text = "minimize" if is_minimize else "maximize"

        print("\n   📊 Ablation Study (Hill Climbing):")
        print(f"      Metric:         {metric_name} ({direction_symbol} {direction_text})")
        print(f"      Baseline CV:    {baseline_score:.4f}")
        print(f"      Component CV:   {cv_score:.4f}")
        print(f"      Improvement:    {improvement:+.4f}")

        min_improvement = 0.001
        should_keep = improvement >= min_improvement

        if not should_keep:
            print("      ❌ Component REJECTED (no improvement or negative impact)")
            print(f"      Reason: Delta ({improvement:+.4f}) < threshold ({min_improvement})")
        else:
            print("      ✅ Component ACCEPTED (positive improvement)")
            if baseline_score not in [float("inf"), float("-inf"), 0]:
                relative_gain = abs(improvement / baseline_score * 100)
                print(f"      Impact: {relative_gain:.2f}% relative improvement")

        return should_keep, cv_score
