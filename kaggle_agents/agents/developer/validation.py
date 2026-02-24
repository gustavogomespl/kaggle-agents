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

    def _infer_metric_from_stdout(self, stdout: str) -> str | None:
        """Infer metric name from stdout patterns.

        Returns:
            Inferred metric name, or None if no pattern matched.
        """
        stdout_lower = stdout.lower()
        metric_patterns = [
            (r"(?:roc[_\s-]?auc|auroc)\s*[:=]", "auc"),
            (r"(?:log[_\s-]?loss|logloss)\s*[:=]", "log_loss"),
            (r"(?:accuracy)\s*[:=]", "accuracy"),
            (r"(?:rmse)\s*[:=]", "rmse"),
            (r"(?:mae)\s*[:=]", "mae"),
            (r"(?:f1[_\s-]?score|f1)\s*[:=]", "f1"),
        ]
        for pattern, metric in metric_patterns:
            if re.search(pattern, stdout_lower):
                return metric
        return None

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

        # When metric is unknown, try to infer from stdout
        metric_unknown = not metric_name or metric_name.lower() in ("unknown", "none", "")
        inferred_metric = None
        if metric_unknown:
            inferred_metric = self._infer_metric_from_stdout(exec_result.stdout)
            if inferred_metric:
                print(f"\n   🔍 Inferred metric from stdout: {inferred_metric}")
                metric_name = inferred_metric

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

        # Detect metric mismatch: scores look like different metrics
        # (e.g., baseline=0.95 AUC vs component=0.27 LogLoss)
        scores_look_mismatched = False
        if baseline_score not in (float("inf"), float("-inf")):
            one_high = max(cv_score, baseline_score) > 0.5
            one_low = min(cv_score, baseline_score) < 0.5
            large_gap = abs(cv_score - baseline_score) > 0.3
            scores_look_mismatched = one_high and one_low and large_gap

        if scores_look_mismatched and metric_unknown and not inferred_metric:
            print("\n   📊 Ablation Study (Hill Climbing):")
            print(f"      Metric:         {metric_name} (unknown)")
            print(f"      Baseline CV:    {baseline_score:.4f}")
            print(f"      Component CV:   {cv_score:.4f}")
            print("      ⚠️  Scores appear to use different metrics (likely mismatch)")
            print("      ✅ Component ACCEPTED (metric mismatch detected, keeping by default)")
            return True, None  # Return None to NOT update baseline with mismatched score

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
