"""
Log Parser for Training Feedback Loop.

This module parses structured logs from training code and extracts
actionable feedback for the LLM to improve model performance.

Log Format:
- [LOG:FOLD] fold=N score=X time=Y
- [LOG:OPTUNA] trial=N score=X time=Y params={...}
- [LOG:TIMING] step=NAME time=X cumulative=Y
- [LOG:FEATURES] top=[...] importances=[...]
- [LOG:MEMORY] current_mb=X peak_mb=Y
- [LOG:HYPERPARAMS] params={...}
- [LOG:CV_SUMMARY] mean=X std=Y scores=[...]
- [LOG:WARNING] message=...
- [LOG:ERROR] message=...
"""

import re
import ast
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainingFeedback:
    """Structured feedback from training logs for LLM improvement."""

    # CV Performance
    fold_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Optuna Trials
    optuna_trials: list[dict[str, Any]] = field(default_factory=list)
    best_optuna_trial: Optional[dict[str, Any]] = None

    # Feature Importances
    top_features: list[str] = field(default_factory=list)
    feature_importances: list[float] = field(default_factory=list)
    zero_importance_features: list[str] = field(default_factory=list)

    # Timing Breakdown
    timing_breakdown: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    slowest_step: str = ""

    # Memory Usage
    memory_current_mb: float = 0.0
    memory_peak_mb: float = 0.0

    # Hyperparameters Used
    hyperparams: dict[str, Any] = field(default_factory=dict)

    # Issues Detected
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Raw logs for context
    raw_output: str = ""

    def has_data(self) -> bool:
        """Check if any meaningful data was parsed."""
        return bool(
            self.fold_scores
            or self.optuna_trials
            or self.timing_breakdown
            or self.hyperparams
        )

    def get_improvement_suggestions(self) -> list[str]:
        """Generate improvement suggestions based on parsed data."""
        suggestions = []

        # High variance detection
        if self.cv_std > 0.02 and self.fold_scores:
            suggestions.append(
                f"High CV variance (std={self.cv_std:.4f}). Consider: "
                "increase regularization (reg_alpha, reg_lambda), reduce max_depth, "
                "or add cross-validation stratification."
            )

        # Low score detection
        if self.cv_mean > 0 and self.cv_mean < 0.6:
            suggestions.append(
                f"Low CV score ({self.cv_mean:.4f}). Consider: "
                "add more features, increase model complexity, "
                "or try a different model architecture."
            )

        # Overfitting detection (fold variance)
        if len(self.fold_scores) >= 3:
            fold_range = max(self.fold_scores) - min(self.fold_scores)
            if fold_range > 0.05:
                suggestions.append(
                    f"Large fold score range ({fold_range:.4f}). Possible overfitting. "
                    "Consider: reduce n_estimators, add early stopping, "
                    "increase min_child_samples."
                )

        # Optuna insights
        if self.optuna_trials and len(self.optuna_trials) >= 2:
            scores = [t.get("score", 0) for t in self.optuna_trials]
            if max(scores) - min(scores) < 0.001:
                suggestions.append(
                    "Optuna trials show minimal score variation. "
                    "Consider: expand hyperparameter search space, "
                    "or the model may have converged."
                )

        # Feature insights
        if self.zero_importance_features:
            suggestions.append(
                f"Found {len(self.zero_importance_features)} zero-importance features. "
                f"Consider removing: {self.zero_importance_features[:5]}"
            )

        # Memory issues
        if self.memory_peak_mb > 8000:  # > 8GB
            suggestions.append(
                f"High memory usage ({self.memory_peak_mb:.0f} MB). "
                "Consider: reduce batch size, use float32 instead of float64, "
                "or process data in chunks."
            )

        # Slow training
        if self.total_time > 600:  # > 10 minutes
            slow_step = self.slowest_step or "unknown"
            suggestions.append(
                f"Long training time ({self.total_time:.0f}s). "
                f"Slowest step: {slow_step}. "
                "Consider: reduce n_estimators for tuning, use GPU, "
                "or subsample data for hyperparameter search."
            )

        # Use best Optuna params
        if self.best_optuna_trial and self.best_optuna_trial.get("params"):
            suggestions.append(
                f"Best Optuna params found: {self.best_optuna_trial['params']}. "
                "Use these as starting point for next iteration."
            )

        return suggestions


def _safe_parse_dict(s: str) -> dict[str, Any]:
    """Safely parse a dictionary string."""
    try:
        # Replace single quotes with double quotes for JSON compatibility
        # But be careful with nested structures
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Try regex extraction for simple key=value patterns
    result = {}
    for match in re.finditer(r"'?(\w+)'?\s*:\s*([^,}]+)", s):
        key = match.group(1)
        value = match.group(2).strip().strip("'\"")
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def _safe_parse_list(s: str) -> list:
    """Safely parse a list string."""
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass

    # Fallback: extract items
    items = re.findall(r"'([^']+)'|\"([^\"]+)\"|([\d.]+)", s)
    result = []
    for groups in items:
        for g in groups:
            if g:
                try:
                    result.append(float(g))
                except ValueError:
                    result.append(g)
                break
    return result


def parse_training_logs(stdout: str) -> TrainingFeedback:
    """
    Parse structured training logs and extract feedback.

    Args:
        stdout: Standard output from training code execution

    Returns:
        TrainingFeedback with parsed data
    """
    feedback = TrainingFeedback(raw_output=stdout)

    for line in stdout.splitlines():
        line = line.strip()

        # Parse [LOG:FOLD]
        if "[LOG:FOLD]" in line:
            match = re.search(
                r"\[LOG:FOLD\]\s*fold=(\d+)\s+score=([\d.]+)\s+time=([\d.]+)",
                line,
            )
            if match:
                score = float(match.group(2))
                feedback.fold_scores.append(score)

        # Parse [LOG:OPTUNA]
        elif "[LOG:OPTUNA]" in line:
            match = re.search(
                r"\[LOG:OPTUNA\]\s*trial=(\d+)\s+score=([\d.]+)\s+time=([\d.]+)\s+params=(.+)",
                line,
            )
            if match:
                trial = {
                    "trial": int(match.group(1)),
                    "score": float(match.group(2)),
                    "time": float(match.group(3)),
                    "params": _safe_parse_dict(match.group(4)),
                }
                feedback.optuna_trials.append(trial)

        # Parse [LOG:TIMING]
        elif "[LOG:TIMING]" in line:
            match = re.search(
                r"\[LOG:TIMING\]\s*step=(\w+)\s+time=([\d.]+)\s+cumulative=([\d.]+)",
                line,
            )
            if match:
                step = match.group(1)
                time_val = float(match.group(2))
                cumulative = float(match.group(3))
                feedback.timing_breakdown[step] = time_val
                feedback.total_time = max(feedback.total_time, cumulative)

        # Parse [LOG:FEATURES]
        elif "[LOG:FEATURES]" in line:
            match = re.search(
                r"\[LOG:FEATURES\]\s*top=(\[.+?\])\s+importances=(\[.+?\])",
                line,
            )
            if match:
                feedback.top_features = _safe_parse_list(match.group(1))
                feedback.feature_importances = _safe_parse_list(match.group(2))

        # Parse [LOG:MEMORY]
        elif "[LOG:MEMORY]" in line:
            match = re.search(
                r"\[LOG:MEMORY\]\s*current_mb=([\d.]+)\s+peak_mb=([\d.]+)",
                line,
            )
            if match:
                feedback.memory_current_mb = float(match.group(1))
                feedback.memory_peak_mb = float(match.group(2))

        # Parse [LOG:HYPERPARAMS]
        elif "[LOG:HYPERPARAMS]" in line:
            match = re.search(r"\[LOG:HYPERPARAMS\]\s*params=(.+)", line)
            if match:
                feedback.hyperparams = _safe_parse_dict(match.group(1))

        # Parse [LOG:CV_SUMMARY]
        elif "[LOG:CV_SUMMARY]" in line:
            match = re.search(
                r"\[LOG:CV_SUMMARY\]\s*mean=([\d.]+)\s+std=([\d.]+)\s+scores=(\[.+?\])",
                line,
            )
            if match:
                feedback.cv_mean = float(match.group(1))
                feedback.cv_std = float(match.group(2))
                scores = _safe_parse_list(match.group(3))
                if scores and not feedback.fold_scores:
                    feedback.fold_scores = [float(s) for s in scores if isinstance(s, (int, float))]

        # Parse [LOG:WARNING]
        elif "[LOG:WARNING]" in line:
            match = re.search(r"\[LOG:WARNING\]\s*message=(.+)", line)
            if match:
                feedback.warnings.append(match.group(1).strip())

        # Parse [LOG:ERROR]
        elif "[LOG:ERROR]" in line:
            match = re.search(r"\[LOG:ERROR\]\s*message=(.+)", line)
            if match:
                feedback.errors.append(match.group(1).strip())

    # Post-processing: calculate derived values
    if feedback.fold_scores:
        if not feedback.cv_mean:
            import numpy as np
            feedback.cv_mean = float(np.mean(feedback.fold_scores))
            feedback.cv_std = float(np.std(feedback.fold_scores))

    if feedback.timing_breakdown:
        feedback.slowest_step = max(
            feedback.timing_breakdown,
            key=feedback.timing_breakdown.get,
        )

    if feedback.optuna_trials:
        feedback.best_optuna_trial = max(
            feedback.optuna_trials,
            key=lambda t: t.get("score", 0),
        )

    # Detect zero-importance features
    if feedback.feature_importances and feedback.top_features:
        for i, imp in enumerate(feedback.feature_importances):
            if imp <= 0 and i < len(feedback.top_features):
                feedback.zero_importance_features.append(feedback.top_features[i])

    return feedback


def format_feedback_for_llm(feedback: TrainingFeedback) -> str:
    """
    Format parsed training feedback into a prompt for LLM improvement.

    Args:
        feedback: Parsed training feedback

    Returns:
        Formatted string for LLM context
    """
    sections = []

    # Header
    sections.append("## Training Results Analysis\n")

    # CV Performance Section
    if feedback.fold_scores:
        sections.append("### CV Performance")
        sections.append(f"- **Mean Score**: {feedback.cv_mean:.6f}")
        sections.append(f"- **Std Dev**: {feedback.cv_std:.6f}")
        sections.append(f"- **Per-fold scores**: {[round(s, 4) for s in feedback.fold_scores]}")
        
        if feedback.cv_std > 0.02:
            sections.append(f"- ⚠️ **High variance detected** (std > 0.02)")
        if feedback.cv_std < 0.005 and feedback.cv_mean > 0.8:
            sections.append(f"- ✅ **Stable and good performance**")
        sections.append("")

    # Optuna Section
    if feedback.optuna_trials:
        sections.append("### Optuna Tuning Results")
        sections.append(f"- **Trials completed**: {len(feedback.optuna_trials)}")
        if feedback.best_optuna_trial:
            sections.append(f"- **Best trial**: Trial {feedback.best_optuna_trial.get('trial', '?')}")
            sections.append(f"- **Best score**: {feedback.best_optuna_trial.get('score', 0):.6f}")
            sections.append(f"- **Best params**: {feedback.best_optuna_trial.get('params', {})}")
        
        # Show score progression
        scores = [t.get("score", 0) for t in feedback.optuna_trials]
        sections.append(f"- **Score range**: {min(scores):.4f} - {max(scores):.4f}")
        sections.append("")

    # Hyperparameters Section
    if feedback.hyperparams:
        sections.append("### Final Hyperparameters Used")
        for k, v in feedback.hyperparams.items():
            sections.append(f"- `{k}`: {v}")
        sections.append("")

    # Feature Importance Section
    if feedback.top_features:
        sections.append("### Top Features by Importance")
        for i, (feat, imp) in enumerate(
            zip(feedback.top_features[:10], feedback.feature_importances[:10]), 1
        ):
            sections.append(f"{i}. `{feat}`: {imp:.4f}")
        
        if feedback.zero_importance_features:
            sections.append(f"\n⚠️ **Zero-importance features** (consider removing):")
            for feat in feedback.zero_importance_features[:10]:
                sections.append(f"- `{feat}`")
        sections.append("")

    # Timing Section
    if feedback.timing_breakdown:
        sections.append("### Execution Time Breakdown")
        for step, time_val in sorted(
            feedback.timing_breakdown.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            sections.append(f"- `{step}`: {time_val:.1f}s")
        sections.append(f"- **Total time**: {feedback.total_time:.1f}s")
        if feedback.slowest_step:
            sections.append(f"- **Bottleneck**: `{feedback.slowest_step}`")
        sections.append("")

    # Memory Section
    if feedback.memory_peak_mb > 0:
        sections.append("### Memory Usage")
        sections.append(f"- **Current**: {feedback.memory_current_mb:.0f} MB")
        sections.append(f"- **Peak**: {feedback.memory_peak_mb:.0f} MB")
        if feedback.memory_peak_mb > 8000:
            sections.append("- ⚠️ High memory usage detected")
        sections.append("")

    # Warnings and Errors
    if feedback.warnings or feedback.errors:
        sections.append("### Issues Detected")
        for warning in feedback.warnings:
            sections.append(f"- ⚠️ {warning}")
        for error in feedback.errors:
            sections.append(f"- ❌ {error}")
        sections.append("")

    # Improvement Suggestions
    suggestions = feedback.get_improvement_suggestions()
    if suggestions:
        sections.append("### Suggested Improvements")
        for i, suggestion in enumerate(suggestions, 1):
            sections.append(f"{i}. {suggestion}")
        sections.append("")

    return "\n".join(sections)

