"""
Detection methods for Meta-Evaluator.

Contains stagnation detection, performance gap detection, and undertrained model detection.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

from ...core.config import is_metric_minimization
from ...utils.csv_utils import read_csv_auto


if TYPE_CHECKING:
    from ...core.state import KaggleState


def _finite_score(value: Any) -> float | None:
    """Coerce a finite trusted score while excluding booleans."""
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _metric_name(state: KaggleState) -> str:
    """Resolve the declared metric name without consulting generated output."""
    metric_contract = state.get("metric_contract") or {}
    if hasattr(metric_contract, "to_dict"):
        metric_contract = metric_contract.to_dict()
    if isinstance(metric_contract, dict) and metric_contract.get("metric_name"):
        return str(metric_contract["metric_name"]).strip().lower()

    competition_info = state.get("competition_info")
    if isinstance(competition_info, dict):
        return str(competition_info.get("evaluation_metric") or "").strip().lower()
    return str(getattr(competition_info, "evaluation_metric", "") or "").strip().lower()


def _metric_direction(state: KaggleState, metric_name: str) -> str:
    """Resolve a known metric direction, otherwise fail closed."""
    explicit = str(state.get("metric_direction") or "").strip().lower()
    if explicit in {"minimize", "maximize"}:
        return explicit

    metric_contract = state.get("metric_contract") or {}
    if hasattr(metric_contract, "to_dict"):
        metric_contract = metric_contract.to_dict()
    if isinstance(metric_contract, dict):
        lower_better = metric_contract.get("is_lower_better")
        if isinstance(lower_better, bool):
            return "minimize" if lower_better else "maximize"

    if not metric_name:
        return "unknown"
    if is_metric_minimization(metric_name):
        return "minimize"
    known_maximize = (
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "average_precision",
        "map",
        "r2",
        "dice",
        "iou",
    )
    if any(name in metric_name for name in known_maximize):
        return "maximize"
    return "unknown"


def _trusted_available_scores(state: KaggleState) -> dict[str, float]:
    """Intersect recomputed scores with explicit OOF eligibility."""
    explicit = state.get("trusted_component_scores")
    availability = state.get("oof_availability")
    if not isinstance(explicit, dict) or not isinstance(availability, dict):
        return {}

    scores: dict[str, float] = {}
    for name, value in explicit.items():
        component_name = str(name)
        if availability.get(name) is not True and availability.get(component_name) is not True:
            continue
        raw_score = value
        if isinstance(value, dict):
            raw_score = value.get("score", value.get("cv_score"))
        score = _finite_score(raw_score)
        if score is not None:
            scores[component_name] = score
    return scores


class DetectionMixin:
    """Mixin providing detection methods for meta-evaluation."""

    def _check_performance_gap_for_debug(self, state: KaggleState) -> dict[str, Any]:
        """
        Compare trusted OOF scores and emit non-blocking diagnostic guidance.

        Args:
            state: Current workflow state

        Returns:
            Advisory comparison. A natural score gap never pauses the workflow.
        """
        model_scores = _trusted_available_scores(state)
        if len(model_scores) < 2:
            return {
                "trigger_debug": False,
                "model_scores": model_scores,
                "abstained": True,
                "reason": "fewer_than_two_trusted_oof_scores",
                "advisory_only": True,
            }

        metric_name = _metric_name(state)
        direction = _metric_direction(state, metric_name)
        if direction == "unknown":
            return {
                "trigger_debug": False,
                "model_scores": model_scores,
                "abstained": True,
                "reason": "metric_direction_unavailable",
                "advisory_only": True,
            }

        if direction == "minimize":
            best_model = min(model_scores, key=model_scores.get)
            worst_model = max(model_scores, key=model_scores.get)
        else:
            best_model = max(model_scores, key=model_scores.get)
            worst_model = min(model_scores, key=model_scores.get)

        directed_regret = (
            model_scores[worst_model] - model_scores[best_model]
            if direction == "minimize"
            else model_scores[best_model] - model_scores[worst_model]
        )
        scale = max(
            abs(model_scores[best_model]),
            abs(model_scores[worst_model]),
            1e-12,
        )
        normalized_regret = max(0.0, directed_regret) / scale
        significant_regret = normalized_regret >= 0.5

        result: dict[str, Any] = {
            "trigger_debug": False,
            "model_scores": model_scores,
            "abstained": False,
            "advisory_only": True,
            "metric_name": metric_name,
            "metric_direction": direction,
            "best_model": best_model,
            "worst_model": worst_model,
            "gap": normalized_regret,
            "absolute_gap": directed_regret,
            "normalized_regret": normalized_regret,
            "significant_regret": significant_regret,
        }
        if significant_regret:
            result.update(
                {
                    "warning": (
                        "Large normalized regret in trusted OOF scores; inspect "
                        "fold diagnostics before deciding whether to revise the model."
                    ),
                    "debug_hints": [
                        "Compare identical OOF folds and metric implementations",
                        "Verify prediction/label alignment using canonical IDs",
                        "Inspect fold-level variance before changing the model",
                    ],
                    "action": "ADVISORY_REVIEW",
                }
            )
        return result

    def _detect_stagnation(self, state: KaggleState) -> dict[str, Any]:
        """
        Detect if progress has stagnated over recent iterations.

        Triggers SOTA search when:
        1. Stagnation: avg improvement < threshold over last N iterations
        2. Score gap: current score is far from target after minimum iterations

        Args:
            state: Current workflow state

        Returns:
            Dict with stagnation info and SOTA search trigger
        """
        iteration_memory = state.get("iteration_memory", [])
        current_iteration = state.get("current_iteration", 0)
        config = self.config.iteration

        # Get stagnation config (more aggressive defaults to detect issues faster)
        # FIX: Lowered thresholds to trigger exploration earlier
        stagnation_window = getattr(config, "stagnation_window", 2)  # Was 3
        stagnation_threshold = getattr(config, "stagnation_threshold", 0.005)  # Was 0.01
        score_gap_threshold = getattr(config, "score_gap_threshold", 0.15)  # Was 0.3

        result = {
            "stagnated": False,
            "trigger_sota_search": False,
            "reason": None,
            "avg_improvement": 0.0,
            "score_gap": 0.0,
            "iterations_checked": 0,
        }

        # Check stagnation: avg improvement over last N iterations
        # Only run if we have enough iterations for meaningful stagnation detection
        if len(iteration_memory) >= stagnation_window:
            recent_improvements = []
            for memory in iteration_memory[-stagnation_window:]:
                # IterationMemory is a dataclass, use attribute access (not dict.get())
                improvement = getattr(memory, "score_improvement", 0)
                if isinstance(improvement, (int, float)):
                    # Direction is already normalized by
                    # calculate_score_improvement: positive means better.
                    # A large regression must trigger exploration, not look
                    # like progress because of an absolute value.
                    recent_improvements.append(float(improvement))

            if recent_improvements:
                avg_improvement = sum(recent_improvements) / len(recent_improvements)
                result["avg_improvement"] = avg_improvement
                result["iterations_checked"] = len(recent_improvements)

                # Stagnation: improvement below threshold
                if avg_improvement < stagnation_threshold:
                    result["stagnated"] = True
                    result["trigger_sota_search"] = True
                    result["reason"] = (
                        f"stagnation: avg_improvement={avg_improvement:.4f} < {stagnation_threshold}"
                    )
                    print(
                        f"\n   📉 STAGNATION DETECTED: avg improvement {avg_improvement:.4f} over last {len(recent_improvements)} iterations"
                    )

        # Check score gap: far from target after minimum iterations
        # NOTE: This runs INDEPENDENTLY of stagnation check, even in early iterations
        if current_iteration >= 2:  # After 2 iterations
            current_score = state.get("current_performance_score", 0.0)
            target_score = state.get("target_score")

            if target_score and isinstance(target_score, (int, float)) and float(target_score) > 0:
                try:
                    score_gap = abs(float(target_score) - float(current_score)) / float(
                        target_score
                    )
                    result["score_gap"] = score_gap

                    if score_gap > score_gap_threshold:
                        result["trigger_sota_search"] = True
                        if result["reason"]:
                            result["reason"] += (
                                f" AND score_gap={score_gap:.1%} > {score_gap_threshold:.0%}"
                            )
                        else:
                            result["reason"] = (
                                f"score_gap: {score_gap:.1%} > {score_gap_threshold:.0%}"
                            )
                        print(
                            f"\n   📊 SCORE GAP DETECTED: {score_gap:.1%} from target after {current_iteration} iterations"
                        )
                except (TypeError, ValueError):
                    pass

        if result["trigger_sota_search"]:
            print(f"   🔍 TRIGGERING SOTA SEARCH: {result['reason']}")

        return result

    def _detect_undertrained_models(
        self,
        state: KaggleState,
    ) -> dict[str, Any] | None:
        """
        Detect if model performance indicates insufficient training.

        Compares CV score against random baseline for the problem type,
        respecting the metric direction (minimize vs maximize).

        Args:
            state: Current workflow state

        Returns:
            Diagnostic dict if undertrained, None otherwise
        """
        metric_name = _metric_name(state)
        normalized_metric = metric_name.replace("-", "_").replace(" ", "_")
        if "logloss" in normalized_metric or "log_loss" in normalized_metric:
            metric_kind = "logloss"
            is_minimize = True
        elif "auc" in normalized_metric:
            metric_kind = "auc"
            is_minimize = False
        elif "accuracy" in normalized_metric:
            metric_kind = "accuracy"
            is_minimize = False
        else:
            # Random baselines are not comparable for unbounded regression
            # metrics (RMSE/MAE) or undeclared/custom metrics.
            return None

        trusted_scores = [
            score
            for field in ("best_single_model_score", "baseline_cv_score")
            if (score := _finite_score(state.get(field))) is not None
        ]
        if not trusted_scores:
            return None
        best_cv_score = min(trusted_scores) if is_minimize else max(trusted_scores)

        n_classes = 2

        # Try to infer n_classes from sample submission
        sample_submission_path = state.get("sample_submission_path")
        if sample_submission_path:
            try:
                sample_sub = read_csv_auto(sample_submission_path)
                n_cols = sample_sub.shape[1]
                if n_cols > 2:
                    n_classes = n_cols - 1  # Subtract ID column
            except Exception:
                pass

        try:
            threshold = float(os.environ.get("KAGGLE_AGENTS_UNDERTRAINED_THRESHOLD", "0.85"))
        except ValueError:
            threshold = 0.85
        threshold = min(max(threshold, 0.0), 1.0)

        if metric_kind == "logloss":
            if best_cv_score < 0:
                return None
            baseline = math.log(max(n_classes, 2))
            is_undertrained = best_cv_score > baseline * threshold
            comparison_msg = (
                f"Trusted score {best_cv_score:.4f} is near or worse than "
                f"the random log-loss baseline {baseline:.4f}"
            )
        else:
            if not 0.0 <= best_cv_score <= 1.0:
                return None
            baseline = 0.5 if metric_kind == "auc" else 1.0 / max(n_classes, 2)
            undertrained_ceiling = baseline + (1 - threshold) * (1.0 - baseline)
            is_undertrained = best_cv_score < undertrained_ceiling
            comparison_msg = (
                f"Trusted score {best_cv_score:.4f} is near the random "
                f"{metric_kind} baseline {baseline:.4f}"
            )

        if is_undertrained:
            direction = "minimize" if is_minimize else "maximize"
            print(f"   ⚠️ UNDERTRAINED MODEL DETECTED ({direction}): {comparison_msg}")
            return {
                "type": "UNDERTRAINED_MODEL",
                "severity": "warning",
                "cv_score": best_cv_score,
                "random_baseline": baseline,
                "n_classes": n_classes,
                "metric_name": metric_name,
                "is_minimize": is_minimize,
                "message": comparison_msg,
                "suggestions": [
                    "Increase training epochs (model may not have converged)",
                    "Verify preprocessing matches model requirements (e.g., preprocess_input for pretrained models)",
                    "Check if learning rate is appropriate (may be too high or too low)",
                    "Ensure data augmentation isn't too aggressive",
                    "Verify class order alignment between predictions and ground truth labels",
                ],
                "planner_directive": (
                    "ADVISORY: trusted classification performance is near "
                    "random; inspect convergence before adding complexity."
                ),
                "developer_directive": (
                    "ADVISORY: inspect preprocessing, convergence, and label "
                    "alignment using trusted OOF diagnostics."
                ),
            }

        return None
