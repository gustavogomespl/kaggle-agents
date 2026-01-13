"""
Detection methods for Meta-Evaluator.

Contains stagnation detection, performance gap detection, and undertrained model detection.
"""

from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING, Any

from ...utils.csv_utils import read_csv_auto


if TYPE_CHECKING:
    from ...core.state import KaggleState


class DetectionMixin:
    """Mixin providing detection methods for meta-evaluation."""

    def _check_performance_gap_for_debug(self, state: KaggleState) -> dict[str, Any]:
        """
        Inner Loop Refinement: Detect when one model performs drastically worse.

        Triggers a dedicated debug iteration when:
        - Two or more models exist
        - Performance gap > 1.0 (for logloss-like metrics)

        This prevents moving forward with broken models in the ensemble.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with trigger_debug, worst_model, gap, debug_hints
        """
        dev_results = state.get("development_results", [])
        ablation_plan = state.get("ablation_plan", [])

        if not dev_results:
            return {"trigger_debug": False}

        model_scores = {}

        for i, result in enumerate(dev_results):
            # Get component info
            component = ablation_plan[i] if i < len(ablation_plan) else None
            component_type = component.component_type if component else "unknown"
            component_name = component.name if component else f"component_{i}"

            if component_type != "model":
                continue

            # Try to extract score from stdout
            # DevelopmentResult is a dataclass - use getattr for safety
            stdout = getattr(result, "stdout", "") or ""

            # Look for common score patterns
            patterns = [
                r"(?:CV|Validation|Val|OOF).*?(?:Score|Loss|logloss|LogLoss|RMSE|MAE|AUC).*?:\s*([\d.]+)",
                r"(?:Score|Loss|logloss|LogLoss).*?:\s*([\d.]+)",
                r"Final.*?(?:Score|Loss|Validation Performance).*?:\s*([\d.]+)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, stdout, re.IGNORECASE)
                if matches:
                    try:
                        model_scores[component_name] = float(matches[-1])
                        break
                    except ValueError:
                        continue

        # Need at least 2 models to compare
        if len(model_scores) < 2:
            return {"trigger_debug": False, "model_scores": model_scores}

        scores = list(model_scores.values())
        max_gap = max(scores) - min(scores)

        # For logloss (lower is better), gap > 1.0 is HUGE
        if max_gap > 1.0:
            # FIX: Respect metric direction when identifying best/worst models
            from ...core.config import is_metric_minimization

            competition_info = state.get("competition_info")
            metric_name = ""
            if competition_info:
                metric_name = getattr(competition_info, "evaluation_metric", "") or ""
            is_minimize = is_metric_minimization(metric_name) if metric_name else True

            if is_minimize:
                # For minimize metrics (logloss, rmse): highest score = worst
                worst_model = max(model_scores, key=model_scores.get)
                best_model = min(model_scores, key=model_scores.get)
            else:
                # For maximize metrics (accuracy, auc): lowest score = worst
                worst_model = min(model_scores, key=model_scores.get)
                best_model = max(model_scores, key=model_scores.get)

            debug_hints = [
                "Check if LabelEncoder class order is consistent with other models",
                "Verify class_weight='balanced' is appropriate for this metric",
                "Compare data preprocessing between models",
                "Check if same train/val splits are used (random_state)",
                "Verify the objective function matches the competition metric",
                "Check for data type mismatches (categorical vs numeric)",
            ]

            print("\n   📊 PERFORMANCE GAP DETECTED:")
            print(f"      Worst: {worst_model} = {model_scores[worst_model]:.4f}")
            print(f"      Best: {best_model} = {model_scores[best_model]:.4f}")
            print(f"      Gap: {max_gap:.2f}")

            return {
                "trigger_debug": True,
                "worst_model": worst_model,
                "best_model": best_model,
                "gap": max_gap,
                "model_scores": model_scores,
                "debug_hints": debug_hints,
                "action": "PAUSE_AND_DEBUG",
            }

        if max_gap > 0.5:
            # Moderate gap - warning only
            print(f"\n   ⚠️  Moderate performance gap ({max_gap:.2f}) between models")
            return {
                "trigger_debug": False,
                "gap": max_gap,
                "model_scores": model_scores,
                "warning": f"Moderate gap of {max_gap:.2f} detected",
            }

        return {"trigger_debug": False, "model_scores": model_scores}

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
                    recent_improvements.append(abs(float(improvement)))

            if recent_improvements:
                avg_improvement = sum(recent_improvements) / len(recent_improvements)
                result["avg_improvement"] = avg_improvement
                result["iterations_checked"] = len(recent_improvements)

                # Stagnation: improvement below threshold
                if avg_improvement < stagnation_threshold:
                    result["stagnated"] = True
                    result["trigger_sota_search"] = True
                    result["reason"] = f"stagnation: avg_improvement={avg_improvement:.4f} < {stagnation_threshold}"
                    print(f"\n   📉 STAGNATION DETECTED: avg improvement {avg_improvement:.4f} over last {len(recent_improvements)} iterations")

        # Check score gap: far from target after minimum iterations
        # NOTE: This runs INDEPENDENTLY of stagnation check, even in early iterations
        if current_iteration >= 2:  # After 2 iterations
            current_score = state.get("current_performance_score", 0.0)
            target_score = state.get("target_score")

            if target_score and isinstance(target_score, (int, float)) and float(target_score) > 0:
                try:
                    score_gap = abs(float(target_score) - float(current_score)) / float(target_score)
                    result["score_gap"] = score_gap

                    if score_gap > score_gap_threshold:
                        result["trigger_sota_search"] = True
                        if result["reason"]:
                            result["reason"] += f" AND score_gap={score_gap:.1%} > {score_gap_threshold:.0%}"
                        else:
                            result["reason"] = f"score_gap: {score_gap:.1%} > {score_gap_threshold:.0%}"
                        print(f"\n   📊 SCORE GAP DETECTED: {score_gap:.1%} from target after {current_iteration} iterations")
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
        from ...core.config import is_metric_minimization

        dev_results = state.get("development_results", [])
        if not dev_results:
            return None

        # Get the best CV score from successful results
        # Note: DevelopmentResult doesn't have cv_score attribute - extract from stdout
        cv_scores = []
        for result in dev_results:
            if result.success and result.stdout:
                # Extract CV score from stdout (pattern: "Final Validation Performance: X.XXXX")
                match = re.search(r"Final Validation Performance[:\s]+([0-9.]+)", result.stdout)
                if match:
                    try:
                        cv_scores.append(float(match.group(1)))
                    except ValueError:
                        pass

        if not cv_scores:
            return None

        # Determine metric and its direction
        competition_info = state.get("competition_info")
        metric_name = ""
        problem_type = ""
        n_classes = 2

        if competition_info:
            metric_name = str(getattr(competition_info, "evaluation_metric", "")).lower()
            problem_type = str(getattr(competition_info, "problem_type", "")).lower()

        # Determine if we're minimizing or maximizing
        is_minimize = is_metric_minimization(metric_name) if metric_name else True

        # Get best score based on metric direction
        if is_minimize:
            best_cv_score = min(cv_scores)
        else:
            best_cv_score = max(cv_scores)

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

        # Calculate random baselines based on metric type
        if is_minimize:
            # Minimization metrics (log_loss, RMSE, etc.)
            random_baselines = {
                "multiclass": -math.log(1 / max(n_classes, 2)),  # log_loss for random
                "binary": 0.693,  # -log(0.5) for binary log_loss
            }
            baseline_key = "multiclass" if n_classes > 2 else "binary"
            baseline = random_baselines.get(baseline_key, 4.0)

            # For minimization: score > threshold * baseline means undertrained
            threshold = float(os.environ.get("KAGGLE_AGENTS_UNDERTRAINED_THRESHOLD", "0.85"))
            is_undertrained = best_cv_score > baseline * threshold
            comparison_msg = f"Score {best_cv_score:.4f} is too high (within {int((1-threshold)*100)}% of random baseline {baseline:.4f})"
        else:
            # Maximization metrics (accuracy, F1, AUC, etc.)
            random_baselines = {
                "accuracy_multiclass": 1.0 / max(n_classes, 2),  # random accuracy
                "accuracy_binary": 0.5,
                "auc": 0.5,  # random AUC
                "f1": 0.0,  # worst F1
            }

            # Determine appropriate baseline
            if "auc" in metric_name or "roc" in metric_name:
                baseline = 0.5
            elif "f1" in metric_name or "precision" in metric_name or "recall" in metric_name:
                baseline = 0.0
            else:
                # Default to accuracy baseline
                baseline = 1.0 / max(n_classes, 2)

            # For maximization: score < threshold * optimal means undertrained
            # Use a different threshold logic: if score is close to random baseline
            threshold = float(os.environ.get("KAGGLE_AGENTS_UNDERTRAINED_THRESHOLD", "0.85"))
            # For maximize metrics, undertrained means score is within 15% above baseline
            # e.g., for binary accuracy: baseline=0.5, threshold=0.85 → 0.5 + 0.15*(1-0.5) = 0.575
            undertrained_ceiling = baseline + (1 - threshold) * (1.0 - baseline)
            is_undertrained = best_cv_score < undertrained_ceiling
            comparison_msg = f"Score {best_cv_score:.4f} is too low (below {undertrained_ceiling:.4f}, near random baseline {baseline:.4f})"

        if is_undertrained:
            direction = "minimize" if is_minimize else "maximize"
            print(f"   ⚠️ UNDERTRAINED MODEL DETECTED ({direction}): {comparison_msg}")
            return {
                "type": "UNDERTRAINED_MODEL",
                "severity": "critical",
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
                "planner_directive": "CRITICAL: Current model is near-random. Prioritize training convergence over new features.",
                "developer_directive": "CRITICAL: Model is undertrained. Check preprocessing, increase epochs, verify label encoding.",
            }

        return None
