"""
Refinement logic for the Developer Agent.

Handles:
- Post-execution refinement iterations
- Training feedback analysis
- Score improvement tracking
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.config import calculate_score_improvement, is_metric_minimization
from ...core.state import AblationComponent, DevelopmentResult, KaggleState
from ...prompts.templates.developer_prompts import (
    DEVELOPER_CORE_IDENTITY,
    HARD_CONSTRAINTS,
)
from ...utils.llm_utils import get_text_content
from ...utils.log_parser import format_feedback_for_llm, parse_training_logs


if TYPE_CHECKING:
    from .agent import DeveloperAgent


class RefinementMixin:
    """Mixin providing refinement capabilities to DeveloperAgent."""

    def _get_refinement_iterations(self: DeveloperAgent, state: KaggleState) -> int:
        """Number of refinement iterations (can be reduced for fast/mlebench runs)."""
        run_mode = str(state.get("run_mode", "")).lower()
        if run_mode == "mlebench":
            return 0
        try:
            return max(0, min(int(os.getenv("REFINEMENT_ITERS", "2")), 3))
        except ValueError:
            return 2

    def _should_run_refinement(
        self: DeveloperAgent,
        component: AblationComponent,
        state: KaggleState,
        new_cv_score: float | None,
        execution_time_s: float | None,
        component_timeout_s: int,
    ) -> bool:
        """Decide whether to run expensive post-success refinements."""
        if component.component_type != "model":
            return False

        if not self.config.ablation.enable_refinement:
            return False

        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", "")).lower()

        # Default to speed-first in MLE-bench / medal objective.
        if run_mode == "mlebench" or "medal" in objective:
            return False

        # If the component already consumed most of its budget, don't re-run it.
        if execution_time_s is not None and component_timeout_s > 0:
            if float(execution_time_s) >= component_timeout_s * 0.70:
                return False

        # If a target score is defined and already reached, skip refinement.
        target_score = state.get("target_score")
        if isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except ValueError:
                target_score = None

        if isinstance(target_score, (int, float)) and isinstance(new_cv_score, (int, float)):
            metric_name = (
                state.get("competition_info").evaluation_metric
                if state.get("competition_info")
                else ""
            )
            if is_metric_minimization(metric_name):
                if float(new_cv_score) <= float(target_score):
                    return False
            elif float(new_cv_score) >= float(target_score):
                return False

        return True

    def _run_refinement_loop(
        self: DeveloperAgent,
        result: DevelopmentResult,
        component: AblationComponent,
        state: KaggleState,
        new_cv_score: float | None,
        working_dir: Path,
        state_updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the refinement loop to improve model performance.

        Args:
            result: Initial development result
            component: Component being refined
            state: Current workflow state
            new_cv_score: Current CV score
            working_dir: Working directory path
            state_updates: State updates dict to modify

        Returns:
            Updated state_updates dict
        """
        competition_info = state["competition_info"]

        print("\nADK Refinement Loop: Trying to improve score...")
        best_code = result.code
        best_score = new_cv_score if new_cv_score is not None else 0.0
        best_stdout = result.stdout

        refinement_iters = self._get_refinement_iterations(state)
        for i in range(refinement_iters):
            print(f"Refinement Iteration {i + 1}/{refinement_iters}")

            # Parse training logs for structured feedback
            training_feedback = parse_training_logs(best_stdout)
            formatted_feedback = ""

            if training_feedback.has_data():
                formatted_feedback = format_feedback_for_llm(training_feedback)
                print("📊 Training feedback extracted from logs")

                if training_feedback.fold_scores:
                    print(
                        f"   CV: {training_feedback.cv_mean:.4f} ± {training_feedback.cv_std:.4f}"
                    )
                if training_feedback.best_optuna_trial:
                    print(
                        f"   Best Optuna trial: {training_feedback.best_optuna_trial.get('score', 0):.4f}"
                    )
                if training_feedback.slowest_step:
                    print(f"   Slowest step: {training_feedback.slowest_step}")

                suggestions = training_feedback.get_improvement_suggestions()
                if suggestions:
                    print("   Suggestions:")
                    for s in suggestions[:3]:
                        print(f"   - {s[:80]}...")

            refine_prompt = f"""
## Current Performance
- CV Score: {best_score:.6f}

{formatted_feedback if formatted_feedback else "No structured training logs available."}

## Improvement Task
Based on the training results above, improve the model to achieve a HIGHER CV score.

**Improvement Guidelines**:
1. If CV std > 0.02: Add regularization or reduce model complexity
2. If overfitting detected: Increase reg_alpha/reg_lambda, reduce max_depth, add dropout
3. If underfitting detected: Increase model complexity, add features, reduce regularization
4. If Optuna best params available: Use them as starting point
5. If zero-importance features found: Remove them
6. If training is slow: Optimize hyperparameters for speed

**IMPORTANT**:
- Keep the same logging format ([LOG:FOLD], [LOG:OPTUNA], etc.) for the next iteration
- Return the complete updated Python code
- Focus on the most impactful change based on the feedback above
"""

            system_prompt = f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}"
            refine_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Here is the current working code:\n```python\n{best_code}\n```\n\n{refine_prompt}"
                ),
            ]

            try:
                refined_response = self.llm.invoke(refine_messages)
                refined_code = self._extract_code_from_response(
                    get_text_content(refined_response.content)
                )

                print("Executing refined code...")
                refined_exec = self.executor.execute(refined_code, working_dir)

                if refined_exec.success:
                    refined_score = self._extract_cv_score(refined_exec.stdout)
                    if refined_score is not None:
                        improvement = calculate_score_improvement(
                            refined_score,
                            best_score,
                            competition_info.evaluation_metric,
                        )
                        if improvement > 0:
                            print(
                                f"🚀 Improvement found: {refined_score:.6f} (was {best_score:.6f})"
                            )
                            best_score = refined_score
                            best_code = refined_code
                            best_stdout = refined_exec.stdout
                            result.code = best_code
                            result.stdout = refined_exec.stdout
                            state_updates["current_code"] = best_code
                            state_updates["baseline_cv_score"] = best_score
                        else:
                            print(f"No improvement ({refined_score:.6f} vs {best_score:.6f})")
                    else:
                        print("Could not extract score from refined code")
                else:
                    print("Refined code failed to execute")
            except Exception as e:
                print(f"Refinement failed: {e}")

        return state_updates
