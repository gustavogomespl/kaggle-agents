"""
Training data collection for Meta-Evaluator.

Contains methods for collecting training data for DSPy optimization.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ...prompts.templates.developer_prompts import format_component_details
from ...prompts.templates.planner_prompts import get_domain_guidance


if TYPE_CHECKING:
    from ...core.state import KaggleState


class TrainingMixin:
    """Mixin providing training data collection methods."""

    def _collect_training_data(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> None:
        """
        Collect training data for DSPy optimization.

        Stores examples with reward signals for later prompt optimization.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis
            reward_signals: Reward signals
        """
        from ...core.state import get_memory_summary_for_planning

        print("\n   💾 Collecting training data for prompt optimization...")

        ablation_plan = state.get("ablation_plan", [])
        dev_results = state.get("development_results", [])

        if not ablation_plan or not dev_results:
            return

        competition_info = state.get("competition_info")
        domain = state.get("domain_detected", "tabular")
        working_dir = state.get("working_directory", "")

        # Build planner inputs matching AblationPlannerSignature
        comp_info_str = ""
        if competition_info:
            comp_info_str = (
                f"Name: {competition_info.name}\n"
                f"Description: {getattr(competition_info, 'description', '')}\n"
                f"Problem Type: {getattr(competition_info, 'problem_type', '')}\n"
                f"Metric: {getattr(competition_info, 'evaluation_metric', '')}\n"
                f"Domain: {domain}\n"
            )

        sota_solutions = state.get("sota_solutions", [])
        if sota_solutions:
            sota_lines = []
            for sol in sota_solutions[:5]:
                title = getattr(sol, "title", "Unknown")
                score = getattr(sol, "score", 0.0)
                votes = getattr(sol, "votes", 0)
                models = getattr(sol, "models_used", []) or []
                strategies = getattr(sol, "strategies", []) or []
                sota_lines.append(
                    f"- {title} (score={score}, votes={votes}) "
                    f"models={models[:3]} strategies={strategies[:3]}"
                )
            sota_summary = "\n".join(sota_lines)
        else:
            sota_summary = "No SOTA solutions available."

        domain_guidance = get_domain_guidance(str(domain))
        memory_summary = get_memory_summary_for_planning(state)

        # Collect planner example
        plan_quality_score = reward_signals["r_combined"]

        self.training_collector.add_example(
            agent_name="planner",
            inputs={
                "competition_info": comp_info_str,
                "domain": str(domain),
                "sota_summary": sota_summary,
                "domain_guidance": domain_guidance,
                "memory_summary": memory_summary,
            },
            outputs={
                "ablation_plan": json.dumps(
                    [
                        {
                            "name": c.name,
                            "component_type": c.component_type,
                            "description": c.code,
                            "estimated_impact": c.estimated_impact,
                            "rationale": "",
                            "code_outline": "",
                        }
                        for c in ablation_plan
                    ],
                    indent=2,
                ),
                "analysis": "",
            },
            score=plan_quality_score,
        )

        # Collect developer examples (one per component)
        for i, result in enumerate(dev_results):
            if i >= len(ablation_plan):
                continue

            component = ablation_plan[i]
            component_score = 1.0 if result.success else 0.0

            self.training_collector.add_example(
                agent_name="developer_generator",
                inputs={
                    "component_details": format_component_details(component),
                    "competition_context": (
                        f"Name: {competition_info.name if competition_info else 'unknown'}\n"
                        f"Domain: {domain}\n"
                        f"Problem Type: {competition_info.problem_type if competition_info else 'unknown'}\n"
                        f"Metric: {competition_info.evaluation_metric if competition_info else 'unknown'}\n"
                    ),
                    "data_paths": (
                        f"Train: {state.get('current_train_path') or state.get('train_data_path', '')}\n"
                        f"Test: {state.get('current_test_path') or state.get('test_data_path', '')}\n"
                        f"Models: {working_dir}/models\n"
                        f"Sample Submission: {state.get('sample_submission_path', '')}\n"
                    ),
                    "requirements": (
                        f"Implement {component.component_type}: {component.name}\n"
                        f"Time budget: {state.get('component_timeout', state.get('testing_timeout', 'unknown'))}\n"
                        "Return a complete, executable script and write a valid submission CSV."
                    ),
                },
                outputs={
                    "code": result.code,
                    "explanation": "",
                },
                score=component_score,
            )

        print("   ✓ Collected training examples for Planner and Developer")
