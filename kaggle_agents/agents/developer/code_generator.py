"""
Code generation logic for the Developer Agent.

Handles:
- Dynamic temperature selection
- Code generation with DSPy or direct LLM
- GRPO reasoning trace integration
- Chain-of-Thought integration
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from ...core.state import AblationComponent, KaggleState, ReasoningTrace
from ...prompts.templates.developer_prompts import (
    build_context,
    build_dynamic_instructions,
    compose_generate_prompt,
    format_component_details,
)
from ...utils.llm_utils import get_text_content


if TYPE_CHECKING:
    from .agent import DeveloperAgent


# Dynamic temperature settings for different contexts
TEMPERATURE_SETTINGS = {
    "initial_generation": 0.1,  # Conservative for initial code generation
    "error_fixing_attempt_1": 0.25,  # Slightly more creative for first fix
    "error_fixing_attempt_2": 0.4,  # More creative after first attempt fails
    "error_fixing_attempt_3": 0.5,  # Maximum creativity for persistent errors
    "debug_mode": 0.45,  # Higher creativity in debug mode
    "ensemble": 0.3,  # Moderate creativity for ensemble strategies
    "feature_engineering": 0.2,  # Some creativity for feature ideas
    "refinement": 0.35,  # Moderate for refinement iterations
}


def get_dynamic_temperature(
    context: str,
    attempt: int = 0,
    component_type: str = "model",
) -> float:
    """
    Get dynamic temperature based on generation context.

    Higher temperatures encourage creativity (useful for error fixing),
    lower temperatures encourage consistency (useful for initial generation).

    Args:
        context: One of 'generation', 'fixing', 'debug', 'refinement'
        attempt: Current attempt number (0-indexed)
        component_type: Type of component being generated

    Returns:
        Appropriate temperature value
    """
    if context == "generation":
        # Use component-specific temperature for generation
        if component_type == "ensemble":
            return TEMPERATURE_SETTINGS["ensemble"]
        if component_type == "feature_engineering":
            return TEMPERATURE_SETTINGS["feature_engineering"]
        return TEMPERATURE_SETTINGS["initial_generation"]

    if context == "fixing":
        # Escalate temperature with each failed attempt
        if attempt <= 0:
            return TEMPERATURE_SETTINGS["error_fixing_attempt_1"]
        if attempt == 1:
            return TEMPERATURE_SETTINGS["error_fixing_attempt_2"]
        return TEMPERATURE_SETTINGS["error_fixing_attempt_3"]

    if context == "debug":
        return TEMPERATURE_SETTINGS["debug_mode"]

    if context == "refinement":
        return TEMPERATURE_SETTINGS["refinement"]

    # Default fallback
    return TEMPERATURE_SETTINGS["initial_generation"]


class CodeGeneratorMixin:
    """Mixin providing code generation capabilities to DeveloperAgent."""

    def _generate_code(
        self: DeveloperAgent,
        component: AblationComponent,
        competition_info,
        working_dir: Path,
        domain: str,
        state: KaggleState = None,
        reasoning_trace: ReasoningTrace = None,
        cot_result=None,  # ChainOfThoughtResult from GRPO
    ) -> str:
        """Generate code for a component with optional GRPO reasoning trace and CoT."""
        component_details = format_component_details(component)

        dataset_info = self._get_dataset_info(working_dir, state)

        # Get domain-specific code template
        domain_template = self._get_domain_template(domain, component.component_type)

        # Resolve key paths from state (preferring downloaded locations)
        resolved_train_path = Path(
            state.get("current_train_path")
            if state and state.get("current_train_path")
            else state.get("train_data_path")
            if state and state.get("train_data_path")
            else working_dir / "train.csv"
        )
        resolved_test_path = Path(
            state.get("current_test_path")
            if state and state.get("current_test_path")
            else state.get("test_data_path")
            if state and state.get("test_data_path")
            else working_dir / "test.csv"
        )
        sample_submission_path = Path(
            state.get("sample_submission_path")
            if state and state.get("sample_submission_path")
            else working_dir / "sample_submission.csv"
        )
        submission_output_path = working_dir / "submission.csv"
        models_dir = working_dir / "models"
        data_files = state.get("data_files", {}) if state else {}
        train_csv_path = data_files.get("train_csv", "")
        test_csv_path = data_files.get("test_csv", "")
        clean_train_path = data_files.get("clean_train", "")

        competition_context = f"""
        Name: {competition_info.name}
        Domain: {domain}
        Problem Type: {competition_info.problem_type}
        Metric: {competition_info.evaluation_metric}
        """

        data_paths = f"""
        Train: {resolved_train_path}
        Clean Train: {clean_train_path}
        Train CSV: {train_csv_path}
        Test: {resolved_test_path}
        Test CSV: {test_csv_path}
        Models: {models_dir}
        Sample Submission: {sample_submission_path}
        Submission Output: {submission_output_path}
        """

        if state is not None:
            requirements = build_dynamic_instructions(
                component=component,
                state=state,
                config=self.config,
                working_dir=str(working_dir),
            )
        else:
            requirements = f"""
            1. Implement {component.component_type}: {component.name}
            2. Save models to models/ directory
            3. Print progress and metrics
            4. Handle errors gracefully
            """

        # GRPO: Inject reasoning trace into requirements
        if reasoning_trace:
            reasoning_guidance = self._format_reasoning_for_prompt(reasoning_trace)
            requirements = reasoning_guidance + "\n\n" + requirements

        # Chain-of-Thought: Inject step-by-step thinking into requirements
        if cot_result:
            cot_guidance = self._format_cot_for_prompt(cot_result)
            requirements = cot_guidance + "\n\n" + requirements

        # Build dynamic context from state (SOTA, feedback, rewards)
        context = build_context(state, component=component) if state else build_context({})

        # Prepare paths dictionary
        # Explicitly distinguish INPUT_DIR (read-only data) from OUTPUT_DIR (writable)
        # This prevents errors in Kaggle Kernels where /kaggle/input is read-only
        input_dir = resolved_train_path.parent  # Parent of train.csv contains data
        output_dir = working_dir  # working_dir is always writable

        paths = {
            "input_dir": str(input_dir),  # READ-ONLY - data files location
            "output_dir": str(output_dir),  # WRITABLE - for models, submission, etc.
            "train": str(resolved_train_path),
            "clean_train": str(clean_train_path),
            "train_csv": str(train_csv_path),
            "test": str(resolved_test_path),
            "test_csv": str(test_csv_path),
            "models": str(models_dir),
            "submission": str(submission_output_path),
        }

        def _generate_with_llm() -> str:
            prompt = compose_generate_prompt(
                component=component,
                competition_info=competition_info,
                paths=paths,
                context=context,
            )

            messages = [
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return self._extract_code_from_response(get_text_content(response.content))

        if self.use_dspy:
            requirements_with_context = requirements
            if context.iteration_num == 0 and context.sota_patterns:
                requirements_with_context += (
                    "\n\n## SOTA Patterns (reference)\n" + context.sota_patterns[:1200]
                )
            if context.previous_feedback:
                requirements_with_context += (
                    "\n\n## Previous Training Feedback\n" + context.previous_feedback[:1200]
                )
            if context.attempt_feedback:
                requirements_with_context += (
                    "\n\n## Prior Attempts (Study + Fix)\n" + context.attempt_feedback[:1600]
                )
            if context.reward_guidance:
                requirements_with_context += (
                    "\n\n## Meta-Evaluator Guidance\n" + context.reward_guidance[:800]
                )

            try:
                result = self.generator_module(
                    component_details=component_details,
                    competition_context=competition_context,
                    data_paths=data_paths,
                    requirements=requirements_with_context,
                )
                code = self._extract_code_from_response(result.code)
            except Exception as exc:
                print(f"⚠️ DSPy generation failed, falling back to base prompt: {exc}")
                code = _generate_with_llm()
        else:
            code = _generate_with_llm()

        return code
