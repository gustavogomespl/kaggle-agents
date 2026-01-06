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

    def _validate_and_resolve_paths(
        self,
        train_path: Path,
        test_path: Path,
        working_dir: Path,
    ) -> tuple[Path, Path]:
        """
        Validate paths exist and search for alternatives if not found.

        For non-standard competition structures (e.g., mlsp-2013-birds with
        essential_data/ subdirectory), the default train.csv path may not exist.
        This method searches subdirectories for actual data files.

        Args:
            train_path: Initial train path
            test_path: Initial test path
            working_dir: Working directory to search

        Returns:
            Tuple of (resolved_train_path, resolved_test_path)
        """
        resolved_train = train_path
        resolved_test = test_path

        # Directories to search for data
        data_subdirs = [
            "train",
            "test",
            "essential_data",
            "supplemental_data",
            "data",
            "audio",
            "audio_data",
        ]
        # Extensions to look for
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        # Check if train path exists
        if not train_path.exists():
            print(f"   ⚠️ Train path not found: {train_path}")

            # First check for train.csv in working_dir
            if (working_dir / "train.csv").exists():
                resolved_train = working_dir / "train.csv"
                print(f"   ✓ Found train.csv in working_dir")
            elif (working_dir / "train").exists():
                resolved_train = working_dir / "train"
                print(f"   ✓ Found train/ directory in working_dir")
            else:
                # Search subdirectories
                for subdir_name in data_subdirs:
                    subdir = working_dir / subdir_name
                    if not subdir.is_dir():
                        continue

                    # Check for train.csv inside
                    if (subdir / "train.csv").exists():
                        resolved_train = subdir / "train.csv"
                        print(f"   ✓ Found train.csv in {subdir_name}/")
                        break

                    # Check for audio/image files (non-tabular data)
                    sample_files = list(subdir.glob("*"))[:50]
                    has_audio = any(
                        f.suffix.lower() in audio_exts for f in sample_files if f.is_file()
                    )
                    has_images = any(
                        f.suffix.lower() in image_exts for f in sample_files if f.is_file()
                    )
                    if has_audio or has_images:
                        resolved_train = subdir
                        dtype = "audio" if has_audio else "image"
                        print(f"   ✓ Found {dtype} data in {subdir_name}/")
                        break

        # Check if test path exists
        if not test_path.exists():
            # First check for test.csv in working_dir
            if (working_dir / "test.csv").exists():
                resolved_test = working_dir / "test.csv"
            elif (working_dir / "test").exists():
                resolved_test = working_dir / "test"
            # For audio/image competitions, test data might be in same dir as train
            elif resolved_train.is_dir() and resolved_train != train_path:
                resolved_test = resolved_train
                print(f"   ℹ️ Using train directory for test data (shared)")

        return resolved_train, resolved_test

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

        # Validate and fix paths if they don't exist
        resolved_train_path, resolved_test_path = self._validate_and_resolve_paths(
            resolved_train_path, resolved_test_path, working_dir
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
            "sample_submission": str(sample_submission_path),
        }

        # Store resolved paths for use by fix/debug functions
        self._resolved_paths = paths

        # Generate path constants header to inject into code
        # This ensures the LLM cannot ignore the correct paths
        path_header = f'''# === PATH CONSTANTS (AUTO-INJECTED - DO NOT MODIFY) ===
from pathlib import Path

TRAIN_PATH = Path("{resolved_train_path}")
TEST_PATH = Path("{resolved_test_path}")
SAMPLE_SUBMISSION_PATH = Path("{sample_submission_path}")
MODELS_DIR = Path("{models_dir}")
OUTPUT_DIR = Path("{working_dir}")

# Create models directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# === END PATH CONSTANTS ===
'''

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

        # Prepend path constants header to ensure LLM-generated code uses correct paths
        return path_header + "\n" + code
