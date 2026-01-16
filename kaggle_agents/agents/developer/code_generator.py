"""
Code generation logic for the Developer Agent.

Handles:
- Dynamic temperature selection
- Code generation with DSPy or direct LLM
- GRPO reasoning trace integration
- Chain-of-Thought integration
"""

from __future__ import annotations

import re
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


# Path constants that should never be redefined by LLM-generated code
IMMUTABLE_PATH_VARS = [
    "TRAIN_PATH",
    "TEST_PATH",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "SAMPLE_SUBMISSION_PATH",
    "SUBMISSION_PATH",
    "AUDIO_SOURCE_DIR",
    "LABEL_FILES",
    # Image competition paths (separate directory and CSV)
    "TRAIN_IMG_DIR",
    "TRAIN_CSV_PATH",
    "TEST_IMG_DIR",
    "TEST_CSV_PATH",
    # Canonical data contract paths
    "CANONICAL_DIR",
    "CANONICAL_TRAIN_IDS_PATH",
    "CANONICAL_Y_PATH",
    "CANONICAL_FOLDS_PATH",
    "CANONICAL_FEATURE_COLS_PATH",
    "CANONICAL_METADATA_PATH",
    # Common base directory patterns
    "BASE_DIR",
    "DATA_DIR",
    "WORKING_DIR",
]


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
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        # Check if train path exists
        if not train_path.exists():
            print(f"   ⚠️ Train path not found: {train_path}")

            # First check for train.csv in working_dir
            if (working_dir / "train.csv").exists():
                resolved_train = working_dir / "train.csv"
                print("   ✓ Found train.csv in working_dir")
            elif (working_dir / "train").exists():
                resolved_train = working_dir / "train"
                print("   ✓ Found train/ directory in working_dir")
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
                print("   ℹ️ Using train directory for test data (shared)")

        return resolved_train, resolved_test

    def _validate_no_path_redefinition(
        self: DeveloperAgent,
        code: str,
        path_header_end_marker: str = "# === END PATH CONSTANTS ===",
    ) -> tuple[bool, list[str]]:
        """
        Detect if LLM-generated code redefines any injected path constants.

        Searches for reassignments of TRAIN_PATH, MODELS_DIR, etc. after the
        injected path constants header.

        Args:
            code: The full generated code (with path header prepended)
            path_header_end_marker: Marker indicating end of injected paths

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Find where the injected header ends
        marker_idx = code.find(path_header_end_marker)
        if marker_idx == -1:
            # No marker found, can't validate
            return True, []

        # Get the code after the injected header
        code_after_header = code[marker_idx + len(path_header_end_marker) :]

        # Check for redefinitions of each immutable path variable
        for var in IMMUTABLE_PATH_VARS:
            # Multiple patterns to catch various redefinition attempts
            patterns = [
                # VAR = Path(...)
                rf"^\s*{var}\s*=\s*Path\s*\(",
                # VAR = "..." or VAR = '...'
                rf"^\s*{var}\s*=\s*['\"]",
                # VAR = something / ... (path concatenation)
                rf"^\s*{var}\s*=\s*\w+\s*/",
                # VAR = BASE_DIR / ...
                rf"^\s*{var}\s*=\s*\w+_DIR\s*/",
                # VAR = os.path.join(...)
                rf"^\s*{var}\s*=\s*os\.path\.join\s*\(",
                # VAR = str(...) (converting path)
                rf"^\s*{var}\s*=\s*str\s*\(",
            ]
            for pattern in patterns:
                if re.search(pattern, code_after_header, re.MULTILINE):
                    violations.append(f"Path redefinition detected: {var}")
                    break  # Only report once per variable

        return len(violations) == 0, violations

    def _strip_path_redefinitions(
        self: DeveloperAgent,
        code: str,
        path_header_end_marker: str = "# === END PATH CONSTANTS ===",
    ) -> str:
        """
        Strip path redefinitions from LLM-generated code.

        Args:
            code: The full generated code (with path header prepended)
            path_header_end_marker: Marker indicating end of injected paths

        Returns:
            Code with path redefinitions commented out
        """
        marker_idx = code.find(path_header_end_marker)
        if marker_idx == -1:
            return code

        header = code[:marker_idx + len(path_header_end_marker)]
        code_after_header = code[marker_idx + len(path_header_end_marker):]

        for var in IMMUTABLE_PATH_VARS:
            # Pattern to match full line with path redefinition
            patterns = [
                rf"^(\s*{var}\s*=\s*Path\s*\([^\)]+\)\s*)$",
                rf"^(\s*{var}\s*=\s*['\"][^'\"]+['\"]\s*)$",
                rf"^(\s*{var}\s*=\s*\w+\s*/[^\n]+)$",
                rf"^(\s*{var}\s*=\s*os\.path\.join\([^\)]+\)\s*)$",
            ]
            for pattern in patterns:
                code_after_header = re.sub(
                    pattern,
                    r"# STRIPPED (path constant): \1",
                    code_after_header,
                    flags=re.MULTILINE,
                )

        return header + code_after_header

    def _strip_nrows_param(
        self: DeveloperAgent,
        code: str,
    ) -> tuple[str, int]:
        """
        Strip nrows parameter from pd.read_csv() calls to prevent data truncation.

        The nrows parameter causes OOF shape mismatches when models are trained on
        different subsets of data, breaking the stacking ensemble. This function
        removes nrows to force all models to use the full canonical dataset.

        Args:
            code: The generated code to sanitize

        Returns:
            Tuple of (sanitized_code, number_of_removals)
        """
        # Universal pattern to match nrows parameter regardless of value type:
        # - nrows=1000000 (numeric literal)
        # - nrows=5_000_000 (underscore separator)
        # - nrows=MAX_ROWS (uppercase constant)
        # - nrows=max_rows (lowercase variable)
        # - nrows=cfg.nrows (attribute access)
        # - nrows=args.nrows (attribute access)
        # - nrows=config['nrows'] (dict access)
        # - nrows=int(...) (function call)
        # - nrows=None (None value - keep this one as it means "no limit")
        #
        # Strategy: Handle different value types with separate patterns.
        # Order matters - function calls must be matched first to handle nested parens.
        patterns = [
            # 1. nrows with simple function call like int(...) or min(...) or len(...)
            #    Match: ", nrows=func_name(...)"
            r",\s*nrows\s*=\s*\w+\([^)]*\)(?=[,)])",
            # 2. nrows with simple value (number, variable, attribute, dict access)
            #    Match: ", nrows=<value>" stopping before ) or ,
            #    Negative lookahead for None (we want to keep nrows=None)
            r",\s*nrows\s*=\s*(?!None\b|none\b)[^,)]+(?=[,)])",
            # 3. nrows at start of kwargs: "nrows=<value>,"
            r"nrows\s*=\s*(?!None\b|none\b)[^,)]+\s*,",
        ]

        removals = 0
        sanitized = code
        for pattern in patterns:
            matches = re.findall(pattern, sanitized)
            removals += len(matches)
            sanitized = re.sub(pattern, "", sanitized)

        return sanitized, removals

    def _rewrite_base_dir_references(
        self: DeveloperAgent,
        code: str,
    ) -> tuple[str, int]:
        """
        Rewrite BASE_DIR references to use correct path constants.

        Does NOT define BASE_DIR - that would mask errors.
        Instead, rewrites specific patterns to correct paths:
        - BASE_DIR / "train*.csv" → TRAIN_PATH
        - BASE_DIR / "test*.csv" → TEST_PATH
        - BASE_DIR / "sample_submission*.csv" → SAMPLE_SUBMISSION_PATH
        - BASE_DIR / anything else → OUTPUT_DIR / "..."

        Args:
            code: The generated code to sanitize

        Returns:
            Tuple of (sanitized_code, number_of_rewrites)
        """
        rewrites = [
            # BASE_DIR / "train*.csv" → TRAIN_PATH (more specific patterns first)
            (r'BASE_DIR\s*/\s*["\']train\.csv["\']', 'TRAIN_PATH'),
            (r'BASE_DIR\s*/\s*["\']train[^"\']*\.csv["\']', 'TRAIN_PATH'),

            # BASE_DIR / "test*.csv" → TEST_PATH
            (r'BASE_DIR\s*/\s*["\']test\.csv["\']', 'TEST_PATH'),
            (r'BASE_DIR\s*/\s*["\']test[^"\']*\.csv["\']', 'TEST_PATH'),

            # BASE_DIR / "sample_submission*.csv" → SAMPLE_SUBMISSION_PATH
            (r'BASE_DIR\s*/\s*["\']sample_submission\.csv["\']', 'SAMPLE_SUBMISSION_PATH'),
            (r'BASE_DIR\s*/\s*["\']sample_submission[^"\']*\.csv["\']', 'SAMPLE_SUBMISSION_PATH'),
            (r'BASE_DIR\s*/\s*["\']sample[^"\']*submission[^"\']*\.csv["\']', 'SAMPLE_SUBMISSION_PATH'),

            # BASE_DIR / "submission.csv" → SUBMISSION_PATH
            (r'BASE_DIR\s*/\s*["\']submission\.csv["\']', 'SUBMISSION_PATH'),

            # BASE_DIR / anything else → OUTPUT_DIR / "..."
            (r'BASE_DIR\s*/\s*(["\'][^"\']+["\'])', r'OUTPUT_DIR / \1'),

            # str(BASE_DIR) → str(OUTPUT_DIR)
            (r'str\s*\(\s*BASE_DIR\s*\)', 'str(OUTPUT_DIR)'),

            # Bare BASE_DIR → OUTPUT_DIR (last, as it's most general)
            (r'\bBASE_DIR\b', 'OUTPUT_DIR'),
        ]

        rewrite_count = 0
        rewritten = code

        for pattern, replacement in rewrites:
            matches = re.findall(pattern, rewritten)
            if matches:
                rewrite_count += len(matches)
                rewritten = re.sub(pattern, replacement, rewritten)

        return rewritten, rewrite_count

    def _validate_audio_label_usage(
        self: DeveloperAgent,
        code: str,
        data_type: str,
    ) -> list[str]:
        """
        Validate that audio competition code uses pre-loaded labels correctly.

        Checks for common LLM mistakes:
        1. Hardcoded label file paths that don't exist
        2. Using pd.read_csv() on label files instead of _PRELOADED_LABELS_DF
        3. Using header=None on files that have headers

        Args:
            code: The generated code to validate
            data_type: Competition data type (audio, image, etc.)

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Only validate for audio competitions
        if data_type not in ("audio", "audio_classification"):
            return warnings

        # Check for hardcoded paths that don't exist
        bad_paths = [
            "rec_labels_train.txt",
            "train_labels.txt",
            "labels_train.txt",
            "train_label.txt",
        ]
        for bad_path in bad_paths:
            if bad_path in code:
                warnings.append(
                    f"⚠️ Hardcoded path '{bad_path}' detected - this file likely doesn't exist! "
                    "Use _PRELOADED_LABELS_DF instead."
                )

        # Check if pre-loaded labels are being ignored
        has_label_parsing = any(
            pattern in code.lower()
            for pattern in ["pd.read_csv", "read_csv", "open("]
        )
        uses_preloaded = "_PRELOADED_LABELS_DF" in code

        if has_label_parsing and not uses_preloaded:
            # Check if the label parsing is happening after the header
            marker_idx = code.find("# === END PATH CONSTANTS ===")
            if marker_idx != -1:
                code_after_header = code[marker_idx:]
                label_file_patterns = [
                    "rec_labels",
                    "train_labels",
                    "label",
                ]
                for pattern in label_file_patterns:
                    if pattern in code_after_header.lower() and "read_csv" in code_after_header:
                        warnings.append(
                            "⚠️ LLM is re-parsing label files instead of using _PRELOADED_LABELS_DF. "
                            "This may cause FileNotFoundError or parsing errors."
                        )
                        break

        return warnings

    def _strip_label_reparsing(
        self: DeveloperAgent,
        code: str,
        path_header_end_marker: str = "# === END PATH CONSTANTS ===",
    ) -> tuple[str, int]:
        """
        Replace LLM-generated label file parsing with pre-loaded label variables.

        The LLM often ignores _PRELOADED_LABELS_DF and re-parses label files,
        causing FileNotFoundError or parsing errors. This function enforces the
        use of pre-loaded labels by REPLACING (not just commenting) the bad code.

        Args:
            code: The full generated code
            path_header_end_marker: Marker indicating end of injected path header

        Returns:
            Tuple of (modified code, number of statements replaced)
        """
        marker_idx = code.find(path_header_end_marker)
        if marker_idx == -1:
            return code, 0

        header = code[: marker_idx + len(path_header_end_marker)]
        code_after_header = code[marker_idx + len(path_header_end_marker) :]

        replace_count = 0

        def make_replacement(match: re.Match) -> str:
            """Extract variable name and create proper replacement assignment."""
            nonlocal replace_count
            full_match = match.group(0)
            indent = match.group(1)  # Preserve original indentation

            # Extract variable name from "varname = pd.read_csv(...)"
            var_match = re.match(r"[\t ]*(\w+)\s*=", full_match)
            if var_match:
                var_name = var_match.group(1)
                replace_count += 1
                # Return proper assignment with same indentation
                return f"{indent}{var_name} = _PRELOADED_LABELS_DF.copy()  # REPLACED: was pd.read_csv on label file"
            # Fallback: just return original if we can't extract var name
            return full_match

        # Single comprehensive pattern to match all label file parsing
        # Using negative lookbehind (?<![a-zA-Z]) to avoid "unlabeled" but match LABEL_FILE, rec_labels_train
        # Matches: label, labels, rec_label, rec_labels, train_label, train_labels (case-insensitive)
        # Group 1: indentation, Group 2: full assignment statement
        pattern = r"([\t ]*)(\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])(?:rec_labels?|train_labels?|labels?)[^)]*\))"

        code_after_header = re.sub(
            pattern,
            make_replacement,
            code_after_header,
            flags=re.IGNORECASE,
        )

        # Note: We intentionally don't handle 'with open()' blocks here because:
        # 1. They're rare for label files (pd.read_csv is the common pattern)
        # 2. Properly removing a with block requires removing the indented body too
        # 3. The validation warnings will catch any remaining issues

        return header + code_after_header, replace_count

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
        # Non-standard label files (e.g., .txt files for MLSP 2013 Birds)
        label_files = data_files.get("label_files", [])
        audio_source_path = data_files.get("audio_source", "")
        data_type = data_files.get("data_type", "tabular")

        competition_context = f"""
        Name: {competition_info.name}
        Domain: {domain}
        Problem Type: {competition_info.problem_type}
        Metric: {competition_info.evaluation_metric}
        """

        # Format label files for prompt
        label_files_str = ", ".join(label_files) if label_files else "None"

        data_paths = f"""
        Train: {resolved_train_path}
        Clean Train: {clean_train_path}
        Train CSV: {train_csv_path}
        Test: {resolved_test_path}
        Test CSV: {test_csv_path}
        Models: {models_dir}
        Sample Submission: {sample_submission_path}
        Submission Output: {submission_output_path}
        Label Files (TXT): {label_files_str}
        Audio Source: {audio_source_path if audio_source_path else "None"}
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
            # Non-standard label files (e.g., MLSP 2013 Birds .txt files)
            "label_files": label_files,
            "audio_source": audio_source_path,
        }

        # Store resolved paths for use by fix/debug functions
        self._resolved_paths = paths

        # Check for canonical data (prepared by canonical_data_preparation_node)
        canonical_dir = working_dir / "canonical"
        has_canonical = canonical_dir.exists() and (canonical_dir / "train_ids.npy").exists()

        # Generate path constants header to inject into code
        # This ensures the LLM cannot ignore the correct paths
        path_header = f'''# === PATH CONSTANTS (AUTO-INJECTED - DO NOT MODIFY) ===
from pathlib import Path
import pandas as pd
import numpy as np
import json

'''
        # Data-type aware path injection
        if data_type == "image":
            # For image competitions: inject BOTH directory paths AND CSV paths
            # TRAIN_IMG_DIR = directory containing images
            # TRAIN_CSV_PATH = CSV file with image IDs and labels
            # TRAIN_PATH = points to CSV for pd.read_csv() compatibility

            # Resolve CSV paths at Python runtime (not in generated code)
            # This fixes the bug where empty strings created Path("") or Path("None")
            resolved_train_csv = train_csv_path if train_csv_path else str(working_dir / "train.csv")
            resolved_test_csv = test_csv_path if test_csv_path else ""

            # Build TEST_CSV_PATH line - only set if we have a valid path
            if resolved_test_csv:
                test_csv_line = f'TEST_CSV_PATH = Path("{resolved_test_csv}")'
            else:
                test_csv_line = "TEST_CSV_PATH = None  # No test CSV available"

            path_header += f'''# === IMAGE COMPETITION PATHS ===
# TRAIN_IMG_DIR: Directory containing training images
# TRAIN_CSV_PATH: CSV file with image IDs and labels (use for pd.read_csv())
TRAIN_IMG_DIR = Path("{resolved_train_path}")
TRAIN_CSV_PATH = Path("{resolved_train_csv}")
TEST_IMG_DIR = Path("{resolved_test_path}")
{test_csv_line}

# COMPATIBILITY: TRAIN_PATH points to CSV for pd.read_csv() calls
# Use TRAIN_IMG_DIR when you need the image directory
TRAIN_PATH = TRAIN_CSV_PATH if TRAIN_CSV_PATH.exists() else Path("{working_dir}/train.csv")
TEST_PATH = TEST_CSV_PATH if TEST_CSV_PATH and TEST_CSV_PATH.exists() else TEST_IMG_DIR
'''
        else:
            # For tabular/audio: original behavior
            path_header += f'''TRAIN_PATH = Path("{resolved_train_path}")
TEST_PATH = Path("{resolved_test_path}")
'''

        path_header += f'''SAMPLE_SUBMISSION_PATH = Path("{sample_submission_path}")
MODELS_DIR = Path("{models_dir}")
OUTPUT_DIR = Path("{working_dir}")
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
COMPONENT_NAME = "{component.name.replace(" ", "_").lower()}"

# Create models directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)
'''
        # Add canonical data paths if available
        if has_canonical:
            path_header += f'''
# === CANONICAL DATA CONTRACT (MANDATORY - DO NOT REDEFINE) ===
# All model components MUST use these artifacts for consistent data handling
CANONICAL_DIR = Path("{canonical_dir}")
CANONICAL_TRAIN_IDS_PATH = CANONICAL_DIR / "train_ids.npy"
CANONICAL_Y_PATH = CANONICAL_DIR / "y.npy"
CANONICAL_FOLDS_PATH = CANONICAL_DIR / "folds.npy"
CANONICAL_FEATURE_COLS_PATH = CANONICAL_DIR / "feature_cols.json"
CANONICAL_METADATA_PATH = CANONICAL_DIR / "metadata.json"

# Load canonical metadata
with open(CANONICAL_METADATA_PATH) as _f:
    CANONICAL_METADATA = json.load(_f)
    N_FOLDS = CANONICAL_METADATA["n_folds"]
    ID_COL = CANONICAL_METADATA.get("id_col", "id")
    TARGET_COL = CANONICAL_METADATA.get("target_col", "target")
    IS_CLASSIFICATION = CANONICAL_METADATA.get("is_classification", True)

print(f"[LOG:INFO] Canonical data loaded: {{CANONICAL_METADATA.get('canonical_rows', 'unknown')}} samples, {{N_FOLDS}} folds")

# === CANONICAL FOLDS (USE IF AVAILABLE) ===
# PREFERRED: Use canonical folds for OOF alignment across all models
# FALLBACK: If canonical folds don't exist, create folds from data (StratifiedKFold)
if CANONICAL_FOLDS_PATH.exists():
    CANONICAL_FOLDS = np.load(CANONICAL_FOLDS_PATH)
    CANONICAL_TRAIN_IDS = np.load(CANONICAL_TRAIN_IDS_PATH, allow_pickle=True)
    CANONICAL_Y = np.load(CANONICAL_Y_PATH, allow_pickle=True)
    CANONICAL_FOLDS_AVAILABLE = True
    print(f"[CANONICAL] Loaded folds.npy: {{len(CANONICAL_FOLDS)}} samples, {{N_FOLDS}} folds")
    # Usage example:
    # for fold in range(N_FOLDS):
    #     train_mask = CANONICAL_FOLDS != fold
    #     val_mask = CANONICAL_FOLDS == fold
    #     train_ids, val_ids = CANONICAL_TRAIN_IDS[train_mask], CANONICAL_TRAIN_IDS[val_mask]
else:
    # Fallback: canonical folds not available, model must create its own
    CANONICAL_FOLDS = None
    CANONICAL_TRAIN_IDS = None
    CANONICAL_Y = None
    CANONICAL_FOLDS_AVAILABLE = False
    print(f"[WARNING] Canonical folds not found at {{CANONICAL_FOLDS_PATH}}")
    print("[WARNING] Model will need to create folds from data (use StratifiedKFold)")
# === END CANONICAL FOLDS ===
'''
        # Add label files paths if available (for non-standard formats like MLSP 2013 Birds)
        if label_files:
            label_paths_code = "\n# Non-standard label files (e.g., .txt files)\nLABEL_FILES = [\n"
            for lf in label_files:
                label_paths_code += f'    Path("{lf}"),\n'
            label_paths_code += "]\n"
            path_header += label_paths_code
            # Add helper function for parsing label files (handles variable-width multi-label rows)
            path_header += """
# MANDATORY: Parse label files - DO NOT use dummy labels (np.zeros)
# This handles VARIABLE-WIDTH multi-label files (e.g., rec_id,label1 vs rec_id,label1,label2,label3)
def parse_label_file(label_path, hidden_marker='?'):
    '''Parse variable-width label file with automatic delimiter detection.

    Returns DataFrame with columns: ['rec_id', 'label'] in long format
    (one row per rec_id-label pair for multi-label files).

    RAISES ValueError if parsing fails - NEVER returns empty DataFrame silently!
    '''
    import csv
    label_path = Path(label_path)
    if not label_path.exists():
        raise ValueError(f"Label file not found: {label_path}")

    content = label_path.read_text(encoding='utf-8', errors='ignore')
    lines = content.strip().split('\\n')
    if len(lines) < 2:
        raise ValueError(f"Label file has insufficient lines ({len(lines)}): {label_path}")

    sample = '\\n'.join(lines[:20])

    # Auto-detect delimiter
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',\\t ;|')
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ',' if ',' in sample else '\\t' if '\\t' in sample else ' '

    # Parse line-by-line to handle variable-width rows
    rows = []
    for line in lines:
        parts = line.strip().split(delimiter)
        if len(parts) < 2:
            continue
        rec_id = parts[0].strip()
        # Skip header row if detected
        if rec_id.lower() in ('rec_id', 'id', 'recording_id', 'filename'):
            continue
        # Each subsequent part is a label
        for label in parts[1:]:
            label = label.strip()
            if label and label != hidden_marker:
                # Try to cast label to int for MultiLabelBinarizer compatibility
                # MLSP-2013-Birds and similar competitions use integer class IDs
                try:
                    label_val = int(label)
                except ValueError:
                    label_val = label  # Keep as string if not numeric
                rows.append({'rec_id': rec_id, 'label': label_val})

    # FAIL LOUDLY instead of returning empty DataFrame
    if not rows:
        raise ValueError(
            f"parse_label_file() failed to parse any rows from {label_path}. "
            f"Detected delimiter: {repr(delimiter)}. First 3 lines: {lines[:3]}. "
            f"If this is a SPARSE multi-label format (e.g., 'rec_id,class1,class5,class12'), "
            f"use parse_mlsp_multilabel() from kaggle_agents.utils.label_parser instead."
        )

    df = pd.DataFrame(rows)
    print(f"[parse_label_file] Parsed {len(df)} label rows from {label_path.name}")
    return df

def parse_id_mapping_file(mapping_path):
    '''Parse ID to filename mapping file (e.g., rec_id2filename.txt).

    Returns dict: {rec_id: filename}
    '''
    import csv
    content = Path(mapping_path).read_text(encoding='utf-8', errors='ignore')
    lines = content.strip().split('\\n')
    sample = '\\n'.join(lines[:20])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',\\t ;|')
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ',' if ',' in sample else '\\t' if '\\t' in sample else ' '

    id_map = {}
    for line in lines:
        parts = line.strip().split(delimiter)
        if len(parts) >= 2:
            rec_id, filename = parts[0].strip(), parts[1].strip()
            if rec_id.lower() not in ('rec_id', 'id', 'recording_id'):
                id_map[rec_id] = filename
    return id_map
"""
            # === PRE-LOAD LABELS IMMEDIATELY (fail fast if broken) ===
            # This forces the LLM to use pre-loaded data instead of generating its own parsing code
            path_header += '''
# ============================================================
# PRE-LOADED LABELS (from LABEL_FILES using parse_label_file)
# ============================================================
def _load_labels_from_files():
    """Load labels from LABEL_FILES using the injected parser.

    Returns tuple: (rec_ids, labels_df, n_classes)
    """
    labels_df = None
    for lf in LABEL_FILES:
        if lf.exists() and 'label' in str(lf).lower():
            try:
                labels_df = parse_label_file(lf)
                print(f"[INFO] Loaded labels from {lf.name}")
                break
            except ValueError as e:
                print(f"[WARNING] Could not parse {lf.name}: {e}")
                continue

    if labels_df is None or len(labels_df) == 0:
        raise ValueError(f"No labels found! LABEL_FILES={LABEL_FILES}")

    rec_ids = labels_df['rec_id'].unique().tolist()
    unique_labels = sorted(labels_df['label'].unique())
    n_classes = len(unique_labels)

    print(f"[INFO] Labels: {len(rec_ids)} recordings, {n_classes} classes")
    return rec_ids, labels_df, n_classes

# === PRE-LOAD LABELS NOW (fail fast if broken) ===
print("="*60)
print("PRE-LOADING LABELS FROM LABEL_FILES...")
print("="*60)
_PRELOADED_REC_IDS, _PRELOADED_LABELS_DF, _PRELOADED_N_CLASSES = _load_labels_from_files()
print(f"Loaded {len(_PRELOADED_REC_IDS)} recording IDs, {_PRELOADED_N_CLASSES} classes")
print("="*60)
# ============================================================
# USE THESE VARIABLES INSTEAD OF PARSING FILES YOURSELF:
#   _PRELOADED_REC_IDS: List of recording IDs
#   _PRELOADED_LABELS_DF: DataFrame with columns ['rec_id', 'label'] (long format)
#   _PRELOADED_N_CLASSES: Number of unique classes
# ============================================================
'''

        # Add audio source path if available
        if audio_source_path:
            path_header += f'\n# Audio source directory\nAUDIO_SOURCE_DIR = Path("{audio_source_path}")\n'

        # CANONICAL_DIR fallback - only for audio competitions when NO canonical data exists
        # This prevents NameError when LLM-generated code references CANONICAL_DIR
        # IMPORTANT: Do NOT override if has_canonical=True (would break canonical contract)
        if data_type in ("audio", "audio_classification") and not has_canonical:
            path_header += f'''
# CANONICAL_DIR fallback (no canonical data detected - creating empty directory)
CANONICAL_DIR = MODELS_DIR / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
'''

        # Inject CVfolds train/test split if available
        # This is CRITICAL for competitions like MLSP 2013 Birds where train/test is defined in CVfolds_*.txt
        test_rec_ids = state.get("test_rec_ids", []) if state else []
        train_rec_ids = state.get("train_rec_ids", []) if state else []
        cv_folds_used = state.get("cv_folds_used", False) if state else False

        if cv_folds_used and test_rec_ids:
            # For large ID lists (>100 items), save to files to avoid bloating generated code
            if len(test_rec_ids) > 100 or len(train_rec_ids) > 100:
                # Save IDs to models directory for loading
                import numpy as np
                models_dir.mkdir(parents=True, exist_ok=True)
                np.save(models_dir / "cvfolds_train_ids.npy", np.array(train_rec_ids))
                np.save(models_dir / "cvfolds_test_ids.npy", np.array(test_rec_ids))
                path_header += f'''
# === CVfolds TRAIN/TEST SPLIT (AUTO-INJECTED - DO NOT OVERRIDE) ===
# These IDs come from CVfolds*.txt file - ALWAYS use these!
# DO NOT infer test count from sample_submission row count!
_cvfolds_train_path = MODELS_DIR / "cvfolds_train_ids.npy"
_cvfolds_test_path = MODELS_DIR / "cvfolds_test_ids.npy"
TRAIN_REC_IDS = np.load(_cvfolds_train_path, allow_pickle=True).tolist() if _cvfolds_train_path.exists() else []
TEST_REC_IDS = np.load(_cvfolds_test_path, allow_pickle=True).tolist() if _cvfolds_test_path.exists() else []
N_TRAIN = {len(train_rec_ids)}
N_TEST = {len(test_rec_ids)}

print(f"[CVfolds] Train: {{N_TRAIN}} recordings, Test: {{N_TEST}} recordings")
# === END CVfolds ===
'''
            else:
                # Small lists can be inlined safely
                path_header += f'''
# === CVfolds TRAIN/TEST SPLIT (AUTO-INJECTED - DO NOT OVERRIDE) ===
# These IDs come from CVfolds*.txt file - ALWAYS use these!
# DO NOT infer test count from sample_submission row count!
TRAIN_REC_IDS = {train_rec_ids}
TEST_REC_IDS = {test_rec_ids}
N_TRAIN = {len(train_rec_ids)}
N_TEST = {len(test_rec_ids)}

print(f"[CVfolds] Train: {{N_TRAIN}} recordings, Test: {{N_TEST}} recordings")
# === END CVfolds ===
'''

        # Inject smart file locator for audio/image competitions
        # This handles missing extensions (e.g., MLSP 2013 Birds where IDs lack .wav extension)
        if data_type in ("audio", "image"):
            path_header += '''
# === SMART FILE LOCATOR (handles missing extensions) ===
# CRITICAL: Use smart_locate_file() when loading audio/image files by ID
# This probes extensions automatically when the exact path doesn't exist
import glob as _glob_module

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]

def smart_locate_file(base_dir, file_id, likely_extensions=None, case_variants=True):
    """
    Robustly locate a file, handling missing extensions and case sensitivity.

    Args:
        base_dir: Directory to search in (Path or str)
        file_id: ID or partial filename (may lack extension)
        likely_extensions: Extensions to try ['.wav', '.mp3'], or None for auto-detect
        case_variants: Try uppercase/lowercase extension variants

    Returns:
        Full path as string if found, None if not found

    Example:
        >>> path = smart_locate_file(audio_dir, "PC1_123")
        '/data/audio/PC1_123.wav'  # Found with .wav extension
    """
    base_dir = Path(base_dir)
    file_id = str(file_id).strip()

    if not file_id or not base_dir.exists():
        return None

    # 1. Direct exact match (ID already has extension)
    direct_path = base_dir / file_id
    if direct_path.exists():
        return str(direct_path)

    # 2. Auto-detect extensions from directory if not provided
    if likely_extensions is None:
        sample_files = list(base_dir.iterdir())[:20]
        found_exts = set(f.suffix.lower() for f in sample_files if f.is_file() and f.suffix)
        likely_extensions = [e for e in AUDIO_EXTENSIONS + IMAGE_EXTENSIONS if e in found_exts]
        if not likely_extensions:
            likely_extensions = AUDIO_EXTENSIONS  # Default fallback

    # 3. Try with extensions
    for ext in likely_extensions:
        ext = f".{ext.lstrip('.')}"  # Normalize: ensure starts with dot

        candidate = base_dir / f"{file_id}{ext}"
        if candidate.exists():
            return str(candidate)

        if case_variants:
            candidate_lower = base_dir / f"{file_id}{ext.lower()}"
            if candidate_lower.exists():
                return str(candidate_lower)
            candidate_upper = base_dir / f"{file_id}{ext.upper()}"
            if candidate_upper.exists():
                return str(candidate_upper)

    # 4. Glob fallback (more expensive)
    # Escape glob special characters in file_id to prevent pattern issues
    escaped_id = _glob_module.escape(file_id)
    matches = list(base_dir.glob(f"{escaped_id}.*"))
    if matches:
        return str(matches[0])

    # 5. Case-insensitive stem match (last resort)
    try:
        for f in base_dir.iterdir():
            if f.is_file() and f.stem.lower() == file_id.lower():
                return str(f)
    except PermissionError:
        pass

    return None


def build_id_to_path_map(id_list, base_dir, extensions=None, verbose=True):
    """
    Build a mapping from IDs to resolved file paths.

    Args:
        id_list: List of file IDs (potentially without extensions)
        base_dir: Directory containing files
        extensions: Extensions to try (None = auto-detect)
        verbose: Print warnings for unresolved IDs

    Returns:
        Tuple of (id_to_path_map, unresolved_ids)
    """
    base_dir = Path(base_dir)
    id_to_path = {}
    unresolved = []

    for file_id in id_list:
        path = smart_locate_file(base_dir, str(file_id), extensions)
        if path:
            id_to_path[str(file_id)] = path
        else:
            unresolved.append(str(file_id))

    if verbose and unresolved:
        print(f"[WARNING] Could not resolve {len(unresolved)}/{len(id_list)} file IDs")
        print(f"[WARNING] Sample unresolved: {unresolved[:5]}")

    return id_to_path, unresolved


print("[INFO] smart_locate_file() available - use for loading audio/image by ID")
'''

        # For audio competitions without label files, inject filename-based label parser
        if data_type == "audio" and not label_files:
            path_header += '''
# === FILENAME-BASED LABEL PARSER (for audio without train.csv) ===
# Use this when labels are embedded in filenames (e.g., train12345_1.aif means label=1)
def create_train_df_from_filenames(audio_dir, label_pattern=r'_(\\d+)\\.'):
    """Parse labels from audio filenames.

    Args:
        audio_dir: Directory containing audio files
        label_pattern: Regex to extract label (default: matches _0., _1., _42., etc.)

    Returns:
        DataFrame with columns: id, path, target
    """
    import re
    AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    audio_files = [f for f in Path(audio_dir).rglob('*') if f.suffix.lower() in AUDIO_EXTS]

    data = []
    for fp in audio_files:
        match = re.search(label_pattern, fp.name)
        if match:
            data.append({'id': fp.stem, 'path': str(fp), 'target': int(match.group(1))})

    if not data:
        raise ValueError(f"No files with label pattern '{label_pattern}' found in {audio_dir}")

    df = pd.DataFrame(data)
    print(f"[INFO] Created train_df from filenames: {len(df)} samples")
    print(f"[INFO] Label distribution: {df['target'].value_counts().to_dict()}")
    return df

# NOTE: For this audio competition, use create_train_df_from_filenames(TRAIN_PATH)
# instead of loading train.csv (which does not exist)
'''

        # Inject submission format hint for multi-label/multi-class competitions
        submission_format = data_files.get("submission_format_info", {})
        if submission_format:
            num_classes = submission_format.get("num_classes", 1)
            id_pattern = submission_format.get("id_pattern", "")
            if num_classes > 1 or id_pattern:
                path_header += f'''
# === SUBMISSION FORMAT (AUTO-DETECTED) ===
# num_classes: {num_classes}
# id_pattern: {id_pattern}
# IMPORTANT: Output shape must be (N_samples, {num_classes})
'''
                if "rec_id * 100" in id_pattern or "* 100 +" in id_pattern:
                    path_header += f'''
# CRITICAL: Submission Id = rec_id * 100 + class_id
# Example: rec_id=5, class=3 → Id=503
NUM_CLASSES = {num_classes}

def create_submission_ids(rec_ids, num_classes={num_classes}):
    """Generate submission IDs in rec_id * 100 + class format."""
    ids = []
    for rec_id in rec_ids:
        for cls in range(num_classes):
            ids.append(rec_id * 100 + cls)
    return ids
'''

        path_header += "\n# === END PATH CONSTANTS ===\n"

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
        full_code = path_header + "\n" + code

        # Validate that LLM did not redefine injected path constants
        is_valid, violations = self._validate_no_path_redefinition(full_code)
        if not is_valid:
            print(f"⚠️  PATH REDEFINITION WARNING: {violations}")
            print("   LLM generated code that redefines injected path constants.")
            print("   Stripping redefinitions to prevent artifacts in wrong locations...")
            # Strip the redefinitions to enforce correct paths
            full_code = self._strip_path_redefinitions(full_code)

        # Strip nrows parameters to prevent data truncation and OOF shape mismatches
        # This is critical for ensemble alignment - all models must use full dataset
        full_code, nrows_removals = self._strip_nrows_param(full_code)
        if nrows_removals > 0:
            print(f"⚠️  NROWS STRIPPED: Removed {nrows_removals} nrows parameter(s) to enforce full dataset usage")
            print("   All models must use the canonical dataset for proper OOF alignment.")

        # Rewrite BASE_DIR references to use correct path constants
        # BASE_DIR is not defined - LLM sometimes generates it from training examples
        full_code, base_dir_rewrites = self._rewrite_base_dir_references(full_code)
        if base_dir_rewrites > 0:
            print(f"⚠️  BASE_DIR REWRITTEN: Replaced {base_dir_rewrites} BASE_DIR reference(s) with correct path constants")
            print("   BASE_DIR is not defined. Use TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH, or OUTPUT_DIR.")

        # Validate audio label usage - warn if LLM is re-parsing files instead of using pre-loaded labels
        audio_warnings = self._validate_audio_label_usage(full_code, data_type)
        for warning in audio_warnings:
            print(warning)
            print("   HINT: Use _PRELOADED_LABELS_DF, _PRELOADED_REC_IDS, _PRELOADED_N_CLASSES instead.")

        # Replace label re-parsing for audio competitions - ENFORCE usage of pre-loaded labels
        # This is stronger than warnings because LLMs often ignore prompt instructions
        if data_type in ("audio", "audio_classification"):
            full_code, replace_count = self._strip_label_reparsing(full_code)
            if replace_count > 0:
                print(f"⚠️  REPLACED {replace_count} label re-parsing statement(s)")
                print("   LLM code tried to re-parse label files instead of using _PRELOADED_LABELS_DF.")
                print("   Replaced with: varname = _PRELOADED_LABELS_DF.copy()")

        return full_code
