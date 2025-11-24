"""Base class for all agents with common functionality."""

import json
import re
import logging
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import pandas as pd

from .api_handler import APIHandler, APISettings
from .state import EnhancedKaggleState, get_restore_dir, get_dir_name
from ..prompts.prompt_base import (
    AGENT_ROLE_TEMPLATE,
    PROMPT_DATA_PREVIEW,
    PROMPT_FEATURE_INFO,
    PROMPT_EACH_EXPERIENCE_WITH_SUGGESTION,
    PROMPT_REORGANIZE_JSON,
    PROMPT_REORGANIZE_EXTRACT_TOOLS,
)

# LangSmith tracing
try:
    from langsmith import traceable
except ImportError:

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


class Agent:
    """Base class for all agents with common patterns and utilities."""

    def __init__(self, role: str, description: str, model: str = "gpt-5-mini"):
        """Initialize agent.

        Args:
            role: Agent's role (e.g., "planner", "developer")
            description: Agent's description and capabilities
            model: LLM model to use
        """
        self.role = role
        self.description = description
        self.model = model
        self.api_handler = APIHandler(model)
        logger.info(f"Agent {self.role} created with model {model}.")

    @traceable(name="Agent_Generate", run_type="chain")
    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_completion_tokens: int = 16000,
        temperature: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Generate response from LLM.

        Args:
            prompt: User prompt
            history: Conversation history
            max_completion_tokens: Maximum tokens to generate (default 16000, increased to prevent incomplete responses)
            temperature: Sampling temperature (if None, uses config default)

        Returns:
            Tuple of (response, updated_history)
        """
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]

        # Use temperature from config if not specified
        if temperature is None:
            from .config_manager import get_config

            config = get_config()
            temperature = config.get_temperature()

        settings = APISettings(
            max_completion_tokens=max_completion_tokens, temperature=temperature
        )

        reply = self.api_handler.get_output(messages=messages, settings=settings)

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})

        return reply, history

    def _gather_experience_with_suggestion(self, state: EnhancedKaggleState) -> str:
        """Gather past experiences with reviewer suggestions.

        Args:
            state: Current state

        Returns:
            Formatted string with experiences and suggestions
        """
        experience_with_suggestion = ""

        # Skip the current (last) memory entry
        memory = state.get("memory", [])
        if not memory:
            return ""

        for i, each_state_memory in enumerate(memory[:-1]):
            # Get this agent's memory from the past state
            act_agent_memory = each_state_memory.get(self.role, {})
            result = act_agent_memory.get("result", "")

            # Get reviewer's feedback for this agent
            reviewer_memory = each_state_memory.get("reviewer", {})
            suggestion = reviewer_memory.get("suggestion", {}).get(
                f"agent {self.role}", ""
            )
            score = reviewer_memory.get("score", {}).get(f"agent {self.role}", 3)

            experience_with_suggestion += PROMPT_EACH_EXPERIENCE_WITH_SUGGESTION.format(
                index=i, experience=result, suggestion=suggestion, score=score
            )

            # For developer: include error messages if available
            if self.role == "developer":
                restore_dir = get_restore_dir(state)
                dir_name = get_dir_name(state)
                error_file = restore_dir / f"{dir_name}_error.txt"
                not_pass_file = restore_dir / f"{dir_name}_not_pass_information.txt"

                if error_file.exists():
                    with open(error_file, "r") as f:
                        error_message = f.read()
                    experience_with_suggestion += (
                        f"\n<ERROR MESSAGE>\n{error_message}\n</ERROR MESSAGE>\n"
                    )
                elif not_pass_file.exists():
                    with open(not_pass_file, "r") as f:
                        not_pass_info = f.read()
                    experience_with_suggestion += f"\n<NOT PASS INFORMATION>\n{not_pass_info}\n</NOT PASS INFORMATION>\n"

        return experience_with_suggestion

    def _read_data(self, state: EnhancedKaggleState, num_lines: int = 11) -> str:
        """Read and format data based on current phase.

        Args:
            state: Current state
            num_lines: Number of lines to read from each file

        Returns:
            Formatted data string
        """

        def read_sample(file_path: Path, num_lines: int) -> str:
            """Read first num_lines from a file."""
            sample_lines = []
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= num_lines:
                        break
                    sample_lines.append(line)
            return "".join(sample_lines)

        competition_dir = Path(state.get("competition_dir", "."))

        # Get target columns
        submission_df = pd.read_csv(competition_dir / "sample_submission.csv")
        submission_columns = submission_df.columns.tolist()
        target_columns = submission_columns[1:]

        result = f"\n#############\n# TARGET VARIABLE #\n{target_columns}"

        # Phase-specific data reading
        phase = state.get("phase", "")
        if phase in [
            "Understand Background",
            "Preliminary Exploratory Data Analysis",
            "Data Cleaning",
        ]:
            train_sample = read_sample(competition_dir / "train.csv", num_lines)
            test_sample = read_sample(competition_dir / "test.csv", num_lines)
            result += f"\n#############\n# TRAIN DATA #\n{train_sample}\n#############\n# TEST DATA #\n{test_sample}"

        elif phase in ["In-depth Exploratory Data Analysis", "Feature Engineering"]:
            cleaned_train = competition_dir / "cleaned_train.csv"
            cleaned_test = competition_dir / "cleaned_test.csv"

            if cleaned_train.exists() and cleaned_test.exists():
                train_sample = read_sample(cleaned_train, num_lines)
                test_sample = read_sample(cleaned_test, num_lines)
                result += f"\n#############\n# CLEANED TRAIN DATA #\n{train_sample}\n#############\n# CLEANED TEST DATA #\n{test_sample}"

        elif phase in ["Model Building, Validation, and Prediction"]:
            processed_train = competition_dir / "processed_train.csv"
            processed_test = competition_dir / "processed_test.csv"

            if processed_train.exists() and processed_test.exists():
                train_sample = read_sample(processed_train, num_lines)
                test_sample = read_sample(processed_test, num_lines)
                submission_sample = read_sample(
                    competition_dir / "sample_submission.csv", num_lines
                )
                result += f"\n#############\n# PROCESSED TRAIN DATA #\n{train_sample}\n#############\n# PROCESSED TEST DATA #\n{test_sample}\n#############\n# SUBMISSION FORMAT #\n{submission_sample}"

                # Extract evaluation metric
                metric = state.get("metric", "")
                if metric:
                    result += f"\n#############\n# EVALUATION METRIC #\n{metric}"

        return result

    def _data_preview(self, state: EnhancedKaggleState, num_lines: int = 11) -> str:
        """Generate a preview of the data using LLM.

        Args:
            state: Current state
            num_lines: Number of lines to read

        Returns:
            Formatted data preview
        """
        data_used_in_preview = self._read_data(state, num_lines=num_lines)
        input_prompt = PROMPT_DATA_PREVIEW.format(data=data_used_in_preview)

        raw_reply, _ = self.generate(input_prompt, [])
        data_preview = self._parse_markdown(raw_reply)

        # Save preview to disk
        restore_dir = get_restore_dir(state)
        preview_file = restore_dir / "data_preview.txt"
        with open(preview_file, "w") as f:
            f.write(data_preview)

        return data_preview

    def _parse_json(self, raw_reply: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with fallback reorganization.

        Args:
            raw_reply: Raw LLM response

        Returns:
            Parsed dictionary
        """

        def try_json_loads(data: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e}")
                return None

        raw_reply = raw_reply.strip()
        logger.info("Attempting to extract JSON from raw reply.")

        # Try to extract JSON from code blocks
        json_match = re.search(r"```json\s*(.+?)\s*```", raw_reply, re.DOTALL)

        if json_match:
            reply_str = json_match.group(1).strip()
            reply = try_json_loads(reply_str)
            if reply is not None:
                return reply

        # Try without code blocks
        json_match = re.search(r"\{.+\}", raw_reply, re.DOTALL)
        if json_match:
            reply_str = json_match.group(0).strip()
            reply = try_json_loads(reply_str)
            if reply is not None:
                return reply

        # Fallback: Ask LLM to reorganize
        logger.info("Failed to parse JSON, attempting reorganization.")

        if self.role == "developer":
            prompt = PROMPT_REORGANIZE_EXTRACT_TOOLS.format(information=raw_reply)
        else:
            prompt = PROMPT_REORGANIZE_JSON.format(information=raw_reply)

        json_reply, _ = self.generate(prompt, history=[])

        json_match = re.search(r"```json\s*(.+?)\s*```", json_reply, re.DOTALL)
        if json_match:
            reply_str = json_match.group(1).strip()
            reply = try_json_loads(reply_str)
            if reply is not None:
                return reply

        logger.error("Final attempt to parse JSON failed.")
        return {}

    def _parse_markdown(self, raw_reply: str) -> str:
        """Parse markdown from LLM response.

        Args:
            raw_reply: Raw LLM response

        Returns:
            Extracted markdown content
        """
        # Try to match ```markdown first
        markdown_match = re.search(r"```markdown\s*(.+?)\s*```", raw_reply, re.DOTALL)
        if markdown_match:
            logger.debug("Found ```markdown block")
            return markdown_match.group(1).strip()

        # Try to match any ``` code block
        generic_match = re.search(r"```\s*(.+?)\s*```", raw_reply, re.DOTALL)
        if generic_match:
            logger.debug("Found generic ``` block")
            content = generic_match.group(1).strip()
            # Remove language identifier if present at the start
            content = re.sub(r"^[a-z]+\n", "", content)
            return content

        # If no code blocks, return as-is (LLM might have returned plain text)
        logger.debug(
            f"No markdown code blocks found. Reply preview: {raw_reply[:200]}..."
        )
        return raw_reply.strip()

    def _parse_code(self, raw_reply: str) -> str:
        """Parse Python code from LLM response.

        Args:
            raw_reply: Raw LLM response

        Returns:
            Extracted Python code
        """
        # Remove any leading/trailing whitespace
        raw_reply = raw_reply.strip()

        # Try to match ```python first (most specific)
        code_match = re.search(r"```python\s*\n?(.*?)\n?```", raw_reply, re.DOTALL)
        if code_match:
            logger.debug("Found ```python block")
            return code_match.group(1).strip()

        # Try to match ```py
        py_match = re.search(r"```py\s*\n?(.*?)\n?```", raw_reply, re.DOTALL)
        if py_match:
            logger.debug("Found ```py block")
            return py_match.group(1).strip()

        # Try to match any ``` code block (might be unlabeled Python code)
        generic_match = re.search(r"```\s*\n?(.*?)\n?```", raw_reply, re.DOTALL)
        if generic_match:
            logger.debug("Found generic ``` block, assuming Python code")
            content = generic_match.group(1).strip()
            # Remove language identifier if present at the start
            content = re.sub(r"^(python|py)\s*\n", "", content)
            return content

        # Check if raw_reply starts with ``` (malformed markdown)
        if raw_reply.startswith("```"):
            logger.warning("Malformed code block detected, attempting to extract...")
            # Remove opening ```python or ```
            content = re.sub(r"^```(?:python|py)?\s*\n?", "", raw_reply)
            # Remove trailing ```
            content = re.sub(r"\n?```\s*$", "", content)
            return content.strip()

        # If no code blocks found, return as-is (might be plain code)
        logger.debug(f"No code blocks found. Reply preview: {raw_reply[:200]}...")
        return raw_reply.strip()

    def _get_feature_info(self, state: EnhancedKaggleState) -> str:
        """Get feature information before and after the current phase.

        Args:
            state: Current state

        Returns:
            Formatted feature information
        """
        # Define file mappings for each phase
        phase_files = {
            "Preliminary Exploratory Data Analysis": (
                "train.csv",
                "test.csv",
                "train.csv",
                "test.csv",
            ),
            "Data Cleaning": (
                "train.csv",
                "test.csv",
                "cleaned_train.csv",
                "cleaned_test.csv",
            ),
            "In-depth Exploratory Data Analysis": (
                "cleaned_train.csv",
                "cleaned_test.csv",
                "cleaned_train.csv",
                "cleaned_test.csv",
            ),
            "Feature Engineering": (
                "cleaned_train.csv",
                "cleaned_test.csv",
                "processed_train.csv",
                "processed_test.csv",
            ),
            "Model Building, Validation, and Prediction": (
                "processed_train.csv",
                "processed_test.csv",
                "processed_train.csv",
                "processed_test.csv",
            ),
        }

        phase = state.get("phase", "")
        file_tuple = phase_files.get(phase)
        if file_tuple is None:
            return "Feature information not available for this phase."

        before_train, before_test, after_train, after_test = file_tuple
        competition_dir = Path(state.get("competition_dir", "."))

        # Read datasets
        before_train_df = pd.read_csv(competition_dir / before_train)
        before_test_df = pd.read_csv(competition_dir / before_test)
        after_train_df = pd.read_csv(competition_dir / after_train)
        after_test_df = pd.read_csv(competition_dir / after_test)

        # Get features
        features_before = list(before_train_df.columns)
        features_after = list(after_train_df.columns)

        # Identify target variable
        target_variable = list(set(features_after) - set(after_test_df.columns))

        if len(target_variable) == 1:
            target_variable = target_variable[0]
        elif len(target_variable) > 1:
            logger.warning(f"Multiple target variables found: {target_variable}")
            target_variable = ", ".join(target_variable)
        else:
            target_variable = "Unknown"

        return PROMPT_FEATURE_INFO.format(
            target_variable=target_variable,
            features_before=features_before,
            features_after=features_after,
        )

    @traceable(name="Agent_Action", run_type="chain")
    def action(self, state: EnhancedKaggleState) -> Dict[str, Any]:
        """Execute agent action (to be implemented by subclasses).

        Args:
            state: Current state

        Returns:
            Dictionary with agent results
        """
        logger.info(f"State {state.get('phase', '')} - Agent {self.role} is executing.")
        role_prompt = AGENT_ROLE_TEMPLATE.format(agent_role=self.role)
        return self._execute(state, role_prompt)

    def _execute(self, state: EnhancedKaggleState, role_prompt: str) -> Dict[str, Any]:
        """Execute agent-specific logic (must be implemented by subclasses).

        Args:
            state: Current state
            role_prompt: Role-specific prompt

        Returns:
            Dictionary with agent results
        """
        raise NotImplementedError("Subclasses must implement _execute method!")
