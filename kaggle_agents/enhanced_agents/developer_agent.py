"""Developer agent with code generation, retry logic, and debugging."""

import json
import logging
from typing import Dict, Any, Tuple
from pathlib import Path

from ..core.agent_base import Agent
from ..core.state import EnhancedKaggleState, get_restore_dir, get_dir_name, get_state_info
from ..core.executor import CodeExecutor
from ..core.config_manager import get_config
from ..prompts.prompt_developer import (
    PROMPT_DEVELOPER,
    PROMPT_DEVELOPER_TASK,
    PROMPT_FIX_CODE,
    PROMPT_DEBUG_CODE
)

logger = logging.getLogger(__name__)


class DeveloperAgent(Agent):
    """Agent responsible for code generation with retry and debugging logic."""

    def __init__(self, model: str = "gpt-5-mini"):
        """Initialize Developer agent.

        Args:
            model: LLM model to use
        """
        super().__init__(
            role="developer",
            description="You are an expert Python programmer for data science and machine learning.",
            model=model
        )
        self.config = get_config()
        self.executor = CodeExecutor()

    def _load_plan(self, state: EnhancedKaggleState) -> str:
        """Load the plan from planner.

        Args:
            state: Current state

        Returns:
            Plan markdown
        """
        restore_dir = get_restore_dir(state)
        plan_file = restore_dir / "markdown_plan.txt"

        if plan_file.exists():
            with open(plan_file, 'r') as f:
                plan_content = f.read()
            logger.info(f"Plan loaded from: {plan_file}")
            return plan_content
        else:
            logger.warning(f"Plan file not found: {plan_file}. Using default plan.")
            # Provide a basic template when plan is missing
            phase = state.get("phase", "")
            return f"""# Default Implementation Plan for {phase}

## Objective
Implement the necessary code for the {phase} phase.

## Steps
1. Load and prepare the data
2. Apply appropriate transformations
3. Save the results
4. Ensure code quality and error handling

Note: This is a fallback plan since the planner's output was not found.
Please implement a reasonable solution based on the phase requirements.
"""

    def _generate_code(
        self,
        state: EnhancedKaggleState,
        plan: str,
        tools: str,
        history: list,
        past_errors: str = ""
    ) -> Tuple[str, list]:
        """Generate code based on plan.

        Args:
            state: Current state
            plan: Implementation plan
            tools: Available tools description
            history: Conversation history
            past_errors: Previous error messages

        Returns:
            Tuple of (generated_code, updated_history)
        """
        # Prepare developer prompt
        data_info = self._read_data(state, num_lines=11)
        experience = self._gather_experience_with_suggestion(state)

        if past_errors:
            experience += f"\n\n# PREVIOUS ERRORS #\n{past_errors}\n"

        phase = state.get("phase", "")
        task = PROMPT_DEVELOPER_TASK.format(phase_name=phase)
        state_info = get_state_info(state)
        restore_dir = get_restore_dir(state)

        input_prompt = PROMPT_DEVELOPER.format(
            phase_name=phase,
            state_info=state_info,
            plan=plan,
            tools=tools,
            data_info=data_info,
            experience=experience,
            task=task,
            restore_dir=restore_dir
        )

        # Generate code (let model use default max_completion_tokens)
        raw_reply, history = self.generate(input_prompt, history)

        # Parse code
        code = self._parse_code(raw_reply)

        return code, history

    def _execute_code(
        self,
        code: str,
        state: EnhancedKaggleState
    ) -> Tuple[bool, str, str]:
        """Execute generated code.

        Args:
            code: Python code to execute
            state: Current state

        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Save code to file
        restore_dir = get_restore_dir(state)
        dir_name = get_dir_name(state)
        code_file = restore_dir / f"{dir_name}_code.py"
        with open(code_file, 'w') as f:
            f.write(code)

        logger.info(f"Executing code: {code_file}")

        # Set working directory to competition directory
        self.executor.working_dir = Path(state.get("competition_dir", "."))

        # Execute code
        timeout = self.config.get_code_timeout()
        success, stdout, stderr = self.executor.execute_code(
            code,
            timeout=timeout,
            capture_output=True
        )

        # Save execution results
        if stdout:
            stdout_file = restore_dir / f"{dir_name}_stdout.txt"
            with open(stdout_file, 'w') as f:
                f.write(stdout)

        if stderr:
            stderr_file = restore_dir / f"{dir_name}_error.txt"
            with open(stderr_file, 'w') as f:
                f.write(stderr)

        return success, stdout, stderr

    def _debug_code(
        self,
        code: str,
        error_output: str,
        history: list
    ) -> Tuple[str, list]:
        """Debug and fix code based on error.

        Args:
            code: Original code
            error_output: Error output from execution
            history: Conversation history

        Returns:
            Tuple of (fixed_code, updated_history)
        """
        # Parse error information
        error_info = self.executor.parse_error_message(error_output)

        # Create debug prompt
        debug_prompt = PROMPT_DEBUG_CODE.format(
            code=code,
            error_output=error_output,
            error_type=error_info['error_type']
        )

        # Get fixed code
        raw_reply, history = self.generate(debug_prompt, history)
        fixed_code = self._parse_code(raw_reply)

        return fixed_code, history

    def _execute(self, state: EnhancedKaggleState, role_prompt: str) -> Dict[str, Any]:
        """Execute developer agent with retry and debugging logic.

        Args:
            state: Current state
            role_prompt: Role-specific prompt

        Returns:
            Dictionary with developer results
        """
        phase = state.get("phase", "")
        logger.info(f"Developer Agent executing for phase: {phase}")

        history = []

        # Initialize system message
        if self.model == 'gpt-5-mini':
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        elif self.model == 'o1-mini':
            history.append({"role": "user", "content": f"{role_prompt}{self.description}"})

        # Load plan
        plan = self._load_plan(state)

        # Get tools
        tools, tool_names = self._get_tools(state)

        task = PROMPT_DEVELOPER_TASK.format(phase_name=phase)

        # Code generation with retry loop
        max_retries = self.config.get_max_code_retries()
        max_debug_iterations = self.config.get_max_debug_iterations()

        code = None
        success = False
        past_errors = ""

        # Retry loop for code generation
        for attempt in range(max_retries):
            logger.info(f"Code generation attempt {attempt + 1}/{max_retries}")

            try:
                # Generate code
                code, history = self._generate_code(
                    state, plan, tools, history, past_errors
                )

                # Save generated code
                restore_dir = get_restore_dir(state)
                dir_name = get_dir_name(state)
                code_file = restore_dir / f"{dir_name}_code.py"
                with open(code_file, 'w') as f:
                    f.write(code)

                # Execute code
                success, stdout, stderr = self._execute_code(code, state)

                if success:
                    logger.info("Code execution successful!")
                    break
                else:
                    logger.warning(f"Code execution failed on attempt {attempt + 1}")
                    past_errors += f"\n\n## Attempt {attempt + 1} Error ##\n{stderr}\n"

                    if attempt < max_retries - 1:
                        # Try to debug and fix
                        logger.info("Attempting to debug code...")
                        code, history = self._debug_code(code, stderr, history)

            except Exception as e:
                logger.error(f"Error during code generation/execution: {e}")
                past_errors += f"\n\n## Attempt {attempt + 1} Exception ##\n{str(e)}\n"

        # If code execution failed, try debugging iterations
        if not success and code:
            logger.info("Starting debugging iterations...")

            for iteration in range(max_debug_iterations):
                logger.info(f"Debug iteration {iteration + 1}/{max_debug_iterations}")

                try:
                    # Execute code again
                    success, stdout, stderr = self._execute_code(code, state)

                    if success:
                        logger.info(f"Code execution successful after {iteration + 1} debug iterations!")
                        break

                    # Debug and fix
                    code, history = self._debug_code(code, stderr, history)

                except Exception as e:
                    logger.error(f"Error during debug iteration: {e}")
                    break

        # Save final history
        restore_dir = get_restore_dir(state)
        history_file = restore_dir / f"{self.role}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        # Prepare result
        if success:
            result_status = "success"
            result = f"Code implementation completed successfully.\n\nCode saved to: {code_file}"
        else:
            result_status = "partial" if code else "failed"
            result = f"Code implementation completed with issues. Review errors in: {restore_dir}"

        input_used_in_review = f"Plan:\n{plan[:1000]}..."

        # Log detailed metrics
        phase = state.get("phase", "")
        logger.info(f"=" * 60)
        logger.info(f"DEVELOPER METRICS - Phase: {phase}")
        logger.info(f"=" * 60)
        logger.info(f"  Status: {result_status.upper()}")
        logger.info(f"  Success: {success}")
        logger.info(f"  Code Length: {len(code) if code else 0} chars")
        logger.info(f"  Max Retries: {max_retries}")
        logger.info(f"=" * 60)

        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": task,
                "input": input_used_in_review,
                "code": code,
                "result": result,
                "status": result_status,
                "success": success
            }
        }

    def _get_tools(self, state: EnhancedKaggleState) -> Tuple[str, list]:
        """Get tools for developer (delegates to parent class).

        Args:
            state: Current state

        Returns:
            Tuple of (tools_description, tool_names)
        """
        # This uses the tool retrieval from planner
        restore_dir = get_restore_dir(state)
        dir_name = get_dir_name(state)
        tools_file = restore_dir / f"tools_used_in_{dir_name}.md"

        if tools_file.exists():
            with open(tools_file, 'r') as f:
                tools = f.read()
            # Extract tool names from file (simple approach)
            tool_names = []
            return tools, tool_names
        else:
            return "No tools specified for this phase.", []


if __name__ == '__main__':
    # Test Developer Agent
    from ..core.state import EnhancedKaggleState

    # Create test state
    state = EnhancedKaggleState(
        competition_name="titanic",
        competition_dir="./test_data/titanic",
        phase="Data Cleaning"
    )

    # Create and run developer
    developer = DeveloperAgent()
    result = developer.action(state)

    print("Developer Result:")
    print(f"Status: {result['developer']['status']}")
    print(f"Success: {result['developer']['success']}")
    print(f"\nCode saved to: {state.restore_dir / f'{state.dir_name}_code.py'}")
