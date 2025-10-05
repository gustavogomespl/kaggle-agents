"""Planner agent with multi-round planning and tool retrieval."""

import json
import logging
from typing import Dict, Any, Tuple, List
from pathlib import Path

from ..core.agent_base import Agent
from ..core.state import EnhancedKaggleState, get_restore_dir, get_dir_name, get_previous_phase, get_state_info, generate_rules, set_background_info, PHASE_TO_DIRECTORY
from ..core.config_manager import get_config
from ..core.tools import OpenaiEmbeddings, RetrieveTool
from ..core.api_handler import load_api_config
from ..prompts.prompt_planner import (
    PROMPT_PLANNER,
    PROMPT_PLANNER_TASK,
    PROMPT_PLANNER_TOOLS,
    PROMPT_PLANNER_REORGANIZE_IN_MARKDOWN,
    PROMPT_PLANNER_REORGANIZE_IN_JSON
)
from ..prompts.prompt_developer import PROMPT_EXTRACT_TOOLS

logger = logging.getLogger(__name__)


class PlannerAgent(Agent):
    """Agent responsible for multi-round planning with tool retrieval."""

    def __init__(self, model: str = "gpt-5-mini"):
        """Initialize Planner agent.

        Args:
            model: LLM model to use
        """
        super().__init__(
            role="planner",
            description="You are good at planning tasks and creating roadmaps.",
            model=model
        )
        self.config = get_config()

    def _get_previous_plan_and_report(self, state: EnhancedKaggleState) -> Tuple[str, str]:
        """Get plans and reports from previous phases.

        Args:
            state: Current state

        Returns:
            Tuple of (previous_plan, previous_report)
        """
        previous_plan = ""
        previous_phases = get_previous_phase(state, type="plan")

        for previous_phase in previous_phases:
            previous_dir_name = PHASE_TO_DIRECTORY[previous_phase]
            previous_plan += f"## {previous_phase.upper()} ##\n"

            plan_file = Path(state.get("competition_dir", ".")) / previous_dir_name / "plan.json"
            if plan_file.exists():
                with open(plan_file, 'r') as f:
                    previous_plan += f.read()
                    previous_plan += '\n'
            else:
                previous_plan += "There is no plan in this phase.\n"

        # Get report from most recent previous phase
        if previous_phases:
            last_phase = previous_phases[-1]
            last_dir_name = PHASE_TO_DIRECTORY[last_phase]
            report_file = Path(state.get("competition_dir", ".")) / last_dir_name / "report.txt"

            if report_file.exists():
                with open(report_file, 'r') as f:
                    previous_report = f.read()
            else:
                previous_report = "There is no report in the previous phase.\n"
        else:
            previous_report = "No previous phases completed yet.\n"

        return previous_plan, previous_report

    def _get_tools(self, state: EnhancedKaggleState) -> Tuple[str, List[str]]:
        """Get relevant tools for the current phase.

        Args:
            state: Current state

        Returns:
            Tuple of (tools_description, tool_names)
        """
        # Get tool names for this phase from config
        phase = state.get("phase", "")
        all_tool_names = self.config.get_phase_tools(phase)

        if not all_tool_names:
            return "There are no pre-defined tools used in this phase.", []

        # Only retrieve tools for phases that need them
        if phase not in ['Data Cleaning', 'Feature Engineering', 'Model Building, Validation, and Prediction']:
            return "There are no pre-defined tools used in this phase.", []

        try:
            # Initialize tool retrieval
            api_key, base_url = load_api_config()
            embeddings = OpenaiEmbeddings(api_key=api_key, base_url=base_url, verify_ssl=False)

            tool_retriever = RetrieveTool(
                embeddings=embeddings,
                doc_path=str(Path(__file__).parent.parent / "tools" / "ml_tools_doc"),
                collection_name="ml_tools"
            )

            # Create database if needed
            tool_retriever.create_db_tools()

            # If we have a plan, extract relevant tools
            restore_dir = get_restore_dir(state)
            plan_file = restore_dir / "markdown_plan.txt"
            if plan_file.exists() and self.role == 'developer':
                with open(plan_file, 'r') as f:
                    markdown_plan = f.read()

                # Ask LLM to extract relevant tools
                input_prompt = PROMPT_EXTRACT_TOOLS.format(
                    document=markdown_plan,
                    all_tool_names=all_tool_names
                )
                raw_reply, _ = self.generate(input_prompt, history=[], max_completion_tokens=4096)

                try:
                    tool_names = self._parse_json(raw_reply)['tool_names']
                except:
                    tool_names = all_tool_names
            else:
                tool_names = all_tool_names

            # Get tool documentation
            tools_description = tool_retriever.get_tools_by_names(tool_names)

            # Save tools used
            dir_name = get_dir_name(state)
            tools_file = restore_dir / f"tools_used_in_{dir_name}.md"
            with open(tools_file, 'w') as f:
                f.write(tools_description)

            return tools_description, tool_names

        except Exception as e:
            logger.error(f"Error retrieving tools: {e}")
            return f"Error retrieving tools: {str(e)}", []

    def _execute(self, state: EnhancedKaggleState, role_prompt: str) -> Dict[str, Any]:
        """Execute planner agent with multi-round planning.

        Args:
            state: Current state
            role_prompt: Role-specific prompt

        Returns:
            Dictionary with planner results
        """
        phase = state.get("phase", "")
        logger.info(f"Planner Agent executing for phase: {phase}")

        # Check if this is a retry and previous plan was acceptable
        memory = state.get("memory", [])
        if len(memory) > 1:
            last_planner_score = memory[-2].get("reviewer", {}).get("score", {}).get("agent planner", 0)
            if last_planner_score >= 3:
                logger.info("Previous plan was acceptable (score >= 3), reusing it")
                return {"planner": memory[-2]["planner"]}

        history = []

        # Initialize system message
        if self.model == 'gpt-5-mini':
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        elif self.model == 'o1-mini':
            history.append({"role": "user", "content": f"{role_prompt}{self.description}"})

        # Get data preview
        data_preview = self._data_preview(state, num_lines=11)
        background_info = f"Data preview:\n{data_preview}"
        set_background_info(state, background_info)
        state_info = get_state_info(state)

        # Define task
        task = PROMPT_PLANNER_TASK.format(phase_name=phase)

        # Round 1: Initial planning
        logger.info("Round 1: Initial planning")
        user_rules = generate_rules(state)
        context = state.get("context", [])
        input_prompt = PROMPT_PLANNER.format(
            phases_in_context=', '.join(context),
            phase_name=phase,
            state_info=state_info,
            user_rules=user_rules,
            background_info=background_info,
            task=task
        )
        _, history = self.generate(input_prompt, history, max_completion_tokens=4096)

        # Round 2: Incorporate previous results and tools
        logger.info("Round 2: Incorporating tools and history")
        previous_plan, previous_report = self._get_previous_plan_and_report(state)
        input_prompt = f"# PREVIOUS PLAN #\n{previous_plan}\n#############\n# PREVIOUS REPORT #\n{previous_report}\n"
        input_prompt += self._read_data(state, num_lines=11)

        # Get relevant tools
        tools, tool_names = self._get_tools(state)
        if tool_names:
            input_prompt += PROMPT_PLANNER_TOOLS.format(tools=tools, tool_names=tool_names)
        else:
            input_prompt += "# AVAILABLE TOOLS #\nThere are no pre-defined tools in this phase. You can use functions from public libraries such as Pandas, NumPy, Scikit-learn, etc.\n"

        raw_plan_reply, history = self.generate(input_prompt, history, max_completion_tokens=4096)

        # Save raw plan
        restore_dir = get_restore_dir(state)
        raw_plan_file = restore_dir / "raw_plan_reply.txt"
        with open(raw_plan_file, 'w') as f:
            f.write(raw_plan_reply)

        # Round 3: Reorganize in Markdown
        logger.info("Round 3: Organizing in Markdown")
        input_prompt = PROMPT_PLANNER_REORGANIZE_IN_MARKDOWN
        organized_markdown_plan, history = self.generate(input_prompt, history, max_completion_tokens=4096)
        markdown_plan = self._parse_markdown(organized_markdown_plan)

        # Save markdown plan
        markdown_plan_file = restore_dir / "markdown_plan.txt"
        with open(markdown_plan_file, 'w') as f:
            f.write(markdown_plan)

        # Round 4: Reorganize in JSON
        logger.info("Round 4: Organizing in JSON")
        input_prompt = PROMPT_PLANNER_REORGANIZE_IN_JSON
        raw_json_plan, history = self.generate(input_prompt, history, max_completion_tokens=4096)

        try:
            json_plan = self._parse_json(raw_json_plan)['final_answer']
        except Exception as e:
            logger.warning(f"Error parsing JSON plan: {e}, using raw dict")
            json_plan = self._parse_json(raw_json_plan)

        # Save JSON plan
        json_plan_file = restore_dir / "json_plan.json"
        with open(json_plan_file, 'w') as f:
            json.dump(json_plan, f, indent=2)

        # User interaction (if enabled)
        if self.config.is_user_interaction_enabled('plan'):
            self._user_interaction(state, markdown_plan_file)

        # Save history
        history_file = restore_dir / f"{self.role}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        input_used_in_review = f"<background_info>\n{background_info}\n</background_info>"

        logger.info("Planner Agent completed")

        # Return results
        plan = markdown_plan
        result = markdown_plan

        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": task,
                "input": input_used_in_review,
                "plan": plan,
                "result": result
            }
        }

    def _user_interaction(self, state: EnhancedKaggleState, markdown_plan_file: Path):
        """Handle user interaction for plan editing.

        Args:
            state: Current state
            markdown_plan_file: Path to markdown plan file
        """
        print("\n" + "="*80)
        print("PLANNER: Plan Generated")
        print("="*80)
        print(f"\nA plan has been generated and saved to: {markdown_plan_file}")
        print("\nYou can now review and modify the plan if needed.")
        print("Options:")
        print("  - Press Enter to continue with the current plan")
        print("  - Type 'edit' to modify the plan file manually")
        print("  - Type 'suggest' to see suggestions for improvement")

        user_input = input("\nYour choice: ").strip().lower()

        if user_input == 'edit':
            print(f"\nPlease edit the file: {markdown_plan_file}")
            print("Save your changes and press Enter when you're done.")
            input("Press Enter to continue...")
            print("Continuing with modified plan...")

        elif user_input == 'suggest':
            restore_dir = get_restore_dir(state)
            dir_name = get_dir_name(state)
            tools_file = restore_dir / f"tools_used_in_{dir_name}.md"
            print(f"\nPlease refer to the tool documentation in: {tools_file}")
            print("Review the 'Notes' section for each tool for important considerations.")
            input("Press Enter to continue...")

        elif user_input:
            print("Invalid input. Continuing with the current plan.")

        else:
            print("Continuing with the current plan.")

        print("="*80 + "\n")


if __name__ == '__main__':
    # Test Planner Agent
    from ..core.state import EnhancedKaggleState

    # Create test state
    state = EnhancedKaggleState(
        competition_name="titanic",
        competition_dir="./test_data/titanic",
        phase="Data Cleaning"
    )

    # Create and run planner
    planner = PlannerAgent()
    result = planner.action(state)

    print("Planner Result:")
    print(f"Plan saved to: {state.restore_dir / 'markdown_plan.txt'}")
    print(f"\nPlan preview:\n{result['planner']['plan'][:500]}...")
