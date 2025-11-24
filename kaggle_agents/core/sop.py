"""Standard Operating Procedure (SOP) - Workflow orchestrator for enhanced agents."""

import logging
from typing import Tuple, Dict, Any
from pathlib import Path

from .state import (
    EnhancedKaggleState,
    should_retry_phase,
    increment_retry_count,
    reset_retry_count,
)
from .config_manager import get_config
from ..enhanced_agents import (
    ReaderAgent,
    PlannerAgent,
    DeveloperAgent,
    ReviewerAgent,
    SummarizerAgent,
)

logger = logging.getLogger(__name__)


class SOP:
    """Standard Operating Procedure for orchestrating the multi-agent workflow."""

    def __init__(self, competition_name: str, model: str = "gpt-5-mini"):
        """Initialize SOP orchestrator.

        Args:
            competition_name: Name of the Kaggle competition
            model: Default LLM model to use
        """
        self.competition_name = competition_name
        self.model = model
        self.config = get_config()

        # Initialize agents
        logger.info(f"Initializing agents with model: {model}")
        self.agents = {
            "reader": ReaderAgent(model),
            "planner": PlannerAgent(model),
            "developer": DeveloperAgent(model),
            "reviewer": ReviewerAgent(model),
            "summarizer": SummarizerAgent(model),
        }

        logger.info(f"SOP initialized for competition: {competition_name}")

    def step(self, state: EnhancedKaggleState) -> Tuple[str, EnhancedKaggleState]:
        """Execute one step of the workflow.

        Args:
            state: Current state (dict/TypedDict from LangGraph)

        Returns:
            Tuple of (status, updated_state) where status is:
                - "Continue": Phase executed successfully
                - "Retry": Phase needs retry
                - "Complete": Workflow completed
                - "Fail": Workflow failed
        """
        logger.info("=" * 80)
        logger.info(f"SOP STEP - Executing phase: {state.get('phase', 'UNKNOWN')}")
        logger.info(
            f"SOP STEP - Retry count: {state.get('retry_count', 0)}/{state.get('max_phase_retries', 3)}"
        )
        logger.info(
            f"SOP STEP - Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 1)}"
        )
        logger.info("=" * 80)

        # Get agents for this phase
        agent_roles = self.config.get_phase_agents(state.get("phase", ""))

        if not agent_roles:
            logger.error(
                f"❌ SOP STEP - No agents configured for phase: {state.get('phase', '')}"
            )
            return "Fail", state

        logger.info(f"✓ SOP STEP - Agent roles for this phase: {agent_roles}")

        # Execute agents in sequence
        phase_results = {}

        # Add initial empty entry to memory so reviewer can access current phase results
        memory = state.get("memory", [])
        current_phase_index = len(memory)
        memory.append(
            {
                "phase": state.get("phase", ""),
                "iteration": state.get("iteration", 0),
                "retry_count": state.get("retry_count", 0),
            }
        )
        state["memory"] = memory

        for agent_role in agent_roles:
            if agent_role not in self.agents:
                logger.error(f"❌ SOP STEP - Agent not found: {agent_role}")
                continue

            logger.info(f"🤖 SOP STEP - Executing agent: {agent_role}")

            try:
                agent = self.agents[agent_role]
                result = agent.action(state)

                # Store result
                phase_results.update(result)

                # Update current memory entry incrementally so reviewer can see results
                state["memory"][current_phase_index].update(result)

                logger.info(f"✅ SOP STEP - Agent {agent_role} completed successfully")

                # Log key result details
                if agent_role in result:
                    agent_result = result[agent_role]
                    if "status" in agent_result:
                        logger.info(f"   └─ Status: {agent_result['status']}")
                    if "success" in agent_result:
                        logger.info(f"   └─ Success: {agent_result['success']}")

            except Exception as e:
                logger.error(
                    f"❌ SOP STEP - Agent {agent_role} failed: {e}", exc_info=True
                )
                # Continue with other agents even if one fails
                phase_results[agent_role] = {
                    "role": agent_role,
                    "error": str(e),
                    "result": f"Agent failed with error: {str(e)}",
                }
                # Update memory entry with error
                state["memory"][current_phase_index][agent_role] = phase_results[
                    agent_role
                ]

        # Memory already updated incrementally above, no need to add again

        # Check if phase was successful
        status = self._evaluate_phase_results(state, phase_results)

        logger.info(
            f"📊 SOP STEP - Phase '{state.get('phase', 'UNKNOWN')}' evaluation result: {status}"
        )
        logger.info("=" * 80)

        return status, state

    def _evaluate_phase_results(
        self, state: EnhancedKaggleState, phase_results: Dict[str, Any]
    ) -> str:
        """Evaluate phase results and determine next action.

        Args:
            state: Current state (dict)
            phase_results: Results from phase execution

        Returns:
            Status string: "Continue", "Retry", "Complete", or "Fail"
        """
        # Check if reviewer was executed
        if "reviewer" in phase_results:
            reviewer_result = phase_results["reviewer"]
            should_proceed = reviewer_result.get("should_proceed", False)
            average_score = reviewer_result.get("average_score", 0)

            logger.info(
                f"📈 EVALUATION - Reviewer score: {average_score:.2f}/5.0, Proceed: {should_proceed}"
            )

            if should_proceed:
                # Phase successful - workflow will handle phase transition
                logger.info("✅ EVALUATION - Phase completed successfully")
                reset_retry_count(state)

                # Don't change phase here - let workflow routing handle it
                # Just return "Continue" to signal success
                return "Continue"
            else:
                # Phase needs retry
                if should_retry_phase(state):
                    logger.warning(
                        f"🔄 EVALUATION - Phase needs retry. Retry count: {state.get('retry_count', 0) + 1}/{state.get('max_phase_retries', 3)}"
                    )
                    increment_retry_count(state)
                    return "Retry"
                else:
                    logger.error("❌ EVALUATION - Max retries reached for phase")
                    return "Fail"

        else:
            # No reviewer (e.g., first phase), assume success
            logger.info("✅ EVALUATION - No reviewer in phase, assuming success")
            reset_retry_count(state)

            # Don't change phase - let workflow handle it
            return "Continue"

    def run(
        self, initial_state: EnhancedKaggleState, max_steps: int = 100
    ) -> EnhancedKaggleState:
        """Run the complete workflow.

        Args:
            initial_state: Initial state
            max_steps: Maximum number of steps to prevent infinite loops

        Returns:
            Final state
        """
        logger.info(f"Starting SOP workflow for: {self.competition_name}")
        logger.info(f"Initial phase: {initial_state.phase}")

        state = initial_state
        step_count = 0

        while step_count < max_steps:
            step_count += 1
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Step {step_count}/{max_steps}")
            logger.info(f"{'=' * 80}\n")

            # Execute one step
            status, state = self.step(state)

            # Save state to disk
            state.save_to_disk()

            # Check status
            if status == "Complete":
                logger.info("Workflow completed successfully!")
                break

            elif status == "Fail":
                logger.error("Workflow failed!")
                break

            elif status == "Retry":
                logger.info("Retrying current phase...")
                continue

            elif status == "Continue":
                logger.info(f"Moving to next phase: {state.phase}")
                continue

            else:
                logger.error(f"Unknown status: {status}")
                break

        if step_count >= max_steps:
            logger.warning(f"Reached maximum steps ({max_steps})")

        logger.info(f"Workflow finished after {step_count} steps")

        return state


if __name__ == "__main__":
    # Test SOP
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("sop_test.log"),
        ],
    )

    # Create test state
    competition_dir = Path("./test_data/titanic")
    competition_dir.mkdir(parents=True, exist_ok=True)

    initial_state = EnhancedKaggleState(
        competition_name="titanic",
        competition_dir=str(competition_dir),
        phase="Understand Background",
    )

    # Create and run SOP
    sop = SOP(competition_name="titanic", model="gpt-5-mini")

    print("Starting SOP test run...")
    print("This will execute the complete workflow for the Titanic competition")
    print("=" * 80)

    final_state = sop.run(initial_state, max_steps=20)

    print("\n" + "=" * 80)
    print("SOP Test Complete")
    print(f"Final phase: {final_state.phase}")
    print(f"Total iterations: {final_state.iteration}")
    print(f"Memory entries: {len(final_state.memory)}")
    print("=" * 80)
