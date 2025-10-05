"""Standard Operating Procedure (SOP) - Workflow orchestrator for enhanced agents."""

import logging
from typing import Tuple, Dict, Any
from pathlib import Path

from .state import EnhancedKaggleState
from .config_manager import get_config
from .memory import Memory
from ..enhanced_agents import (
    ReaderAgent,
    PlannerAgent,
    DeveloperAgent,
    ReviewerAgent,
    SummarizerAgent
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
            "summarizer": SummarizerAgent(model)
        }

        logger.info(f"SOP initialized for competition: {competition_name}")

    def _state_to_dict(self, state_obj: EnhancedKaggleState) -> dict:
        """Convert EnhancedKaggleState object to dict for LangGraph.

        Args:
            state_obj: EnhancedKaggleState object

        Returns:
            Dictionary representation of state
        """
        # Create dict with all state fields
        return {
            "messages": state_obj.messages if hasattr(state_obj, 'messages') else [],
            "competition_name": state_obj.competition_name,
            "competition_type": state_obj.competition_type,
            "metric": state_obj.metric,
            "competition_dir": state_obj.competition_dir,
            "train_data_path": state_obj.train_data_path,
            "test_data_path": state_obj.test_data_path,
            "sample_submission_path": state_obj.sample_submission_path,
            "eda_summary": state_obj.eda_summary,
            "data_insights": state_obj.data_insights,
            "features_engineered": state_obj.features_engineered,
            "feature_importance": state_obj.feature_importance,
            "models_trained": state_obj.models_trained,
            "best_model": state_obj.best_model,
            "cv_scores": state_obj.cv_scores,
            "submission_path": state_obj.submission_path,
            "submission_score": state_obj.submission_score,
            "leaderboard_rank": state_obj.leaderboard_rank,
            "iteration": state_obj.iteration,
            "max_iterations": state_obj.max_iterations,
            "errors": state_obj.errors,
            "phase": state_obj.phase,
            "memory": state_obj.memory,
            "background_info": state_obj.background_info,
            "rules": state_obj.rules,
            "retry_count": state_obj.retry_count,
            "max_phase_retries": state_obj.max_phase_retries,
            "status": state_obj.status,
        }

    def step(self, state: dict) -> Tuple[str, dict]:
        """Execute one step of the workflow.

        Args:
            state: Current state (dict from LangGraph)

        Returns:
            Tuple of (status, updated_state_dict) where status is:
                - "Continue": Phase executed successfully
                - "Retry": Phase needs retry
                - "Complete": Workflow completed
                - "Fail": Workflow failed
        """
        # Convert dict to EnhancedKaggleState object for processing
        state_obj = EnhancedKaggleState(**state)

        logger.info(f"="*80)
        logger.info(f"Executing phase: {state_obj.phase}")
        logger.info(f"Retry count: {state_obj.retry_count}/{state_obj.max_phase_retries}")
        logger.info(f"="*80)

        # Get agents for this phase
        agent_roles = self.config.get_phase_agents(state_obj.phase)

        if not agent_roles:
            logger.warning(f"No agents configured for phase: {state_obj.phase}")
            # Convert back to dict before returning
            return "Fail", self._state_to_dict(state_obj)

        # Execute agents in sequence
        phase_results = {}

        for agent_role in agent_roles:
            if agent_role not in self.agents:
                logger.error(f"Agent not found: {agent_role}")
                continue

            logger.info(f"Executing agent: {agent_role}")

            try:
                agent = self.agents[agent_role]
                result = agent.action(state_obj)

                # Store result
                phase_results.update(result)

                logger.info(f"Agent {agent_role} completed successfully")

            except Exception as e:
                logger.error(f"Agent {agent_role} failed: {e}", exc_info=True)
                # Continue with other agents even if one fails
                phase_results[agent_role] = {
                    "role": agent_role,
                    "error": str(e),
                    "result": f"Agent failed with error: {str(e)}"
                }

        # Add phase results to memory
        state_obj.add_memory(phase_results)

        # Check if phase was successful
        status = self._evaluate_phase_results(state_obj, phase_results)

        logger.info(f"Phase evaluation result: {status}")

        # Convert state object back to dict before returning
        return status, self._state_to_dict(state_obj)

    def _evaluate_phase_results(
        self,
        state: EnhancedKaggleState,
        phase_results: Dict[str, Any]
    ) -> str:
        """Evaluate phase results and determine next action.

        Args:
            state: Current state
            phase_results: Results from phase execution

        Returns:
            Status string: "Continue", "Retry", "Complete", or "Fail"
        """
        # Check if reviewer was executed
        if "reviewer" in phase_results:
            reviewer_result = phase_results["reviewer"]
            should_proceed = reviewer_result.get("should_proceed", False)
            average_score = reviewer_result.get("average_score", 0)

            logger.info(f"Reviewer score: {average_score:.2f}/5.0, Proceed: {should_proceed}")

            if should_proceed:
                # Phase successful, move to next phase
                logger.info("Phase completed successfully")
                state.next_phase()
                state.reset_retry_count()

                # Check if workflow is complete
                if state.phase == "Complete":
                    return "Complete"

                return "Continue"
            else:
                # Phase needs retry
                if state.should_retry_phase():
                    logger.warning(f"Phase needs retry. Retry count: {state.retry_count + 1}/{state.max_phase_retries}")
                    state.increment_retry_count()
                    return "Retry"
                else:
                    logger.error("Max retries reached for phase")
                    return "Fail"

        else:
            # No reviewer (e.g., first phase), assume success
            logger.info("No reviewer in phase, assuming success")
            state.next_phase()
            state.reset_retry_count()

            if state.phase == "Complete":
                return "Complete"

            return "Continue"

    def run(self, initial_state: EnhancedKaggleState, max_steps: int = 100) -> EnhancedKaggleState:
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
            logger.info(f"\n{'='*80}")
            logger.info(f"Step {step_count}/{max_steps}")
            logger.info(f"{'='*80}\n")

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


if __name__ == '__main__':
    # Test SOP
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sop_test.log')
        ]
    )

    # Create test state
    competition_dir = Path("./test_data/titanic")
    competition_dir.mkdir(parents=True, exist_ok=True)

    initial_state = EnhancedKaggleState(
        competition_name="titanic",
        competition_dir=str(competition_dir),
        phase="Understand Background"
    )

    # Create and run SOP
    sop = SOP(competition_name="titanic", model="gpt-5-mini")

    print("Starting SOP test run...")
    print("This will execute the complete workflow for the Titanic competition")
    print("="*80)

    final_state = sop.run(initial_state, max_steps=20)

    print("\n" + "="*80)
    print("SOP Test Complete")
    print(f"Final phase: {final_state.phase}")
    print(f"Total iterations: {final_state.iteration}")
    print(f"Memory entries: {len(final_state.memory)}")
    print("="*80)
