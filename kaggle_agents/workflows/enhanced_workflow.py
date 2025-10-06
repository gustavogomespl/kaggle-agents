"""Enhanced LangGraph workflow integrating multi-agent system with feedback loops."""

from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.state import EnhancedKaggleState
from ..core.sop import SOP
from ..core.config_manager import get_config

import logging

logger = logging.getLogger(__name__)


def create_enhanced_workflow(
    competition_name: str,
    model: str = "gpt-5-mini",
    checkpointer: Optional[MemorySaver] = None
) -> StateGraph:
    """Create enhanced Kaggle workflow with multi-agent system.

    This workflow integrates the SOP orchestrator with LangGraph for
    advanced features like checkpointing and visualization.

    Args:
        competition_name: Name of Kaggle competition
        model: LLM model to use
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled LangGraph workflow
    """
    config = get_config()

    # Initialize SOP
    sop = SOP(competition_name=competition_name, model=model)

    # Define phase nodes
    def execute_understand_background(state: dict) -> dict:
        """Execute Understand Background phase."""
        logger.info("📖 WORKFLOW - Executing: Understand Background")
        logger.info(f"📖 WORKFLOW - State keys: {list(state.keys())}")
        logger.info(f"📖 WORKFLOW - Current iteration: {state.get('iteration', 'UNKNOWN')}/{state.get('max_iterations', 'UNKNOWN')}")

        # Set phase before execution
        state["phase"] = "Understand Background"

        # Execute SOP step with dict
        logger.info("📖 WORKFLOW - Calling sop.step()")
        status, updated_state = sop.step(state)
        logger.info(f"📖 WORKFLOW - SOP returned status: {status}")

        # Return dict with updates (using .get() for safe access)
        # Keep current phase - routing will change it
        return {
            "phase": "Understand Background",  # Keep current, router will change
            "status": status,
            "memory": updated_state.get("memory", []),
            "background_info": updated_state.get("background_info", ""),
            "competition_type": updated_state.get("competition_type", ""),
            "metric": updated_state.get("metric", ""),
        }

    def execute_preliminary_eda(state: dict) -> dict:
        """Execute Preliminary EDA phase."""
        logger.info("🔍 Executing: Preliminary Exploratory Data Analysis")

        # Set phase before execution
        state["phase"] = "Preliminary Exploratory Data Analysis"

        # Execute SOP step with dict
        status, updated_state = sop.step(state)

        # Return dict with updates - keep current phase
        return {
            "phase": "Preliminary Exploratory Data Analysis",
            "status": status,
            "memory": updated_state.get("memory", []),
            "eda_summary": updated_state.get("eda_summary", {}),
        }

    def execute_data_cleaning(state: dict) -> dict:
        """Execute Data Cleaning phase."""
        logger.info("🧹 Executing: Data Cleaning")

        # Set phase before execution
        state["phase"] = "Data Cleaning"

        # Execute SOP step with dict
        status, updated_state = sop.step(state)

        # Return dict with updates - keep current phase
        return {
            "phase": "Data Cleaning",
            "status": status,
            "memory": updated_state.get("memory", []),
        }

    def execute_deep_eda(state: dict) -> dict:
        """Execute Deep EDA phase."""
        logger.info("🔬 Executing: In-depth Exploratory Data Analysis")

        # Set phase before execution
        state["phase"] = "In-depth Exploratory Data Analysis"

        # Execute SOP step with dict
        status, updated_state = sop.step(state)

        # Return dict with updates - keep current phase
        return {
            "phase": "In-depth Exploratory Data Analysis",
            "status": status,
            "memory": updated_state.get("memory", []),
            "data_insights": updated_state.get("data_insights", []),
        }

    def execute_feature_engineering(state: dict) -> dict:
        """Execute Feature Engineering phase."""
        logger.info("⚙️ Executing: Feature Engineering")

        # Set phase before execution
        state["phase"] = "Feature Engineering"

        # Execute SOP step with dict
        status, updated_state = sop.step(state)

        # Return dict with updates - keep current phase
        return {
            "phase": "Feature Engineering",
            "status": status,
            "memory": updated_state.get("memory", []),
            "features_engineered": updated_state.get("features_engineered", []),
        }

    def execute_model_building(state: dict) -> dict:
        """Execute Model Building phase."""
        logger.info("🎯 Executing: Model Building, Validation, and Prediction")

        # Set phase before execution
        state["phase"] = "Model Building, Validation, and Prediction"

        # Execute SOP step with dict
        status, updated_state = sop.step(state)

        # If this is the last phase and it succeeded, mark as Complete
        if status == "Continue":
            status = "Complete"
            logger.info("🎉 Model Building complete - marking workflow as Complete")

        # Return dict with updates - keep current phase
        return {
            "phase": "Model Building, Validation, and Prediction",
            "status": status,
            "memory": updated_state.get("memory", []),
            "models_trained": updated_state.get("models_trained", []),
            "best_model": updated_state.get("best_model", {}),
            "submission_path": updated_state.get("submission_path", ""),
        }

    # Routing function
    def route_after_phase(state: dict) -> str:
        """Route to next phase or handle retry/failure.

        Args:
            state: Current state (dict)

        Returns:
            Next node name
        """
        status = state.get("status", "Continue")
        current_phase = state.get("phase", "")

        logger.info(f"🔀 ROUTING - Status: {status}, Current Phase: {current_phase}")

        if status == "Complete":
            logger.info("✅ ROUTING - Workflow complete! Going to END")
            return END

        elif status == "Fail":
            logger.error("❌ ROUTING - Workflow failed! Going to END")
            return END

        elif status == "Retry":
            # Retry current phase
            phase_to_node = {
                "Understand Background": "understand_background",
                "Preliminary Exploratory Data Analysis": "preliminary_eda",
                "Data Cleaning": "data_cleaning",
                "In-depth Exploratory Data Analysis": "deep_eda",
                "Feature Engineering": "feature_engineering",
                "Model Building, Validation, and Prediction": "model_building"
            }
            next_node = phase_to_node.get(current_phase, END)
            logger.info(f"🔄 ROUTING - Retrying phase: {current_phase} -> {next_node}")
            return next_node

        elif status == "Continue":
            # Move to next phase
            next_node = None
            if current_phase == "Understand Background":
                next_node = "preliminary_eda"
            elif current_phase == "Preliminary Exploratory Data Analysis":
                next_node = "data_cleaning"
            elif current_phase == "Data Cleaning":
                next_node = "deep_eda"
            elif current_phase == "In-depth Exploratory Data Analysis":
                next_node = "feature_engineering"
            elif current_phase == "Feature Engineering":
                next_node = "model_building"
            elif current_phase == "Model Building, Validation, and Prediction":
                next_node = END
            elif current_phase == "Complete":
                next_node = END
            else:
                logger.warning(f"⚠️ ROUTING - Unknown phase: {current_phase}, going to END")
                next_node = END

            logger.info(f"➡️ ROUTING - Continue from '{current_phase}' to '{next_node}'")
            return next_node

        else:
            logger.error(f"❌ ROUTING - Unknown status: {status}, going to END")
            return END

    # Build workflow graph using dict-based state
    from typing_extensions import TypedDict
    from typing import Annotated, List, Dict, Any
    from operator import add
    from langgraph.graph import MessagesState

    def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries, with right values taking precedence."""
        if not left:
            return right
        if not right:
            return left
        return {**left, **right}

    class WorkflowState(MessagesState):
        """State schema for workflow - uses dict representation."""
        competition_name: str
        competition_type: str
        metric: str
        competition_dir: str
        train_data_path: str
        test_data_path: str
        sample_submission_path: str
        eda_summary: Annotated[Dict[str, Any], merge_dict]
        data_insights: Annotated[List[str], add]
        features_engineered: Annotated[List[str], add]
        feature_importance: Annotated[Dict[str, float], merge_dict]
        models_trained: Annotated[List[Dict[str, Any]], add]
        best_model: Annotated[Dict[str, Any], merge_dict]
        cv_scores: Annotated[List[float], add]
        submission_path: str
        submission_score: float
        leaderboard_rank: int
        iteration: int
        max_iterations: int
        errors: Annotated[List[str], add]
        phase: str
        memory: Annotated[List[Dict[str, Any]], add]
        background_info: str
        rules: Dict[str, Any]
        retry_count: int
        max_phase_retries: int
        status: str

    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("understand_background", execute_understand_background)
    workflow.add_node("preliminary_eda", execute_preliminary_eda)
    workflow.add_node("data_cleaning", execute_data_cleaning)
    workflow.add_node("deep_eda", execute_deep_eda)
    workflow.add_node("feature_engineering", execute_feature_engineering)
    workflow.add_node("model_building", execute_model_building)

    # Add edges
    workflow.add_edge(START, "understand_background")

    # Add conditional routing from each phase
    workflow.add_conditional_edges(
        "understand_background",
        route_after_phase,
        {
            "understand_background": "understand_background",  # Retry
            "preliminary_eda": "preliminary_eda",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "preliminary_eda",
        route_after_phase,
        {
            "preliminary_eda": "preliminary_eda",  # Retry
            "data_cleaning": "data_cleaning",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "data_cleaning",
        route_after_phase,
        {
            "data_cleaning": "data_cleaning",  # Retry
            "deep_eda": "deep_eda",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "deep_eda",
        route_after_phase,
        {
            "deep_eda": "deep_eda",  # Retry
            "feature_engineering": "feature_engineering",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "feature_engineering",
        route_after_phase,
        {
            "feature_engineering": "feature_engineering",  # Retry
            "model_building": "model_building",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "model_building",
        route_after_phase,
        {
            "model_building": "model_building",  # Retry
            END: END
        }
    )

    # Compile with optional checkpointer
    return workflow.compile(checkpointer=checkpointer)


if __name__ == '__main__':
    # Test enhanced workflow
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create test state
    competition_dir = Path("./test_data/titanic")
    competition_dir.mkdir(parents=True, exist_ok=True)

    initial_state = {
        "messages": [],
        "competition_name": "titanic",
        "competition_dir": str(competition_dir),
        "phase": "Understand Background",
        "memory": [],
        "retry_count": 0,
        "iteration": 0,
        "max_iterations": 5,
        "errors": []
    }

    # Create workflow
    workflow = create_enhanced_workflow(
        competition_name="titanic",
        model="gpt-5-mini"
    )

    print("Enhanced Workflow Created")
    print("="*80)
    print("Starting workflow execution...")
    print("="*80 + "\n")

    # Run workflow
    try:
        final_state = workflow.invoke(initial_state)

        print("\n" + "="*80)
        print("Workflow Complete")
        print(f"Final phase: {final_state.get('phase', 'Unknown')}")
        print(f"Memory entries: {len(final_state.get('memory', []))}")
        print("="*80)

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()
