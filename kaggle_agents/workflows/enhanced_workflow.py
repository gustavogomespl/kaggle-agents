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
    def execute_understand_background(state: EnhancedKaggleState) -> EnhancedKaggleState:
        """Execute Understand Background phase."""
        logger.info("📖 Executing: Understand Background")
        state.phase = "Understand Background"
        status, updated_state = sop.step(state)
        updated_state.status = status
        return updated_state

    def execute_preliminary_eda(state: EnhancedKaggleState) -> EnhancedKaggleState:
        """Execute Preliminary EDA phase."""
        logger.info("Executing: Preliminary Exploratory Data Analysis")
        state.phase = "Preliminary Exploratory Data Analysis"
        status, updated_state = sop.step(state)
        updated_state.status = status
        return updated_state

    def execute_data_cleaning(state: EnhancedKaggleState) -> EnhancedKaggleState:
        """Execute Data Cleaning phase."""
        logger.info("🧹 Executing: Data Cleaning")
        state.phase = "Data Cleaning"
        status, updated_state = sop.step(state)
        updated_state.status = status
        return updated_state

    def execute_deep_eda(state: EnhancedKaggleState) -> EnhancedKaggleState:
        """Execute Deep EDA phase."""
        logger.info("Executing: In-depth Exploratory Data Analysis")
        state.phase = "In-depth Exploratory Data Analysis"
        status, updated_state = sop.step(state)
        updated_state.status = status
        return updated_state

    def execute_feature_engineering(state: EnhancedKaggleState) -> EnhancedKaggleState:
        """Execute Feature Engineering phase."""
        logger.info("Executing: Feature Engineering")
        state.phase = "Feature Engineering"
        status, updated_state = sop.step(state)
        updated_state.status = status
        return updated_state

    def execute_model_building(state: EnhancedKaggleState) -> EnhancedKaggleState:
        """Execute Model Building phase."""
        logger.info("Executing: Model Building, Validation, and Prediction")
        state.phase = "Model Building, Validation, and Prediction"
        status, updated_state = sop.step(state)
        updated_state.status = status
        return updated_state

    # Routing function
    def route_after_phase(state: EnhancedKaggleState) -> str:
        """Route to next phase or handle retry/failure.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        status = getattr(state, 'status', 'Continue')

        if status == "Complete":
            logger.info("Workflow complete!")
            return END

        elif status == "Fail":
            logger.error("Workflow failed!")
            return END

        elif status == "Retry":
            # Retry current phase
            current_phase = state.phase
            phase_to_node = {
                "Understand Background": "understand_background",
                "Preliminary Exploratory Data Analysis": "preliminary_eda",
                "Data Cleaning": "data_cleaning",
                "In-depth Exploratory Data Analysis": "deep_eda",
                "Feature Engineering": "feature_engineering",
                "Model Building, Validation, and Prediction": "model_building"
            }
            return phase_to_node.get(current_phase, END)

        elif status == "Continue":
            # Move to next phase
            current_phase = state.phase

            if current_phase == "Understand Background":
                return "preliminary_eda"
            elif current_phase == "Preliminary Exploratory Data Analysis":
                return "data_cleaning"
            elif current_phase == "Data Cleaning":
                return "deep_eda"
            elif current_phase == "In-depth Exploratory Data Analysis":
                return "feature_engineering"
            elif current_phase == "Feature Engineering":
                return "model_building"
            elif current_phase == "Model Building, Validation, and Prediction":
                return END
            elif current_phase == "Complete":
                return END
            else:
                logger.warning(f"Unknown phase: {current_phase}")
                return END

        else:
            logger.error(f"Unknown status: {status}")
            return END

    # Build workflow graph
    workflow = StateGraph(EnhancedKaggleState)

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
