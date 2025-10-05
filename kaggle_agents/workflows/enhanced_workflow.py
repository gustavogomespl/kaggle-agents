"""Enhanced LangGraph workflow integrating multi-agent system with feedback loops."""

from typing import Literal, Optional, TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

from ..core.state import EnhancedKaggleState
from ..core.sop import SOP
from ..core.config_manager import get_config

import logging

logger = logging.getLogger(__name__)


def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with right values taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


class EnhancedWorkflowState(MessagesState):
    """State schema for Enhanced Workflow compatible with LangGraph."""

    # Competition Metadata
    competition_name: str
    competition_type: str
    metric: str
    competition_dir: str

    # Data Paths
    train_data_path: str
    test_data_path: str
    sample_submission_path: str

    # EDA Results
    eda_summary: Annotated[Dict[str, Any], merge_dict]
    data_insights: Annotated[List[str], add]

    # Feature Engineering
    features_engineered: Annotated[List[str], add]
    feature_importance: Annotated[Dict[str, float], merge_dict]

    # Model Training
    models_trained: Annotated[List[Dict[str, Any]], add]
    best_model: Annotated[Dict[str, Any], merge_dict]
    cv_scores: Annotated[List[float], add]

    # Submission
    submission_path: str
    submission_score: float
    leaderboard_rank: int

    # Workflow Control
    iteration: int
    max_iterations: int

    # Error Tracking
    errors: Annotated[List[str], add]

    # Enhanced Fields
    phase: str
    memory: Annotated[List[Dict[str, Any]], add]
    background_info: str
    rules: Dict[str, Any]
    retry_count: int
    max_phase_retries: int

    # Status for routing
    status: str


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

    # Helper function to convert dict state to EnhancedKaggleState
    def dict_to_state(state_dict: dict, phase: str) -> EnhancedKaggleState:
        """Convert workflow dict state to EnhancedKaggleState object."""
        return EnhancedKaggleState(
            messages=state_dict.get("messages", []),
            competition_name=state_dict["competition_name"],
            competition_dir=state_dict["competition_dir"],
            competition_type=state_dict.get("competition_type", ""),
            metric=state_dict.get("metric", ""),
            train_data_path=state_dict.get("train_data_path", ""),
            test_data_path=state_dict.get("test_data_path", ""),
            sample_submission_path=state_dict.get("sample_submission_path", ""),
            eda_summary=state_dict.get("eda_summary", {}),
            data_insights=state_dict.get("data_insights", []),
            features_engineered=state_dict.get("features_engineered", []),
            feature_importance=state_dict.get("feature_importance", {}),
            models_trained=state_dict.get("models_trained", []),
            best_model=state_dict.get("best_model", {}),
            cv_scores=state_dict.get("cv_scores", []),
            submission_path=state_dict.get("submission_path", ""),
            submission_score=state_dict.get("submission_score", 0.0),
            leaderboard_rank=state_dict.get("leaderboard_rank", 0),
            iteration=state_dict.get("iteration", 0),
            max_iterations=state_dict.get("max_iterations", 5),
            errors=state_dict.get("errors", []),
            phase=phase,
            memory=state_dict.get("memory", []),
            background_info=state_dict.get("background_info", ""),
            rules=state_dict.get("rules", {}),
            retry_count=state_dict.get("retry_count", 0),
            max_phase_retries=state_dict.get("max_phase_retries", 3)
        )

    # Define phase nodes
    def execute_understand_background(state: EnhancedWorkflowState) -> dict:
        """Execute Understand Background phase."""
        logger.info("📖 Executing: Understand Background")

        # Convert dict to EnhancedKaggleState for SOP
        state_obj = dict_to_state(state, "Understand Background")

        # Execute SOP step
        status, updated_state = sop.step(state_obj)

        # Return dict update
        return {
            "phase": updated_state.phase,
            "status": status,
            "memory": updated_state.memory,
            "background_info": updated_state.background_info,
            "competition_type": updated_state.competition_type,
            "metric": updated_state.metric
        }

    def execute_preliminary_eda(state: EnhancedWorkflowState) -> dict:
        """Execute Preliminary EDA phase."""
        logger.info("🔍 Executing: Preliminary Exploratory Data Analysis")

        state_obj = dict_to_state(state, "Preliminary Exploratory Data Analysis")
        status, updated_state = sop.step(state_obj)

        return {
            "phase": updated_state.phase,
            "status": status,
            "memory": updated_state.memory,
            "eda_summary": updated_state.eda_summary
        }

    def execute_data_cleaning(state: EnhancedWorkflowState) -> dict:
        """Execute Data Cleaning phase."""
        logger.info("🧹 Executing: Data Cleaning")

        state_obj = dict_to_state(state, "Data Cleaning")
        status, updated_state = sop.step(state_obj)

        return {
            "phase": updated_state.phase,
            "status": status,
            "memory": updated_state.memory
        }

    def execute_deep_eda(state: EnhancedWorkflowState) -> dict:
        """Execute Deep EDA phase."""
        logger.info("🔬 Executing: In-depth Exploratory Data Analysis")

        state_obj = dict_to_state(state, "In-depth Exploratory Data Analysis")
        status, updated_state = sop.step(state_obj)

        return {
            "phase": updated_state.phase,
            "status": status,
            "memory": updated_state.memory,
            "data_insights": updated_state.data_insights
        }

    def execute_feature_engineering(state: EnhancedWorkflowState) -> dict:
        """Execute Feature Engineering phase."""
        logger.info("⚙️ Executing: Feature Engineering")

        state_obj = dict_to_state(state, "Feature Engineering")
        status, updated_state = sop.step(state_obj)

        return {
            "phase": updated_state.phase,
            "status": status,
            "memory": updated_state.memory,
            "features_engineered": updated_state.features_engineered
        }

    def execute_model_building(state: EnhancedWorkflowState) -> dict:
        """Execute Model Building phase."""
        logger.info("🎯 Executing: Model Building, Validation, and Prediction")

        state_obj = dict_to_state(state, "Model Building, Validation, and Prediction")
        status, updated_state = sop.step(state_obj)

        return {
            "phase": updated_state.phase,
            "status": status,
            "memory": updated_state.memory,
            "models_trained": updated_state.models_trained,
            "best_model": updated_state.best_model,
            "submission_path": updated_state.submission_path
        }

    # Routing function
    def route_after_phase(state: EnhancedWorkflowState) -> str:
        """Route to next phase or handle retry/failure.

        Args:
            state: Current state (dict)

        Returns:
            Next node name
        """
        status = state.get('status', 'Continue')
        current_phase = state.get('phase', '')

        if status == "Complete":
            logger.info("✅ Workflow complete!")
            return END

        elif status == "Fail":
            logger.error("❌ Workflow failed!")
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
            logger.info(f"🔄 Retrying phase: {current_phase}")
            return phase_to_node.get(current_phase, END)

        elif status == "Continue":
            # Move to next phase
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
                logger.warning(f"⚠️ Unknown phase: {current_phase}")
                return END

        else:
            logger.error(f"❌ Unknown status: {status}")
            return END

    # Build workflow graph
    workflow = StateGraph(EnhancedWorkflowState)

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
