"""LangGraph workflow for orchestrating Kaggle agents."""

from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from ..agents.data_collector import DataCollectorAgent
from ..agents.eda_agent import EDAAgent
from ..agents.strategy_agent import StrategyAgent
from ..agents.feature_engineer import FeatureEngineeringAgent
from ..agents.model_trainer import ModelTrainingAgent
from ..agents.ensemble_agent import EnsembleAgent
from ..agents.submission_agent import SubmissionAgent
from ..agents.leaderboard_monitor import LeaderboardMonitorAgent
from ..utils.state import KaggleState


def create_kaggle_workflow(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Create the Kaggle multi-agent workflow.

    Implements a directed graph where agents use Command objects for routing,
    following LangGraph best practices for control flow and state management.

    Args:
        checkpointer: Optional checkpointer for persistence (e.g., MemorySaver)

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize agents as nodes
    data_collector = DataCollectorAgent()
    eda_agent = EDAAgent()
    strategy_agent = StrategyAgent()
    feature_engineer = FeatureEngineeringAgent()
    model_trainer = ModelTrainingAgent()
    ensemble_agent = EnsembleAgent()
    submission_agent = SubmissionAgent()
    leaderboard_monitor = LeaderboardMonitorAgent()

    # Build workflow graph
    workflow = StateGraph(KaggleState)

    # Add agent nodes
    workflow.add_node("data_collection", data_collector)
    workflow.add_node("eda", eda_agent)
    workflow.add_node("strategy", strategy_agent)
    workflow.add_node("feature_engineering", feature_engineer)
    workflow.add_node("model_training", model_trainer)
    workflow.add_node("ensemble", ensemble_agent)
    workflow.add_node("submission", submission_agent)
    workflow.add_node("leaderboard", leaderboard_monitor)

    # Define workflow edges
    workflow.add_edge(START, "data_collection")
    workflow.add_edge("data_collection", "eda")
    workflow.add_edge("eda", "strategy")
    workflow.add_edge("strategy", "feature_engineering")
    workflow.add_edge("feature_engineering", "model_training")
    workflow.add_edge("model_training", "ensemble")
    workflow.add_edge("ensemble", "submission")
    workflow.add_edge("submission", "leaderboard")

    # Conditional edge from leaderboard for iteration or completion
    def should_continue(state: KaggleState) -> Literal["feature_engineering", "__end__"]:
        """Determine whether to iterate or complete based on performance."""
        # Check if we should iterate to improve
        if state.leaderboard_rank > 0:
            # Calculate percentile (assuming top 1000 teams)
            percentile = (state.leaderboard_rank / 1000) * 100

            # Iterate if not in top 20% and haven't exceeded max iterations
            if percentile > 20 and state.iteration < state.max_iterations:
                return "feature_engineering"

        return END

    workflow.add_conditional_edges(
        "leaderboard",
        should_continue,
        {
            "feature_engineering": "feature_engineering",
            END: END,
        },
    )

    # Compile with optional checkpointer for persistence
    return workflow.compile(checkpointer=checkpointer)
