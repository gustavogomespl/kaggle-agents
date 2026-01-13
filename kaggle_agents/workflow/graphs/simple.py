"""Simplified workflow graph for testing the Kaggle Agents pipeline."""

from langgraph.graph import END, StateGraph

from ...agents import (
    developer_agent_node,
    planner_agent_node,
    search_agent_node,
)
from ...core.state import KaggleState
from ..nodes import (
    canonical_data_preparation_node,
    data_audit_node,
    data_download_node,
    data_format_discovery_node,
    data_validation_node,
    domain_detection_node,
)


def create_simple_workflow() -> StateGraph:
    """
    Create a simplified workflow for testing (no iterations).

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("data_download", data_download_node)
    workflow.add_node("data_format_discovery", data_format_discovery_node)
    workflow.add_node("data_validation", data_validation_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("data_audit", data_audit_node)
    workflow.add_node("canonical_data_preparation", canonical_data_preparation_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)

    # Linear flow
    workflow.set_entry_point("data_download")
    workflow.add_edge("data_download", "data_format_discovery")
    workflow.add_edge("data_format_discovery", "data_validation")
    workflow.add_edge("data_validation", "domain_detection")
    workflow.add_edge("domain_detection", "data_audit")
    workflow.add_edge("data_audit", "canonical_data_preparation")
    workflow.add_edge("canonical_data_preparation", "search")
    workflow.add_edge("search", "planner")
    workflow.add_edge("planner", "developer")
    workflow.add_edge("developer", END)

    return workflow.compile()
