"""
LangGraph Workflow for Autonomous Kaggle Competition Solving.

This module defines the complete agent workflow using LangGraph's StateGraph,
implementing the full pipeline from SOTA search to submission.
"""

# Graph creation functions
from .graphs import (
    compile_workflow,
    create_mlebench_workflow,
    create_simple_workflow,
    create_workflow,
)

# Node functions (for backward compatibility and direct access)
from .nodes import (
    auto_sota_search_node,
    canonical_data_preparation_node,
    data_audit_node,
    data_download_node,
    data_format_discovery_node,
    data_validation_node,
    domain_detection_node,
    iteration_control_node,
    performance_evaluation_node,
)

# Routing functions
from .routing import (
    route_after_developer,
    route_after_iteration_control,
    route_after_meta_evaluator,
    route_after_submission,
    should_continue_workflow,
    should_retry_component,
)

# Workflow runner functions
from .runner import (
    run_simple_workflow,
    run_workflow,
)


__all__ = [
    # Graph creation
    "create_workflow",
    "compile_workflow",
    "create_mlebench_workflow",
    "create_simple_workflow",
    # Workflow execution
    "run_workflow",
    "run_simple_workflow",
    # Routing functions
    "should_continue_workflow",
    "should_retry_component",
    "route_after_developer",
    "route_after_submission",
    "route_after_iteration_control",
    "route_after_meta_evaluator",
    # Node functions
    "data_download_node",
    "data_format_discovery_node",
    "data_validation_node",
    "domain_detection_node",
    "data_audit_node",
    "canonical_data_preparation_node",
    "iteration_control_node",
    "performance_evaluation_node",
    "auto_sota_search_node",
]
