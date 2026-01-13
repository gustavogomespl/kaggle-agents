"""Main workflow graph creation for the Kaggle Agents pipeline."""

from langgraph.graph import END, StateGraph

from ...agents import (
    developer_agent_node,
    ensemble_agent_node,
    planner_agent_node,
    robustness_agent_node,
    search_agent_node,
)
from ...agents.meta_evaluator_agent import meta_evaluator_node
from ...agents.reporting_agent import reporting_agent_node
from ...agents.submission_agent import submission_agent_node
from ...core.state import KaggleState
from ...nodes.curriculum_learning import (
    curriculum_learning_node,
    inject_subtask_guidance,
)
from ...nodes.prompt_refinement import prompt_refinement_node
from ..nodes import (
    auto_sota_search_node,
    canonical_data_preparation_node,
    data_audit_node,
    data_download_node,
    data_exploration_node,
    data_format_discovery_node,
    data_validation_node,
    domain_detection_node,
    iteration_control_node,
    performance_evaluation_node,
)
from ..routing import (
    route_after_developer,
    route_after_iteration_control,
    route_after_meta_evaluator,
    route_after_submission,
)


def create_workflow() -> StateGraph:
    """
    Create the complete LangGraph workflow.

    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("data_download", data_download_node)
    workflow.add_node("data_format_discovery", data_format_discovery_node)
    workflow.add_node("data_validation", data_validation_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("data_audit", data_audit_node)
    workflow.add_node("canonical_data_preparation", canonical_data_preparation_node)
    workflow.add_node("data_exploration", data_exploration_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("iteration_control", iteration_control_node)
    workflow.add_node("performance_evaluation", performance_evaluation_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)
    workflow.add_node("auto_sota_search", auto_sota_search_node)
    workflow.add_node("curriculum_learning", curriculum_learning_node)
    workflow.add_node("inject_curriculum", inject_subtask_guidance)
    workflow.add_node("prompt_refinement", prompt_refinement_node)
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("reporting", reporting_agent_node)

    # Define edges
    # Start → Data Download
    workflow.set_entry_point("data_download")

    # Data Download → Data Format Discovery → Data Validation → Domain Detection → Data Audit
    workflow.add_edge("data_download", "data_format_discovery")
    workflow.add_edge("data_format_discovery", "data_validation")
    workflow.add_edge("data_validation", "domain_detection")
    workflow.add_edge("domain_detection", "data_audit")

    # Data Audit → Canonical Data Preparation → Data Exploration → Search
    workflow.add_edge("data_audit", "canonical_data_preparation")
    workflow.add_edge("canonical_data_preparation", "data_exploration")
    workflow.add_edge("data_exploration", "search")

    # Search → Planner
    workflow.add_edge("search", "planner")

    # Planner → Developer
    workflow.add_edge("planner", "developer")

    # Developer → Conditional (more components or done?)
    workflow.add_conditional_edges(
        "developer",
        route_after_developer,
        {
            "iterate": "developer",
            "end": "robustness",
        },
    )

    # Robustness → Ensemble
    workflow.add_edge("robustness", "ensemble")

    # Ensemble → Submission
    workflow.add_edge("ensemble", "submission")

    # Submission → Conditional (valid or retry?)
    workflow.add_conditional_edges(
        "submission",
        route_after_submission,
        {
            "retry_developer": "developer",
            "continue": "performance_evaluation",
        },
    )

    # Performance Evaluation → Meta-Evaluator (RL analysis)
    workflow.add_edge("performance_evaluation", "meta_evaluator")

    # Meta-Evaluator → Conditional (SOTA search, curriculum, or continue?)
    workflow.add_conditional_edges(
        "meta_evaluator",
        route_after_meta_evaluator,
        {
            "sota_search": "auto_sota_search",
            "curriculum": "curriculum_learning",
            "continue": "prompt_refinement",
        },
    )

    # SOTA Search → Curriculum Learning
    workflow.add_edge("auto_sota_search", "curriculum_learning")

    # Curriculum Learning → Inject Guidance → Prompt Refinement
    workflow.add_edge("curriculum_learning", "inject_curriculum")
    workflow.add_edge("inject_curriculum", "prompt_refinement")

    # Prompt Refinement → Iteration Control
    workflow.add_edge("prompt_refinement", "iteration_control")

    # Iteration Control → Conditional (refine or done?)
    workflow.add_conditional_edges(
        "iteration_control",
        route_after_iteration_control,
        {
            "refine": "planner",
            "end": "reporting",
        },
    )

    # Reporting → END
    workflow.add_edge("reporting", END)

    return workflow


def compile_workflow(checkpointer=None):
    """
    Compile the workflow with optional checkpointing.

    Args:
        checkpointer: Optional checkpointer (e.g., MemorySaver())

    Returns:
        Compiled workflow
    """
    workflow = create_workflow()

    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
