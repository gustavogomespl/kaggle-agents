"""MLE-bench workflow graph creation for the Kaggle Agents pipeline."""

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
)


def create_mlebench_workflow() -> StateGraph:
    """
    Create a workflow for MLE-bench evaluation.

    This workflow skips data_download_node since MLE-bench data
    is already prepared and loaded into the state.

    The flow is:
        domain_detection → search → planner → developer (loop) →
        robustness → ensemble → submission → performance_evaluation →
        meta_evaluator → [curriculum_learning] → prompt_refinement →
        iteration_control → [refine → planner | end → reporting]

    Features:
        - WEBRL: Curriculum learning from failures (auto sub-tasks)
        - Iteration loop for refinement

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes (skip data_download)
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
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("performance_evaluation", performance_evaluation_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)
    workflow.add_node("auto_sota_search", auto_sota_search_node)
    workflow.add_node("curriculum_learning", curriculum_learning_node)
    workflow.add_node("inject_curriculum", inject_subtask_guidance)
    workflow.add_node("prompt_refinement", prompt_refinement_node)
    workflow.add_node("iteration_control", iteration_control_node)
    workflow.add_node("reporting", reporting_agent_node)

    # Entry point: data_format_discovery (data already loaded but may need format discovery)
    workflow.set_entry_point("data_format_discovery")

    # Data Format Discovery → Data Validation → Domain Detection → Data Audit → Canonical → EDA → Search
    workflow.add_edge("data_format_discovery", "data_validation")
    workflow.add_edge("data_validation", "domain_detection")
    workflow.add_edge("domain_detection", "data_audit")
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

    # Submission → Performance Evaluation → Meta-Evaluator
    workflow.add_edge("submission", "performance_evaluation")
    workflow.add_edge("performance_evaluation", "meta_evaluator")

    # Meta-Evaluator → Conditional (WEBRL: curriculum, SOTA search, or continue?)
    workflow.add_conditional_edges(
        "meta_evaluator",
        route_after_meta_evaluator,
        {
            "sota_search": "auto_sota_search",
            "curriculum": "curriculum_learning",
            "continue": "prompt_refinement",
        },
    )

    # Auto SOTA Search → Curriculum Learning (with SOTA guidance)
    workflow.add_edge("auto_sota_search", "curriculum_learning")

    # Curriculum Learning → Inject Guidance → Prompt Refinement
    workflow.add_edge("curriculum_learning", "inject_curriculum")
    workflow.add_edge("inject_curriculum", "prompt_refinement")

    # Prompt Refinement → Iteration Control
    workflow.add_edge("prompt_refinement", "iteration_control")

    # Iteration Control → Conditional (refine or end?)
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

    return workflow.compile()
