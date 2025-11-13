"""
LangGraph Workflow for Autonomous Kaggle Competition Solving.

This module defines the complete agent workflow using LangGraph's StateGraph,
implementing the full pipeline from SOTA search to submission.
"""

from typing import Dict, Any, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .core.state import KaggleState, create_initial_state
from .core.config import get_config
from .domain import detect_competition_domain
from .agents import (
    search_agent_node,
    planner_agent_node,
    developer_agent_node,
    robustness_agent_node,
    submission_agent_node,
)


# ==================== Agent Nodes ====================

def domain_detection_node(state: KaggleState) -> Dict[str, Any]:
    """
    Detect competition domain.

    Args:
        state: Current state

    Returns:
        State updates with domain detection
    """
    print("\n" + "="*60)
    print("= DOMAIN DETECTION")
    print("="*60)

    competition_info = state["competition_info"]
    working_dir = state["working_directory"]

    domain, confidence = detect_competition_domain(competition_info, working_dir)

    print(f"\n Domain Detected: {domain}")
    print(f"   Confidence: {confidence:.1%}")

    return {
        "domain_detected": domain,
        "domain_confidence": confidence,
        "last_updated": datetime.now(),
    }


def iteration_control_node(state: KaggleState) -> Dict[str, Any]:
    """
    Control iteration and check termination conditions.

    Args:
        state: Current state

    Returns:
        State updates with iteration control
    """
    print("\n" + "="*60)
    print("= ITERATION CONTROL")
    print("="*60)

    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    best_score = state.get("best_score", 0.0)
    target_percentile = state.get("target_percentile", 20.0)

    # Increment iteration
    new_iteration = current_iteration + 1

    print(f"\n=� Iteration: {new_iteration}/{max_iterations}")
    print(f"   Best Score: {best_score:.4f}")
    print(f"   Target: Top {target_percentile}%")

    # Check if we should continue
    should_continue = new_iteration < max_iterations

    # Check if goal achieved
    # Note: In real scenario, would check actual percentile
    # For now, continue until max iterations

    termination_reason = None
    if not should_continue:
        termination_reason = "max_iterations_reached"

    return {
        "current_iteration": new_iteration,
        "should_continue": should_continue,
        "termination_reason": termination_reason,
        "last_updated": datetime.now(),
    }


# ==================== Conditional Functions ====================

def should_continue_workflow(state: KaggleState) -> Literal["continue", "end"]:
    """
    Decide whether to continue or end the workflow.

    Args:
        state: Current state

    Returns:
        "continue" or "end"
    """
    should_continue = state.get("should_continue", True)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # End conditions
    if not should_continue:
        return "end"

    if current_iteration >= max_iterations:
        return "end"

    # Check if we have components to implement
    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    if current_component_index >= len(ablation_plan):
        # All components implemented, could iterate or end
        return "end"

    return "continue"


def should_retry_component(state: KaggleState) -> Literal["retry", "next"]:
    """
    Decide whether to retry current component or move to next.

    Args:
        state: Current state

    Returns:
        "retry" or "next"
    """
    development_results = state.get("development_results", [])

    if not development_results:
        return "next"

    # Check last result
    last_result = development_results[-1]

    if last_result.success:
        return "next"

    # Check retry count
    code_retry_count = state.get("code_retry_count", 0)
    max_retries = 3  # Max retries at workflow level

    if code_retry_count < max_retries:
        return "retry"

    # Max retries reached, move to next component
    return "next"


def route_after_developer(state: KaggleState) -> Literal["iterate", "end"]:
    """
    Route after developer agent completes.

    Args:
        state: Current state

    Returns:
        "iterate" to continue implementing components, or "end" if done
    """
    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    # Check if more components to implement
    if current_component_index < len(ablation_plan):
        return "iterate"

    # All components done
    return "end"


# ==================== Workflow Construction ====================

def create_workflow() -> StateGraph:
    """
    Create the complete LangGraph workflow.

    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("iteration_control", iteration_control_node)

    # Define edges
    # Start → Domain Detection
    workflow.set_entry_point("domain_detection")

    # Domain Detection → Search
    workflow.add_edge("domain_detection", "search")

    # Search → Planner
    workflow.add_edge("search", "planner")

    # Planner → Developer
    workflow.add_edge("planner", "developer")

    # Developer → Conditional (more components or done?)
    workflow.add_conditional_edges(
        "developer",
        route_after_developer,
        {
            "iterate": "developer",      # More components to implement
            "end": "robustness",          # All components done → validate
        }
    )

    # Robustness → Submission
    workflow.add_edge("robustness", "submission")

    # Submission → Iteration Control
    workflow.add_edge("submission", "iteration_control")

    # Iteration Control → Conditional (continue or end?)
    workflow.add_conditional_edges(
        "iteration_control",
        should_continue_workflow,
        {
            "continue": "search",  # New iteration
            "end": END,            # Workflow complete
        }
    )

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

    if checkpointer:
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        compiled = workflow.compile()

    return compiled


# ==================== Workflow Execution ====================

def run_workflow(
    competition_name: str,
    working_dir: str,
    competition_info: Dict[str, Any],
    max_iterations: int = 3,
    use_checkpointing: bool = False,
) -> KaggleState:
    """
    Run the complete workflow for a competition.

    Args:
        competition_name: Name of the Kaggle competition
        working_dir: Working directory for artifacts
        competition_info: Competition metadata
        max_iterations: Maximum workflow iterations
        use_checkpointing: Whether to use checkpointing

    Returns:
        Final state
    """
    print("="*70)
    print(f"=� KAGGLE AGENTS WORKFLOW: {competition_name}")
    print("="*70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    # Set competition info
    from .core.state import CompetitionInfo
    state["competition_info"] = CompetitionInfo(**competition_info)

    # Set iteration config
    state["max_iterations"] = max_iterations

    # Create workflow
    if use_checkpointing:
        checkpointer = MemorySaver()
        workflow = compile_workflow(checkpointer=checkpointer)

        # Run with config for checkpointing
        config = {"configurable": {"thread_id": competition_name}}
        final_state = workflow.invoke(state, config)
    else:
        workflow = compile_workflow()
        final_state = workflow.invoke(state)

    print("\n" + "="*70)
    print("<� WORKFLOW COMPLETE")
    print("="*70)

    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Iterations: {final_state.get('current_iteration', 0)}")
    print(f"   SOTA Solutions: {len(final_state.get('sota_solutions', []))}")
    print(f"   Components Planned: {len(final_state.get('ablation_plan', []))}")
    print(f"   Components Implemented: {len(final_state.get('development_results', []))}")

    # Success count
    dev_results = final_state.get("development_results", [])
    successful = sum(1 for r in dev_results if r.success)
    if dev_results:
        print(f"   Success Rate: {successful}/{len(dev_results)} ({successful/len(dev_results)*100:.0f}%)")

    # Validation summary
    validation_score = final_state.get("overall_validation_score")
    if validation_score is not None:
        print(f"   Validation Score: {validation_score:.1%}")

    # Submission summary
    submissions = final_state.get("submissions", [])
    if submissions:
        latest_sub = submissions[-1]
        if latest_sub.public_score is not None:
            print(f"   Public Score: {latest_sub.public_score:.4f}")
            if latest_sub.percentile is not None:
                print(f"   Percentile: {latest_sub.percentile:.1f}%")

    print(f"\n   Termination: {final_state.get('termination_reason', 'unknown')}")

    return final_state


# ==================== Simplified Workflow (for testing) ====================

def create_simple_workflow() -> StateGraph:
    """
    Create a simplified workflow for testing (no iterations).

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)

    # Linear flow
    workflow.set_entry_point("domain_detection")
    workflow.add_edge("domain_detection", "search")
    workflow.add_edge("search", "planner")
    workflow.add_edge("planner", "developer")
    workflow.add_edge("developer", END)

    return workflow.compile()


def run_simple_workflow(
    competition_name: str,
    working_dir: str,
    competition_info: Dict[str, Any],
) -> KaggleState:
    """
    Run simplified workflow (one pass, no iterations).

    Args:
        competition_name: Competition name
        working_dir: Working directory
        competition_info: Competition metadata

    Returns:
        Final state
    """
    print("="*70)
    print(f"=� SIMPLE WORKFLOW: {competition_name}")
    print("="*70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    from .core.state import CompetitionInfo
    state["competition_info"] = CompetitionInfo(**competition_info)

    # Run workflow
    workflow = create_simple_workflow()
    final_state = workflow.invoke(state)

    print("\n" + "="*70)
    print(" WORKFLOW COMPLETE")
    print("="*70)

    return final_state
