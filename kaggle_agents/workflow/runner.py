"""Workflow execution functions for the Kaggle Agents pipeline."""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from ..core.config import get_config
from ..core.state import CompetitionInfo, KaggleState, create_initial_state
from .graphs import compile_workflow, create_simple_workflow


def run_workflow(
    competition_name: str,
    working_dir: str,
    competition_info: dict[str, Any],
    max_iterations: int = 5,
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
    print("=" * 70)
    print(f"KAGGLE AGENTS WORKFLOW: {competition_name}")
    print("=" * 70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    # Set competition info
    state["competition_info"] = CompetitionInfo(**competition_info)

    # Set iteration config
    state["max_iterations"] = max_iterations

    # Create workflow
    # Get centralized recursion_limit from config (default 300)
    agent_config = get_config()
    recursion_limit = agent_config.iteration.langgraph_recursion_limit

    if use_checkpointing:
        checkpointer = MemorySaver()
        workflow = compile_workflow(checkpointer=checkpointer)

        # Run with config for checkpointing
        config = {
            "configurable": {"thread_id": competition_name},
            "recursion_limit": recursion_limit,
            "metadata": {
                "competition": competition_name,
                "project": "default",
                "type": "autonomous-run",
            },
        }
        final_state = workflow.invoke(state, config)
    else:
        workflow = compile_workflow()
        config = {
            "recursion_limit": recursion_limit,
            "metadata": {
                "competition": competition_name,
                "project": "default",
                "type": "autonomous-run",
            },
        }
        final_state = workflow.invoke(state, config)

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)

    # Print summary
    print("\n📊 Summary:")
    print(f"   Iterations: {final_state.get('current_iteration', 0)}")
    print(f"   SOTA Solutions: {len(final_state.get('sota_solutions', []))}")
    print(f"   Components Planned: {len(final_state.get('ablation_plan', []))}")
    print(f"   Components Implemented: {len(final_state.get('development_results', []))}")

    # Success count
    dev_results = final_state.get("development_results", [])
    successful = sum(1 for r in dev_results if r.success)
    if dev_results:
        print(
            f"   Success Rate: {successful}/{len(dev_results)} ({successful / len(dev_results) * 100:.0f}%)"
        )

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


def run_simple_workflow(
    competition_name: str,
    working_dir: str,
    competition_info: dict[str, Any],
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
    print("=" * 70)
    print(f"SIMPLE WORKFLOW: {competition_name}")
    print("=" * 70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    state["competition_info"] = CompetitionInfo(**competition_info)

    # Run workflow
    workflow = create_simple_workflow()
    final_state = workflow.invoke(state)

    print("\n" + "=" * 70)
    print(" WORKFLOW COMPLETE")
    print("=" * 70)

    return final_state
