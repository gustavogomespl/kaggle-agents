"""
LangGraph Workflow for Autonomous Kaggle Competition Solving.

This module defines the complete agent workflow using LangGraph's StateGraph,
implementing the full pipeline from SOTA search to submission.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agents import (
    developer_agent_node,
    ensemble_agent_node,  # Ensemble Strategy
    planner_agent_node,
    robustness_agent_node,
    search_agent_node,
)
from .agents.meta_evaluator_agent import meta_evaluator_node  # Meta-Evaluator with RL
from .agents.reporting_agent import reporting_agent_node
from .agents.submission_agent import submission_agent_node
from .core.state import KaggleState, create_initial_state
from .domain import detect_competition_domain
from .nodes.prompt_refinement import prompt_refinement_node
from .tools.kaggle_api import KaggleAPIClient


# ==================== Agent Nodes ====================

def data_download_node(state: KaggleState) -> dict[str, Any]:
    """
    Download competition data from Kaggle.

    Args:
        state: Current state

    Returns:
        State updates with data file paths
    """
    print("\n" + "="*60)
    print("= DATA DOWNLOAD")
    print("="*60)

    competition_info = state["competition_info"]
    working_dir = Path(state["working_directory"])

    print(f"\n📥 Downloading data for: {competition_info.name}")
    print(f"   Destination: {working_dir}")

    try:
        # Initialize Kaggle API client
        kaggle_client = KaggleAPIClient()

        # Download competition data
        data_files = kaggle_client.download_competition_data(
            competition=competition_info.name,
            path=str(working_dir),
            quiet=False
        )

        print("\n✓ Download complete!")
        print(f"   Train: {data_files.get('train', 'N/A')}")
        print(f"   Test: {data_files.get('test', 'N/A')}")
        target_col = "target" # Default
        if data_files.get('sample_submission'):
            print(f"   Sample Submission: {data_files['sample_submission']}")
            try:
                # Infer target column from sample submission (usually 2nd column)
                sample_sub = pd.read_csv(data_files['sample_submission'])
                if len(sample_sub.columns) >= 2:
                    target_col = sample_sub.columns[1]
                    print(f"   🎯 Target Column Detected: {target_col}")
            except Exception as e:
                print(f"   ⚠️ Could not read sample submission to infer target: {e}")

        # GENERATE FIXED FOLDS (Consistent CV)
        if data_files.get('train'):
            try:
                from .utils.cross_validation import generate_folds
                folds_path = str(working_dir / "folds.csv")
                # Use train_csv if available (for image competitions where 'train' is a dir/zip)
                train_path_for_folds = data_files.get('train_csv', data_files['train'])
                
                generate_folds(
                    train_path=train_path_for_folds,
                    target_col=target_col,
                    output_path=folds_path,
                    n_folds=5,
                    seed=42
                )
                data_files['folds'] = folds_path
            except Exception as e:
                print(f"   ⚠️ Failed to generate fixed folds: {e}")

        return {
            "data_files": data_files,
            "train_data_path": data_files.get("train", ""),
            "test_data_path": data_files.get("test", ""),
            "sample_submission_path": data_files.get("sample_submission", ""),
            "target_col": target_col,
            "last_updated": datetime.now(),
        }

    except RuntimeError as e:
        # Authentication error
        error_msg = str(e)
        print("\n❌ Kaggle API Authentication Failed")
        print(f"   {error_msg}")
        print("\n💡 To fix:")
        print("   1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("   2. Or create ~/.kaggle/kaggle.json with your credentials")
        print("   3. Get credentials from: https://www.kaggle.com/settings/account")

        return {
            "errors": [f"Kaggle authentication failed: {error_msg}"],
            "last_updated": datetime.now(),
        }

    except Exception as e:
        # Download error
        error_msg = str(e)
        print("\n❌ Data Download Failed")
        print(f"   {error_msg}")
        print("\n💡 Possible causes:")
        print(f"   - Competition '{competition_info.name}' doesn't exist")
        print("   - You haven't accepted the competition rules")
        print("   - Network connectivity issues")

        return {
            "errors": [f"Data download failed: {error_msg}"],
            "last_updated": datetime.now(),
        }


def domain_detection_node(state: KaggleState) -> dict[str, Any]:
    """
    Detect competition domain.

    Args:
        state: Current state

    Returns:
        State updates with domain detection
    """
    print("\n" + "="*60)
    print("= DOMAIN DETECTION")
    print("="*60)

    competition_info = state["competition_info"]
    working_dir = state["working_directory"]

    # Fast-path: in MLE-bench mode (and some pipelines) we already know the raw data type
    data_type = (state.get("data_files") or {}).get("data_type")
    problem_type = (competition_info.problem_type or "").lower()

    if data_type in {"image", "audio", "text"}:
        if data_type == "image":
            domain = "image_regression" if "regression" in problem_type else "image_classification"
            confidence = 0.95
        elif data_type == "audio":
            domain = "audio_regression" if "regression" in problem_type else "audio_classification"
            confidence = 0.95
        else:  # text
            if "seq" in problem_type or "seq2seq" in problem_type:
                domain = "seq_to_seq"
                confidence = 0.95
            elif "regression" in problem_type:
                domain = "text_regression"
                confidence = 0.95
            else:
                domain = "text_classification"
                confidence = 0.95
    else:
        domain, confidence = detect_competition_domain(competition_info, working_dir)

    print(f"\n Domain Detected: {domain}")
    print(f"   Confidence: {confidence:.1%}")

    return {
        "domain_detected": domain,
        "domain_confidence": confidence,
        "last_updated": datetime.now(),
    }


def iteration_control_node(state: KaggleState) -> dict[str, Any]:
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

    print(f"\nIteration: {new_iteration}/{max_iterations}")
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

    # Reset component index for refinement iterations (iteration > 1)
    # This ensures new components from refined plan are implemented
    updates = {
        "current_iteration": new_iteration,
        "should_continue": should_continue,
        "termination_reason": termination_reason,
        "last_updated": datetime.now(),
    }

    # If this is a refinement iteration (> 1), reset component index
    if new_iteration > 1:
        print("   🔄 Starting refinement iteration - resetting component index")
        updates["current_component_index"] = 0

    return updates


def performance_evaluation_node(state: KaggleState) -> dict[str, Any]:
    """
    Evaluate performance and decide if refinement is needed.

    Args:
        state: Current state

    Returns:
        State updates with refinement decision
    """
    print("\n" + "="*60)
    print("= PERFORMANCE EVALUATION")
    print("="*60)

    current_score = state.get("best_score", 0.0)
    # Dynamic target_score from state (set by MLE-bench or config), fallback to top 20% threshold
    target_score = state.get("target_score")
    if target_score is None:
        target_score = 1.0
    elif isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # Get submission results if available
    submissions = state.get("submissions", [])
    public_score = None
    if submissions:
        latest_sub = submissions[-1]
        public_score = latest_sub.public_score
        if public_score:
            print(f"\n📊 Public Score: {public_score:.4f}")
            # Use public score if available
            current_score = max(current_score, public_score)

    print(f"\nCurrent Score: {current_score:.4f}")
    print(f"Target Score:  {target_score:.4f}")
    print(f"Gap:           {target_score - current_score:.4f}")

    # Analyze component performance
    dev_results = state.get("development_results", [])
    successful_components = [r for r in dev_results if r.success]

    print(f"\n📈 Component Success Rate: {len(successful_components)}/{len(dev_results)} ({len(successful_components)/len(dev_results)*100:.0f}%)" if dev_results else "\n📈 No components tested yet")

    # Decision: should we refine?
    needs_refinement = False
    refinement_reason = None

    if current_score >= target_score:
        print(f"\n🎉 Target achieved! ({current_score:.4f} >= {target_score:.4f})")
        needs_refinement = False
    elif current_iteration >= max_iterations:
        print(f"\n⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        needs_refinement = False
    else:
        # Check if we have room for improvement
        improvement_potential = target_score - current_score

        if improvement_potential > 0.001:  # 0.1% gap
            print(f"\n🔄 Refinement needed (gap: {improvement_potential:.4f})")
            needs_refinement = True
            refinement_reason = "score_below_target"
        else:
            print("\n✅ Close enough to target")
            needs_refinement = False

    return {
        "needs_refinement": needs_refinement,
        "refinement_reason": refinement_reason,
        "current_performance_score": current_score,
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
    # Explicit early-stop flag (e.g., set by DeveloperAgent after MLE-bench grading)
    if state.get("skip_remaining_components"):
        print("\n⏩ skip_remaining_components=True - Stopping component iteration early")
        return "end"

    # Check for medal achievement in MLE-bench mode (early exit)
    mlebench_grade = state.get("mlebench_grade")
    run_mode = str(state.get("run_mode", "")).lower()

    if run_mode == "mlebench" and isinstance(mlebench_grade, dict):
        if mlebench_grade.get("valid_submission"):
            if any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
                print("\n🏅 MEDAL ACHIEVED - Stopping component iteration early")
                return "end"

            # Also allow stopping once a configured target_score is reached.
            from .core.config import is_metric_minimization

            metric_name = ""
            try:
                metric_name = state["competition_info"].evaluation_metric
            except Exception:
                metric_name = ""

            current_score = state.get("current_performance_score", 0.0)
            target_score = state.get("target_score")

            if target_score is None:
                target_score = 1.0

            if isinstance(current_score, str):
                try:
                    current_score = float(current_score)
                except ValueError:
                    current_score = 0.0

            if isinstance(target_score, str):
                try:
                    target_score = float(target_score)
                except ValueError:
                    target_score = 1.0

            if isinstance(current_score, (int, float)) and isinstance(target_score, (int, float)):
                if is_metric_minimization(metric_name):
                    if float(current_score) <= float(target_score):
                        print("\n🎯 Target reached (min metric) - Stopping component iteration early")
                        return "end"
                else:
                    if float(current_score) >= float(target_score):
                        print("\n🎯 Target reached - Stopping component iteration early")
                        return "end"

    # Check for critical errors (e.g., data download failed)
    errors = state.get("errors", [])
    if errors:
        # Check if error is about missing data files
        for error in errors:
            if "Data download failed" in error or "authentication failed" in error.lower():
                print("\n⚠️ Critical error detected, stopping workflow")
                return "end"

    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    # Check if more components to implement
    if current_component_index < len(ablation_plan):
        # Check if we're stuck on the same component (prevent infinite loop)
        dev_results = state.get("development_results", [])
        if len(dev_results) >= 3:
            # Check if last 3 results all failed on same component
            recent_failures = [r for r in dev_results[-3:] if not r.success]
            if len(recent_failures) == 3:
                # Check if all have same error about data files
                data_errors = [r for r in recent_failures if "Data files not found" in (r.stderr or "")]
                if len(data_errors) == 3:
                    print("\n⚠️ Repeated data file errors, stopping workflow")
                    return "end"

        return "iterate"

    # All components done
    return "end"


def route_after_iteration_control(state: KaggleState) -> Literal["refine", "end"]:
    """
    Route after iteration control - decide if we refine or end.

    Args:
        state: Current state

    Returns:
        "refine" to start refinement iteration, or "end" if done
    """
    needs_refinement = state.get("needs_refinement", False)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    print("\n🔀 Routing decision:")
    print(f"   Current iteration: {current_iteration}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Needs refinement: {needs_refinement}")

    if state.get("skip_remaining_components"):
        print("   ⏩ skip_remaining_components=True - Ending iteration")
        return "end"

    # Check for medal achievement in MLE-bench mode (early exit)
    mlebench_grade = state.get("mlebench_grade")
    run_mode = str(state.get("run_mode", "")).lower()

    if run_mode == "mlebench" and isinstance(mlebench_grade, dict):
        if mlebench_grade.get("valid_submission"):
            if any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
                print("   🏅 MEDAL ACHIEVED - Ending iteration")
                return "end"

    # Check termination conditions first
    if current_iteration >= max_iterations:
        print("   ➡️  Ending (max iterations reached)")
        return "end"

    # Check if goal already achieved - dynamic target_score
    current_score = state.get("current_performance_score", 0.0)
    target_score = state.get("target_score")
    if target_score is None:
        target_score = 1.0
    elif isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0

    # Respect metric direction when available
    from .core.config import is_metric_minimization

    metric_name = ""
    try:
        metric_name = state["competition_info"].evaluation_metric
    except Exception:
        metric_name = ""

    if isinstance(current_score, str):
        try:
            current_score = float(current_score)
        except ValueError:
            current_score = 0.0

    if isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1

    if isinstance(current_score, (int, float)) and isinstance(target_score, (int, float)):
        if is_metric_minimization(metric_name):
            if float(current_score) <= float(target_score):
                print(
                    f"   ➡️  Ending (goal achieved: {current_score:.4f} <= {target_score:.4f})"
                )
                return "end"
        else:
            if float(current_score) >= float(target_score):
                print(
                    f"   ➡️  Ending (goal achieved: {current_score:.4f} >= {target_score:.4f})"
                )
                return "end"

    # Decide based on refinement flag
    if needs_refinement and current_iteration > 0:
        print(f"   ➡️  Refining (iteration {current_iteration})")
        return "refine"
    print("   ➡️  Ending (no refinement needed or first iteration)")
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
    workflow.add_node("data_download", data_download_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("iteration_control", iteration_control_node)
    workflow.add_node("performance_evaluation", performance_evaluation_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)  # RL-based meta-evaluation
    workflow.add_node("prompt_refinement", prompt_refinement_node)  # RLPrompt/DSPy optimization
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("reporting", reporting_agent_node)

    # Define edges
    # Start → Data Download
    workflow.set_entry_point("data_download")

    # Data Download → Domain Detection
    workflow.add_edge("data_download", "domain_detection")

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

    # Robustness → Ensemble
    workflow.add_edge("robustness", "ensemble")

    # Ensemble → Submission
    workflow.add_edge("ensemble", "submission")

    # Submission → Performance Evaluation
    workflow.add_edge("submission", "performance_evaluation")

    # Performance Evaluation → Meta-Evaluator (RL analysis)
    workflow.add_edge("performance_evaluation", "meta_evaluator")

    # Meta-Evaluator → Prompt Refinement → Iteration Control
    workflow.add_edge("meta_evaluator", "prompt_refinement")
    workflow.add_edge("prompt_refinement", "iteration_control")

    # Iteration Control → Conditional (refine or done?)
    workflow.add_conditional_edges(
        "iteration_control",
        route_after_iteration_control,
        {
            "refine": "planner",  # Start refinement cycle
            "end": "reporting", # Goal achieved or max iterations -> Explain
        }
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



# ==================== Workflow Execution ====================

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
    print("="*70)
    print(f"KAGGLE AGENTS WORKFLOW: {competition_name}")
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
        config = {
            "configurable": {"thread_id": competition_name},
            "recursion_limit": 150,
            "metadata": {
                "competition": competition_name,
                "project": "default",
                "type": "autonomous-run"
            }
        }
        final_state = workflow.invoke(state, config)
    else:
        workflow = compile_workflow()
        config = {
            "recursion_limit": 150,
            "metadata": {
                "competition": competition_name,
                "project": "default",
                "type": "autonomous-run"
            }
        }
        final_state = workflow.invoke(state, config)

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)

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


# ==================== MLE-bench Workflow ====================

def create_mlebench_workflow() -> StateGraph:
    """
    Create a workflow for MLE-bench evaluation.

    This workflow skips data_download_node since MLE-bench data
    is already prepared and loaded into the state.

    The flow is:
        domain_detection → search → planner → developer (loop) →
        robustness → ensemble → submission → reporting

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes (skip data_download)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)
    workflow.add_node("prompt_refinement", prompt_refinement_node)
    workflow.add_node("reporting", reporting_agent_node)

    # Entry point: domain_detection (data already loaded)
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

    # Robustness → Ensemble
    workflow.add_edge("robustness", "ensemble")

    # Ensemble → Submission
    workflow.add_edge("ensemble", "submission")

    # Submission → Meta-Evaluator → Prompt Refinement → Reporting
    workflow.add_edge("submission", "meta_evaluator")
    workflow.add_edge("meta_evaluator", "prompt_refinement")
    workflow.add_edge("prompt_refinement", "reporting")

    # Reporting → END
    workflow.add_edge("reporting", END)

    return workflow.compile()


# ==================== Simplified Workflow (for testing) ====================

def create_simple_workflow() -> StateGraph:
    """
    Create a simplified workflow for testing (no iterations).

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("data_download", data_download_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)

    # Linear flow
    workflow.set_entry_point("data_download")
    workflow.add_edge("data_download", "domain_detection")
    workflow.add_edge("domain_detection", "search")
    workflow.add_edge("search", "planner")
    workflow.add_edge("planner", "developer")
    workflow.add_edge("developer", END)

    return workflow.compile()


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
    print("="*70)
    print(f"SIMPLE WORKFLOW: {competition_name}")
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
