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
from .nodes.curriculum_learning import (
    curriculum_learning_node,
    inject_subtask_guidance,
)
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
    print("\n" + "=" * 60)
    print("= DATA DOWNLOAD")
    print("=" * 60)

    competition_info = state["competition_info"]
    working_dir = Path(state["working_directory"])

    print(f"\n📥 Downloading data for: {competition_info.name}")
    print(f"   Destination: {working_dir}")

    try:
        # Initialize Kaggle API client
        kaggle_client = KaggleAPIClient()

        # Download competition data
        data_files = kaggle_client.download_competition_data(
            competition=competition_info.name, path=str(working_dir), quiet=False
        )

        print("\n✓ Download complete!")
        print(f"   Train: {data_files.get('train', 'N/A')}")
        print(f"   Test: {data_files.get('test', 'N/A')}")
        target_col = "target"  # Default
        if data_files.get("sample_submission"):
            print(f"   Sample Submission: {data_files['sample_submission']}")
            try:
                # Infer target column from sample submission (usually 2nd column)
                sample_sub = pd.read_csv(data_files["sample_submission"])
                if len(sample_sub.columns) >= 2:
                    target_col = sample_sub.columns[1]
                    print(f"   🎯 Target Column Detected: {target_col}")
            except Exception as e:
                print(f"   ⚠️ Could not read sample submission to infer target: {e}")

        # GENERATE FIXED FOLDS (Consistent CV)
        if data_files.get("train"):
            try:
                from .utils.cross_validation import generate_folds

                folds_path = str(working_dir / "folds.csv")
                # Use train_csv if available (for image competitions where 'train' is a dir/zip)
                train_path_for_folds = data_files.get("train_csv", data_files["train"])

                generate_folds(
                    train_path=train_path_for_folds,
                    target_col=target_col,
                    output_path=folds_path,
                    n_folds=5,
                    seed=42,
                )
                data_files["folds"] = folds_path
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
    Detect competition domain using LLM-First approach.

    Args:
        state: Current state

    Returns:
        State updates with domain detection
    """
    print("\n" + "=" * 60)
    print("= DOMAIN DETECTION")
    print("=" * 60)

    competition_info = state["competition_info"]
    working_dir = state["working_directory"]

    submission_format_type = None
    submission_format_metadata: dict[str, Any] = {}

    # Get LLM for domain detection (use planner's LLM with low temperature)
    from .core.config import get_llm_for_role

    try:
        llm = get_llm_for_role(role="planner", temperature=0.0)
    except Exception as e:
        print(f"   Warning: Could not get LLM for domain detection: {e}")
        llm = None

    # Use LLM-First domain detection (delegated to detector.py)
    domain, confidence = detect_competition_domain(competition_info, working_dir, llm=llm)

    # Check for image-to-image overrides (pixel-level submission format)
    data_files = state.get("data_files") or {}
    data_type = data_files.get("data_type")

    if data_type == "image":
        # Check for clean/target directory (e.g., train_cleaned for denoising)
        clean_train_path = data_files.get("clean_train", "")
        if clean_train_path and Path(clean_train_path).exists():
            domain = "image_to_image"
            confidence = 0.95
            print("   Override: Detected clean/target directory -> image_to_image")

        # Check for pixel-level submission format (many rows per image)
        sample_sub_path = data_files.get("sample_submission", "") or state.get(
            "sample_submission_path", ""
        )
        test_path = data_files.get("test", "") or state.get("test_data_path", "")
        if sample_sub_path and Path(sample_sub_path).exists():
            from .domain import detect_submission_format

            sub_format, sub_meta = detect_submission_format(
                sample_sub_path, test_path if test_path else None, competition_info
            )
            submission_format_type = sub_format
            submission_format_metadata = sub_meta
            if sub_format == "pixel_level":
                domain = "image_to_image"
                confidence = 0.95
                print("   Override: Detected pixel-level submission format -> image_to_image")
                print(f"      Expected rows: {sub_meta.get('expected_rows', 'unknown')}")
                print(f"      ID pattern: {sub_meta.get('id_pattern', 'unknown')}")

    print(f"\n Domain Detected: {domain}")
    print(f"   Confidence: {confidence:.1%}")

    return {
        "domain_detected": domain,
        "domain_confidence": confidence,
        "submission_format_type": submission_format_type,
        "submission_format_metadata": submission_format_metadata,
        "submission_format": {"type": submission_format_type, **submission_format_metadata}
        if submission_format_type
        else {},
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
    print("\n" + "=" * 60)
    print("= ITERATION CONTROL")
    print("=" * 60)

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
    print("\n" + "=" * 60)
    print("= PERFORMANCE EVALUATION")
    print("=" * 60)

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
        if public_score is not None:
            print(f"\n📊 Public Score: {public_score:.4f}")
            # Use metric direction for score selection
            from .core.config import compare_scores

            if current_score == 0.0:
                current_score = public_score
            else:
                try:
                    metric_name = state["competition_info"].evaluation_metric
                except Exception:
                    metric_name = ""
                current_score = compare_scores(current_score, public_score, metric_name)

    from .core.config import is_metric_minimization

    metric_name = ""
    try:
        metric_name = state["competition_info"].evaluation_metric
    except Exception:
        metric_name = ""

    minimize = is_metric_minimization(metric_name)
    gap = (current_score - target_score) if minimize else (target_score - current_score)

    print(f"\nCurrent Score: {current_score:.4f}")
    print(f"Target Score:  {target_score:.4f}")
    print(f"Gap:           {gap:.4f} ({'minimize' if minimize else 'maximize'})")

    # Analyze component performance
    dev_results = state.get("development_results", [])
    successful_components = [r for r in dev_results if r.success]

    print(
        f"\n📈 Component Success Rate: {len(successful_components)}/{len(dev_results)} ({len(successful_components) / len(dev_results) * 100:.0f}%)"
        if dev_results
        else "\n📈 No components tested yet"
    )

    # Decision: should we refine?
    needs_refinement = False
    refinement_reason = None

    if minimize:
        target_achieved = current_score <= target_score
    else:
        target_achieved = current_score >= target_score

    if target_achieved:
        comparator = "<=" if minimize else ">="
        print(f"\n🎉 Target achieved! ({current_score:.4f} {comparator} {target_score:.4f})")
        needs_refinement = False
    elif current_iteration >= max_iterations:
        print(f"\n⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        needs_refinement = False
    else:
        # Check if we have room for improvement
        improvement_potential = gap

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

    Simplified routing logic - only stops for:
    1. Explicit skip_remaining_components flag
    2. Medal achievement (MLE-bench success)
    3. Critical errors (data download failed, auth issues)
    4. All components implemented

    Target score checking is delegated to iteration_control to allow
    multiple refinement iterations with meta-evaluator insights.

    Args:
        state: Current state

    Returns:
        "iterate" to continue implementing components, or "end" if done
    """
    # Explicit early-stop flag (e.g., set by DeveloperAgent)
    if state.get("skip_remaining_components"):
        print("\n⏩ skip_remaining_components=True - Moving to validation")
        return "end"

    # Check for medal achievement in MLE-bench mode (immediate success)
    mlebench_grade = state.get("mlebench_grade")
    run_mode = str(state.get("run_mode", "")).lower()

    if run_mode == "mlebench" and isinstance(mlebench_grade, dict):
        if mlebench_grade.get("valid_submission"):
            if any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
                print("\n🏅 MEDAL ACHIEVED - Moving to validation")
                return "end"

    # Check for critical errors (data download failed, auth issues)
    errors = state.get("errors", [])
    if errors:
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
                data_errors = [
                    r for r in recent_failures if "Data files not found" in (r.stderr or "")
                ]
                if len(data_errors) == 3:
                    print("\n⚠️ Repeated data file errors, stopping workflow")
                    return "end"

        remaining = len(ablation_plan) - current_component_index
        print(f"\n🔄 {remaining} component(s) remaining - continuing iteration")
        return "iterate"

    # All components done - move to validation
    print(f"\n✅ All {len(ablation_plan)} components implemented - moving to validation")
    return "end"


def route_after_submission(state: KaggleState) -> Literal["retry_developer", "continue"]:
    """
    Route after submission agent - retry if submission is invalid.

    Checks if the submission passed validation. If not, routes back to
    the developer to regenerate with the error context.

    Args:
        state: Current state

    Returns:
        "retry_developer" if submission invalid and retries remaining,
        "continue" otherwise
    """
    submissions = state.get("submissions", [])

    if not submissions:
        # No submission generated at all - retry
        retry_count = state.get("retry_submission_count", 0)
        if retry_count < 3:
            state["retry_submission_count"] = retry_count + 1
            state["submission_validation_error"] = "No submission file generated"
            print(f"⚠️ No submission generated, retrying... ({retry_count + 1}/3)")
            return "retry_developer"
        return "continue"

    last_submission = submissions[-1]

    # Check if submission is valid (handle both dict and object)
    is_valid = True
    error_msg = None

    if isinstance(last_submission, dict):
        is_valid = last_submission.get("valid", True)
        error_msg = last_submission.get("error")
    else:
        # Object with attributes
        is_valid = getattr(last_submission, "valid", True)
        error_msg = getattr(last_submission, "error", None)

    if not is_valid and error_msg:
        retry_count = state.get("retry_submission_count", 0)

        if retry_count < 3:
            state["retry_submission_count"] = retry_count + 1
            state["submission_validation_error"] = error_msg
            print(f"⚠️ Invalid submission: {error_msg[:100]}...")
            print(f"   Retrying with error context... ({retry_count + 1}/3)")
            return "retry_developer"
        print("⚠️ Max submission retries reached, continuing...")

    return "continue"


def route_after_iteration_control(state: KaggleState) -> Literal["refine", "end"]:
    """
    Route after iteration control - decide if we refine or end.

    In MLE-bench mode, aggressively refines until:
    1. Medal is achieved
    2. Max iterations reached

    Args:
        state: Current state

    Returns:
        "refine" to start refinement iteration, or "end" if done
    """
    needs_refinement = state.get("needs_refinement", False)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    run_mode = str(state.get("run_mode", "")).lower()
    mlebench_grade = state.get("mlebench_grade")

    print("\n🔀 Routing decision:")
    print(f"   Current iteration: {current_iteration}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Needs refinement: {needs_refinement}")
    print(f"   Run mode: {run_mode}")

    # Explicit skip flag
    if state.get("skip_remaining_components"):
        print("   ⏩ skip_remaining_components=True - Ending")
        return "end"

    # Check for medal achievement (success - stop)
    if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
        if any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
            print("   🏅 MEDAL ACHIEVED - Success!")
            return "end"

    # Max iterations reached
    if current_iteration >= max_iterations:
        print(f"   ⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        return "end"

    # MLE-bench mode: aggressively refine until medal or max_iterations
    if run_mode == "mlebench":
        # Log refinement guidance if available
        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance:
            print("   📋 Refinement guidance available from meta-evaluator")
            if refinement_guidance.get("planner_guidance"):
                print(f"      Planner: {refinement_guidance['planner_guidance'][:80]}...")
            if refinement_guidance.get("developer_guidance"):
                print(f"      Developer: {refinement_guidance['developer_guidance'][:80]}...")

        print(f"   🔄 MLE-bench mode: Starting refinement iteration {current_iteration + 1}")
        return "refine"

    # Standard Kaggle mode: check target_score
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
            target_score = 1.0

    if isinstance(current_score, (int, float)) and isinstance(target_score, (int, float)):
        if is_metric_minimization(metric_name):
            if float(current_score) <= float(target_score):
                print(f"   ✅ Goal achieved: {current_score:.4f} <= {target_score:.4f}")
                return "end"
        elif float(current_score) >= float(target_score):
            print(f"   ✅ Goal achieved: {current_score:.4f} >= {target_score:.4f}")
            return "end"

    # Decide based on refinement flag
    if needs_refinement:
        print(f"   🔄 Starting refinement iteration {current_iteration + 1}")
        return "refine"

    print("   ✅ No refinement needed")
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
            "iterate": "developer",  # More components to implement
            "end": "robustness",  # All components done → validate
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
            "retry_developer": "developer",  # Invalid submission → regenerate
            "continue": "performance_evaluation",  # Valid → continue
        },
    )

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
            "end": "reporting",  # Goal achieved or max iterations -> Explain
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
    print("=" * 70)
    print(f"KAGGLE AGENTS WORKFLOW: {competition_name}")
    print("=" * 70)

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
                "type": "autonomous-run",
            },
        }
        final_state = workflow.invoke(state, config)
    else:
        workflow = compile_workflow()
        config = {
            "recursion_limit": 150,
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


# ==================== MLE-bench Workflow ====================


def route_after_meta_evaluator(state: KaggleState) -> Literal["curriculum", "continue"]:
    """
    Route after meta-evaluator - check if curriculum learning is needed.

    WEBRL-style: If there are critical failures, generate sub-tasks first.

    Args:
        state: Current state

    Returns:
        "curriculum" if subtasks needed, "continue" otherwise
    """
    failure_analysis = state.get("failure_analysis", {})
    error_patterns = failure_analysis.get("error_patterns", [])
    failed_components = failure_analysis.get("failed_components", [])

    # Check for critical errors that need curriculum learning
    critical_errors = ["memory_error", "timeout_error", "import_error", "syntax_error"]
    has_critical = any(e in critical_errors for e in error_patterns)

    # Only trigger curriculum if we have failures and this is a refinement iteration
    current_iteration = state.get("current_iteration", 0)

    if has_critical and current_iteration > 0 and len(failed_components) > 0:
        print("\n   WEBRL: Critical failures detected - triggering curriculum learning")
        return "curriculum"

    return "continue"


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
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("performance_evaluation", performance_evaluation_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)
    workflow.add_node("curriculum_learning", curriculum_learning_node)  # WEBRL
    workflow.add_node("inject_curriculum", inject_subtask_guidance)  # WEBRL guidance injection
    workflow.add_node("prompt_refinement", prompt_refinement_node)
    workflow.add_node("iteration_control", iteration_control_node)
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
            "iterate": "developer",  # More components to implement
            "end": "robustness",  # All components done → validate
        },
    )

    # Robustness → Ensemble
    workflow.add_edge("robustness", "ensemble")

    # Ensemble → Submission
    workflow.add_edge("ensemble", "submission")

    # Submission → Performance Evaluation → Meta-Evaluator
    workflow.add_edge("submission", "performance_evaluation")
    workflow.add_edge("performance_evaluation", "meta_evaluator")

    # Meta-Evaluator → Conditional (WEBRL: curriculum or continue?)
    workflow.add_conditional_edges(
        "meta_evaluator",
        route_after_meta_evaluator,
        {
            "curriculum": "curriculum_learning",  # WEBRL: Generate sub-tasks
            "continue": "prompt_refinement",  # Standard path
        },
    )

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
            "refine": "planner",  # Start refinement cycle (back to planner with guidance)
            "end": "reporting",  # Goal achieved or max iterations → Report
        },
    )

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
    print("=" * 70)
    print(f"SIMPLE WORKFLOW: {competition_name}")
    print("=" * 70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    from .core.state import CompetitionInfo

    state["competition_info"] = CompetitionInfo(**competition_info)

    # Run workflow
    workflow = create_simple_workflow()
    final_state = workflow.invoke(state)

    print("\n" + "=" * 70)
    print(" WORKFLOW COMPLETE")
    print("=" * 70)

    return final_state
