"""Main entry point for Kaggle Agents."""

import os
import sys


# Fix matplotlib backend for Google Colab and other environments
# This must be set before any imports that use matplotlib (e.g., LightGBM)
if 'MPLBACKEND' in os.environ:
    if os.environ['MPLBACKEND'] == 'module://matplotlib_inline.backend_inline':
        os.environ['MPLBACKEND'] = 'Agg'
else:
    # Set a safe default backend for headless environments
    os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from .utils.config import Config
from .utils.state import KaggleState
from .workflows.enhanced_workflow import create_enhanced_workflow
from .workflows.kaggle_workflow import create_kaggle_workflow


def initialize_state(competition: str, max_iterations: int = 5) -> KaggleState:
    """Initialize the workflow state (simple mode).

    Args:
        competition: Kaggle competition name
        max_iterations: Maximum number of improvement iterations

    Returns:
        Initial state dictionary
    """
    return {
        "messages": [HumanMessage(content=f"Starting Kaggle competition: {competition}")],
        "competition_name": competition,
        "competition_type": "",
        "metric": "",
        "train_data_path": "",
        "test_data_path": "",
        "sample_submission_path": "",
        "eda_summary": {},
        "data_insights": [],
        "features_engineered": [],
        "feature_importance": {},
        "models_trained": [],
        "best_model": {},
        "cv_scores": [],
        "submission_path": "",
        "submission_score": 0.0,
        "leaderboard_rank": 0,
        "next_agent": "data_collection",
        "iteration": 0,
        "max_iterations": max_iterations,
        "errors": [],
    }


def initialize_enhanced_state(competition: str, data_dir: str, max_iterations: int = 5) -> dict:
    """Initialize the enhanced workflow state.

    Args:
        competition: Kaggle competition name
        data_dir: Directory containing competition data
        max_iterations: Maximum number of improvement iterations

    Returns:
        Initial enhanced state dictionary
    """
    competition_dir = Path(data_dir) / competition
    competition_dir.mkdir(parents=True, exist_ok=True)

    return {
        "messages": [HumanMessage(content=f"Starting Kaggle competition: {competition}")],
        "competition_name": competition,
        "competition_type": "",
        "metric": "",
        "competition_dir": str(competition_dir),
        "train_data_path": "",
        "test_data_path": "",
        "sample_submission_path": "",
        "eda_summary": {},
        "data_insights": [],
        "features_engineered": [],
        "feature_importance": {},
        "models_trained": [],
        "best_model": {},
        "cv_scores": [],
        "submission_path": "",
        "submission_score": 0.0,
        "leaderboard_rank": 0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "errors": [],
        "phase": "Understand Background",
        "memory": [],
        "background_info": "",
        "rules": {},
        "retry_count": 0,
        "max_phase_retries": 3,
        "status": "Continue",
    }


def extract_competition_slug(competition_input: str) -> str:
    """Extract competition slug from URL or name.

    Args:
        competition_input: Competition name or URL

    Returns:
        Competition slug (e.g., 'titanic')

    Examples:
        'titanic' -> 'titanic'
        'https://www.kaggle.com/competitions/titanic' -> 'titanic'
        'https://www.kaggle.com/c/titanic' -> 'titanic'
    """
    # Remove trailing slash
    competition_input = competition_input.rstrip('/')

    # If it's a URL, extract the slug
    if 'kaggle.com' in competition_input:
        # Handle both /competitions/ and /c/ URLs
        parts = competition_input.split('/')
        # Find 'competitions' or 'c' in the URL
        for i, part in enumerate(parts):
            if part in ('competitions', 'c') and i + 1 < len(parts):
                return parts[i + 1]

    # Otherwise, assume it's already a slug
    return competition_input


def main():
    """Main function to run Kaggle agents."""
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parser = argparse.ArgumentParser(
        description="AutoKaggle - Multi-agent framework for autonomous Kaggle competitions"
    )
    parser.add_argument(
        "competition",
        type=str,
        help="Kaggle competition name or URL (e.g., 'titanic' or 'https://www.kaggle.com/competitions/titanic')",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of improvement iterations (default: 5)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the workflow graph",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "enhanced"],
        default="enhanced",
        help="Workflow mode: 'simple' (basic LangGraph) or 'enhanced' (multi-agent with feedback) (default: enhanced)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)",
    )

    args = parser.parse_args()

    # Extract competition slug from URL or name
    competition_slug = extract_competition_slug(args.competition)

    # Validate and configure
    try:
        Config.validate()
        Config.configure_tracing()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set up your .env file with required API keys.")
        print("Copy .env.example to .env and fill in your credentials.")
        return

    # Create necessary directories
    Path(Config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.SUBMISSIONS_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Kaggle Agents - Multi-Agent Competition Framework")
    print("=" * 80)
    print(f"Competition: {competition_slug}")
    if args.competition != competition_slug:
        print(f"  (from URL: {args.competition})")
    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {args.model}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 80)
    print()

    # Create workflow based on mode
    if args.mode == "enhanced":
        print("Using ENHANCED mode with multi-agent system and feedback loops")
        workflow = create_enhanced_workflow(
            competition_name=competition_slug,
            model=args.model
        )
        initial_state = initialize_enhanced_state(
            competition_slug,
            Config.DATA_DIR,
            args.max_iterations
        )
    else:
        print("Using SIMPLE mode with basic LangGraph workflow")
        workflow = create_kaggle_workflow()
        initial_state = initialize_state(competition_slug, args.max_iterations)

    # Visualize if requested
    if args.visualize:
        try:
            from IPython.display import Image, display
            display(Image(workflow.get_graph().draw_mermaid_png()))
        except ImportError:
            print("Warning: Visualization requires IPython. Install with: uv add ipython")

    # Run workflow
    try:
        print("Starting workflow...\n")
        final_state = workflow.invoke(initial_state)

        print()
        print("=" * 80)
        print("Workflow completed")
        print("=" * 80)

        # Print summary
        if final_state.get("errors"):
            print("\nErrors encountered:")
            for error in final_state["errors"]:
                print(f"  - {error}")

        if final_state.get("best_model"):
            print(f"\nBest Model: {final_state['best_model']['name']}")
            print(f"CV Score: {final_state['best_model']['mean_cv_score']:.4f}")

        if final_state.get("submission_path"):
            print(f"\nSubmission: {final_state['submission_path']}")

        if final_state.get("leaderboard_rank"):
            print(f"\nLeaderboard Rank: {final_state['leaderboard_rank']}")
            print(f"Public Score: {final_state.get('submission_score', 0.0)}")

        print(f"\nIterations completed: {final_state.get('iteration', 0)}/{args.max_iterations}")
        print()

    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
    except Exception as e:
        print(f"\n\nWorkflow failed: {e!s}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
