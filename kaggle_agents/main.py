"""Main entry point for Kaggle Agents."""

import argparse
from pathlib import Path
from langchain_core.messages import HumanMessage
from .workflows.kaggle_workflow import create_kaggle_workflow
from .utils.config import Config
from .utils.state import KaggleState


def initialize_state(competition: str, max_iterations: int = 5) -> KaggleState:
    """Initialize the workflow state.

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


def main():
    """Main function to run Kaggle agents."""
    parser = argparse.ArgumentParser(
        description="AutoKaggle - Multi-agent framework for autonomous Kaggle competitions"
    )
    parser.add_argument(
        "competition",
        type=str,
        help="Kaggle competition name (e.g., 'titanic')",
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

    args = parser.parse_args()

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
    print(f"Competition: {args.competition}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 80)
    print()

    # Create workflow
    workflow = create_kaggle_workflow()

    # Visualize if requested
    if args.visualize:
        try:
            from IPython.display import Image, display
            display(Image(workflow.get_graph().draw_mermaid_png()))
        except ImportError:
            print("⚠️  Visualization requires IPython. Install with: uv add ipython")

    # Initialize state
    initial_state = initialize_state(args.competition, args.max_iterations)

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
        print(f"\n\nWorkflow failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
