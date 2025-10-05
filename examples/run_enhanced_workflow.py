"""Example script demonstrating the enhanced workflow."""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle_agents.core.state import EnhancedKaggleState
from kaggle_agents.core.sop import SOP
from kaggle_agents.workflows.enhanced_workflow import create_enhanced_workflow


def setup_logging(log_file: str = "enhanced_workflow.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def run_with_sop(competition_name: str, competition_dir: str):
    """Run workflow using SOP orchestrator directly.

    Args:
        competition_name: Name of the Kaggle competition
        competition_dir: Directory containing competition data
    """
    print("\n" + "="*80)
    print("Running Enhanced Workflow with SOP")
    print("="*80 + "\n")

    # Create initial state
    initial_state = EnhancedKaggleState(
        competition_name=competition_name,
        competition_dir=competition_dir,
        phase="Understand Background"
    )

    # Create and run SOP
    sop = SOP(competition_name=competition_name, model="gpt-4o")
    final_state = sop.run(initial_state, max_steps=50)

    print("\n" + "="*80)
    print("SOP Workflow Complete")
    print("="*80)
    print(f"Final Phase: {final_state.phase}")
    print(f"Total Memory Entries: {len(final_state.memory)}")
    print(f"Competition Type: {final_state.competition_type}")
    print(f"Metric: {final_state.metric}")
    print("="*80 + "\n")

    return final_state


def run_with_langgraph(competition_name: str, competition_dir: str):
    """Run workflow using LangGraph integration.

    Args:
        competition_name: Name of the Kaggle competition
        competition_dir: Directory containing competition data
    """
    print("\n" + "="*80)
    print("Running Enhanced Workflow with LangGraph")
    print("="*80 + "\n")

    # Create initial state dict
    initial_state = {
        "messages": [],
        "competition_name": competition_name,
        "competition_dir": competition_dir,
        "phase": "Understand Background",
        "memory": [],
        "retry_count": 0,
        "iteration": 0,
        "max_iterations": 5,
        "errors": [],
        "competition_type": "",
        "metric": "",
        "background_info": "",
        "rules": {},
    }

    # Create workflow
    workflow = create_enhanced_workflow(
        competition_name=competition_name,
        model="gpt-4o"
    )

    # Run workflow
    final_state = workflow.invoke(initial_state)

    print("\n" + "="*80)
    print("LangGraph Workflow Complete")
    print("="*80)
    print(f"Final Phase: {final_state.get('phase', 'Unknown')}")
    print(f"Total Memory Entries: {len(final_state.get('memory', []))}")
    print(f"Competition Type: {final_state.get('competition_type', 'Unknown')}")
    print(f"Metric: {final_state.get('metric', 'Unknown')}")
    print("="*80 + "\n")

    return final_state


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run enhanced Kaggle agents workflow"
    )
    parser.add_argument(
        "competition",
        type=str,
        help="Competition name (e.g., 'titanic')"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing competition data (default: ./data)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sop", "langgraph"],
        default="sop",
        help="Execution method: 'sop' (direct) or 'langgraph' (integrated) (default: sop)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="enhanced_workflow.log",
        help="Log file path (default: enhanced_workflow.log)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    # Setup competition directory
    competition_dir = Path(args.data_dir) / args.competition
    competition_dir.mkdir(parents=True, exist_ok=True)

    print(f"Competition: {args.competition}")
    print(f"Data Directory: {competition_dir}")
    print(f"Method: {args.method.upper()}")
    print(f"Log File: {args.log_file}")

    # Run workflow
    try:
        if args.method == "sop":
            final_state = run_with_sop(args.competition, str(competition_dir))
        else:
            final_state = run_with_langgraph(args.competition, str(competition_dir))

        print("Workflow completed successfully!")

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
