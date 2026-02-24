"""
Enhanced CLI for Kaggle Agents.

Simple, clean, and efficient command-line interface with Rich output.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core import get_config, solve_competition


console = Console()


def print_header():
    """Print application header."""
    console.print(
        Panel.fit(
            "[bold]Kaggle Agents[/bold]\nAutonomous Competition Solving",
            border_style="blue",
        )
    )


def print_error(message: str):
    """Print error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str):
    """Print success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_info(message: str):
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def start_competition(
    competition_name: str,
    description: str | None = None,
    problem_type: str = "binary_classification",
    evaluation_metric: str = "auto",
    max_iterations: int = 3,
):
    """
    Start solving a Kaggle competition.

    Args:
        competition_name: Name of the Kaggle competition
        description: Competition description
        problem_type: Type of problem (classification, regression, etc.)
        evaluation_metric: Metric used for evaluation
        max_iterations: Maximum number of iterations
    """
    print_header()

    console.print(f"\n[bold]Competition:[/bold] {competition_name}")
    console.print(f"[bold]Problem Type:[/bold] {problem_type}")
    console.print(f"[bold]Metric:[/bold] {evaluation_metric}")
    console.print(f"[bold]Max Iterations:[/bold] {max_iterations}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting workflow...", total=None)

            # Run competition
            results = solve_competition(
                competition_name=competition_name,
                competition_description=description or f"Solve {competition_name}",
                problem_type=problem_type,
                evaluation_metric=evaluation_metric,
                max_iterations=max_iterations,
            )

            progress.update(task, description="Complete!")

        # Print results
        print_results(results)

    except KeyboardInterrupt:
        print_error("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to run competition: {e!s}")
        sys.exit(1)


def print_results(results):
    """Print workflow results in a formatted table."""
    console.print("\n[bold green]Workflow Complete![/bold green]\n")

    # Create results table
    table = Table(title="Results Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Competition", results.competition_name)
    table.add_row("Success", "✓" if results.success else "✗")
    table.add_row("Iterations", str(results.iterations))
    table.add_row("SOTA Solutions", str(results.sota_solutions_found))
    table.add_row("Components Planned", str(results.components_planned))
    table.add_row("Components Implemented", str(results.components_implemented))
    table.add_row("Success Rate", f"{results.success_rate:.1%}")
    table.add_row("Total Time", f"{results.total_time:.1f}s")
    table.add_row("Termination", results.termination_reason)

    console.print(table)

    # Additional details
    state = results.final_state

    if state.get("validation_results"):
        val_score = state.get("overall_validation_score", 0)
        console.print(f"\n[bold]Validation Score:[/bold] {val_score:.1%}")

    if state.get("submissions"):
        latest = state["submissions"][-1]
        if latest.public_score:
            console.print(f"[bold]Public Score:[/bold] {latest.public_score:.4f}")
        if latest.percentile:
            console.print(f"[bold]Percentile:[/bold] {latest.percentile:.1f}%")


def show_config():
    """Display current configuration."""
    print_header()

    config = get_config()

    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("LLM Model", config.llm.model)
    table.add_row("Temperature", str(config.llm.temperature))
    table.add_row("Max Iterations", str(config.iteration.max_iterations))
    table.add_row("Target Percentile", f"{config.iteration.target_percentile}%")
    table.add_row("Auto Submit", "✓" if config.kaggle.auto_submit else "✗")
    table.add_row("Work Directory", str(config.paths.work_dir))

    console.print(table)


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kaggle Agents - Autonomous Competition Solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start solving a competition")
    start_parser.add_argument("competition", help="Competition name")
    start_parser.add_argument("--description", "-d", help="Competition description", default=None)
    start_parser.add_argument(
        "--problem-type",
        "-p",
        help="Problem type",
        default="binary_classification",
        choices=[
            "binary_classification",
            "multiclass_classification",
            "regression",
        ],
    )
    start_parser.add_argument("--metric", "-m", help="Evaluation metric (default: auto-detect from Kaggle API)", default="auto")
    start_parser.add_argument(
        "--max-iterations",
        "-i",
        type=int,
        help="Maximum iterations",
        default=3,
    )

    # Config command
    subparsers.add_parser("config", help="Show current configuration")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "start":
        start_competition(
            competition_name=args.competition,
            description=args.description,
            problem_type=args.problem_type,
            evaluation_metric=args.metric,
            max_iterations=args.max_iterations,
        )
    elif args.command == "config":
        show_config()
    else:
        # No command - show help
        print_header()
        parser.print_help()


if __name__ == "__main__":
    main()
