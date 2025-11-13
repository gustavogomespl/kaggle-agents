"""
Orchestrator for managing the Kaggle Agents workflow.

This module provides a high-level interface for running
the complete agent pipeline with progress tracking and control.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .state import KaggleState, CompetitionInfo, create_initial_state
from .config import get_config, get_competition_dir
from ..workflow import run_workflow, run_simple_workflow


console = Console()


@dataclass
class WorkflowResults:
    """Results from workflow execution."""

    competition_name: str
    success: bool
    iterations: int
    sota_solutions_found: int
    components_planned: int
    components_implemented: int
    success_rate: float
    total_time: float
    final_state: KaggleState
    termination_reason: str


class KaggleOrchestrator:
    """
    High-level orchestrator for Kaggle competition solving.

    This class provides:
    - Simple API for running workflows
    - Progress tracking with Rich
    - Result summarization
    - Error handling and recovery
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.config = get_config()
        self.console = Console()

    def solve_competition(
        self,
        competition_name: str,
        competition_description: str = "",
        problem_type: str = "unknown",
        evaluation_metric: str = "unknown",
        max_iterations: int = 3,
        simple_mode: bool = False,
    ) -> WorkflowResults:
        """
        Solve a Kaggle competition end-to-end.

        Args:
            competition_name: Name of the competition
            competition_description: Description of the problem
            problem_type: Type (classification, regression, etc.)
            evaluation_metric: Evaluation metric used
            max_iterations: Maximum workflow iterations
            simple_mode: Use simple workflow (no iterations)

        Returns:
            WorkflowResults with execution summary
        """
        start_time = time.time()

        # Setup
        working_dir = str(get_competition_dir(competition_name))

        competition_info = {
            "name": competition_name,
            "description": competition_description,
            "evaluation_metric": evaluation_metric,
            "problem_type": problem_type,
        }

        # Display header
        self._display_header(competition_name, problem_type, evaluation_metric)

        try:
            # Run workflow
            if simple_mode:
                final_state = run_simple_workflow(
                    competition_name=competition_name,
                    working_dir=working_dir,
                    competition_info=competition_info,
                )
            else:
                final_state = run_workflow(
                    competition_name=competition_name,
                    working_dir=working_dir,
                    competition_info=competition_info,
                    max_iterations=max_iterations,
                    use_checkpointing=False,
                )

            # Calculate results
            results = self._create_results(
                competition_name=competition_name,
                final_state=final_state,
                total_time=time.time() - start_time,
            )

            # Display results
            self._display_results(results)

            return results

        except Exception as e:
            console.print(f"\n[red]L Workflow failed: {str(e)}[/red]")
            raise

    def _create_results(
        self,
        competition_name: str,
        final_state: KaggleState,
        total_time: float,
    ) -> WorkflowResults:
        """Create results summary from final state."""
        dev_results = final_state.get("development_results", [])
        successful = sum(1 for r in dev_results if r.success)
        success_rate = (successful / len(dev_results) * 100) if dev_results else 0.0

        return WorkflowResults(
            competition_name=competition_name,
            success=final_state.get("should_continue", False) == False,
            iterations=final_state.get("current_iteration", 0),
            sota_solutions_found=len(final_state.get("sota_solutions", [])),
            components_planned=len(final_state.get("ablation_plan", [])),
            components_implemented=len(dev_results),
            success_rate=success_rate,
            total_time=total_time,
            final_state=final_state,
            termination_reason=final_state.get("termination_reason", "unknown"),
        )

    def _display_header(self, competition: str, problem_type: str, metric: str):
        """Display workflow header."""
        header = f"""
[bold cyan]< KAGGLE AGENTS - Autonomous Competition Solving[/bold cyan]

[bold]Competition:[/bold] {competition}
[bold]Problem Type:[/bold] {problem_type}
[bold]Metric:[/bold] {metric}
[bold]Goal:[/bold] Top 20% (Percentile d 20)
"""
        console.print(Panel(header, border_style="cyan"))

    def _display_results(self, results: WorkflowResults):
        """Display workflow results."""
        # Create results table
        table = Table(title="Workflow Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")

        table.add_row("Competition", results.competition_name)
        table.add_row("Total Time", f"{results.total_time:.1f}s")
        table.add_row("Iterations", str(results.iterations))
        table.add_row("SOTA Solutions", str(results.sota_solutions_found))
        table.add_row("Components Planned", str(results.components_planned))
        table.add_row("Components Implemented", str(results.components_implemented))
        table.add_row("Success Rate", f"{results.success_rate:.0f}%")
        table.add_row("Termination", results.termination_reason)

        console.print("\n")
        console.print(table)

        # Success/failure message
        if results.success:
            console.print("\n[bold green] Workflow completed successfully![/bold green]")
        else:
            console.print("\n[bold yellow]  Workflow incomplete[/bold yellow]")

    def get_workflow_status(self, competition_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a running or completed workflow.

        Args:
            competition_name: Competition name

        Returns:
            Status dictionary or None
        """
        # TODO: Implement status tracking
        # Would require persistent storage of workflow state
        console.print("[yellow]Status tracking not yet implemented[/yellow]")
        return None


# ==================== Convenience Function ====================

def solve_competition(
    competition_name: str,
    competition_description: str = "",
    problem_type: str = "unknown",
    evaluation_metric: str = "unknown",
    max_iterations: int = 3,
) -> WorkflowResults:
    """
    Solve a Kaggle competition (convenience function).

    Args:
        competition_name: Competition name
        competition_description: Problem description
        problem_type: Problem type
        evaluation_metric: Metric used
        max_iterations: Max iterations

    Returns:
        WorkflowResults
    """
    orchestrator = KaggleOrchestrator()
    return orchestrator.solve_competition(
        competition_name=competition_name,
        competition_description=competition_description,
        problem_type=problem_type,
        evaluation_metric=evaluation_metric,
        max_iterations=max_iterations,
    )
