"""
MLE-bench Integration Module.

This module provides tools for running kaggle-agents on MLE-bench
competitions with proper data handling, grading, and evaluation.

Example:
    >>> from kaggle_agents.mlebench import solve_mlebench
    >>> result = solve_mlebench(
    ...     competition_id="aerial-cactus-identification",
    ...     problem_type="binary_classification",
    ...     evaluation_metric="auc",
    ... )
    >>> print(f"Valid: {result.valid_submission}, Score: {result.score}")
"""

from .data_adapter import MLEBenchDataAdapter, MLEBenchDataInfo
from .runner import MLEBenchResult, MLEBenchRunner, solve_mlebench


__all__ = [
    "MLEBenchDataAdapter",
    "MLEBenchDataInfo",
    "MLEBenchResult",
    "MLEBenchRunner",
    "solve_mlebench",
]
