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

from __future__ import annotations

from .data_adapter import MLEBenchDataAdapter, MLEBenchDataInfo


__all__ = [
    "MLEBenchDataAdapter",
    "MLEBenchDataInfo",
    "MLEBenchResult",
    "MLEBenchRunner",
    "solve_mlebench",
]


def __getattr__(name: str):
    """
    Lazy attribute loading.

    Importing `kaggle_agents.mlebench` should not require Kaggle credentials or
    heavyweight workflow imports when only the data adapter is needed (e.g., tests).
    """
    if name in {"MLEBenchResult", "MLEBenchRunner", "solve_mlebench"}:
        from .runner import MLEBenchResult, MLEBenchRunner, solve_mlebench

        return {
            "MLEBenchResult": MLEBenchResult,
            "MLEBenchRunner": MLEBenchRunner,
            "solve_mlebench": solve_mlebench,
        }[name]
    raise AttributeError(name)
