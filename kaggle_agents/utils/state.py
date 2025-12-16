"""
Legacy state object used by older workflows and tests.

The actively maintained workflow uses `kaggle_agents/core/state.py`. This module
keeps the legacy API stable for unit tests and compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def merge_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries, with `right` keys overwriting `left`."""
    return {**left, **right}


@dataclass
class KaggleState:
    """Lightweight state container with sensible defaults."""

    # Core context
    messages: list[Any] = field(default_factory=list)
    competition_name: str = ""
    competition_type: str = ""
    metric: str = ""

    # Paths
    train_data_path: str = ""
    test_data_path: str = ""
    sample_submission_path: str = ""

    # EDA / insights
    eda_summary: dict[str, Any] = field(default_factory=dict)
    data_insights: list[str] = field(default_factory=list)

    # Feature engineering
    features_engineered: list[str] = field(default_factory=list)
    feature_importance: dict[str, Any] = field(default_factory=dict)

    # Modeling
    models_trained: list[dict[str, Any]] = field(default_factory=list)
    best_model: dict[str, Any] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)

    # Submission / leaderboard
    submission_path: str = ""
    submission_score: float = 0.0
    leaderboard_rank: int = 0

    # Iteration control
    next_agent: str = ""
    iteration: int = 0
    max_iterations: int = 5

    # Errors
    errors: list[str] = field(default_factory=list)

