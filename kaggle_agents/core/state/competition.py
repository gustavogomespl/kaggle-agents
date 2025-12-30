"""
Competition-related data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import DomainType, SubmissionFormatType


@dataclass
class CompetitionInfo:
    """Competition metadata and configuration."""

    name: str
    description: str
    evaluation_metric: str
    problem_type: str  # classification, regression, ranking, etc.
    domain: DomainType | None = None
    data_files: list[str] = field(default_factory=list)
    submission_format: dict[str, Any] = field(default_factory=dict)
    submission_format_type: SubmissionFormatType | None = None
    submission_format_metadata: dict[str, Any] = field(default_factory=dict)
    deadline: datetime | None = None


@dataclass
class SOTASolution:
    """State-of-the-art solution retrieved from search."""

    source: str  # notebook_id or discussion_url
    title: str
    score: float
    votes: int
    code_snippets: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    feature_engineering: list[str] = field(default_factory=list)
    ensemble_approach: str | None = None


@dataclass
class AblationComponent:
    """A code component identified for ablation testing."""

    name: str
    component_type: str  # feature_engineering, model, preprocessing, ensemble
    code: str
    estimated_impact: float = 0.0
    tested: bool = False
    actual_impact: float | None = None


def merge_competition_info(
    existing: CompetitionInfo | None, new: CompetitionInfo
) -> CompetitionInfo:
    """Merge competition info, preferring new values when provided."""
    if existing is None:
        return new

    return CompetitionInfo(
        name=new.name if new.name else existing.name,
        description=new.description if new.description else existing.description,
        evaluation_metric=new.evaluation_metric
        if new.evaluation_metric
        else existing.evaluation_metric,
        problem_type=new.problem_type if new.problem_type else existing.problem_type,
        domain=new.domain if new.domain is not None else existing.domain,
        data_files=new.data_files if new.data_files else existing.data_files,
        submission_format=new.submission_format
        if new.submission_format
        else existing.submission_format,
        submission_format_type=new.submission_format_type
        if new.submission_format_type is not None
        else existing.submission_format_type,
        submission_format_metadata=new.submission_format_metadata
        if new.submission_format_metadata
        else existing.submission_format_metadata,
        deadline=new.deadline if new.deadline is not None else existing.deadline,
    )
