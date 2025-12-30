"""
Result data structures from various workflow stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class DevelopmentResult:
    """Result from code development and execution."""

    code: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    artifacts_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    run_fidelity: Literal["full", "debug"] = "full"


@dataclass
class CodeAttempt:
    """A single code execution attempt for learning and prompt feedback."""

    component_name: str
    component_type: str
    stage: Literal["generate", "fix", "debug", "refine"]
    attempt: int
    success: bool
    cv_score: float | None = None
    error: str | None = None
    meta_feedback: str | None = None
    code_excerpt: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    execution_time: float = 0.0
    run_fidelity: Literal["full", "debug"] = "full"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result from robustness validation."""

    module: str  # debugging, leakage, data_usage, format
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmissionResult:
    """Result from Kaggle submission."""

    submission_id: str | None
    public_score: float | None
    private_score: float | None = None
    percentile: float | None = None
    cv_score: float | None = None
    file_path: str | None = None
    valid: bool = True
    error: str | None = None
    submitted_at: datetime = field(default_factory=datetime.now)
