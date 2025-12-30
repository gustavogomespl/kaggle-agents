"""
Result data structures from various workflow stages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional


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
    cv_score: Optional[float] = None
    error: Optional[str] = None
    meta_feedback: Optional[str] = None
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

    submission_id: Optional[str]
    public_score: Optional[float]
    private_score: Optional[float] = None
    percentile: Optional[float] = None
    cv_score: Optional[float] = None
    file_path: Optional[str] = None
    valid: bool = True
    error: Optional[str] = None
    submitted_at: datetime = field(default_factory=datetime.now)
