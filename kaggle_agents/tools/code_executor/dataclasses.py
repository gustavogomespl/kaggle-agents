"""
Data classes for code execution.

Contains ExecutionResult and ExecutionProgress dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Result from code execution."""

    success: bool
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int
    artifacts_created: list[str]
    errors: list[str]


@dataclass
class ExecutionProgress:
    """
    Track execution progress for checkpoint/resume support.

    This class captures incremental progress during long-running executions,
    enabling intelligent timeout decisions and partial result recovery.
    """

    # Fold tracking (for cross-validation)
    folds_completed: int = 0
    total_folds: int = 5
    fold_scores: list = None  # type: ignore  # list[float]

    # Score tracking
    current_cv_score: float = None  # type: ignore  # Optional[float]
    best_fold_score: float = None  # type: ignore  # Optional[float]

    # Artifact tracking
    models_saved: list = None  # type: ignore  # list[str]
    oof_predictions_saved: bool = False
    test_predictions_saved: bool = False

    # Time tracking
    elapsed_seconds: float = 0.0
    avg_fold_time: float = None  # type: ignore  # Optional[float]
    estimated_remaining: float = None  # type: ignore  # Optional[float]

    # Status
    current_phase: str = "initializing"  # initializing, training, validating, predicting
    last_output: str = ""

    def __post_init__(self):
        if self.fold_scores is None:
            self.fold_scores = []
        if self.models_saved is None:
            self.models_saved = []

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage based on folds completed."""
        if self.total_folds == 0:
            return 0.0
        return (self.folds_completed / self.total_folds) * 100

    @property
    def can_create_partial_submission(self) -> bool:
        """Check if we have enough data for a partial submission."""
        return self.folds_completed >= 1 and self.oof_predictions_saved

    def update_from_stdout(self, line: str) -> bool:
        """
        Update progress from a stdout line.

        Returns True if progress was updated.
        """
        updated = False
        line_lower = line.lower()

        # Detect fold completion patterns
        # Pattern: "Fold X best AUC: Y.YYYY" or "Fold X score: Y.YYYY"
        fold_patterns = [
            r"fold\s*(\d+)\s*(?:best\s+)?(?:auc|score|accuracy|f1|rmse|mae)[:=]\s*([\d.]+)",
            r"=+\s*fold\s*(\d+)\s*=+",
            r"fold\s*(\d+)\s*(?:of\s*\d+)?\s*(?:completed?|finished?|done)",
        ]

        for pattern in fold_patterns:
            match = re.search(pattern, line_lower)
            if match:
                try:
                    fold_num = int(match.group(1))
                    # Handle 0-indexed vs 1-indexed folds
                    completed = fold_num + 1 if fold_num < self.total_folds else fold_num
                    if completed > self.folds_completed:
                        self.folds_completed = min(completed, self.total_folds)
                        updated = True

                    # Try to extract score if present
                    if len(match.groups()) >= 2:
                        score = float(match.group(2))
                        self.fold_scores.append(score)
                        self.best_fold_score = max(self.fold_scores)
                except (ValueError, IndexError):
                    pass

        # Detect Final Validation Performance
        if "final validation performance" in line_lower:
            match = re.search(r"([\d.]+)", line)
            if match:
                self.current_cv_score = float(match.group(1))
                updated = True

        # Detect OOF/test prediction saves
        if "saved oof" in line_lower or "oof_pred" in line_lower:
            self.oof_predictions_saved = True
            updated = True

        if "saved test" in line_lower or "test_pred" in line_lower:
            self.test_predictions_saved = True
            updated = True

        # Detect model saves
        if (
            "saved model" in line_lower
            or "saving model" in line_lower
            or ".pkl" in line_lower
            or ".joblib" in line_lower
            or ".pth" in line_lower
        ):
            # Extract model name if possible
            match = re.search(r"(\w+\.(pkl|joblib|pth|pt|h5))", line_lower)
            if match and match.group(1) not in self.models_saved:
                self.models_saved.append(match.group(1))
                updated = True

        # Detect phase changes
        if "loading" in line_lower or "preparing" in line_lower:
            self.current_phase = "initializing"
        elif "training" in line_lower or "fitting" in line_lower:
            self.current_phase = "training"
        elif "validating" in line_lower or "evaluating" in line_lower:
            self.current_phase = "validating"
        elif "predicting" in line_lower or "inference" in line_lower:
            self.current_phase = "predicting"

        self.last_output = line.strip()[:200]
        return updated

    def estimate_remaining_time(self):
        """Estimate remaining time based on fold progress."""
        if self.folds_completed == 0 or self.elapsed_seconds == 0:
            return None

        self.avg_fold_time = self.elapsed_seconds / self.folds_completed
        remaining_folds = self.total_folds - self.folds_completed
        self.estimated_remaining = self.avg_fold_time * remaining_folds

        return self.estimated_remaining

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "folds_completed": self.folds_completed,
            "total_folds": self.total_folds,
            "fold_scores": self.fold_scores,
            "current_cv_score": self.current_cv_score,
            "best_fold_score": self.best_fold_score,
            "models_saved": self.models_saved,
            "oof_predictions_saved": self.oof_predictions_saved,
            "test_predictions_saved": self.test_predictions_saved,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_fold_time": self.avg_fold_time,
            "estimated_remaining": self.estimated_remaining,
            "current_phase": self.current_phase,
            "progress_percent": self.progress_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionProgress:
        """Create from dictionary."""
        return cls(
            folds_completed=data.get("folds_completed", 0),
            total_folds=data.get("total_folds", 5),
            fold_scores=data.get("fold_scores", []),
            current_cv_score=data.get("current_cv_score"),
            best_fold_score=data.get("best_fold_score"),
            models_saved=data.get("models_saved", []),
            oof_predictions_saved=data.get("oof_predictions_saved", False),
            test_predictions_saved=data.get("test_predictions_saved", False),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            avg_fold_time=data.get("avg_fold_time"),
            estimated_remaining=data.get("estimated_remaining"),
            current_phase=data.get("current_phase", "unknown"),
        )
