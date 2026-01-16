"""
PiML Cross-Competition Persistent Memory Dataclasses.

Stores winning strategies and learnings across competitions for
transfer learning and cold-start optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class DatasetFingerprint:
    """
    Fingerprint of a dataset for similarity matching.

    Used to find similar competitions based on data characteristics.
    """

    n_samples: int
    n_features: int
    missing_rate: float  # Percentage of missing values (0-1)
    imbalance_ratio: float  # Max class ratio / min class ratio
    target_type: str  # "classification", "regression", "multi-label"
    feature_types: dict[str, int] = field(default_factory=dict)  # {"numeric": N, "categorical": M}
    n_classes: Optional[int] = None  # For classification tasks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_rate": self.missing_rate,
            "imbalance_ratio": self.imbalance_ratio,
            "target_type": self.target_type,
            "feature_types": self.feature_types,
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetFingerprint":
        """Create from dictionary."""
        return cls(
            n_samples=data.get("n_samples", 0),
            n_features=data.get("n_features", 0),
            missing_rate=data.get("missing_rate", 0.0),
            imbalance_ratio=data.get("imbalance_ratio", 1.0),
            target_type=data.get("target_type", "unknown"),
            feature_types=data.get("feature_types", {}),
            n_classes=data.get("n_classes"),
        )

    def similarity_score(self, other: "DatasetFingerprint") -> float:
        """
        Compute similarity score between two fingerprints.

        Returns a value between 0 (completely different) and 1 (identical).
        """
        if self.target_type != other.target_type:
            return 0.0  # Different task types are not comparable

        # Size similarity (log scale to handle large differences)
        import math

        size_sim = 1 - min(1, abs(math.log10(max(1, self.n_samples)) - math.log10(max(1, other.n_samples))) / 3)
        feat_sim = 1 - min(1, abs(math.log10(max(1, self.n_features)) - math.log10(max(1, other.n_features))) / 2)

        # Missing rate similarity
        missing_sim = 1 - abs(self.missing_rate - other.missing_rate)

        # Imbalance similarity (log scale)
        imb_sim = 1 - min(1, abs(math.log10(max(1, self.imbalance_ratio)) - math.log10(max(1, other.imbalance_ratio))) / 2)

        # Weighted average
        weights = [0.25, 0.25, 0.2, 0.3]  # size, features, missing, imbalance
        return weights[0] * size_sim + weights[1] * feat_sim + weights[2] * missing_sim + weights[3] * imb_sim


@dataclass
class WinningStrategy:
    """
    A successful strategy from a competition.

    Stores the key decisions and configurations that led to success.
    """

    model_type: str  # "xgboost", "lightgbm", "neural_net", "ensemble", etc.
    preprocessing_steps: list[str] = field(default_factory=list)  # ["normalize", "fillna_median", ...]
    feature_engineering: list[str] = field(default_factory=list)  # ["target_encoding", "date_features", ...]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    ensemble_strategy: Optional[str] = None  # "stacking", "weighted_avg", etc.
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    cv_strategy: str = "kfold"  # "kfold", "stratified", "group", "timeseries"
    n_folds: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type,
            "preprocessing_steps": self.preprocessing_steps,
            "feature_engineering": self.feature_engineering,
            "hyperparameters": self.hyperparameters,
            "ensemble_strategy": self.ensemble_strategy,
            "ensemble_weights": self.ensemble_weights,
            "cv_strategy": self.cv_strategy,
            "n_folds": self.n_folds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WinningStrategy":
        """Create from dictionary."""
        return cls(
            model_type=data.get("model_type", "unknown"),
            preprocessing_steps=data.get("preprocessing_steps", []),
            feature_engineering=data.get("feature_engineering", []),
            hyperparameters=data.get("hyperparameters", {}),
            ensemble_strategy=data.get("ensemble_strategy"),
            ensemble_weights=data.get("ensemble_weights", {}),
            cv_strategy=data.get("cv_strategy", "kfold"),
            n_folds=data.get("n_folds", 5),
        )


@dataclass
class CrossCompetitionMemory:
    """
    Memory record for a completed competition.

    Stores everything needed to inform future competitions.
    """

    competition_id: str
    competition_name: str
    domain: str  # "tabular", "cv", "nlp", "audio", "timeseries"
    fingerprint: DatasetFingerprint
    winning_strategy: WinningStrategy
    failed_approaches: list[str] = field(default_factory=list)  # Approaches that didn't work
    error_patterns: list[str] = field(default_factory=list)  # Common errors encountered
    final_score: float = 0.0
    medal: Optional[str] = None  # "gold", "silver", "bronze", None
    iterations_used: int = 0
    total_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "competition_id": self.competition_id,
            "competition_name": self.competition_name,
            "domain": self.domain,
            "fingerprint": self.fingerprint.to_dict(),
            "winning_strategy": self.winning_strategy.to_dict(),
            "failed_approaches": self.failed_approaches,
            "error_patterns": self.error_patterns,
            "final_score": self.final_score,
            "medal": self.medal,
            "iterations_used": self.iterations_used,
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossCompetitionMemory":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            competition_id=data.get("competition_id", ""),
            competition_name=data.get("competition_name", ""),
            domain=data.get("domain", "unknown"),
            fingerprint=DatasetFingerprint.from_dict(data.get("fingerprint", {})),
            winning_strategy=WinningStrategy.from_dict(data.get("winning_strategy", {})),
            failed_approaches=data.get("failed_approaches", []),
            error_patterns=data.get("error_patterns", []),
            final_score=data.get("final_score", 0.0),
            medal=data.get("medal"),
            iterations_used=data.get("iterations_used", 0),
            total_time_seconds=data.get("total_time_seconds", 0.0),
            timestamp=timestamp,
            notes=data.get("notes", ""),
        )


@dataclass
class StrategyRecommendation:
    """
    A recommendation based on historical competition memory.

    Returned when querying similar competitions.
    """

    source_competition: str  # Competition this recommendation came from
    similarity_score: float  # How similar the source competition was (0-1)
    recommended_strategy: WinningStrategy
    approaches_to_avoid: list[str]  # Failed approaches from similar competition
    expected_score_range: Optional[tuple[float, float]] = None  # (min, max) expected

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_competition": self.source_competition,
            "similarity_score": self.similarity_score,
            "recommended_strategy": self.recommended_strategy.to_dict(),
            "approaches_to_avoid": self.approaches_to_avoid,
            "expected_score_range": self.expected_score_range,
        }
