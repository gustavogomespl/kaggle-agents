"""
Memory-related data structures for tracking learning across iterations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class IterationMemory:
    """Enhanced iteration memory with structured learning signals."""

    iteration: int
    phase: str
    actions_taken: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    score_improvement: float = 0.0

    # Structured what worked/failed
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)

    # Detailed component performance
    component_scores: dict[str, float] = field(default_factory=dict)

    # Hyperparameter configurations used
    hyperparameters_used: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Error patterns with attempted solutions
    error_solutions: list[dict[str, Any]] = field(default_factory=list)

    # LLM-generated insights from this iteration
    key_insights: list[str] = field(default_factory=list)

    # Time tracking
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformanceRecord:
    """Track individual model performance across iterations."""

    model_name: str
    model_type: str  # lightgbm, xgboost, catboost, neural_net, sklearn, etc.
    cv_score: float
    public_lb_score: float | None = None

    # Hyperparameters that achieved this score
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Training characteristics
    training_time_seconds: float = 0.0
    memory_usage_mb: float | None = None

    # Feature information
    features_used: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Context
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataInsights:
    """Persistent EDA insights about the competition data."""

    # Basic stats
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    n_classes: int | None = None

    # Target distribution
    target_distribution: dict[str, float] = field(default_factory=dict)
    is_imbalanced: bool = False
    imbalance_ratio: float | None = None

    # Feature types
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    datetime_features: list[str] = field(default_factory=list)
    text_features: list[str] = field(default_factory=list)

    # Data quality
    missing_value_cols: dict[str, float] = field(default_factory=dict)
    high_cardinality_cols: list[str] = field(default_factory=list)
    constant_cols: list[str] = field(default_factory=list)

    # Correlations and insights
    highly_correlated_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    important_features_eda: list[str] = field(default_factory=list)

    # LLM-generated insights (hybrid approach)
    llm_insights: list[str] = field(default_factory=list)


@dataclass
class ErrorPatternMemory:
    """Memory of error patterns and their solutions."""

    error_type: str
    error_pattern: str = ""

    occurrences: int = 1

    # Solutions that were tried
    solutions_tried: list[str] = field(default_factory=list)
    successful_solutions: list[str] = field(default_factory=list)

    # Context
    affected_models: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)

    # LLM analysis (hybrid approach)
    root_cause: str = ""
    prevention_strategy: str = ""

    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class HyperparameterHistory:
    """Track hyperparameter configurations and their outcomes."""

    model_type: str  # lightgbm, xgboost, catboost, etc.
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Outcome
    cv_score: float = 0.0
    success: bool = True

    # Issues encountered
    issues: list[str] = field(default_factory=list)

    # Context
    data_size: int = 0
    n_classes: int | None = None
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


def merge_error_pattern_memory(
    existing: list["ErrorPatternMemory"],
    new: list["ErrorPatternMemory"],
) -> list["ErrorPatternMemory"]:
    """Merge error pattern memory entries by (error_type, error_pattern)."""

    def _normalize(entry: Any) -> Optional["ErrorPatternMemory"]:
        if isinstance(entry, ErrorPatternMemory):
            return entry
        if isinstance(entry, dict):
            try:
                return ErrorPatternMemory(**entry)
            except TypeError:
                return None
        return None

    def _uniq(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    merged: dict[tuple[str, str], ErrorPatternMemory] = {}
    for entry in list(existing) + list(new):
        em = _normalize(entry)
        if em is None:
            continue
        key = (em.error_type, em.error_pattern)
        if key not in merged:
            merged[key] = em
            continue
        current = merged[key]
        current.occurrences = current.occurrences + em.occurrences
        current.first_seen = min(current.first_seen, em.first_seen)
        current.last_seen = max(current.last_seen, em.last_seen)
        current.solutions_tried = _uniq(current.solutions_tried + em.solutions_tried)
        current.successful_solutions = _uniq(current.successful_solutions + em.successful_solutions)
        current.affected_models = _uniq(current.affected_models + em.affected_models)
        current.affected_components = _uniq(current.affected_components + em.affected_components)
        if em.root_cause:
            current.root_cause = em.root_cause
        if em.prevention_strategy:
            current.prevention_strategy = em.prevention_strategy
        merged[key] = current

    return list(merged.values())
