"""
State Contracts for Kaggle Agents.

These contracts provide single source of truth for:
- Metrics (evaluation, comparison, deltas)
- Canonical data (paths, hashes, validation)
- Submission format (schema, validation)
- Evaluation fidelity (comparability between experiments)
- Data usage (tracking which assets were used)

All contracts are JSON-serializable for LangGraph checkpointing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@dataclass
class MetricContract:
    """Single source of truth for competition metric.

    CRITICAL: best_score initialized as None, not 0.0!
    For minimize metrics, 0.0 would be "very good" and never update.

    Attributes:
        metric_name: Name of the metric (e.g., "auc", "rmse", "accuracy")
        is_lower_better: True for RMSE/logloss, False for AUC/accuracy
        target_score: Threshold for success (optional)
        best_score: Best score achieved so far (None until first evaluation)
        baseline_score: Baseline score for comparison (optional)
    """

    metric_name: str
    is_lower_better: bool
    target_score: float | None = None
    best_score: float | None = None
    baseline_score: float | None = None

    def compare(self, a: float, b: float) -> int:
        """Compare two scores.

        Returns:
            1 if a is better than b
            -1 if b is better than a
            0 if equal

        Note:
            For minimize (is_lower_better=True), a < b means a is BETTER -> return +1
            For maximize (is_lower_better=False), a > b means a is BETTER -> return +1
        """
        if self.is_lower_better:
            # Minimize: lower is better, so a < b -> a wins -> +1
            return 1 if a < b else (-1 if a > b else 0)
        else:
            # Maximize: higher is better, so a > b -> a wins -> +1
            return 1 if a > b else (-1 if a < b else 0)

    def is_improvement(self, new: float, old: float) -> bool:
        """Check if new score is better than old."""
        return self.compare(new, old) > 0

    def compute_delta(self, new: float, old: float) -> float:
        """Compute delta such that delta > 0 ALWAYS means improvement.

        This standardizes delta_score across all metrics:
        - For minimize metrics: old - new (lower new is better, so positive delta)
        - For maximize metrics: new - old (higher new is better, so positive delta)
        """
        if self.is_lower_better:
            return old - new  # Lower is better, so old - new > 0 means improved
        else:
            return new - old  # Higher is better, so new - old > 0 means improved

    def update_best(self, score: float) -> bool:
        """Update best_score if improved.

        Args:
            score: New score to compare against best

        Returns:
            True if best_score was updated, False otherwise
        """
        import math

        # Validate score is a valid number
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            return False

        if self.best_score is None or self.is_improvement(score, self.best_score):
            self.best_score = score
            return True
        return False

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        direction = "minimize" if self.is_lower_better else "maximize"
        best = f"{self.best_score:.4f}" if self.best_score is not None else "None"
        return f"MetricContract({self.metric_name}, {direction}, best={best})"

    def to_dict(self) -> dict:
        """Serialize for checkpointing (LangGraph compatible)."""
        return {
            "metric_name": self.metric_name,
            "is_lower_better": self.is_lower_better,
            "target_score": self.target_score,
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MetricContract:
        """Deserialize from checkpoint."""
        return cls(
            metric_name=data["metric_name"],
            is_lower_better=data["is_lower_better"],
            target_score=data.get("target_score"),
            best_score=data.get("best_score"),
            baseline_score=data.get("baseline_score"),
        )


@dataclass
class CanonicalDataContract:
    """Immutable contract for canonical data paths.

    Provides fingerprinting via MD5 hashes to detect:
    - Folds changes (folds_hash)
    - Target changes (y_hash)
    - Row order/ID changes (train_ids_hash)
    - Schema changes (train_schema_hash)

    Attributes:
        canonical_dir: Directory containing canonical data files
        train_ids_path: Path to train_ids.npy
        y_path: Path to y.npy
        folds_path: Path to folds.npy
        feature_cols_path: Path to feature_cols.json
        metadata_path: Path to metadata.json
        n_train: Number of training samples
        n_test: Number of test samples
        n_folds: Number of CV folds
        id_col: Name of ID column
        target_col: Name of target column
        is_classification: Whether this is a classification task
        folds_hash: MD5 hash of folds.npy
        y_hash: MD5 hash of y.npy
        train_ids_hash: MD5 hash of train_ids.npy
        train_schema_hash: MD5 hash of (columns + dtypes)
    """

    canonical_dir: str
    train_ids_path: str
    y_path: str
    folds_path: str
    feature_cols_path: str
    metadata_path: str

    # Core metadata (loaded once, immutable)
    n_train: int
    n_test: int
    n_folds: int
    id_col: str
    target_col: str
    is_classification: bool

    # Dataset fingerprinting
    folds_hash: str
    y_hash: str
    train_ids_hash: str
    train_schema_hash: str

    def validate(self) -> tuple[bool, list[str]]:
        """Validate all canonical files exist and match checksums.

        Uses compute_array_hash() for consistency with contract creation,
        which handles object/string arrays correctly.

        Returns:
            Tuple of (is_valid, list of violations)
        """
        import numpy as np

        violations = []

        # Check file existence
        for attr in ["train_ids_path", "y_path", "folds_path"]:
            path = Path(getattr(self, attr))
            if not path.exists():
                violations.append(f"Missing: {path}")

        # Verify hashes if files exist
        # Use compute_array_hash() for consistency with contract creation
        # Use allow_pickle=True for object arrays (common for ID columns)
        folds_path = Path(self.folds_path)
        if folds_path.exists():
            arr = np.load(folds_path, allow_pickle=True)
            actual_hash = self.compute_array_hash(arr)
            if actual_hash != self.folds_hash:
                violations.append(
                    f"folds.npy hash mismatch: {actual_hash[:8]}... != {self.folds_hash[:8]}..."
                )

        y_path = Path(self.y_path)
        if y_path.exists():
            arr = np.load(y_path, allow_pickle=True)
            actual_hash = self.compute_array_hash(arr)
            if actual_hash != self.y_hash:
                violations.append(
                    f"y.npy hash mismatch: {actual_hash[:8]}... != {self.y_hash[:8]}..."
                )

        train_ids_path = Path(self.train_ids_path)
        if train_ids_path.exists():
            arr = np.load(train_ids_path, allow_pickle=True)
            actual_hash = self.compute_array_hash(arr)
            if actual_hash != self.train_ids_hash:
                violations.append(
                    f"train_ids.npy hash mismatch: {actual_hash[:8]}... != {self.train_ids_hash[:8]}..."
                )

        return len(violations) == 0, violations

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "canonical_dir": self.canonical_dir,
            "train_ids_path": self.train_ids_path,
            "y_path": self.y_path,
            "folds_path": self.folds_path,
            "feature_cols_path": self.feature_cols_path,
            "metadata_path": self.metadata_path,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_folds": self.n_folds,
            "id_col": self.id_col,
            "target_col": self.target_col,
            "is_classification": self.is_classification,
            "folds_hash": self.folds_hash,
            "y_hash": self.y_hash,
            "train_ids_hash": self.train_ids_hash,
            "train_schema_hash": self.train_schema_hash,
        }

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        task = "classification" if self.is_classification else "regression"
        return (
            f"CanonicalDataContract(n_train={self.n_train}, n_folds={self.n_folds}, "
            f"{task}, folds_hash={self.folds_hash[:8]}...)"
        )

    @classmethod
    def from_dict(cls, data: dict) -> CanonicalDataContract:
        """Deserialize from checkpoint."""
        return cls(**data)

    @staticmethod
    def compute_array_hash(arr: "np.ndarray") -> str:
        """Compute deterministic MD5 hash of a numpy array.

        Handles both numeric and object/string arrays correctly.
        For object arrays (common for ID columns and string targets),
        tobytes() would encode PyObject pointer addresses which change
        on every load. Instead, we serialize to a stable string format.

        Args:
            arr: Numpy array to hash

        Returns:
            MD5 hexdigest of the array content
        """
        import numpy as np

        # For object/string arrays, convert to stable string representation
        if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
            # Convert to string array and join with delimiter
            # Use repr() to handle None, NaN, and special characters
            stable_repr = "\x00".join(repr(x) for x in arr.flat)
            return hashlib.md5(stable_repr.encode("utf-8")).hexdigest()

        # For numeric arrays, tobytes() is stable and efficient
        return hashlib.md5(arr.tobytes()).hexdigest()

    @staticmethod
    def compute_schema_hash(columns: list[str], dtypes: list[str]) -> str:
        """Compute MD5 hash of schema (columns + dtypes)."""
        schema_str = ",".join(f"{c}:{d}" for c, d in zip(columns, dtypes))
        return hashlib.md5(schema_str.encode()).hexdigest()


@dataclass
class SubmissionContract:
    """Contract for submission format.

    Validates that submissions match expected schema:
    - Correct ID column
    - Correct target columns
    - Correct number of rows
    - Correct format type (label, wide, multi_target)

    Attributes:
        id_col: Name of ID column
        target_cols: List of target column names
        expected_rows: Expected number of rows in submission
        format_type: Type of submission format
        class_order: Order of classes for classification (optional)
        sample_submission_path: Path to sample_submission.csv
    """

    id_col: str
    target_cols: list[str]
    expected_rows: int
    format_type: Literal["label", "wide", "multi_target"]
    class_order: list[str] | None
    sample_submission_path: str

    def validate_submission(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate submission matches contract.

        Args:
            df: Submission DataFrame to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        if len(df) != self.expected_rows:
            errors.append(f"Row count: {len(df)} != expected {self.expected_rows}")

        if self.id_col not in df.columns:
            errors.append(f"Missing ID column: {self.id_col}")

        for col in self.target_cols:
            if col not in df.columns:
                errors.append(f"Missing target column: {col}")

        # Check for NaN values in predictions
        for col in self.target_cols:
            if col in df.columns and df[col].isna().any():
                n_nan = df[col].isna().sum()
                errors.append(f"NaN values in {col}: {n_nan} rows")

        return len(errors) == 0, errors

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"SubmissionContract({self.format_type}, "
            f"rows={self.expected_rows}, cols={len(self.target_cols)})"
        )

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "id_col": self.id_col,
            "target_cols": self.target_cols,
            "expected_rows": self.expected_rows,
            "format_type": self.format_type,
            "class_order": self.class_order,
            "sample_submission_path": self.sample_submission_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SubmissionContract:
        """Deserialize from checkpoint."""
        return cls(**data)


@dataclass
class EvalFidelityContract:
    """Contract for evaluation fidelity level.

    MLE-STAR uses subsampling in fast phases, then removes it for final.
    Scores from different fidelities should NOT be directly compared.

    Attributes:
        fidelity: Level of evaluation fidelity
        timeout_s: Timeout in seconds (optional)
        max_rows: Maximum rows to use (None = all rows)
        max_features: Maximum features to use (optional)
        n_folds_used: Number of folds used (may differ from canonical n_folds)
    """

    fidelity: Literal["debug", "fast_cv", "full_cv", "train_all"]
    timeout_s: int | None = None
    max_rows: int | None = None
    max_features: int | None = None
    n_folds_used: int | None = None

    def is_comparable_to(self, other: EvalFidelityContract) -> bool:
        """Check if two fidelities can be compared directly.

        Only scores from the same fidelity level and row count
        should be compared.
        """
        return self.fidelity == other.fidelity and self.max_rows == other.max_rows

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        parts = [f"EvalFidelityContract({self.fidelity}"]
        if self.max_rows:
            parts.append(f", max_rows={self.max_rows}")
        if self.n_folds_used:
            parts.append(f", folds={self.n_folds_used}")
        parts.append(")")
        return "".join(parts)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "fidelity": self.fidelity,
            "timeout_s": self.timeout_s,
            "max_rows": self.max_rows,
            "max_features": self.max_features,
            "n_folds_used": self.n_folds_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalFidelityContract:
        """Deserialize from checkpoint."""
        return cls(**data)


@dataclass
class DataUsageContract:
    """Contract for data assets usage tracking.

    MLE-STAR checks if LLM actually uses all provided data files.
    LLMs often ignore auxiliary files and only load train.csv.

    Attributes:
        provided_assets: All files in input directory
        required_assets: Files that SHOULD be used
        used_assets_evidence: Evidence of usage {path: log_ref}
    """

    provided_assets: list[str]
    required_assets: list[str]
    used_assets_evidence: dict[str, str] = field(default_factory=dict)

    def check_data_usage(self) -> tuple[bool, list[str]]:
        """Check if all required assets were used.

        Returns:
            Tuple of (all_used, list of unused assets)
        """
        unused = [a for a in self.required_assets if a not in self.used_assets_evidence]
        return len(unused) == 0, unused

    def record_usage(self, asset_path: str, evidence: str) -> None:
        """Record that an asset was used.

        Args:
            asset_path: Path to the asset
            evidence: Reference to log/code showing usage
        """
        self.used_assets_evidence[asset_path] = evidence

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        n_used = len(self.used_assets_evidence)
        n_required = len(self.required_assets)
        status = "OK" if n_used >= n_required else f"MISSING {n_required - n_used}"
        return f"DataUsageContract({n_used}/{n_required} assets, {status})"

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "provided_assets": self.provided_assets,
            "required_assets": self.required_assets,
            "used_assets_evidence": self.used_assets_evidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataUsageContract:
        """Deserialize from checkpoint."""
        return cls(
            provided_assets=data["provided_assets"],
            required_assets=data["required_assets"],
            used_assets_evidence=data.get("used_assets_evidence", {}),
        )


# Helper functions for contract creation


def create_metric_contract(
    metric_name: str,
    is_lower_better: bool | None = None,
    target_score: float | None = None,
) -> MetricContract:
    """Create a MetricContract with sensible defaults.

    If is_lower_better is not specified, it is inferred from the metric name.

    Args:
        metric_name: Name of the metric
        is_lower_better: Whether lower is better (optional, inferred if not given)
        target_score: Target score threshold (optional)

    Returns:
        MetricContract instance
    """
    # Infer is_lower_better from metric name if not specified
    if is_lower_better is None:
        lower_better_metrics = {
            "rmse",
            "mse",
            "mae",
            "mape",
            "logloss",
            "log_loss",
            "cross_entropy",
            "error",
            "loss",
        }
        metric_lower = metric_name.lower()
        is_lower_better = any(m in metric_lower for m in lower_better_metrics)

    return MetricContract(
        metric_name=metric_name,
        is_lower_better=is_lower_better,
        target_score=target_score,
    )


def create_submission_contract_from_sample(
    sample_submission_path: str,
) -> SubmissionContract:
    """Create a SubmissionContract from sample_submission.csv.

    Args:
        sample_submission_path: Path to sample_submission.csv

    Returns:
        SubmissionContract instance
    """
    import pandas as pd

    sample_sub = pd.read_csv(sample_submission_path)
    cols = sample_sub.columns.tolist()

    if len(cols) < 2:
        raise ValueError(
            f"sample_submission must have at least 2 columns (id + target), got: {cols}"
        )

    id_col = cols[0]
    target_cols = cols[1:]
    expected_rows = len(sample_sub)

    # Determine format type
    if len(target_cols) == 1:
        format_type = "label"
        class_order = None
    else:
        # Check if binary (multi-label) or continuous (multi-target)
        sample_values = sample_sub[target_cols].iloc[:100]
        is_binary = sample_values.isin([0, 1, 0.0, 1.0]).all().all()
        format_type = "wide" if is_binary else "multi_target"
        class_order = target_cols

    return SubmissionContract(
        id_col=id_col,
        target_cols=target_cols,
        expected_rows=expected_rows,
        format_type=format_type,
        class_order=class_order,
        sample_submission_path=sample_submission_path,
    )
