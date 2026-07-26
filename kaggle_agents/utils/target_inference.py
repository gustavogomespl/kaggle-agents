"""
Target Column Inference for Kaggle Competitions.

This module provides automatic detection of:
- Single-target classification/regression
- Multi-label classification (independent binary targets)
- Multi-target regression (multiple continuous targets)

The target type affects:
- Loss function (softmax vs sigmoid)
- Metric calculation (per-class vs averaged)
- Submission validation (row sums, value ranges)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


TargetType = Literal["single", "multi_label", "multi_target"]


@dataclass
class TargetInfo:
    """
    Information about target columns detected from sample_submission.

    Attributes:
        target_cols: List of target column names (can have multiple)
        target_type: Type of target ("single", "multi_label", "multi_target")
        id_col: Name of the ID column
    """

    target_cols: list[str]
    target_type: TargetType
    id_col: str
    type_source: str = "submission_schema"

    @property
    def is_multi_output(self) -> bool:
        """Check if this is a multi-output problem."""
        return self.target_type in ("multi_label", "multi_target")

    @property
    def n_targets(self) -> int:
        """Number of target columns."""
        return len(self.target_cols)


class TargetInferenceError(ValueError):
    """Raised when public schema/labels cannot prove a target contract."""


def infer_target_type_from_train(
    train_data: pd.DataFrame,
    target_cols: list[str],
    *,
    problem_type: str | None = None,
    explicit_target_type: TargetType | None = None,
) -> tuple[TargetType, str]:
    """Classify an ordered training-target matrix from real public labels.

    Sample-submission values are deliberately excluded: templates commonly
    contain all-zero placeholders and therefore carry no semantic evidence
    about multilabel classification versus multi-target regression.
    """
    if not target_cols:
        raise TargetInferenceError("Target contract must contain at least one column")
    if len(target_cols) != len(set(target_cols)):
        raise TargetInferenceError("Target contract contains duplicate columns")
    missing = [column for column in target_cols if column not in train_data.columns]
    if missing:
        raise TargetInferenceError(
            f"Training data is missing declared target columns: {missing}"
        )
    if len(target_cols) == 1:
        if explicit_target_type not in (None, "single"):
            raise TargetInferenceError(
                f"Explicit target_type={explicit_target_type!r} requires multiple targets"
            )
        return "single", (
            "explicit_target_type"
            if explicit_target_type == "single"
            else "single_training_target"
        )

    targets = train_data.loc[:, target_cols]
    if targets.isna().any().any():
        raise TargetInferenceError(
            "Training target columns contain missing labels; canonical scoring "
            "cannot establish a complete target matrix"
        )

    numeric_targets = targets.apply(pd.to_numeric, errors="coerce")
    all_numeric = bool(numeric_targets.notna().all().all())
    finite_numeric = bool(
        all_numeric
        and np.isfinite(numeric_targets.to_numpy(dtype=float)).all()
    )
    binary_indicators = bool(
        finite_numeric
        and all(
            set(np.unique(numeric_targets[column].to_numpy(dtype=float))).issubset(
                {0.0, 1.0}
            )
            for column in target_cols
        )
    )

    normalized_problem = (
        str(problem_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    declares_multilabel = any(
        marker in normalized_problem
        for marker in ("multi_label", "multilabel")
    )
    declares_classification = (
        declares_multilabel or "classification" in normalized_problem
    )
    declares_multi_target = any(
        marker in normalized_problem
        for marker in ("multi_target", "multioutput_regression")
    )
    declares_regression = (
        declares_multi_target
        or "regression" in normalized_problem
        or "forecast" in normalized_problem
    )
    if declares_classification and declares_regression:
        raise TargetInferenceError(
            f"Conflicting public problem type for target inference: {problem_type!r}"
        )

    requested = explicit_target_type
    if requested is None and declares_multilabel:
        requested = "multi_label"
    elif requested is None and declares_multi_target:
        requested = "multi_target"

    if requested == "single":
        raise TargetInferenceError(
            "Explicit single-target type is incompatible with multiple target columns"
        )
    if requested == "multi_label":
        if not binary_indicators:
            raise TargetInferenceError(
                "Explicit multi_label contract requires real training labels to "
                "be independent binary indicators"
            )
        return "multi_label", "explicit_target_type"
    if requested == "multi_target":
        if not finite_numeric:
            raise TargetInferenceError(
                "Explicit multi_target contract requires finite numeric training labels"
            )
        return "multi_target", "explicit_target_type"

    if declares_classification:
        if not binary_indicators:
            raise TargetInferenceError(
                "Multiple classification targets are not binary indicator "
                "columns; multi-output multiclass semantics are ambiguous"
            )
        return "multi_label", "public_problem_type_and_training_labels"
    if declares_regression:
        if not finite_numeric:
            raise TargetInferenceError(
                "Declared regression targets must be finite numeric columns"
            )
        return "multi_target", "public_problem_type_and_training_labels"

    if binary_indicators:
        return "multi_label", "observed_binary_training_matrix"
    if finite_numeric:
        values = numeric_targets.to_numpy(dtype=float)
        integer_like = bool(np.allclose(values, np.round(values)))
        cardinality_limit = max(20, int(np.sqrt(len(targets))))
        continuous_like = (not integer_like) or any(
            int(numeric_targets[column].nunique(dropna=False))
            > cardinality_limit
            for column in target_cols
        )
        if continuous_like:
            return "multi_target", "observed_continuous_training_matrix"

    raise TargetInferenceError(
        "Cannot distinguish multi-label classification from multi-target "
        "regression using real training labels alone. Supply an explicit public "
        "problem_type/target_type contract."
    )


def infer_target_columns(
    sample_submission_path: str | Path,
    *,
    train_data: str | Path | pd.DataFrame | None = None,
    problem_type: str | None = None,
    target_col: str | None = None,
    target_cols: list[str] | None = None,
    target_type: TargetType | None = None,
) -> TargetInfo:
    """
    Resolve ordered training targets from public submission/train contracts.

    Logic:
    1. If only 1 target column (after ID) -> single target
    2. Preserve sample-submission column order only when those columns exist in
       real training labels
    3. Classify multiple targets from training values/dtypes plus an optional
       public problem-type contract

    Args:
        sample_submission_path: Path to sample_submission.csv

    Returns:
        TargetInfo with detected target columns and type

    Examples:
        >>> info = infer_target_columns("sample_submission.csv")
        >>> print(info.target_type)  # "single", "multi_label", or "multi_target"
        >>> print(info.target_cols)  # ["target"] or ["class_0", "class_1", ...]
    """
    sample_sub = pd.read_csv(sample_submission_path, nrows=0)
    cols = [str(column) for column in sample_sub.columns]

    if len(cols) < 2:
        raise ValueError(
            f"sample_submission must have at least 2 columns (id + target), got: {cols}"
        )

    id_col = cols[0]
    submission_target_cols = cols[1:]
    explicit_cols = [
        str(column)
        for column in list(target_cols or [])
        if isinstance(column, str) and column
    ]

    if train_data is None:
        if len(submission_target_cols) != 1 and target_type is None:
            raise TargetInferenceError(
                "A multi-column submission template does not identify "
                "multi_label versus multi_target semantics. Provide real public "
                "training labels and problem_type, or an explicit target_type."
            )
        resolved_cols = explicit_cols or submission_target_cols
        resolved_type = target_type or "single"
        return TargetInfo(
            target_cols=resolved_cols,
            target_type=resolved_type,
            id_col=id_col,
            type_source=(
                "explicit_target_type"
                if target_type is not None
                else "single_submission_column"
            ),
        )

    train_df = (
        train_data.copy()
        if isinstance(train_data, pd.DataFrame)
        else pd.read_csv(Path(train_data))
    )
    if explicit_cols and all(column in train_df.columns for column in explicit_cols):
        if (
            len(submission_target_cols) > 1
            and set(explicit_cols) == set(submission_target_cols)
        ):
            resolved_cols = submission_target_cols
        else:
            resolved_cols = explicit_cols
    elif (
        len(submission_target_cols) > 1
        and all(column in train_df.columns for column in submission_target_cols)
    ):
        resolved_cols = submission_target_cols
    elif target_col and target_col in train_df.columns:
        resolved_cols = [str(target_col)]
    elif (
        len(submission_target_cols) == 1
        and submission_target_cols[0] in train_df.columns
    ):
        resolved_cols = submission_target_cols
    else:
        raise TargetInferenceError(
            "Submission schema does not resolve to training target columns; "
            f"submission outputs={submission_target_cols!r}, "
            f"declared target={target_col!r}"
        )

    resolved_type, type_source = infer_target_type_from_train(
        train_df,
        resolved_cols,
        problem_type=problem_type,
        explicit_target_type=target_type,
    )
    return TargetInfo(
        target_cols=resolved_cols,
        target_type=resolved_type,
        id_col=id_col,
        type_source=type_source,
    )


def get_target_type_constraints(target_type: TargetType) -> str:
    """
    Get constraints/instructions for a specific target type.

    Used to inject into developer prompts.

    Args:
        target_type: The type of target

    Returns:
        String with constraints for code generation
    """
    if target_type == "multi_label":
        return """
## Multi-Label Classification (CRITICAL)

**MANDATORY**: Use sigmoid PER CLASS, NOT softmax:
- Softmax: classes are mutually exclusive (single-label)
- Sigmoid: each class is independent (multi-label)

```python
# CORRECT for multi-label
predictions = torch.sigmoid(logits)  # Independent per class
# or
predictions = 1 / (1 + np.exp(-logits))

# WRONG for multi-label (DO NOT use)
predictions = torch.softmax(logits, dim=1)  # Sum = 1, exclusive classes
```

**Metric**: Log-loss per column, then average:
```python
from sklearn.metrics import log_loss
import numpy as np

scores = [log_loss(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]
final_score = np.mean(scores)
print(f"Final Validation Performance: {final_score:.6f}")
```

**Binary threshold** (if needed for submission):
```python
binary_preds = (predictions > 0.5).astype(int)
```
"""

    if target_type == "multi_target":
        return """
## Multi-Target Regression

Multiple continuous targets require:
- Train one model per target, OR
- Use multi-output regressor

**Metric**: RMSE per column, then average:
```python
import numpy as np

rmse_scores = [np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2)) for i in range(n_targets)]
final_score = np.mean(rmse_scores)
print(f"Final Validation Performance: {final_score:.6f}")
```
"""

    # single
    return """
## Single Target

Standard classification/regression:
- Use softmax for multiclass (probabilities sum to 1)
- Use sigmoid for binary classification
"""


def validate_predictions_shape(
    predictions: np.ndarray,
    target_info: TargetInfo,
    stage: str = "validation",
) -> tuple[bool, str]:
    """
    Validate that predictions have correct shape for target type.

    Args:
        predictions: Prediction array
        target_info: Target information
        stage: "validation" or "submission"

    Returns:
        Tuple of (is_valid, error_message)
    """
    n_targets = target_info.n_targets

    # Check dimensions
    if predictions.ndim == 1:
        if n_targets != 1:
            return False, f"Expected 2D array with {n_targets} columns, got 1D array"
        return True, ""

    if predictions.ndim != 2:
        return False, f"Expected 2D array, got {predictions.ndim}D"

    if predictions.shape[1] != n_targets:
        return False, f"Expected {n_targets} columns, got {predictions.shape[1]}"

    # Check value ranges for multi-label
    if target_info.target_type == "multi_label":
        if predictions.min() < 0 or predictions.max() > 1:
            return False, "Multi-label predictions must be in [0, 1] range (sigmoid probabilities)"

        # Check that rows don't sum to 1 (would indicate softmax was used incorrectly)
        row_sums = predictions.sum(axis=1)
        if np.allclose(row_sums, 1.0, atol=0.01):
            return False, (
                "Multi-label predictions should NOT sum to 1 (use sigmoid, not softmax). "
                "Each class probability should be independent."
            )

    return True, ""
