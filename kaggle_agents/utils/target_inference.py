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

    @property
    def is_multi_output(self) -> bool:
        """Check if this is a multi-output problem."""
        return self.target_type in ("multi_label", "multi_target")

    @property
    def n_targets(self) -> int:
        """Number of target columns."""
        return len(self.target_cols)


def infer_target_columns(sample_submission_path: str | Path) -> TargetInfo:
    """
    Detect target columns and type from sample_submission.csv.

    Logic:
    1. If only 1 target column (after ID) -> single target
    2. If multiple columns with binary values (0/1) -> multi_label
    3. If multiple columns with continuous values -> multi_target

    Args:
        sample_submission_path: Path to sample_submission.csv

    Returns:
        TargetInfo with detected target columns and type

    Examples:
        >>> info = infer_target_columns("sample_submission.csv")
        >>> print(info.target_type)  # "single", "multi_label", or "multi_target"
        >>> print(info.target_cols)  # ["target"] or ["class_0", "class_1", ...]
    """
    sample_sub = pd.read_csv(sample_submission_path)
    cols = sample_sub.columns.tolist()

    if len(cols) < 2:
        raise ValueError(
            f"sample_submission must have at least 2 columns (id + target), got: {cols}"
        )

    id_col = cols[0]
    target_cols = cols[1:]

    # Single target
    if len(target_cols) == 1:
        return TargetInfo(
            target_cols=target_cols,
            target_type="single",
            id_col=id_col,
        )

    # Multiple targets - check if binary (multi-label) or continuous (multi-target)
    sample_values = sample_sub[target_cols].iloc[:100]

    # Check if values are binary (0/1)
    is_binary = sample_values.isin([0, 1, 0.0, 1.0]).all().all()

    if is_binary:
        return TargetInfo(
            target_cols=target_cols,
            target_type="multi_label",
            id_col=id_col,
        )

    return TargetInfo(
        target_cols=target_cols,
        target_type="multi_target",
        id_col=id_col,
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
    predictions: np.ndarray,  # type: ignore
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
    import numpy as np

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
