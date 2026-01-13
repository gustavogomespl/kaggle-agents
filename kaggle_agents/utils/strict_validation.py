"""
Strict validation module for kaggle-agents.

This module provides centralized validation logic with configurable strictness
to detect broken components early (fail-fast) rather than in the ensemble phase.

Environment Variables:
    KAGGLE_AGENTS_STRICT_MODE: Enable hard failures (default: 0)
    KAGGLE_AGENTS_REQUIRE_CLASS_ORDER: Require class_order.npy (default: 0)
    KAGGLE_AGENTS_REQUIRE_TRAIN_IDS: Require train_ids.npy (default: 0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class StrictValidationConfig:
    """Environment-based configuration for validation strictness."""

    strict_mode: bool = False
    require_class_order: bool = False
    require_train_ids: bool = False
    probability_tolerance: float = 0.01
    empty_row_threshold: float = 0.0  # Fraction of empty rows allowed (0 = none)

    @classmethod
    def from_env(cls) -> StrictValidationConfig:
        """Load configuration from environment variables."""
        return cls(
            strict_mode=os.getenv("KAGGLE_AGENTS_STRICT_MODE", "0").lower()
            in {"1", "true", "yes"},
            require_class_order=os.getenv(
                "KAGGLE_AGENTS_REQUIRE_CLASS_ORDER", "0"
            ).lower()
            in {"1", "true", "yes"},
            require_train_ids=os.getenv("KAGGLE_AGENTS_REQUIRE_TRAIN_IDS", "0").lower()
            in {"1", "true", "yes"},
        )


@dataclass
class ValidationResult:
    """Result of model artifact validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_verified: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (doesn't affect validity in lenient mode)."""
        self.warnings.append(msg)


def validate_model_artifacts(
    working_dir: Path,
    component_name: str,
    expected_n_train: int | None = None,
    expected_n_test: int | None = None,
    expected_class_order: Sequence[str] | None = None,
    problem_type: str = "classification",
    config: StrictValidationConfig | None = None,
) -> ValidationResult:
    """
    Post-execution validation of model artifacts.

    Validates:
    1. OOF file exists and has correct shape
    2. Test file exists and has correct shape
    3. Class order file exists (if multiclass and required)
    4. Probabilities in [0, 1]
    5. No empty rows (sum=0)
    6. Multiclass: rows sum to 1.0

    Args:
        working_dir: Competition working directory
        component_name: Name of the component (e.g., "xgboost_baseline")
        expected_n_train: Expected number of training samples
        expected_n_test: Expected number of test samples
        expected_class_order: Expected class order from sample_submission
        problem_type: "classification", "regression", or "multilabel"
        config: Validation configuration (loads from env if None)

    Returns:
        ValidationResult with errors, warnings, and verified files
    """
    if config is None:
        config = StrictValidationConfig.from_env()

    result = ValidationResult()
    models_dir = working_dir / "models"

    # 1. Check OOF file exists
    oof_path = models_dir / f"oof_{component_name}.npy"
    if not oof_path.exists():
        result.add_error(f"Missing OOF file: {oof_path.name}")
        return result  # Can't continue without OOF

    result.files_verified.append(oof_path.name)

    # 2. Check test file exists
    test_path = models_dir / f"test_{component_name}.npy"
    if not test_path.exists():
        result.add_error(f"Missing test file: {test_path.name}")
        return result  # Can't continue without test predictions

    result.files_verified.append(test_path.name)

    # 3. Load and validate OOF predictions
    try:
        oof_preds = np.load(oof_path)
    except Exception as e:
        result.add_error(f"Failed to load OOF file: {e}")
        return result

    # 4. Load and validate test predictions
    try:
        test_preds = np.load(test_path)
    except Exception as e:
        result.add_error(f"Failed to load test file: {e}")
        return result

    # 5. Validate OOF shape
    if expected_n_train is not None:
        if oof_preds.shape[0] != expected_n_train:
            result.add_error(
                f"OOF row count mismatch: {oof_preds.shape[0]} vs expected {expected_n_train}"
            )

    # 6. Validate test shape
    if expected_n_test is not None:
        if test_preds.shape[0] != expected_n_test:
            result.add_error(
                f"Test row count mismatch: {test_preds.shape[0]} vs expected {expected_n_test}"
            )

    # 7. For classification, validate probabilities
    # Debug: Log the problem_type being used for validation
    print(f"   [VALIDATION] problem_type={problem_type}, validating {component_name}")

    if problem_type in ("classification", "multilabel"):
        # Check range [0, 1]
        oof_min, oof_max = oof_preds.min(), oof_preds.max()
        test_min, test_max = test_preds.min(), test_preds.max()

        if oof_min < -config.probability_tolerance or oof_max > 1 + config.probability_tolerance:
            result.add_error(
                f"OOF probabilities out of range: min={oof_min:.4f}, max={oof_max:.4f}"
            )

        if test_min < -config.probability_tolerance or test_max > 1 + config.probability_tolerance:
            result.add_error(
                f"Test probabilities out of range: min={test_min:.4f}, max={test_max:.4f}"
            )

        # Check for NaN/Inf
        if np.any(~np.isfinite(oof_preds)):
            result.add_error("OOF predictions contain NaN or Inf values")

        if np.any(~np.isfinite(test_preds)):
            result.add_error("Test predictions contain NaN or Inf values")

        # Check for empty rows (all zeros - indicates unfilled OOF)
        # For classification: a row of all zeros means no prediction was made
        if oof_preds.ndim > 1:
            empty_oof_rows = np.sum(oof_preds.sum(axis=1) == 0)
        else:
            # For 1D predictions (binary classification), 0 is a valid probability
            # Only flag if prediction is EXACTLY 0.0 AND this is truly empty
            empty_oof_rows = 0  # 1D classification predictions of 0 are valid

        if empty_oof_rows > 0:
            empty_fraction = empty_oof_rows / oof_preds.shape[0]
            if empty_fraction > config.empty_row_threshold:
                result.add_warning(
                    f"{empty_oof_rows} OOF rows have all-zero predictions ({empty_fraction:.1%})"
                )
                if config.strict_mode:
                    result.add_error(
                        f"Empty OOF rows exceed threshold: {empty_oof_rows} rows"
                    )

        # For multiclass (not multilabel), check row sums = 1.0
        if (
            problem_type == "classification"
            and oof_preds.ndim > 1
            and oof_preds.shape[1] > 2
        ):
            oof_row_sums = oof_preds.sum(axis=1)
            bad_oof_rows = np.sum(np.abs(oof_row_sums - 1.0) > config.probability_tolerance)
            if bad_oof_rows > 0:
                result.add_warning(
                    f"{bad_oof_rows} OOF rows do not sum to 1.0 (not normalized)"
                )

            test_row_sums = test_preds.sum(axis=1)
            bad_test_rows = np.sum(np.abs(test_row_sums - 1.0) > config.probability_tolerance)
            if bad_test_rows > 0:
                result.add_warning(
                    f"{bad_test_rows} test rows do not sum to 1.0 (not normalized)"
                )

    # 7b. For regression, validate prediction sanity
    elif problem_type == "regression":
        oof_min, oof_max = oof_preds.min(), oof_preds.max()
        test_min, test_max = test_preds.min(), test_preds.max()

        # Check for NaN/Inf values
        if np.any(~np.isfinite(oof_preds)):
            result.add_error("OOF predictions contain NaN or Inf values")

        if np.any(~np.isfinite(test_preds)):
            result.add_error("Test predictions contain NaN or Inf values")

        # Warn about extreme prediction ranges (may indicate undertrained model)
        pred_range = oof_max - oof_min
        if pred_range > 1000:
            result.add_warning(
                f"Large OOF prediction range: {oof_min:.2f} to {oof_max:.2f} (range={pred_range:.2f})"
            )

        test_range = test_max - test_min
        if test_range > 1000:
            result.add_warning(
                f"Large test prediction range: {test_min:.2f} to {test_max:.2f} (range={test_range:.2f})"
            )

        # Warn about negative predictions (invalid for many regression targets)
        # Common cases: prices, fares, counts, durations - all should be >= 0
        if oof_min < 0:
            result.add_warning(
                f"Negative OOF predictions detected: min={oof_min:.4f}"
            )
        if test_min < 0:
            result.add_warning(
                f"Negative test predictions detected: min={test_min:.4f}"
            )

        # Check for constant predictions (model not learning)
        if np.std(oof_preds) < 1e-6:
            result.add_error(
                f"OOF predictions are constant (std={np.std(oof_preds):.2e})"
            )

    # 8. Check class order file (for multiclass)
    if expected_class_order is not None and len(expected_class_order) > 2:
        class_order_path = models_dir / f"class_order_{component_name}.npy"
        global_class_order_path = models_dir / "class_order.npy"

        class_order_found = False
        if class_order_path.exists():
            try:
                saved_order = np.load(class_order_path, allow_pickle=True).tolist()
                if saved_order != list(expected_class_order):
                    result.add_error(
                        f"Class order mismatch: model has {saved_order[:3]}..., "
                        f"expected {list(expected_class_order)[:3]}..."
                    )
                else:
                    result.files_verified.append(class_order_path.name)
                    class_order_found = True
            except Exception as e:
                result.add_warning(f"Failed to verify class order: {e}")
        elif global_class_order_path.exists():
            try:
                saved_order = np.load(global_class_order_path, allow_pickle=True).tolist()
                if saved_order != list(expected_class_order):
                    result.add_error(
                        f"Global class order mismatch: has {saved_order[:3]}..., "
                        f"expected {list(expected_class_order)[:3]}..."
                    )
                else:
                    result.files_verified.append("class_order.npy")
                    class_order_found = True
            except Exception as e:
                result.add_warning(f"Failed to verify global class order: {e}")

        if not class_order_found:
            msg = f"Missing class_order file for {component_name} (multiclass alignment unknown)"
            if config.require_class_order:
                result.add_error(msg)
            else:
                result.add_warning(msg)

    # 9. Check train IDs file (optional)
    if config.require_train_ids:
        train_ids_path = models_dir / f"train_ids_{component_name}.npy"
        if not train_ids_path.exists():
            result.add_error(f"Missing train IDs file: {train_ids_path.name}")
        else:
            result.files_verified.append(train_ids_path.name)

    return result


def validate_prediction_quality(
    preds: np.ndarray,
    y_true: np.ndarray | None = None,
    problem_type: str = "classification",
) -> tuple[bool, list[str]]:
    """
    Detect random/broken predictions.

    Checks:
    - Constant predictions (all same value)
    - Near-uniform predictions (suspiciously close to 1/n_classes)
    - AUC close to 0.5 if y_true provided (indicates random guessing)

    Args:
        preds: Prediction array
        y_true: Ground truth labels (optional, enables AUC check)
        problem_type: "classification" or "regression"

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues: list[str] = []

    # Handle edge cases
    if preds is None or preds.size == 0:
        issues.append("Predictions array is empty or None")
        return False, issues

    # Check for constant predictions
    if preds.ndim == 1:
        unique_vals = np.unique(preds)
        if len(unique_vals) <= 3:
            issues.append(
                f"Near-constant predictions: only {len(unique_vals)} unique values"
            )
    else:
        # For multiclass, check if all rows are nearly identical
        row_variance = np.var(preds, axis=0)
        if np.max(row_variance) < 0.001:
            issues.append(
                "All prediction rows are nearly identical (variance < 0.001)"
            )

    # Check for uniform predictions (1/n_classes for all) - multiclass only
    if preds.ndim > 1 and preds.shape[1] > 1:
        expected_uniform = 1.0 / preds.shape[1]
        mean_preds = preds.mean(axis=0)
        max_deviation = np.max(np.abs(mean_preds - expected_uniform))
        if max_deviation < 0.05:
            issues.append(
                f"Predictions are suspiciously uniform "
                f"(max deviation from {expected_uniform:.3f} is {max_deviation:.4f})"
            )

    # Check AUC if ground truth provided
    if y_true is not None and problem_type == "classification":
        try:
            from sklearn.metrics import roc_auc_score

            # Handle different prediction shapes
            if preds.ndim > 1 and preds.shape[1] == 2:
                # Binary with both classes - use positive class
                auc = roc_auc_score(y_true, preds[:, 1])
            elif preds.ndim == 1:
                # Binary with single column
                auc = roc_auc_score(y_true, preds)
            elif preds.ndim > 1 and preds.shape[1] > 2:
                # Multiclass
                auc = roc_auc_score(
                    y_true, preds, multi_class="ovr", average="weighted"
                )
            else:
                auc = None

            if auc is not None and 0.48 < auc < 0.52:
                issues.append(
                    f"AUC is {auc:.4f} - suspiciously close to random guessing (0.5)"
                )

        except Exception:
            # If AUC calculation fails, don't add an issue
            pass

    return len(issues) == 0, issues


def quick_oof_validation(
    working_path: Path,
    component_name: str | None = None,
    artifacts_created: list[str] | None = None,
) -> list[str]:
    """
    Quick post-execution validation of OOF files.

    Used by code_executor to detect issues immediately after execution.

    Args:
        working_path: Working directory
        component_name: Component name (auto-detected if None)
        artifacts_created: List of created artifacts (for auto-detection)

    Returns:
        List of issues found (empty if valid)
    """
    issues: list[str] = []
    models_dir = working_path / "models"

    if not models_dir.exists():
        issues.append("models/ directory not found")
        return issues

    # Auto-detect component name from artifacts if not provided
    if component_name is None and artifacts_created:
        for artifact in artifacts_created:
            if artifact.startswith("oof_") and artifact.endswith(".npy"):
                component_name = artifact[4:-4]  # Remove "oof_" and ".npy"
                break

    if component_name is None:
        # Look for any OOF file
        oof_files = list(models_dir.glob("oof_*.npy"))
        if not oof_files:
            issues.append("No OOF prediction files found in models/")
            return issues
        component_name = oof_files[0].stem.replace("oof_", "")

    # Check OOF file
    oof_path = models_dir / f"oof_{component_name}.npy"
    if not oof_path.exists():
        issues.append(f"OOF file not found: oof_{component_name}.npy")
    else:
        try:
            oof = np.load(oof_path)
            if np.any(~np.isfinite(oof)):
                issues.append("OOF contains NaN or Inf values")
            if oof.ndim > 1:
                empty_rows = np.sum(oof.sum(axis=1) == 0)
                if empty_rows > 0:
                    issues.append(f"{empty_rows} OOF rows are all zeros (unfilled)")
        except Exception as e:
            issues.append(f"Failed to load OOF: {e}")

    # Check test file
    test_path = models_dir / f"test_{component_name}.npy"
    if not test_path.exists():
        issues.append(f"Test file not found: test_{component_name}.npy")
    else:
        try:
            test = np.load(test_path)
            if np.any(~np.isfinite(test)):
                issues.append("Test predictions contain NaN or Inf values")
        except Exception as e:
            issues.append(f"Failed to load test predictions: {e}")

    return issues
