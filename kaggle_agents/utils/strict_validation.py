"""
Strict validation module for kaggle-agents.

This module provides centralized validation logic with configurable strictness
to detect broken components early (fail-fast) rather than in the ensemble phase.

Environment Variables:
    KAGGLE_AGENTS_STRICT_MODE: Enable hard failures (default: 0)
    KAGGLE_AGENTS_REQUIRE_CLASS_ORDER: Require class_order.npy (default: 0)
    KAGGLE_AGENTS_REQUIRE_TRAIN_IDS: Require train_ids.npy (default: 0)
    KAGGLE_AGENTS_REQUIRE_TEST_IDS: Require test_ids.npy (default: 0)
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
    require_component_class_order: bool = False
    require_train_ids: bool = False
    require_test_ids: bool = False
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
            require_component_class_order=os.getenv(
                "KAGGLE_AGENTS_REQUIRE_COMPONENT_CLASS_ORDER", "0"
            ).lower()
            in {"1", "true", "yes"},
            require_train_ids=os.getenv("KAGGLE_AGENTS_REQUIRE_TRAIN_IDS", "0").lower()
            in {"1", "true", "yes"},
            require_test_ids=os.getenv("KAGGLE_AGENTS_REQUIRE_TEST_IDS", "0").lower()
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


_CLASSIFICATION_PROBLEM_TYPES = {
    "classification",
    "binary_classification",
    "multiclass_classification",
    "tabular_classification",
    "image_classification",
    "audio_classification",
    "text_classification",
}
_MULTILABEL_PROBLEM_TYPES = {
    "multilabel",
    "multi_label",
    "multilabel_classification",
    "multi_label_classification",
}
_REGRESSION_PROBLEM_TYPES = {
    "regression",
    "tabular_regression",
    "image_regression",
    "audio_regression",
    "text_regression",
    "time_series_forecasting",
    "forecasting",
}
_SEQ2SEQ_PROBLEM_TYPES = {
    "seq2seq",
    "seq_to_seq",
    "sequence_to_sequence",
    "text_normalization",
    "translation",
    "summarization",
}


def _normalize_problem_type(problem_type: str) -> str:
    """Map concrete workflow problem types to validation families."""
    normalized = (
        str(problem_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    if normalized in _MULTILABEL_PROBLEM_TYPES:
        return "multilabel"
    if normalized in _CLASSIFICATION_PROBLEM_TYPES:
        return "classification"
    if normalized in _REGRESSION_PROBLEM_TYPES:
        return "regression"
    if normalized in _SEQ2SEQ_PROBLEM_TYPES:
        return "seq2seq"
    return normalized


def _prediction_width(preds: np.ndarray) -> int | None:
    """Return the prediction width for supported 1-D/2-D artifacts."""
    if preds.ndim == 1:
        return 1
    if preds.ndim == 2:
        return int(preds.shape[1])
    return None


def _validate_basic_array(
    preds: np.ndarray,
    *,
    label: str,
    result: ValidationResult,
    allow_text: bool = False,
) -> None:
    """Validate properties that are mandatory for every problem type."""
    if preds.ndim not in {1, 2}:
        result.add_error(
            f"{label} predictions must be a 1-D or 2-D array, got shape {preds.shape}"
        )
        return
    if preds.shape[0] == 0 or (preds.ndim == 2 and preds.shape[1] == 0):
        result.add_error(f"{label} predictions are empty (shape {preds.shape})")
        return
    if not np.issubdtype(preds.dtype, np.number):
        if allow_text:
            if any(value is None for value in preds.reshape(-1).tolist()):
                result.add_error(f"{label} text predictions contain null values")
            return
        result.add_error(
            f"{label} predictions must be numeric, got dtype {preds.dtype}"
        )
        return
    if np.any(~np.isfinite(preds)):
        result.add_error(f"{label} predictions contain NaN or Inf values")


def _validate_id_artifact(
    path: Path,
    *,
    label: str,
    expected_rows: int,
    expected_ids: Sequence[object] | None,
    required: bool,
    result: ValidationResult,
) -> None:
    """Validate the IDs that define prediction row order."""
    if not path.exists():
        if required or expected_ids is not None:
            result.add_error(f"Missing {label} IDs file: {path.name}")
        return
    try:
        ids = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    except Exception as exc:
        result.add_error(f"Failed to load {label} IDs: {exc}")
        return

    if len(ids) != expected_rows:
        result.add_error(
            f"{label} ID row count mismatch: {len(ids)} vs predictions {expected_rows}"
        )
        return
    normalized = [str(value) for value in ids.tolist()]
    if len(set(normalized)) != len(normalized):
        result.add_error(f"{label} IDs contain duplicates")
        return
    if expected_ids is not None:
        expected = [str(value) for value in expected_ids]
        if normalized != expected:
            result.add_error(
                f"{label} IDs do not match canonical IDs in exact row order"
            )
            return
    result.files_verified.append(path.name)


def validate_model_artifacts(
    working_dir: Path,
    component_name: str,
    expected_n_train: int | None = None,
    expected_n_test: int | None = None,
    expected_class_order: Sequence[str] | None = None,
    expected_train_ids: Sequence[object] | None = None,
    expected_test_ids: Sequence[object] | None = None,
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
        expected_train_ids: Canonical IDs in exact OOF row order
        expected_test_ids: Canonical IDs in exact test-prediction row order
        problem_type: "classification", "regression", or "multilabel"
        config: Validation configuration (loads from env if None)

    Returns:
        ValidationResult with errors, warnings, and verified files
    """
    if config is None:
        config = StrictValidationConfig.from_env()

    result = ValidationResult()
    models_dir = working_dir / "models"
    validation_family = _normalize_problem_type(problem_type)
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
        oof_preds = np.load(oof_path, allow_pickle=False)
    except Exception as e:
        result.add_error(f"Failed to load OOF file: {e}")
        return result

    # 4. Load and validate test predictions
    try:
        test_preds = np.load(test_path, allow_pickle=False)
    except Exception as e:
        result.add_error(f"Failed to load test file: {e}")
        return result

    oof_preds = np.asarray(oof_preds)
    test_preds = np.asarray(test_preds)
    allow_text = validation_family == "seq2seq"

    # Temporal forward chaining reserves the oldest block as training-only
    # history. It has no honest OOF prediction, so the full-shape artifact must
    # retain NaN there and validation operates only on the canonical mask.
    oof_eligible_mask = np.ones(oof_preds.shape[0], dtype=bool)
    oof_mask_path = working_dir / "canonical" / "oof_eligible_mask.npy"
    if oof_mask_path.is_file():
        try:
            oof_eligible_mask = np.asarray(
                np.load(oof_mask_path, allow_pickle=False), dtype=bool
            )
        except Exception as exc:
            result.add_error(f"Failed to load canonical OOF eligibility mask: {exc}")
            return result
        if oof_eligible_mask.shape != (oof_preds.shape[0],):
            result.add_error(
                "Canonical OOF eligibility mask shape mismatch: "
                f"{oof_eligible_mask.shape} vs {(oof_preds.shape[0],)}"
            )
            return result
        warmup_oof = oof_preds[~oof_eligible_mask]
        if warmup_oof.size and (
            not np.issubdtype(warmup_oof.dtype, np.number)
            or not np.isnan(warmup_oof).all()
        ):
            result.add_error(
                "Temporal warm-up OOF rows must remain NaN and must not be "
                "fabricated as validation coverage"
            )
    oof_for_validation = oof_preds[oof_eligible_mask]
    if oof_for_validation.shape[0] == 0:
        result.add_error("Canonical OOF eligibility mask selects no rows")
        return result

    _validate_basic_array(
        oof_for_validation,
        label="OOF-eligible",
        result=result,
        allow_text=allow_text,
    )
    _validate_basic_array(
        test_preds, label="Test", result=result, allow_text=allow_text
    )

    # Unsupported dimensions do not have a meaningful row/column contract.
    if oof_preds.ndim not in {1, 2} or test_preds.ndim not in {1, 2}:
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

    oof_width = _prediction_width(oof_preds)
    test_width = _prediction_width(test_preds)
    if oof_width != test_width:
        result.add_error(
            f"OOF/test prediction width mismatch: {oof_width} vs {test_width}"
        )

    # 7. For classification, validate probabilities
    print(
        f"   [VALIDATION] problem_type={problem_type} "
        f"(family={validation_family}), validating {component_name}"
    )

    if validation_family in {"classification", "multilabel"}:
        # Non-finite arrays were already rejected. Avoid deriving misleading
        # extrema or row sums from them.
        arrays_are_finite = bool(
            np.all(np.isfinite(oof_for_validation))
            and np.all(np.isfinite(test_preds))
        )

        # Check range [0, 1]
        if arrays_are_finite:
            oof_min, oof_max = (
                oof_for_validation.min(),
                oof_for_validation.max(),
            )
            test_min, test_max = test_preds.min(), test_preds.max()

            if (
                oof_min < -config.probability_tolerance
                or oof_max > 1 + config.probability_tolerance
            ):
                result.add_error(
                    f"OOF probabilities out of range: "
                    f"min={oof_min:.4f}, max={oof_max:.4f}"
                )

            if (
                test_min < -config.probability_tolerance
                or test_max > 1 + config.probability_tolerance
            ):
                result.add_error(
                    f"Test probabilities out of range: "
                    f"min={test_min:.4f}, max={test_max:.4f}"
                )

        # Check for empty rows (all zeros - indicates unfilled OOF)
        # For classification: a row of all zeros means no prediction was made
        if oof_for_validation.ndim > 1 and arrays_are_finite:
            empty_oof_rows = int(
                np.sum(oof_for_validation.sum(axis=1) == 0)
            )
        else:
            # For 1D predictions (binary classification), 0 is a valid probability
            # Only flag if prediction is EXACTLY 0.0 AND this is truly empty
            empty_oof_rows = 0  # 1D classification predictions of 0 are valid

        if empty_oof_rows > 0:
            empty_fraction = empty_oof_rows / oof_for_validation.shape[0]
            if empty_fraction > config.empty_row_threshold:
                result.add_warning(
                    f"{empty_oof_rows} OOF rows have all-zero predictions ({empty_fraction:.1%})"
                )
                if config.strict_mode:
                    result.add_error(
                        f"Empty OOF rows exceed threshold: {empty_oof_rows} rows"
                    )

        if test_preds.ndim > 1 and arrays_are_finite:
            empty_test_rows = int(np.sum(test_preds.sum(axis=1) == 0))
            if empty_test_rows > 0:
                empty_fraction = empty_test_rows / test_preds.shape[0]
                result.add_warning(
                    f"{empty_test_rows} test rows have all-zero predictions "
                    f"({empty_fraction:.1%})"
                )
                if (
                    config.strict_mode
                    and empty_fraction > config.empty_row_threshold
                ):
                    result.add_error(
                        f"Empty test rows exceed threshold: {empty_test_rows} rows"
                    )

        # For multiclass (not multilabel), check row sums = 1.0
        if (
            validation_family == "classification"
            and oof_for_validation.ndim > 1
            and oof_for_validation.shape[1] > 1
            and arrays_are_finite
        ):
            oof_row_sums = oof_for_validation.sum(axis=1)
            bad_oof_rows = np.sum(np.abs(oof_row_sums - 1.0) > config.probability_tolerance)
            if bad_oof_rows > 0:
                result.add_warning(
                    f"{bad_oof_rows} OOF rows do not sum to 1.0 (not normalized)"
                )
                if config.strict_mode:
                    result.add_error(
                        f"{bad_oof_rows} OOF rows violate the probability-sum contract"
                    )

            test_row_sums = test_preds.sum(axis=1)
            bad_test_rows = np.sum(np.abs(test_row_sums - 1.0) > config.probability_tolerance)
            if bad_test_rows > 0:
                result.add_warning(
                    f"{bad_test_rows} test rows do not sum to 1.0 (not normalized)"
                )
                if config.strict_mode:
                    result.add_error(
                        f"{bad_test_rows} test rows violate the probability-sum contract"
                    )

        if (
            expected_class_order is not None
            and len(expected_class_order) > 1
            and oof_width is not None
            and validation_family in {"classification", "multilabel"}
            and oof_width != len(expected_class_order)
        ):
            result.add_error(
                f"Prediction column count mismatch: {oof_width} vs "
                f"{len(expected_class_order)} expected classes/targets"
            )

    # 7b. For regression, validate prediction sanity
    elif validation_family == "regression":
        if (
            not np.all(np.isfinite(oof_for_validation))
            or not np.all(np.isfinite(test_preds))
        ):
            return result

        oof_min, oof_max = (
            oof_for_validation.min(),
            oof_for_validation.max(),
        )
        test_min, test_max = test_preds.min(), test_preds.max()

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
        if np.std(oof_for_validation) < 1e-6:
            result.add_error(
                "OOF predictions are constant "
                f"(std={np.std(oof_for_validation):.2e})"
            )

    # 8. Check class order file (for multiclass)
    if expected_class_order is not None and len(expected_class_order) > 2:
        class_order_path = models_dir / f"class_order_{component_name}.npy"
        global_class_order_path = models_dir / "class_order.npy"

        class_order_found = False
        if class_order_path.exists():
            try:
                saved_order = np.load(
                    class_order_path,
                    allow_pickle=False,
                ).tolist()
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
        elif (
            not config.require_component_class_order
            and global_class_order_path.exists()
        ):
            try:
                saved_order = np.load(
                    global_class_order_path,
                    allow_pickle=False,
                ).tolist()
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
            qualifier = (
                "component-specific "
                if config.require_component_class_order
                else ""
            )
            msg = (
                f"Missing {qualifier}class_order file for {component_name} "
                "(multiclass alignment unknown)"
            )
            if config.require_class_order:
                result.add_error(msg)
            else:
                result.add_warning(msg)

    # 9. Prediction order is part of the artifact contract.
    _validate_id_artifact(
        models_dir / f"train_ids_{component_name}.npy",
        label="Train",
        expected_rows=oof_preds.shape[0],
        expected_ids=expected_train_ids,
        required=config.require_train_ids,
        result=result,
    )
    _validate_id_artifact(
        models_dir / f"test_ids_{component_name}.npy",
        label="Test",
        expected_rows=test_preds.shape[0],
        expected_ids=expected_test_ids,
        required=config.require_test_ids,
        result=result,
    )

    return result


def validate_prediction_quality(
    preds: np.ndarray,
    y_true: np.ndarray | None = None,
    problem_type: str = "classification",
) -> tuple[bool, list[str]]:
    """
    Detect structurally broken predictions.

    Checks:
    - Non-finite values
    - Exactly constant predictions

    Args:
        preds: Prediction array
        y_true: Ground truth labels (reserved for metric-aware diagnostics)
        problem_type: "classification" or "regression"

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues: list[str] = []

    # Handle edge cases
    if preds is None or preds.size == 0:
        issues.append("Predictions array is empty or None")
        return False, issues

    preds = np.asarray(preds)
    if preds.ndim not in {1, 2}:
        issues.append(
            f"Predictions must be a 1-D or 2-D array, got shape {preds.shape}"
        )
        return False, issues
    validation_family = _normalize_problem_type(problem_type)
    if not np.issubdtype(preds.dtype, np.number):
        if validation_family == "seq2seq":
            if any(value is None for value in preds.reshape(-1).tolist()):
                issues.append("Text predictions contain null values")
            return len(issues) == 0, issues
        issues.append(f"Predictions must be numeric, got dtype {preds.dtype}")
        return False, issues
    if np.any(~np.isfinite(preds)):
        issues.append("Predictions contain NaN or Inf values")
        return False, issues

    # Artifact validation should reject broken constants, not weak or balanced
    # models. Class-balanced predictions legitimately have a uniform *mean*.
    if float(np.max(np.std(preds, axis=0))) < 1e-12:
        issues.append("Predictions are constant")

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
            oof = np.load(oof_path, allow_pickle=False)
            eligible_mask_path = (
                working_path / "canonical" / "oof_eligible_mask.npy"
            )
            if eligible_mask_path.is_file():
                eligible_mask = np.asarray(
                    np.load(eligible_mask_path, allow_pickle=False), dtype=bool
                )
                if eligible_mask.shape != (len(oof),):
                    issues.append(
                        "Canonical OOF eligibility mask shape mismatch"
                    )
                    eligible_oof = np.asarray([])
                else:
                    warmup_oof = oof[~eligible_mask]
                    if warmup_oof.size and not np.isnan(warmup_oof).all():
                        issues.append(
                            "Temporal warm-up OOF rows must remain NaN"
                        )
                    eligible_oof = oof[eligible_mask]
            else:
                eligible_oof = oof
            if eligible_oof.size and np.any(~np.isfinite(eligible_oof)):
                issues.append("Eligible OOF contains NaN or Inf values")
            if eligible_oof.ndim > 1:
                empty_rows = np.sum(eligible_oof.sum(axis=1) == 0)
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
            test = np.load(test_path, allow_pickle=False)
            if np.any(~np.isfinite(test)):
                issues.append("Test predictions contain NaN or Inf values")
        except Exception as e:
            issues.append(f"Failed to load test predictions: {e}")

    return issues
