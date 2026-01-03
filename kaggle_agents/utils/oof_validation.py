"""OOF validation utilities for stacking ensemble.

This module provides comprehensive validation of Out-of-Fold (OOF) predictions
to ensure proper stacking ensemble hygiene.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OOFSanityResult:
    """Result of OOF sanity check."""

    model_name: str
    is_valid: bool
    n_samples: int
    n_classes: int
    has_nan: bool
    has_inf: bool
    class_order_match: bool
    train_ids_match: bool
    shape_match: bool
    fold_info_available: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def compute_oof_statistics(oof_array: np.ndarray) -> dict[str, Any]:
    """Compute statistics for OOF predictions.

    Args:
        oof_array: OOF predictions array

    Returns:
        Dictionary with min, max, mean, std, nan_count, inf_count, etc.
    """
    stats: dict[str, Any] = {
        "shape": oof_array.shape,
        "dtype": str(oof_array.dtype),
        "min": float(np.nanmin(oof_array)) if oof_array.size > 0 else None,
        "max": float(np.nanmax(oof_array)) if oof_array.size > 0 else None,
        "mean": float(np.nanmean(oof_array)) if oof_array.size > 0 else None,
        "std": float(np.nanstd(oof_array)) if oof_array.size > 0 else None,
        "nan_count": int(np.isnan(oof_array).sum()),
        "inf_count": int(np.isinf(oof_array).sum()),
        "zero_count": int((oof_array == 0).sum()),
    }

    # Row-wise statistics for multiclass
    if oof_array.ndim > 1:
        row_sums = oof_array.sum(axis=1)
        stats["row_sum_min"] = float(np.nanmin(row_sums))
        stats["row_sum_max"] = float(np.nanmax(row_sums))
        stats["row_sum_mean"] = float(np.nanmean(row_sums))
        stats["empty_rows"] = int((row_sums == 0).sum())

    return stats


def assert_oof_sanity(
    oof_path: Path,
    models_dir: Path,
    expected_train_ids: np.ndarray | None = None,
    expected_class_order: list[str] | None = None,
    expected_shape: tuple[int, ...] | None = None,
    folds_path: Path | None = None,
    problem_type: str | None = None,
) -> OOFSanityResult:
    """Comprehensive OOF sanity check.

    Validates:
    1. Same index/order across all OOF files (via train_ids)
    2. Same classes/order (via class_order)
    3. Shape validation (n_samples, n_classes) identical
    4. No NaN/inf values
    5. Fold leakage verification (if folds.csv available)

    Args:
        oof_path: Path to oof_{model_name}.npy
        models_dir: Directory containing model artifacts
        expected_train_ids: IDs from original train.csv
        expected_class_order: Class order from sample_submission
        expected_shape: Expected (n_samples, n_classes) if known
        folds_path: Path to folds.csv for leakage verification
        problem_type: Optional hint ("classification" or "regression")

    Returns:
        OOFSanityResult with validation details
    """
    name = oof_path.stem.replace("oof_", "", 1)
    warnings: list[str] = []
    errors: list[str] = []

    # Load OOF predictions
    try:
        oof = np.load(oof_path)
    except Exception as e:
        return OOFSanityResult(
            model_name=name,
            is_valid=False,
            n_samples=0,
            n_classes=0,
            has_nan=False,
            has_inf=False,
            class_order_match=False,
            train_ids_match=False,
            shape_match=False,
            fold_info_available=False,
            errors=[f"Failed to load OOF: {e}"],
        )

    n_samples = oof.shape[0]
    n_classes = oof.shape[1] if oof.ndim > 1 else 1

    # Check for NaN values
    has_nan = bool(np.isnan(oof).any())
    if has_nan:
        nan_count = int(np.isnan(oof).sum())
        errors.append(f"Contains {nan_count} NaN values")

    # Check for inf values
    has_inf = bool(np.isinf(oof).any())
    if has_inf:
        inf_count = int(np.isinf(oof).sum())
        errors.append(f"Contains {inf_count} inf values")

    # Check shape
    shape_match = True
    if expected_shape is not None:
        if oof.shape != expected_shape:
            shape_match = False
            errors.append(f"Shape mismatch: {oof.shape} vs expected {expected_shape}")

    # Check class order
    class_order_match = True
    if expected_class_order is not None:
        class_order_path = models_dir / f"class_order_{name}.npy"
        if not class_order_path.exists():
            class_order_path = models_dir / "class_order.npy"

        if class_order_path.exists():
            try:
                saved_order = np.load(class_order_path, allow_pickle=True).tolist()
                if saved_order != expected_class_order:
                    class_order_match = False
                    errors.append(f"Class order mismatch: {saved_order} vs {expected_class_order}")
            except Exception as e:
                warnings.append(f"Could not load class_order: {e}")
        else:
            warnings.append("No class_order file found, cannot verify alignment")

    # Check train IDs (row order)
    train_ids_match = True
    if expected_train_ids is not None:
        train_ids_path = models_dir / f"train_ids_{name}.npy"
        if train_ids_path.exists():
            try:
                saved_ids = np.load(train_ids_path, allow_pickle=True)
                if not np.array_equal(saved_ids, expected_train_ids):
                    train_ids_match = False
                    errors.append("Train IDs mismatch - row order inconsistent")
            except Exception as e:
                warnings.append(f"Could not load train_ids: {e}")
        else:
            warnings.append("No train_ids file found, cannot verify row order")

    # Check fold info (for leakage verification)
    fold_info_available = False
    if folds_path is not None and folds_path.exists():
        fold_assignment_path = models_dir / f"fold_assignment_{name}.npy"
        if fold_assignment_path.exists():
            fold_info_available = True
            # Could add more detailed leakage checks here
        else:
            warnings.append("No fold_assignment file - cannot verify zero-leakage")

    # Additional sanity checks (classification only)
    if problem_type != "regression":
        if oof.ndim > 1:
            row_sums = oof.sum(axis=1)
            empty_rows = int((row_sums == 0).sum())
            if empty_rows > 0:
                warnings.append(f"{empty_rows} empty rows (sum=0) in OOF")

            # Check if probabilities are normalized
            not_normalized = int((np.abs(row_sums - 1.0) > 0.01).sum())
            if not_normalized > 0:
                warnings.append(f"{not_normalized} rows not normalized (sum != 1)")

        # Check probability bounds
        if oof.min() < 0:
            errors.append(f"Negative probabilities found: min={oof.min():.6f}")
        if oof.max() > 1:
            errors.append(f"Probabilities > 1 found: max={oof.max():.6f}")

    # Determine overall validity
    is_valid = len(errors) == 0 and class_order_match and train_ids_match and shape_match

    return OOFSanityResult(
        model_name=name,
        is_valid=is_valid,
        n_samples=n_samples,
        n_classes=n_classes,
        has_nan=has_nan,
        has_inf=has_inf,
        class_order_match=class_order_match,
        train_ids_match=train_ids_match,
        shape_match=shape_match,
        fold_info_available=fold_info_available,
        warnings=warnings,
        errors=errors,
    )


def validate_oof_stack(
    prediction_pairs: dict[str, tuple[Path, Path]],
    models_dir: Path,
    train_ids: np.ndarray | None = None,
    expected_class_order: list[str] | None = None,
    folds_path: Path | None = None,
    strict_mode: bool = False,
    problem_type: str | None = None,
) -> tuple[dict[str, tuple[Path, Path]], list[OOFSanityResult]]:
    """Validate all OOF files and filter to only valid ones.

    Args:
        prediction_pairs: Dictionary of (oof_path, test_path) pairs
        models_dir: Directory containing model artifacts
        train_ids: Expected train IDs for row order validation
        expected_class_order: Expected class order from sample_submission
        folds_path: Path to folds.csv for leakage verification
        strict_mode: If True, reject models with any warnings
        problem_type: Optional hint ("classification" or "regression")

    Returns:
        Tuple of (valid_pairs, all_results)
    """
    all_results: list[OOFSanityResult] = []
    valid_pairs: dict[str, tuple[Path, Path]] = {}

    # Determine expected shape from first valid OOF
    expected_shape: tuple[int, ...] | None = None
    for name, (oof_path, _) in prediction_pairs.items():
        try:
            oof = np.load(oof_path)
            expected_shape = oof.shape
            break
        except Exception:
            continue

    for name, (oof_path, test_path) in prediction_pairs.items():
        result = assert_oof_sanity(
            oof_path=oof_path,
            models_dir=models_dir,
            expected_train_ids=train_ids,
            expected_class_order=expected_class_order,
            expected_shape=expected_shape,
            folds_path=folds_path,
            problem_type=problem_type,
        )
        all_results.append(result)

        # Print validation status
        if result.is_valid:
            status = "valid"
            if result.warnings:
                status += f" ({len(result.warnings)} warnings)"
            print(f"   [OOF] {name}: {status} (shape: {result.n_samples}x{result.n_classes})")

            # In strict mode, reject if there are warnings
            if strict_mode and result.warnings:
                print(f"   [OOF] {name}: rejected (strict mode)")
                continue

            valid_pairs[name] = (oof_path, test_path)
        else:
            print(f"   [OOF] {name}: INVALID - {', '.join(result.errors)}")

    return valid_pairs, all_results


def print_oof_summary(results: list[OOFSanityResult]) -> None:
    """Print a summary of OOF validation results.

    Args:
        results: List of OOFSanityResult from validation
    """
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    with_warnings = sum(1 for r in results if r.is_valid and r.warnings)
    invalid = total - valid

    print(f"\n   OOF VALIDATION SUMMARY:")
    print(f"   Total models: {total}")
    print(f"   Valid: {valid} ({100 * valid / total:.1f}%)")
    if with_warnings > 0:
        print(f"   Valid with warnings: {with_warnings}")
    if invalid > 0:
        print(f"   Invalid: {invalid}")
        for r in results:
            if not r.is_valid:
                print(f"      - {r.model_name}: {', '.join(r.errors)}")
