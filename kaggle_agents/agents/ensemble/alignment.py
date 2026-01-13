"""OOF alignment and validation functions."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import class_orders_match, normalize_class_order


def validate_oof_alignment(
    models_dir: Path,
    train_ids: np.ndarray,
    expected_class_order: list[str],
) -> dict[str, tuple[Path, Path]]:
    """Validate and filter OOFs by row and class alignment.

    This method tracks and reports all skip reasons for better debugging
    of ensemble validation failures.

    Args:
        models_dir: Directory containing model artifacts
        train_ids: IDs from original train.csv (expected row order)
        expected_class_order: Class order from sample_submission

    Returns:
        Dictionary of valid prediction pairs (name -> (oof_path, test_path))
    """
    valid_pairs: dict[str, tuple[Path, Path]] = {}
    skip_reasons: list[str] = []  # Track WHY each model was skipped

    strict_mode = os.getenv("KAGGLE_AGENTS_STRICT_MODE", "0").lower() in {"1", "true", "yes"}

    for oof_path in models_dir.glob("oof_*.npy"):
        name = oof_path.stem.replace("oof_", "", 1)
        test_path = models_dir / f"test_{name}.npy"

        # Check test file exists
        if not test_path.exists():
            skip_reasons.append(f"{name}: Missing test_{name}.npy")
            continue

        # 1. Verify class_order
        class_order_path = models_dir / f"class_order_{name}.npy"
        class_order_validated = False

        if class_order_path.exists():
            try:
                saved_order = np.load(class_order_path, allow_pickle=True).tolist()
                if not class_orders_match(saved_order, expected_class_order):
                    skip_reasons.append(
                        f"{name}: Class order mismatch - "
                        f"model has {normalize_class_order(saved_order)[:2]}..., "
                        f"expected {normalize_class_order(expected_class_order)[:2]}..."
                    )
                    continue
                class_order_validated = True
            except Exception as e:
                skip_reasons.append(f"{name}: Failed to load class_order: {e}")
                continue
        else:
            # Try global class_order.npy as fallback
            global_class_order = models_dir / "class_order.npy"
            if global_class_order.exists():
                try:
                    saved_order = np.load(global_class_order, allow_pickle=True).tolist()
                    if not class_orders_match(saved_order, expected_class_order):
                        skip_reasons.append(
                            f"{name}: Global class order mismatch - "
                            f"has {normalize_class_order(saved_order)[:2]}..., "
                            f"expected {normalize_class_order(expected_class_order)[:2]}..."
                        )
                        continue
                    class_order_validated = True
                except Exception as e:
                    skip_reasons.append(f"{name}: Failed to load global class_order: {e}")
                    continue

        # Warn about missing class order (but don't skip in lenient mode)
        if not class_order_validated:
            msg = f"{name}: Missing class_order file (alignment cannot be verified)"
            if strict_mode:
                skip_reasons.append(msg)
                continue
            print(f"   Warning: {msg} - including with caution")

        # 2. Verify train_ids (row order)
        train_ids_path = models_dir / f"train_ids_{name}.npy"
        if train_ids_path.exists():
            try:
                saved_ids = np.load(train_ids_path, allow_pickle=True)
                if not np.array_equal(saved_ids, train_ids):
                    skip_reasons.append(f"{name}: Train IDs mismatch (row order differs)")
                    continue
            except Exception as e:
                skip_reasons.append(f"{name}: Failed to load train_ids: {e}")
                continue
        # Metadata missing: warn but include in lenient mode
        elif strict_mode:
            skip_reasons.append(f"{name}: Missing train_ids file (strict mode)")
            continue
        else:
            print(f"   Warning: {name}: Missing train_ids file - including with caution")

        valid_pairs[name] = (oof_path, test_path)

    # === PRINT SKIP REASON SUMMARY ===
    if skip_reasons:
        print("\n   ENSEMBLE ALIGNMENT VALIDATION - SKIPPED MODELS:")
        print(f"   Total skipped: {len(skip_reasons)}")
        for reason in skip_reasons[:10]:  # Show first 10
            print(f"      - {reason}")
        if len(skip_reasons) > 10:
            print(f"      ... and {len(skip_reasons) - 10} more")
        print()

    return valid_pairs


def align_oof_by_canonical_ids(
    oof: np.ndarray,
    model_train_ids: np.ndarray,
    canonical_train_ids: np.ndarray,
    model_name: str = "unknown",
) -> np.ndarray | None:
    """Align OOF predictions to canonical ID order with strict validation.

    Args:
        oof: OOF predictions from model
        model_train_ids: Train IDs corresponding to oof rows
        canonical_train_ids: Target canonical ID order
        model_name: Name of the model (for error messages)

    Returns:
        OOF aligned to canonical ID order, or None if alignment is impossible
    """
    # Create ID to index mapping for model predictions
    model_id_to_idx = {id_val: idx for idx, id_val in enumerate(model_train_ids)}

    # Calculate overlap BEFORE alignment
    common_ids = set(model_train_ids) & set(canonical_train_ids)
    overlap_pct = len(common_ids) / len(canonical_train_ids) * 100

    # Strict validation - reject if overlap is too low
    if overlap_pct < 50:
        print(f"      CRITICAL: {model_name} has only {overlap_pct:.1f}% ID overlap!")
        print("         Model trained on different data - EXCLUDING from ensemble")
        return None

    if overlap_pct < 80:
        print(f"      WARNING: {model_name} has low ID overlap ({overlap_pct:.1f}%)")
        print("         Ensemble may be degraded - model used different sampling")

    # Initialize aligned OOF with zeros
    if oof.ndim > 1:
        aligned_oof = np.zeros((len(canonical_train_ids), oof.shape[1]))
    else:
        aligned_oof = np.zeros(len(canonical_train_ids))

    # Track how many IDs we successfully aligned
    aligned_count = 0

    # Map model predictions to canonical order
    for canonical_idx, canonical_id in enumerate(canonical_train_ids):
        if canonical_id in model_id_to_idx:
            model_idx = model_id_to_idx[canonical_id]
            aligned_oof[canonical_idx] = oof[model_idx]
            aligned_count += 1

    print(f"      OK: ID alignment: {aligned_count}/{len(canonical_train_ids)} ({overlap_pct:.1f}%) IDs matched")

    return aligned_oof


def load_and_align_oof(
    oof_path: Path,
    train_ids_path: Path,
    reference_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load OOF predictions and align to reference ID order.

    Uses vectorized pandas Index operations for efficiency.

    Args:
        oof_path: Path to oof_*.npy
        train_ids_path: Path to train_ids_*.npy (same order as oof)
        reference_ids: Target ID order (from train.csv)

    Returns:
        Tuple of (aligned_oof, valid_mask) - mask is True where alignment succeeded
    """
    oof = np.load(oof_path)

    if not train_ids_path.exists():
        print(f"      [LOG:WARNING] No train_ids file for {oof_path.name}, assuming aligned")
        return oof, np.ones(len(oof), dtype=bool)

    train_ids = np.load(train_ids_path, allow_pickle=True)

    if len(train_ids) != len(oof):
        raise ValueError(f"train_ids length {len(train_ids)} != oof length {len(oof)}")

    # Convert to pandas Index for vectorized lookup
    oof_index = pd.Index(train_ids)
    ref_index = pd.Index(reference_ids)

    # Get positions of reference IDs in OOF index (-1 for missing)
    indexer = oof_index.get_indexer(ref_index)

    # Create valid mask (where alignment succeeded)
    valid_mask = indexer >= 0
    n_missing = (~valid_mask).sum()

    if n_missing > 0:
        print(f"      [LOG:WARNING] {n_missing}/{len(ref_index)} IDs not found in OOF predictions")

    # Allocate aligned array
    if oof.ndim == 1:
        aligned_oof = np.zeros(len(ref_index), dtype=oof.dtype)
    else:
        aligned_oof = np.zeros((len(ref_index), oof.shape[1]), dtype=oof.dtype)

    # Fill valid positions using vectorized indexing
    aligned_oof[valid_mask] = oof[indexer[valid_mask]]

    return aligned_oof, valid_mask


def stack_with_alignment(
    oof_paths: list[Path],
    train_ids_paths: list[Path],
    reference_ids: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack multiple OOF predictions with proper alignment.

    Args:
        oof_paths: List of paths to oof_*.npy files
        train_ids_paths: List of paths to train_ids_*.npy files
        reference_ids: Reference ID order (from train.csv)
        y_true: Target values in reference order

    Returns:
        Tuple of (X_meta, y_aligned, combined_mask) - only rows with ALL predictions
    """
    all_oof = []
    all_masks = []

    for oof_path, ids_path in zip(oof_paths, train_ids_paths):
        oof, mask = load_and_align_oof(oof_path, ids_path, reference_ids)
        all_oof.append(oof)
        all_masks.append(mask)

    # Combined mask: only where ALL OOF predictions exist
    combined_mask = np.all(all_masks, axis=0)
    n_valid = combined_mask.sum()
    print(f"      [LOG:INFO] Stacking {n_valid}/{len(combined_mask)} rows with complete OOF predictions")

    # Stack features (only for valid rows)
    X_meta = np.column_stack([oof[combined_mask] for oof in all_oof])
    y_aligned = y_true[combined_mask]

    return X_meta, y_aligned, combined_mask
