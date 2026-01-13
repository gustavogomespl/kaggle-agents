"""Fallback and recovery functions for ensemble creation."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import validate_oof_alignment
from .prediction_pairs import find_prediction_pairs


def recover_from_checkpoints(
    models_dir: Path,
    component_names: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Recover OOF/test predictions from fold checkpoints.

    Scans checkpoint directories for partial CV results and reconstructs
    OOF predictions from completed folds.

    Args:
        models_dir: Directory containing model artifacts
        component_names: Optional list of component names to check

    Returns:
        Dict mapping component name to (oof, test) prediction arrays
    """
    recovered = {}
    checkpoints_dir = models_dir / "checkpoints"

    if not checkpoints_dir.exists():
        return recovered

    print("\n   CHECKPOINT RECOVERY:")
    print(f"      Scanning {checkpoints_dir}")

    # Find checkpoint state files
    state_files = list(checkpoints_dir.glob("*_checkpoint_state.json"))

    for state_file in state_files:
        try:
            with open(state_file) as f:
                state = json.load(f)

            component_name = state.get("component_name", "unknown")

            # Skip if not in requested list
            if component_names and component_name not in component_names:
                continue

            n_completed = len(state.get("completed_folds", []))
            min_folds = state.get("min_folds", 2)

            if n_completed < min_folds:
                print(f"      {component_name}: Only {n_completed}/{min_folds} folds, skipping")
                continue

            # Load partial OOF
            partial_oof_path = checkpoints_dir / f"{component_name}_oof_partial.npy"
            if not partial_oof_path.exists():
                print(f"      {component_name}: No partial OOF found")
                continue

            oof = np.load(partial_oof_path)

            # Check for test predictions (may not exist for partial)
            test_path = models_dir / f"test_{component_name}.npy"
            if test_path.exists():
                test = np.load(test_path)
            else:
                # Generate test predictions from fold models
                test = _generate_test_from_fold_models(
                    checkpoints_dir, component_name, state
                )

            if test is not None:
                recovered[component_name] = (oof, test)
                print(f"      {component_name}: Recovered {n_completed} folds, OOF shape {oof.shape}")

        except Exception as e:
            print(f"      Error recovering from {state_file}: {e}")

    return recovered


def _generate_test_from_fold_models(
    checkpoints_dir: Path,
    component_name: str,
    state: dict,
) -> np.ndarray | None:
    """Generate test predictions by averaging fold model predictions.

    Args:
        checkpoints_dir: Directory containing fold checkpoints
        component_name: Name of the component
        state: Checkpoint state dictionary

    Returns:
        Test predictions array or None if not possible
    """
    # This would require loading test data and averaging predictions
    # For now, return None to indicate test predictions need to be generated
    return None


def fallback_to_best_single_model(
    models_dir: Path,
    problem_type: str = "classification",
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    """Fallback to using the best single model when ensemble fails.

    Args:
        models_dir: Directory containing model artifacts
        problem_type: 'classification' or 'regression'

    Returns:
        Tuple of (ensemble_dict, final_predictions) or (None, None)
    """
    print("\n   FALLBACK TO BEST SINGLE MODEL:")

    # Find all test prediction files
    test_files = list(models_dir.glob("test_*.npy"))
    if not test_files:
        print("      No test predictions found")
        return None, None

    # Find corresponding OOF files and pick the one with best coverage
    best_model = None
    best_coverage = 0
    best_test = None

    for test_path in test_files:
        name = test_path.stem.replace("test_", "", 1)
        oof_path = models_dir / f"oof_{name}.npy"

        if not oof_path.exists():
            continue

        try:
            oof = np.load(oof_path)
            test = np.load(test_path)

            # Calculate coverage (non-zero predictions)
            if oof.ndim > 1:
                coverage = (oof.sum(axis=1) != 0).mean()
            else:
                coverage = (np.abs(oof) > 1e-10).mean()

            if coverage > best_coverage:
                best_coverage = coverage
                best_model = name
                best_test = test

        except Exception as e:
            print(f"      Error loading {name}: {e}")

    if best_model is None:
        print("      No valid models found")
        return None, None

    print(f"      Selected: {best_model} (coverage: {best_coverage:.1%})")

    ensemble = {
        "method": "single_model_fallback",
        "model_name": best_model,
        "coverage": best_coverage,
    }

    return ensemble, best_test


def create_ensemble_with_fallback(
    models_dir: Path,
    y: np.ndarray,
    problem_type: str,
    metric_name: str,
    expected_class_order: list[str] | None = None,
    train_ids: np.ndarray | None = None,
    min_models: int = 2,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    """Create ensemble with graceful fallback for partial models.

    This method attempts to create an ensemble, falling back through:
    1. Standard ensemble with all valid models
    2. Recovered checkpoints for partial OOF
    3. Best single model if ensemble not possible

    Args:
        models_dir: Directory containing model artifacts
        y: Target values
        problem_type: 'classification' or 'regression'
        metric_name: Metric name for scoring
        expected_class_order: Expected class order for classification
        train_ids: Training sample IDs for alignment
        min_models: Minimum models required for ensemble

    Returns:
        Tuple of (ensemble_dict, final_predictions) or (None, None)
    """
    print(f"\n   ENSEMBLE WITH FALLBACK (min_models={min_models}):")

    # Step 1: Try standard ensemble
    prediction_pairs = find_prediction_pairs(models_dir)
    print(f"      Found {len(prediction_pairs)} prediction pairs")

    if len(prediction_pairs) >= min_models:
        # Validate pairs
        valid_pairs = validate_oof_alignment(
            models_dir, train_ids, expected_class_order
        )

        if len(valid_pairs) >= min_models:
            print(f"      {len(valid_pairs)} valid pairs, proceeding with standard ensemble")
            # Use existing ensemble creation logic
            return None, None  # Will fall through to standard method

    # Step 2: Try to recover from checkpoints
    print(f"      Insufficient valid pairs ({len(prediction_pairs)}), trying checkpoint recovery")
    recovered = recover_from_checkpoints(models_dir)

    if recovered:
        # Add recovered models to prediction pairs
        for name, (oof, test) in recovered.items():
            if name not in prediction_pairs:
                # Save recovered predictions
                oof_path = models_dir / f"oof_{name}_recovered.npy"
                test_path = models_dir / f"test_{name}_recovered.npy"
                np.save(oof_path, oof)
                if test is not None:
                    np.save(test_path, test)
                    prediction_pairs[f"{name}_recovered"] = (oof_path, test_path)

        if len(prediction_pairs) >= min_models:
            print(f"      After recovery: {len(prediction_pairs)} pairs available")
            return None, None  # Will fall through to standard method

    # Step 3: Fallback to best single model
    print("      Still insufficient models, falling back to best single model")
    return fallback_to_best_single_model(models_dir, problem_type)
