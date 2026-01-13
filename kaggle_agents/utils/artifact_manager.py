"""
Artifact Manager for Kaggle Agents.

Provides centralized management of model artifacts (OOF predictions,
test predictions, metadata files) with validation and discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelArtifacts:
    """Container for a single model's artifacts."""

    name: str
    oof_path: Path
    test_path: Path
    train_ids_path: Path | None = None
    class_order_path: Path | None = None
    fold_assignment_path: Path | None = None
    model_path: Path | None = None
    is_valid: bool = True
    validation_issues: list[str] = field(default_factory=list)


class ArtifactManager:
    """Centralized artifact management for model ensemble."""

    # Required artifacts for a model to be included in ensemble
    REQUIRED_ARTIFACTS = [
        "oof_{name}.npy",
        "test_{name}.npy",
        "train_ids_{name}.npy",
    ]

    # Optional but recommended artifacts
    OPTIONAL_ARTIFACTS = [
        "class_order_{name}.npy",
        "fold_assignment_{name}.npy",
        "model_{name}.pkl",
        "model_{name}.pth",
        "model_{name}.pt",
    ]

    def __init__(self, models_dir: str | Path):
        """Initialize artifact manager.

        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = Path(models_dir)
        self._discovered_models: dict[str, ModelArtifacts] = {}

    def discover_models(self) -> dict[str, ModelArtifacts]:
        """
        Discover all models with complete artifact sets.

        Returns:
            Dict mapping model name to ModelArtifacts
        """
        if not self.models_dir.exists():
            return {}

        # Find all OOF files
        oof_files = list(self.models_dir.glob("oof_*.npy"))

        for oof_path in oof_files:
            name = oof_path.stem.replace("oof_", "", 1)

            # Check for required test file
            test_path = self.models_dir / f"test_{name}.npy"
            if not test_path.exists():
                continue

            # Check for train_ids
            train_ids_path = self.models_dir / f"train_ids_{name}.npy"
            if not train_ids_path.exists():
                train_ids_path = None

            # Check for optional artifacts
            class_order_path = self.models_dir / f"class_order_{name}.npy"
            if not class_order_path.exists():
                class_order_path = None

            fold_assignment_path = self.models_dir / f"fold_assignment_{name}.npy"
            if not fold_assignment_path.exists():
                fold_assignment_path = None

            # Check for model file (multiple extensions)
            model_path = None
            for ext in [".pkl", ".pth", ".pt", ".joblib"]:
                candidate = self.models_dir / f"model_{name}{ext}"
                if candidate.exists():
                    model_path = candidate
                    break

            self._discovered_models[name] = ModelArtifacts(
                name=name,
                oof_path=oof_path,
                test_path=test_path,
                train_ids_path=train_ids_path,
                class_order_path=class_order_path,
                fold_assignment_path=fold_assignment_path,
                model_path=model_path,
            )

        return self._discovered_models

    def validate_model_artifacts(
        self,
        name: str,
        expected_n_train: int | None = None,
        expected_n_test: int | None = None,
        expected_class_order: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate a model's artifacts for correctness.

        Args:
            name: Model name
            expected_n_train: Expected number of training samples
            expected_n_test: Expected number of test samples
            expected_class_order: Expected class order for multiclass

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if name not in self._discovered_models:
            self.discover_models()

        if name not in self._discovered_models:
            return False, [f"Model '{name}' not found"]

        artifacts = self._discovered_models[name]

        # Load and validate OOF
        try:
            oof = np.load(artifacts.oof_path)

            # Shape check
            if expected_n_train and oof.shape[0] != expected_n_train:
                issues.append(
                    f"OOF shape mismatch: {oof.shape[0]} vs expected {expected_n_train}"
                )

            # NaN/Inf check
            if not np.isfinite(oof).all():
                n_invalid = (~np.isfinite(oof)).sum()
                issues.append(f"OOF contains {n_invalid} NaN/Inf values")

            # Empty rows check
            if oof.ndim > 1:
                empty_mask = oof.sum(axis=1) == 0
            else:
                empty_mask = np.abs(oof) < 1e-10
            n_empty = empty_mask.sum()
            if n_empty > 0:
                issues.append(f"OOF has {n_empty} empty/zero rows")

            # Probability range check (for classification)
            if oof.ndim > 1 or (oof.min() >= 0 and oof.max() <= 1):
                if oof.min() < 0 or oof.max() > 1:
                    issues.append(
                        f"OOF values outside [0,1]: min={oof.min():.4f}, max={oof.max():.4f}"
                    )

        except Exception as e:
            issues.append(f"Failed to load OOF: {e}")

        # Load and validate test predictions
        try:
            test = np.load(artifacts.test_path)

            if expected_n_test and test.shape[0] != expected_n_test:
                issues.append(
                    f"Test shape mismatch: {test.shape[0]} vs expected {expected_n_test}"
                )

            if not np.isfinite(test).all():
                n_invalid = (~np.isfinite(test)).sum()
                issues.append(f"Test contains {n_invalid} NaN/Inf values")

        except Exception as e:
            issues.append(f"Failed to load test predictions: {e}")

        # Validate class order if expected
        if expected_class_order and artifacts.class_order_path:
            try:
                saved_order = np.load(
                    artifacts.class_order_path, allow_pickle=True
                ).tolist()
                if saved_order != expected_class_order:
                    issues.append(
                        f"Class order mismatch: {saved_order[:3]}... vs {expected_class_order[:3]}..."
                    )
            except Exception as e:
                issues.append(f"Failed to load class order: {e}")

        # Update artifact validity
        artifacts.is_valid = len(issues) == 0
        artifacts.validation_issues = issues

        return len(issues) == 0, issues

    def get_valid_models(
        self,
        expected_n_train: int | None = None,
        expected_n_test: int | None = None,
        expected_class_order: list[str] | None = None,
        require_train_ids: bool = True,
    ) -> dict[str, ModelArtifacts]:
        """
        Get all valid models after validation.

        Args:
            expected_n_train: Expected training samples
            expected_n_test: Expected test samples
            expected_class_order: Expected class order
            require_train_ids: Whether to require train_ids file

        Returns:
            Dict of valid models
        """
        self.discover_models()

        valid_models = {}
        for name, artifacts in self._discovered_models.items():
            # Check train_ids requirement
            if require_train_ids and artifacts.train_ids_path is None:
                continue

            is_valid, _ = self.validate_model_artifacts(
                name, expected_n_train, expected_n_test, expected_class_order
            )

            if is_valid:
                valid_models[name] = artifacts

        return valid_models

    def load_oof_stack(
        self,
        model_names: list[str] | None = None,
        align_by_ids: bool = False,
        canonical_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load OOF predictions as a stacked array.

        Args:
            model_names: Specific models to load (None = all valid)
            align_by_ids: Whether to align by train IDs
            canonical_ids: Canonical train IDs for alignment

        Returns:
            Tuple of (stacked_oof, model_names_used)
        """
        if model_names is None:
            valid = self.get_valid_models(require_train_ids=align_by_ids)
            model_names = list(valid.keys())

        oofs = []
        names_used = []

        for name in model_names:
            if name not in self._discovered_models:
                continue

            artifacts = self._discovered_models[name]
            oof = np.load(artifacts.oof_path)

            if align_by_ids and canonical_ids is not None:
                if artifacts.train_ids_path is None:
                    print(f"   Skipping {name}: no train_ids for alignment")
                    continue

                model_ids = np.load(artifacts.train_ids_path, allow_pickle=True)
                oof = self._align_by_id(oof, model_ids, canonical_ids)

            oofs.append(oof)
            names_used.append(name)

        if not oofs:
            raise ValueError("No valid OOF predictions to stack")

        # Stack: (n_models, n_samples, n_classes) or (n_models, n_samples)
        stacked = np.stack(oofs, axis=0)

        return stacked, names_used

    def load_test_stack(
        self,
        model_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load test predictions as a stacked array.

        Args:
            model_names: Specific models to load (None = all valid)

        Returns:
            Tuple of (stacked_test, model_names_used)
        """
        if model_names is None:
            valid = self.get_valid_models()
            model_names = list(valid.keys())

        tests = []
        names_used = []

        for name in model_names:
            if name not in self._discovered_models:
                continue

            artifacts = self._discovered_models[name]
            test = np.load(artifacts.test_path)
            tests.append(test)
            names_used.append(name)

        if not tests:
            raise ValueError("No valid test predictions to stack")

        stacked = np.stack(tests, axis=0)
        return stacked, names_used

    def _align_by_id(
        self,
        predictions: np.ndarray,
        model_ids: np.ndarray,
        canonical_ids: np.ndarray,
    ) -> np.ndarray:
        """Align predictions to canonical ID order."""
        model_id_to_idx = {id_val: idx for idx, id_val in enumerate(model_ids)}

        if predictions.ndim > 1:
            aligned = np.zeros((len(canonical_ids), predictions.shape[1]))
        else:
            aligned = np.zeros(len(canonical_ids))

        for canonical_idx, canonical_id in enumerate(canonical_ids):
            if canonical_id in model_id_to_idx:
                model_idx = model_id_to_idx[canonical_id]
                aligned[canonical_idx] = predictions[model_idx]

        return aligned

    def summary(self) -> dict[str, Any]:
        """Get summary of discovered artifacts."""
        self.discover_models()

        return {
            "n_models": len(self._discovered_models),
            "models": list(self._discovered_models.keys()),
            "with_train_ids": sum(
                1 for m in self._discovered_models.values() if m.train_ids_path
            ),
            "with_class_order": sum(
                1 for m in self._discovered_models.values() if m.class_order_path
            ),
            "with_fold_assignment": sum(
                1 for m in self._discovered_models.values() if m.fold_assignment_path
            ),
        }

    def print_summary(self) -> None:
        """Print a summary of discovered artifacts."""
        summary = self.summary()
        print("\n   Artifact Summary:")
        print(f"      Total models: {summary['n_models']}")
        print(f"      With train_ids: {summary['with_train_ids']}")
        print(f"      With class_order: {summary['with_class_order']}")
        print(f"      With fold_assignment: {summary['with_fold_assignment']}")
        print(f"      Models: {', '.join(summary['models'])}")
