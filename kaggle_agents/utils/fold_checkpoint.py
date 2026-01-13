"""
Fold Checkpointing for Graceful Degradation.

Provides mechanisms to save progress after each CV fold and recover
partial ensembles when training times out or fails.

Research: Partial ensembles (3/5 folds) typically provide 90%+ of full ensemble performance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class FoldCheckpoint:
    """Container for a single fold's checkpoint data."""

    fold_idx: int
    model_path: Path
    oof_predictions: np.ndarray
    val_indices: np.ndarray
    score: float
    elapsed_time: float
    n_iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fold_idx": self.fold_idx,
            "model_path": str(self.model_path),
            "oof_shape": list(self.oof_predictions.shape),
            "val_indices_len": len(self.val_indices),
            "score": self.score,
            "elapsed_time": self.elapsed_time,
            "n_iterations": self.n_iterations,
            "metadata": self.metadata,
        }


class FoldCheckpointManager:
    """Checkpoint and recover partial CV training.

    Saves progress after each fold completes, enabling recovery when:
    - Training times out mid-CV
    - A later fold fails but earlier folds succeeded
    - Need to resume training after interruption

    Usage:
        manager = FoldCheckpointManager(
            checkpoint_dir=Path("models/checkpoints"),
            component_name="lgb_baseline",
            n_samples=10000,
            n_classes=5,
            min_folds=2,
        )

        for fold_idx in range(n_folds):
            model, oof_preds, val_idx, score = train_fold(...)
            manager.save_fold(fold_idx, model, oof_preds, val_idx, score)

        # Later, if training was interrupted:
        oof, completed_folds = manager.recover_partial_ensemble()
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        component_name: str,
        n_samples: int,
        n_classes: int = 1,
        min_folds: int = 2,
    ):
        """Initialize fold checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            component_name: Name of the component (for file naming)
            n_samples: Total number of training samples
            n_classes: Number of classes (1 for regression/binary)
            min_folds: Minimum folds required for valid ensemble
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.component_name = component_name
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.min_folds = min_folds
        self.checkpoints: dict[int, FoldCheckpoint] = {}
        self.start_time = time.time()

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Try to load existing checkpoints
        self._load_checkpoint_state()

    @property
    def completed_folds(self) -> list[int]:
        """List of completed fold indices."""
        return sorted(self.checkpoints.keys())

    @property
    def n_completed(self) -> int:
        """Number of completed folds."""
        return len(self.checkpoints)

    @property
    def has_valid_ensemble(self) -> bool:
        """Whether we have enough folds for a valid ensemble."""
        return self.n_completed >= self.min_folds

    @property
    def total_elapsed_time(self) -> float:
        """Total time spent training all folds."""
        return sum(ckpt.elapsed_time for ckpt in self.checkpoints.values())

    @property
    def average_fold_time(self) -> float:
        """Average time per fold."""
        if not self.checkpoints:
            return 0.0
        return self.total_elapsed_time / len(self.checkpoints)

    def _get_state_path(self) -> Path:
        """Get path to checkpoint state file."""
        return self.checkpoint_dir / f"{self.component_name}_checkpoint_state.json"

    def _get_fold_paths(self, fold_idx: int) -> tuple[Path, Path, Path]:
        """Get paths for fold model, OOF, and indices."""
        base = self.checkpoint_dir / f"{self.component_name}_fold_{fold_idx}"
        return (
            base.with_suffix(".pkl"),  # model
            base.with_name(f"{base.name}_oof.npy"),  # OOF predictions
            base.with_name(f"{base.name}_val_idx.npy"),  # validation indices
        )

    def save_fold(
        self,
        fold_idx: int,
        model: Any,
        oof_predictions: np.ndarray,
        val_indices: np.ndarray,
        score: float,
        n_iterations: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint after a fold completes.

        Args:
            fold_idx: Index of the completed fold
            model: Trained model object
            oof_predictions: OOF predictions for this fold
            val_indices: Indices of validation samples
            score: Validation score for this fold
            n_iterations: Number of training iterations
            metadata: Optional additional metadata
        """
        elapsed = time.time() - self.start_time

        # Get paths
        model_path, oof_path, idx_path = self._get_fold_paths(fold_idx)

        # Save model
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            print(f"   Warning: Could not save model for fold {fold_idx}: {e}")
            # Create empty placeholder
            model_path.touch()

        # Save OOF predictions
        np.save(oof_path, oof_predictions)

        # Save validation indices
        np.save(idx_path, val_indices)

        # Create checkpoint
        checkpoint = FoldCheckpoint(
            fold_idx=fold_idx,
            model_path=model_path,
            oof_predictions=oof_predictions,
            val_indices=val_indices,
            score=score,
            elapsed_time=elapsed - self.total_elapsed_time,  # Time for this fold
            n_iterations=n_iterations,
            metadata=metadata or {},
        )

        self.checkpoints[fold_idx] = checkpoint

        # Save state
        self._save_checkpoint_state()

        print(f"   Checkpoint saved: fold {fold_idx}, score={score:.6f}, elapsed={checkpoint.elapsed_time:.1f}s")

    def _save_checkpoint_state(self) -> None:
        """Save checkpoint state to disk."""
        state = {
            "component_name": self.component_name,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "min_folds": self.min_folds,
            "completed_folds": self.completed_folds,
            "total_elapsed_time": self.total_elapsed_time,
            "checkpoints": {
                fold_idx: ckpt.to_dict() for fold_idx, ckpt in self.checkpoints.items()
            },
        }

        state_path = self._get_state_path()
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_checkpoint_state(self) -> None:
        """Load checkpoint state from disk if available."""
        state_path = self._get_state_path()
        if not state_path.exists():
            return

        try:
            with open(state_path) as f:
                state = json.load(f)

            # Verify compatibility
            if state.get("n_samples") != self.n_samples:
                print("   Warning: Checkpoint n_samples mismatch, ignoring")
                return

            # Load checkpoints
            for fold_idx_str, ckpt_data in state.get("checkpoints", {}).items():
                fold_idx = int(fold_idx_str)
                model_path, oof_path, idx_path = self._get_fold_paths(fold_idx)

                if oof_path.exists() and idx_path.exists():
                    oof = np.load(oof_path)
                    val_idx = np.load(idx_path)

                    self.checkpoints[fold_idx] = FoldCheckpoint(
                        fold_idx=fold_idx,
                        model_path=model_path,
                        oof_predictions=oof,
                        val_indices=val_idx,
                        score=ckpt_data.get("score", 0.0),
                        elapsed_time=ckpt_data.get("elapsed_time", 0.0),
                        n_iterations=ckpt_data.get("n_iterations", 0),
                        metadata=ckpt_data.get("metadata", {}),
                    )

            if self.checkpoints:
                print(f"   Loaded {len(self.checkpoints)} existing checkpoints: folds {self.completed_folds}")

        except Exception as e:
            print(f"   Warning: Could not load checkpoint state: {e}")

    def recover_partial_ensemble(self) -> tuple[np.ndarray, list[int]]:
        """Recover OOF predictions from completed folds.

        Returns:
            Tuple of (oof_predictions, completed_fold_indices)

        Raises:
            ValueError: If not enough folds completed
        """
        if not self.has_valid_ensemble:
            raise ValueError(
                f"Only {self.n_completed} folds completed, need {self.min_folds}. "
                f"Completed: {self.completed_folds}"
            )

        print("\n   RECOVERING PARTIAL ENSEMBLE:")
        print(f"      Completed folds: {self.completed_folds} ({self.n_completed}/{self.min_folds} minimum)")

        # Initialize full OOF array
        if self.n_classes > 1:
            oof = np.zeros((self.n_samples, self.n_classes))
        else:
            oof = np.zeros(self.n_samples)

        # Track which samples have predictions
        has_prediction = np.zeros(self.n_samples, dtype=bool)

        # Stitch together OOF from completed folds
        total_score = 0.0
        for fold_idx, ckpt in sorted(self.checkpoints.items()):
            oof[ckpt.val_indices] = ckpt.oof_predictions
            has_prediction[ckpt.val_indices] = True
            total_score += ckpt.score
            print(f"      Fold {fold_idx}: {len(ckpt.val_indices)} samples, score={ckpt.score:.6f}")

        coverage = has_prediction.sum() / self.n_samples
        avg_score = total_score / self.n_completed

        print(f"      Coverage: {coverage:.1%} of samples have predictions")
        print(f"      Average fold score: {avg_score:.6f}")

        if coverage < 0.5:
            print(f"      WARNING: Low coverage ({coverage:.1%}), ensemble may be unreliable")

        return oof, self.completed_folds

    def load_fold_models(self) -> dict[int, Any]:
        """Load trained models from completed folds.

        Returns:
            Dict mapping fold_idx to model object
        """
        models = {}
        for fold_idx, ckpt in self.checkpoints.items():
            if ckpt.model_path.exists() and ckpt.model_path.stat().st_size > 0:
                try:
                    models[fold_idx] = joblib.load(ckpt.model_path)
                except Exception as e:
                    print(f"   Warning: Could not load model for fold {fold_idx}: {e}")
        return models

    def estimate_remaining_time(self, remaining_folds: int) -> float:
        """Estimate time needed for remaining folds.

        Args:
            remaining_folds: Number of folds still to train

        Returns:
            Estimated time in seconds
        """
        if not self.checkpoints:
            return float("inf")

        # Use average fold time with some buffer
        return self.average_fold_time * remaining_folds * 1.2

    def should_continue_training(self, remaining_budget: float, remaining_folds: int) -> bool:
        """Check if we should continue training more folds.

        Args:
            remaining_budget: Remaining time budget in seconds
            remaining_folds: Number of folds still to train

        Returns:
            True if we should continue, False if we should stop and use partial ensemble
        """
        if remaining_folds == 0:
            return False

        estimated_time = self.estimate_remaining_time(remaining_folds)

        if estimated_time > remaining_budget * 0.9:
            print(f"\n   TIME CHECK: Estimated {estimated_time:.1f}s needed for {remaining_folds} folds")
            print(f"      Available: {remaining_budget:.1f}s")
            print(f"      Decision: Stop and use partial ensemble ({self.n_completed} folds)")
            return False

        return True

    def get_fold_scores(self) -> dict[int, float]:
        """Get scores for all completed folds."""
        return {fold_idx: ckpt.score for fold_idx, ckpt in self.checkpoints.items()}

    def get_best_fold(self) -> tuple[int, float]:
        """Get the fold with the best score.

        Returns:
            Tuple of (fold_idx, score)
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        scores = self.get_fold_scores()
        best_fold = min(scores, key=scores.get)  # Assuming lower is better
        return best_fold, scores[best_fold]

    def cleanup(self) -> None:
        """Remove all checkpoint files."""
        for fold_idx in list(self.checkpoints.keys()):
            model_path, oof_path, idx_path = self._get_fold_paths(fold_idx)
            for path in [model_path, oof_path, idx_path]:
                if path.exists():
                    path.unlink()

        state_path = self._get_state_path()
        if state_path.exists():
            state_path.unlink()

        self.checkpoints.clear()
        print(f"   Cleaned up checkpoints for {self.component_name}")

    def summary(self) -> dict[str, Any]:
        """Get summary of checkpoint state."""
        return {
            "component_name": self.component_name,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "min_folds": self.min_folds,
            "completed_folds": self.completed_folds,
            "n_completed": self.n_completed,
            "has_valid_ensemble": self.has_valid_ensemble,
            "total_elapsed_time": self.total_elapsed_time,
            "average_fold_time": self.average_fold_time,
            "fold_scores": self.get_fold_scores(),
        }

    def print_summary(self) -> None:
        """Print summary of checkpoint state."""
        summary = self.summary()
        print(f"\n   FOLD CHECKPOINT SUMMARY ({self.component_name}):")
        print(f"      Completed: {summary['n_completed']}/{summary['min_folds']} minimum folds")
        print(f"      Folds: {summary['completed_folds']}")
        print(f"      Valid ensemble: {summary['has_valid_ensemble']}")
        print(f"      Total time: {summary['total_elapsed_time']:.1f}s")
        print(f"      Avg fold time: {summary['average_fold_time']:.1f}s")
        if summary['fold_scores']:
            scores = list(summary['fold_scores'].values())
            print(f"      Scores: mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")


def create_fold_checkpointing_code(checkpoint_dir: str, component_name: str) -> str:
    """Generate code snippet for fold checkpointing.

    This can be injected into generated model training code.

    Args:
        checkpoint_dir: Directory for checkpoints
        component_name: Component name for file naming

    Returns:
        Python code snippet as string
    """
    return f'''
# === FOLD CHECKPOINTING (AUTO-INJECTED) ===
from kaggle_agents.utils.fold_checkpoint import FoldCheckpointManager

_fold_checkpoint_manager = None

def _init_fold_checkpointing(n_samples, n_classes=1, min_folds=2):
    """Initialize fold checkpointing."""
    global _fold_checkpoint_manager
    _fold_checkpoint_manager = FoldCheckpointManager(
        checkpoint_dir=Path("{checkpoint_dir}"),
        component_name="{component_name}",
        n_samples=n_samples,
        n_classes=n_classes,
        min_folds=min_folds,
    )
    return _fold_checkpoint_manager

def _save_fold_checkpoint(fold_idx, model, oof_preds, val_idx, score, n_iterations=0):
    """Save checkpoint after fold completes."""
    if _fold_checkpoint_manager is not None:
        _fold_checkpoint_manager.save_fold(
            fold_idx=fold_idx,
            model=model,
            oof_predictions=oof_preds,
            val_indices=val_idx,
            score=score,
            n_iterations=n_iterations,
        )

def _recover_partial_ensemble():
    """Recover OOF from completed folds."""
    if _fold_checkpoint_manager is not None and _fold_checkpoint_manager.has_valid_ensemble:
        return _fold_checkpoint_manager.recover_partial_ensemble()
    return None, []
# === END FOLD CHECKPOINTING ===
'''
