"""
Checkpoint Manager - Save and restore execution checkpoints.

This module provides utilities for saving and restoring execution state,
enabling recovery from timeouts and creation of partial submissions.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .code_executor import ExecutionProgress


class CheckpointManager:
    """
    Manage execution checkpoints for recovery and partial submissions.

    This class handles:
    - Saving/loading execution progress
    - Detecting partial results (OOF predictions, models)
    - Creating submissions from partial results
    """

    CHECKPOINT_FILE = "checkpoint.json"
    OOF_PATTERN = "oof_*.npy"
    TEST_PATTERN = "test_*.npy"

    def __init__(self, working_dir: Path | str):
        """
        Initialize checkpoint manager.

        Args:
            working_dir: Working directory for the competition
        """
        self.working_dir = Path(working_dir) if isinstance(working_dir, str) else working_dir
        self.models_dir = self.working_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        component_name: str,
        progress: ExecutionProgress,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Save current progress checkpoint.

        Args:
            component_name: Name of the component being executed
            progress: Current execution progress
            additional_data: Optional additional data to save

        Returns:
            Path to checkpoint file
        """
        checkpoint_path = self.models_dir / f"checkpoint_{component_name}.json"

        checkpoint = {
            "component_name": component_name,
            "timestamp": datetime.now().isoformat(),
            "progress": progress.to_dict(),
            "additional_data": additional_data or {},
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"   Checkpoint saved: {checkpoint_path.name}")
        return checkpoint_path

    def load_checkpoint(self, component_name: str) -> Optional[tuple[ExecutionProgress, dict]]:
        """
        Load checkpoint for a component.

        Args:
            component_name: Name of the component

        Returns:
            Tuple of (ExecutionProgress, additional_data) or None if not found
        """
        checkpoint_path = self.models_dir / f"checkpoint_{component_name}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            progress = ExecutionProgress.from_dict(checkpoint.get("progress", {}))
            additional_data = checkpoint.get("additional_data", {})

            return progress, additional_data

        except Exception as e:
            print(f"   Warning: Could not load checkpoint: {e}")
            return None

    def find_partial_results(self) -> dict[str, Any]:
        """
        Find any partial results in the working directory.

        Returns:
            Dictionary with found partial results
        """
        results = {
            "oof_predictions": [],
            "test_predictions": [],
            "models": [],
            "submissions": [],
        }

        # Find OOF predictions
        for oof_file in self.models_dir.glob(self.OOF_PATTERN):
            results["oof_predictions"].append(str(oof_file))

        # Find test predictions
        for test_file in self.models_dir.glob(self.TEST_PATTERN):
            results["test_predictions"].append(str(test_file))

        # Find model files
        model_extensions = [".pkl", ".joblib", ".pth", ".pt", ".h5"]
        for ext in model_extensions:
            for model_file in self.models_dir.glob(f"*{ext}"):
                results["models"].append(str(model_file))

        # Find existing submissions
        for sub_file in self.working_dir.glob("submission*.csv"):
            results["submissions"].append(str(sub_file))

        return results

    def create_partial_submission(
        self,
        sample_submission_path: Path | str,
        output_path: Optional[Path | str] = None,
    ) -> Optional[Path]:
        """
        Create a submission from partial test predictions.

        This method:
        1. Finds all test prediction files
        2. Averages predictions across available folds
        3. Creates submission CSV

        Args:
            sample_submission_path: Path to sample submission template
            output_path: Optional output path (default: submission_partial.csv)

        Returns:
            Path to created submission or None if not possible
        """
        sample_sub_path = Path(sample_submission_path)
        if not sample_sub_path.exists():
            print("   Warning: Sample submission not found")
            return None

        # Find test predictions
        test_preds_files = list(self.models_dir.glob(self.TEST_PATTERN))

        if not test_preds_files:
            print("   No test predictions found for partial submission")
            return None

        print(f"   Found {len(test_preds_files)} test prediction files")

        try:
            # Load sample submission
            sample_sub = pd.read_csv(sample_sub_path)
            id_col = sample_sub.columns[0]
            target_col = sample_sub.columns[1]

            # Load and average predictions
            all_preds = []
            for pred_file in test_preds_files:
                try:
                    preds = np.load(pred_file)
                    all_preds.append(preds)
                    print(f"      Loaded: {pred_file.name} (shape: {preds.shape})")
                except Exception as e:
                    print(f"      Warning: Could not load {pred_file.name}: {e}")

            if not all_preds:
                print("   No valid predictions to average")
                return None

            # Average predictions
            avg_preds = np.mean(all_preds, axis=0)
            print(f"   Averaged {len(all_preds)} prediction files")

            # Handle shape mismatch
            if len(avg_preds) != len(sample_sub):
                print(f"   Warning: Prediction length ({len(avg_preds)}) != sample_sub length ({len(sample_sub)})")
                # Truncate or pad
                if len(avg_preds) > len(sample_sub):
                    avg_preds = avg_preds[:len(sample_sub)]
                else:
                    # Pad with mean
                    pad_length = len(sample_sub) - len(avg_preds)
                    avg_preds = np.concatenate([avg_preds, np.full(pad_length, avg_preds.mean())])

            # Create submission
            submission = sample_sub.copy()
            submission[target_col] = avg_preds

            # Clip predictions if binary classification
            if submission[target_col].min() >= 0 and submission[target_col].max() <= 1:
                submission[target_col] = submission[target_col].clip(0, 1)

            # Save
            if output_path is None:
                output_path = self.working_dir / "submission_partial.csv"
            else:
                output_path = Path(output_path)

            submission.to_csv(output_path, index=False)
            print(f"   Partial submission saved: {output_path.name}")

            return output_path

        except Exception as e:
            print(f"   Error creating partial submission: {e}")
            return None

    def cleanup_checkpoints(self, component_name: Optional[str] = None):
        """
        Remove checkpoint files.

        Args:
            component_name: Specific component to clean up, or None for all
        """
        if component_name:
            checkpoint_path = self.models_dir / f"checkpoint_{component_name}.json"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"   Cleaned up checkpoint: {checkpoint_path.name}")
        else:
            for checkpoint_file in self.models_dir.glob("checkpoint_*.json"):
                checkpoint_file.unlink()
                print(f"   Cleaned up checkpoint: {checkpoint_file.name}")

    def get_recovery_status(self) -> dict[str, Any]:
        """
        Get overall recovery status for the working directory.

        Returns:
            Dictionary with recovery status and available options
        """
        partial_results = self.find_partial_results()

        # Check what we can recover
        can_create_partial = len(partial_results["test_predictions"]) > 0
        has_existing_submission = len(partial_results["submissions"]) > 0

        # Find checkpoints
        checkpoints = list(self.models_dir.glob("checkpoint_*.json"))

        status = {
            "has_checkpoints": len(checkpoints) > 0,
            "checkpoint_count": len(checkpoints),
            "checkpoints": [c.stem.replace("checkpoint_", "") for c in checkpoints],
            "can_create_partial_submission": can_create_partial,
            "has_existing_submission": has_existing_submission,
            "partial_results": partial_results,
        }

        return status
