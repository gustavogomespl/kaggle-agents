"""Canonical data preparation node for the Kaggle Agents workflow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...core.state import KaggleState
from ...core.state.contracts import CanonicalDataContract
from ...utils.data_contract import prepare_canonical_data


def canonical_data_preparation_node(state: KaggleState) -> dict[str, Any]:
    """
    Prepare canonical data contract for consistent data handling.

    This node creates the canonical data artifacts that ALL model components
    must consume. This ensures consistent:
    - Row sampling (same train_ids across all components)
    - Fold assignments (same folds.npy for all CV)
    - Feature columns (intersection of train/test to prevent schema mismatch)
    - Target alignment (y.npy aligned with train_ids)

    Args:
        state: Current state

    Returns:
        State updates with canonical data paths
    """
    print("\n" + "=" * 60)
    print("= CANONICAL DATA PREPARATION")
    print("=" * 60)

    working_dir = Path(state["working_directory"])
    data_files = state.get("data_files", {})
    target_col = state.get("target_col", "target")

    # Get train and test paths
    train_path = data_files.get("train_csv") or data_files.get("train")
    test_path = data_files.get("test_csv") or data_files.get("test")

    # Handle non-tabular data (images, audio)
    data_type = str(data_files.get("data_type", "")).lower()
    if data_type == "image":
        print(f"   Skipping canonical data prep for {data_type} data type")
        print("   (Image competitions use different data flow)")
        return {
            "canonical_data_prepared": False,
            "canonical_data_skipped_reason": f"{data_type} data type",
            "last_updated": datetime.now(),
        }

    # For audio: try filename-based label extraction if no train.csv
    if data_type == "audio":
        train_csv_path = working_dir / "train.csv"
        if not train_csv_path.exists():
            print("   Audio competition without train.csv detected")
            print("   Attempting to create canonical data from audio filenames...")

            # Import detection mixin for filename-based label extraction
            from ...mlebench.data_adapter.detection import DetectionMixin

            detector = DetectionMixin()
            train_dir = working_dir / "train"

            if train_dir.exists():
                result = detector.create_canonical_from_audio_filenames(
                    audio_dir=train_dir,
                    canonical_dir=working_dir / "canonical",
                    n_folds=5,
                )

                if result.get("success"):
                    print(f"   Created canonical data from {result['metadata']['canonical_rows']} audio files")
                    return {
                        "canonical_data_prepared": True,
                        "canonical_dir": result["canonical_dir"],
                        "canonical_train_ids_path": result["train_ids_path"],
                        "canonical_y_path": result["y_path"],
                        "canonical_folds_path": result["folds_path"],
                        "canonical_metadata": result["metadata"],
                        "canonical_data_skipped_reason": None,
                        "last_updated": datetime.now(),
                    }
                else:
                    print(f"   Failed to extract labels from filenames: {result.get('error')}")

            # Fallback: skip canonical data for audio without labels
            print("   Skipping canonical data prep for audio (no train.csv or filename labels)")
            return {
                "canonical_data_prepared": False,
                "canonical_data_skipped_reason": "audio without train.csv or filename labels",
                "last_updated": datetime.now(),
            }

    # Validate paths exist
    if not train_path or not Path(train_path).exists():
        print(f"   Warning: Train path not found: {train_path}")
        return {
            "canonical_data_prepared": False,
            "canonical_data_skipped_reason": "train path not found",
            "last_updated": datetime.now(),
        }

    if not test_path or not Path(test_path).exists():
        print(f"   Warning: Test path not found: {test_path}")
        # Continue anyway - test path is optional for canonical prep

    # Determine max_rows for sampling based on config
    fast_mode = state.get("fast_mode", False)
    timeout_s = state.get("timeout_s")

    # Budget-aware sampling thresholds
    max_rows = None
    if fast_mode:
        max_rows = 50_000
        print(f"   Fast mode: sampling to {max_rows:,} rows")
    elif timeout_s and timeout_s < 1800:  # Less than 30 min
        max_rows = 200_000
        print(f"   Short timeout ({timeout_s}s): sampling to {max_rows:,} rows")

    # Detect task type from domain for seq2seq handling
    domain = state.get("domain_detected", "tabular")
    competition_name = state.get("competition_name", "").lower()
    seq2seq_domains = {"seq_to_seq", "text_normalization", "translation", "summarization"}

    # Determine task_type with priority:
    # 1. Specific text_normalization detection from competition name
    # 2. Domain detected as seq_to_seq variant
    # 3. Default to tabular
    text_norm_keywords = ["normalization", "normalize", "text-norm", "tts"]
    is_text_norm = any(kw in competition_name for kw in text_norm_keywords)

    if is_text_norm:
        task_type = "text_normalization"
        print("   Task type: text_normalization (detected from competition name)")
    elif domain in seq2seq_domains:
        # Map generic seq_to_seq to specific type if possible
        task_type = domain if domain != "seq_to_seq" else "seq2seq"
        print(f"   Task type from domain: {task_type}")
    else:
        task_type = "tabular"

    try:
        canonical_result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path if test_path and Path(test_path).exists() else train_path,
            target_col=target_col,
            output_dir=working_dir,
            max_rows=max_rows,
            fast_mode=fast_mode,
            timeout_s=timeout_s,
            task_type=task_type,
        )

        # CRITICAL: Validate canonical data is not empty
        canonical_rows = canonical_result.get("metadata", {}).get("canonical_rows", 0)
        if canonical_rows == 0:
            print("\n   ⚠️ CRITICAL: canonical_rows == 0, data discovery failed!")
            print("   Attempting fallback: scan for audio files with labels in filenames...")

            # Try audio filename-based fallback
            from ...mlebench.data_adapter.detection import DetectionMixin

            detector = DetectionMixin()
            train_dir = working_dir / "train"

            if train_dir.exists():
                fallback_result = detector.create_canonical_from_audio_filenames(
                    audio_dir=train_dir,
                    canonical_dir=working_dir / "canonical",
                    n_folds=5,
                )

                if fallback_result.get("success"):
                    print(f"   ✅ Fallback succeeded: {fallback_result['metadata']['canonical_rows']} files")
                    return {
                        "canonical_data_prepared": True,
                        "canonical_dir": fallback_result["canonical_dir"],
                        "canonical_train_ids_path": fallback_result["train_ids_path"],
                        "canonical_y_path": fallback_result["y_path"],
                        "canonical_folds_path": fallback_result["folds_path"],
                        "canonical_metadata": fallback_result["metadata"],
                        "last_updated": datetime.now(),
                    }

            print("   ❌ Fallback failed: No audio files with labels found")
            return {
                "canonical_data_prepared": False,
                "canonical_data_skipped_reason": "canonical_rows=0 and no filename labels",
                "last_updated": datetime.now(),
            }

        print("\n   Canonical data artifacts created:")
        print(f"      train_ids: {canonical_result['metadata']['canonical_rows']:,} rows")
        print(f"      n_folds: {canonical_result['metadata']['n_folds']}")
        print(f"      n_features: {canonical_result['metadata']['n_features']}")

        if canonical_result["metadata"].get("sampled"):
            print(f"      Sampled from {canonical_result['metadata']['original_rows']:,} original rows")

        if canonical_result["metadata"].get("group_col"):
            print(f"      Group column: {canonical_result['metadata']['group_col']} (GroupKFold)")

        # Compute hashes for fingerprinting
        train_ids_arr = np.load(canonical_result["train_ids_path"], allow_pickle=True)
        y_arr = np.load(canonical_result["y_path"], allow_pickle=True)
        folds_arr = np.load(canonical_result["folds_path"])

        train_ids_hash = CanonicalDataContract.compute_array_hash(train_ids_arr)
        y_hash = CanonicalDataContract.compute_array_hash(y_arr)
        folds_hash = CanonicalDataContract.compute_array_hash(folds_arr)

        # Compute schema hash from feature columns
        # NOTE: Schema hash currently uses placeholder dtypes ("unknown") instead of actual dtypes.
        # This means the hash will detect column additions/removals/reordering but NOT dtype changes.
        # Loading actual dtypes would require reading the full dataset which is expensive.
        # For full dtype validation, compare against the original train.csv at runtime.
        with open(canonical_result["feature_cols_path"]) as f:
            feature_cols = json.load(f)
        train_schema_hash = CanonicalDataContract.compute_schema_hash(
            columns=feature_cols, dtypes=["unknown"] * len(feature_cols)
        )

        # Create CanonicalDataContract
        metadata = canonical_result["metadata"]
        canonical_contract = CanonicalDataContract(
            canonical_dir=canonical_result["canonical_dir"],
            train_ids_path=canonical_result["train_ids_path"],
            y_path=canonical_result["y_path"],
            folds_path=canonical_result["folds_path"],
            feature_cols_path=canonical_result["feature_cols_path"],
            metadata_path=str(Path(canonical_result["canonical_dir"]) / "metadata.json"),
            n_train=metadata["canonical_rows"],
            n_test=0,  # Will be updated when test data is processed
            n_folds=metadata["n_folds"],
            id_col=metadata["id_col"],
            target_col=metadata["target_col"],
            is_classification=metadata["is_classification"],
            folds_hash=folds_hash,
            y_hash=y_hash,
            train_ids_hash=train_ids_hash,
            train_schema_hash=train_schema_hash,
        )

        print(f"      folds_hash: {folds_hash[:8]}...")
        print(f"      y_hash: {y_hash[:8]}...")
        print(f"      train_ids_hash: {train_ids_hash[:8]}...")

        return {
            "canonical_data_prepared": True,
            "canonical_dir": canonical_result["canonical_dir"],
            "canonical_train_ids_path": canonical_result["train_ids_path"],
            "canonical_y_path": canonical_result["y_path"],
            "canonical_folds_path": canonical_result["folds_path"],
            "canonical_feature_cols_path": canonical_result["feature_cols_path"],
            "canonical_metadata": canonical_result["metadata"],
            "canonical_contract": canonical_contract.to_dict(),
            "last_updated": datetime.now(),
        }

    except Exception as e:
        print(f"\n   Error preparing canonical data: {e}")
        print("   Continuing without canonical data contract...")
        return {
            "canonical_data_prepared": False,
            "canonical_data_error": str(e),
            "last_updated": datetime.now(),
        }
