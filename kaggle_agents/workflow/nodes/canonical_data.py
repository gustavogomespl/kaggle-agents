"""Canonical data preparation node for the Kaggle Agents workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.state import KaggleState
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

    # Skip for non-tabular data (images, audio)
    data_type = str(data_files.get("data_type", "")).lower()
    if data_type in {"image", "audio"}:
        print(f"   Skipping canonical data prep for {data_type} data type")
        print("   (Image/audio competitions use different data flow)")
        return {
            "canonical_data_prepared": False,
            "canonical_data_skipped_reason": f"{data_type} data type",
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

        print("\n   Canonical data artifacts created:")
        print(f"      train_ids: {canonical_result['metadata']['canonical_rows']:,} rows")
        print(f"      n_folds: {canonical_result['metadata']['n_folds']}")
        print(f"      n_features: {canonical_result['metadata']['n_features']}")

        if canonical_result["metadata"].get("sampled"):
            print(f"      Sampled from {canonical_result['metadata']['original_rows']:,} original rows")

        if canonical_result["metadata"].get("group_col"):
            print(f"      Group column: {canonical_result['metadata']['group_col']} (GroupKFold)")

        return {
            "canonical_data_prepared": True,
            "canonical_dir": canonical_result["canonical_dir"],
            "canonical_train_ids_path": canonical_result["train_ids_path"],
            "canonical_y_path": canonical_result["y_path"],
            "canonical_folds_path": canonical_result["folds_path"],
            "canonical_feature_cols_path": canonical_result["feature_cols_path"],
            "canonical_metadata": canonical_result["metadata"],
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
