"""Domain detection node for the Kaggle Agents workflow."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.state import KaggleState
from ...domain import detect_competition_domain


# Domain type mappings for environment variable forcing
FORCE_TYPE_TO_DOMAIN = {
    # Seq2seq / text normalization
    "seq2seq": "seq_to_seq",
    "seq_to_seq": "seq_to_seq",
    "text_normalization": "seq_to_seq",
    "text-normalization": "seq_to_seq",
    "textnormalization": "seq_to_seq",
    # Image domains
    "image": "image_classification",
    "image_classification": "image_classification",
    "image-classification": "image_classification",
    "image_to_image": "image_to_image",
    "image-to-image": "image_to_image",
    # Audio domains
    "audio": "audio_classification",
    "audio_classification": "audio_classification",
    "audio-classification": "audio_classification",
    "audio_tagging": "audio_tagging",
    "audio-tagging": "audio_tagging",
    # Text domains
    "text": "text_classification",
    "text_classification": "text_classification",
    "text-classification": "text_classification",
    "nlp": "text_classification",
    # Tabular domains
    "tabular": "tabular_classification",
    "tabular_classification": "tabular_classification",
    "tabular-classification": "tabular_classification",
    "tabular_regression": "tabular_regression",
    "tabular-regression": "tabular_regression",
    "regression": "tabular_regression",
}


def domain_detection_node(state: KaggleState) -> dict[str, Any]:
    """
    Detect competition domain using LLM-First approach.

    Args:
        state: Current state

    Returns:
        State updates with domain detection
    """
    print("\n" + "=" * 60)
    print("= DOMAIN DETECTION")
    print("=" * 60)

    competition_info = state["competition_info"]
    working_dir = state["working_directory"]

    submission_format_type = None
    submission_format_metadata: dict[str, Any] = {}

    # ========================================================================
    # FORCE DOMAIN OVERRIDE via environment variable
    # This takes ABSOLUTE precedence over all detection methods
    # ========================================================================
    env_force_data_type = os.getenv("KAGGLE_AGENTS_FORCE_DATA_TYPE", "")
    env_data_type = os.getenv("KAGGLE_AGENTS_DATA_TYPE", "")
    env_force_domain = os.getenv("KAGGLE_AGENTS_FORCE_DOMAIN", "")

    forced_type = (env_force_data_type or env_data_type or env_force_domain).strip().lower()

    # Debug: show which env vars are set
    if env_force_data_type or env_data_type or env_force_domain:
        print(f"   [ENV] KAGGLE_AGENTS_FORCE_DATA_TYPE={env_force_data_type!r}")
        print(f"   [ENV] KAGGLE_AGENTS_DATA_TYPE={env_data_type!r}")
        print(f"   [ENV] KAGGLE_AGENTS_FORCE_DOMAIN={env_force_domain!r}")
        print(f"   [ENV] Resolved forced_type={forced_type!r}")

    if forced_type and forced_type in FORCE_TYPE_TO_DOMAIN:
        forced_domain = FORCE_TYPE_TO_DOMAIN[forced_type]
        print(f"   ⚠️ Domain FORCED via env var: {forced_domain}")
        print(f"      (KAGGLE_AGENTS_FORCE_DATA_TYPE={forced_type})")
        competition_info.domain = forced_domain
        return {
            "domain_detected": forced_domain,
            "domain_confidence": 1.0,  # User explicitly requested
            "submission_format_type": None,
            "submission_format_metadata": {},
            "submission_format": {},
            "last_updated": datetime.now(),
        }
    # ========================================================================

    # Get LLM for domain detection (use planner's LLM with low temperature)
    from ...core.config import get_llm_for_role

    try:
        llm = get_llm_for_role(role="planner", temperature=0.0)
    except Exception as e:
        print(f"   Warning: Could not get LLM for domain detection: {e}")
        llm = None

    # Use multi-signal domain detection (delegated to detector.py)
    # Pass state for SOTA tags extraction if available
    domain, confidence = detect_competition_domain(
        competition_info, working_dir, llm=llm, state=state
    )

    # Check for image-to-image overrides (pixel-level submission format)
    data_files = state.get("data_files") or {}
    data_type = data_files.get("data_type")

    if data_type == "image":
        # Check for clean/target directory (e.g., train_cleaned for denoising)
        clean_train_path = data_files.get("clean_train", "")
        if clean_train_path and Path(clean_train_path).exists():
            domain = "image_to_image"
            confidence = 0.95
            print("   Override: Detected clean/target directory -> image_to_image")

        # Check for pixel-level submission format (many rows per image)
        sample_sub_path = data_files.get("sample_submission", "") or state.get(
            "sample_submission_path", ""
        )
        test_path = data_files.get("test", "") or state.get("test_data_path", "")
        if sample_sub_path and Path(sample_sub_path).exists():
            from ...domain import detect_submission_format

            sub_format, sub_meta = detect_submission_format(
                sample_sub_path, test_path if test_path else None, competition_info
            )
            submission_format_type = sub_format
            submission_format_metadata = sub_meta
            if sub_format == "pixel_level":
                domain = "image_to_image"
                confidence = 0.95
                print("   Override: Detected pixel-level submission format -> image_to_image")
                print(f"      Expected rows: {sub_meta.get('expected_rows', 'unknown')}")
                print(f"      ID pattern: {sub_meta.get('id_pattern', 'unknown')}")

    if not domain or confidence < 0.5:
        data_type = str(data_files.get("data_type", "")).lower()
        fallback_domain = None
        if data_type == "image":
            fallback_domain = "image_classification"
        elif data_type == "audio":
            fallback_domain = "audio_classification"
        elif data_type == "text":
            fallback_domain = "text_classification"
        elif data_type == "tabular":
            fallback_domain = "tabular"
        if fallback_domain:
            print(f"   Fallback domain from data_type='{data_type}': {fallback_domain}")
            domain = fallback_domain
            confidence = max(confidence, 0.6)

    print(f"\n Domain Detected: {domain}")
    print(f"   Confidence: {confidence:.1%}")

    competition_info.domain = domain

    return {
        "domain_detected": domain,
        "domain_confidence": confidence,
        "submission_format_type": submission_format_type,
        "submission_format_metadata": submission_format_metadata,
        "submission_format": {"type": submission_format_type, **submission_format_metadata}
        if submission_format_type
        else {},
        "last_updated": datetime.now(),
    }
