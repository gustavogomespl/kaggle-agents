"""Data audit node for the Kaggle Agents workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.state import KaggleState
from ...utils.data_audit import AuditFailedError, audit_audio_competition, print_audit_report


def data_audit_node(state: KaggleState) -> dict[str, Any]:
    """
    Audit competition data before expensive processing begins.

    For audio competitions, validates that audio files exist and labels are parseable.
    FAIL-FAST: Raises AuditFailedError if critical data is missing.

    Args:
        state: Current state

    Returns:
        State updates with audit results
    """
    print("\n" + "=" * 60)
    print("= DATA AUDIT")
    print("=" * 60)

    domain = state.get("domain_detected", "")
    data_files = state.get("data_files", {})
    working_dir = Path(state.get("working_directory", "."))

    # Only run domain-specific audits for supported domains
    if domain and "audio" in domain.lower():
        print("   Running audio competition audit...")

        audio_source = data_files.get("audio_source")
        audio_source_dir = Path(audio_source) if audio_source else None

        train_path = Path(data_files.get("train", "")) if data_files.get("train") else None
        test_path = Path(data_files.get("test", "")) if data_files.get("test") else None

        label_files = data_files.get("label_files", [])
        label_paths = [Path(lf) for lf in label_files] if label_files else []

        try:
            result = audit_audio_competition(
                working_dir=working_dir,
                audio_source_dir=audio_source_dir,
                label_files=label_paths,
                train_path=train_path,
                test_path=test_path,
                min_audio_files=10,
                strict=True,  # Fail-fast by default
            )
            print_audit_report(result)

            return {
                "data_audit_result": {
                    "is_valid": result.is_valid,
                    "audio_files_found": result.audio_files_found,
                    "audio_source_dir": str(result.audio_source_dir) if result.audio_source_dir else None,
                    "label_files_found": [str(lf) for lf in result.label_files_found],
                    "warnings": result.warnings,
                },
                "last_updated": datetime.now(),
            }

        except AuditFailedError as e:
            print(f"\n   AUDIT FAILED: {e}")
            print("   Stopping execution to prevent wasted compute.")
            # Re-raise to halt the workflow
            raise

    else:
        print(f"   Skipping domain-specific audit for domain: {domain}")
        print("   (Audio audit only runs for audio_* domains)")

    return {
        "data_audit_result": {"is_valid": True, "skipped": True},
        "last_updated": datetime.now(),
    }
