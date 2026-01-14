"""
Data audit utilities for audio competition validation.

Provides fail-fast validation to prevent wasted compute on broken pipelines.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AuditFailedError(Exception):
    """Raised when data audit fails and execution should stop."""

    def __init__(self, message: str, audit_result: AudioAuditResult | None = None):
        super().__init__(message)
        self.audit_result = audit_result


@dataclass
class AudioAuditResult:
    """Results from audio competition data audit."""

    audio_files_found: int
    audio_extensions: set[str]
    label_files_found: list[Path]
    label_format_detected: str | None  # 'csv', 'txt_space', 'txt_tab', etc.
    sample_labels_preview: list[str]
    warnings: list[str]
    is_valid: bool
    failure_reason: str | None = None
    audio_source_dir: Path | None = None
    train_samples: int = 0
    test_samples: int = 0

    def __post_init__(self) -> None:
        # Convert set to frozenset for hashing if needed
        if isinstance(self.audio_extensions, set):
            self.audio_extensions = frozenset(self.audio_extensions)


# Supported audio file extensions (case-insensitive)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}


def find_audio_files(
    directory: Path,
    extensions: set[str] | None = None,
    case_insensitive: bool = True,
) -> list[Path]:
    """
    Recursively find all audio files in a directory.

    Uses rglob for recursive search with case-insensitive extension matching.

    Args:
        directory: Root directory to search (must be a directory, not a file)
        extensions: Audio extensions to look for (default: AUDIO_EXTENSIONS)
        case_insensitive: Whether to match extensions case-insensitively

    Returns:
        List of audio file paths found, sorted by name
    """
    if extensions is None:
        extensions = AUDIO_EXTENSIONS

    if not directory.exists():
        return []

    # Guard against file paths (e.g., train.csv) - only search directories
    if not directory.is_dir():
        return []

    audio_files = []
    for f in directory.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower() if case_insensitive else f.suffix
        if ext in extensions:
            audio_files.append(f)

    return sorted(audio_files)


def detect_label_format(file_path: Path, sample_lines: int = 20) -> dict[str, Any]:
    """
    Auto-detect label file format using csv.Sniffer with fallbacks.

    Args:
        file_path: Path to label file
        sample_lines: Number of lines to sample for detection

    Returns:
        Dictionary with format info:
        {
            'delimiter': str,
            'has_header': bool,
            'num_columns': int,
            'format_type': str,  # 'csv', 'txt_space', 'txt_tab', etc.
            'encoding': str,
        }
    """
    result = {
        "delimiter": ",",
        "has_header": False,
        "num_columns": 0,
        "format_type": "unknown",
        "encoding": "utf-8",
    }

    try:
        # Try different encodings
        content = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                content = file_path.read_text(encoding=encoding, errors="strict")
                result["encoding"] = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        lines = content.strip().split("\n")[:sample_lines]
        if not lines:
            return result

        sample = "\n".join(lines)

        # Try csv.Sniffer first
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;|")
            result["delimiter"] = dialect.delimiter
            result["has_header"] = csv.Sniffer().has_header(sample)
        except csv.Error:
            # Fallback: detect delimiter by counting occurrences
            delimiters = [(",", "csv"), ("\t", "txt_tab"), (" ", "txt_space"), (";", "csv_semicolon")]
            max_count = 0
            for delim, fmt_type in delimiters:
                count = lines[0].count(delim)
                if count > max_count:
                    max_count = count
                    result["delimiter"] = delim
                    result["format_type"] = fmt_type

        # Determine format type from delimiter
        if result["delimiter"] == ",":
            result["format_type"] = "csv"
        elif result["delimiter"] == "\t":
            result["format_type"] = "txt_tab"
        elif result["delimiter"] == " ":
            result["format_type"] = "txt_space"
        else:
            result["format_type"] = f"delimited_{result['delimiter']!r}"

        # Count columns
        if lines:
            first_line = lines[0]
            result["num_columns"] = len(first_line.split(result["delimiter"]))

        # Check for header by seeing if first line looks different from second
        if len(lines) > 1:
            first_fields = lines[0].split(result["delimiter"])
            second_fields = lines[1].split(result["delimiter"])

            # If first line has all non-numeric fields and second has numeric, likely header
            first_all_text = all(not _is_numeric(f.strip()) for f in first_fields if f.strip())
            second_has_numeric = any(_is_numeric(f.strip()) for f in second_fields if f.strip())

            if first_all_text and second_has_numeric:
                result["has_header"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def _is_numeric(s: str) -> bool:
    """Check if string is numeric (int or float)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def check_id_integrity(
    sample_ids: list[str],
    data_dir: Path,
    extensions: list[str] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate that IDs match files in a directory.

    CRITICAL: Call this early to catch ID-extension mismatches before wasting GPU time.
    This prevents the common error where ID mappings don't include file extensions.

    Args:
        sample_ids: List of file IDs to check (typically from rec_id2filename.txt)
        data_dir: Directory where files should exist
        extensions: Extensions to probe (None = common audio/image types)

    Returns:
        Tuple of (is_valid, message, details_dict)
        - is_valid: True if IDs match files directly
        - message: Human-readable explanation
        - details_dict: Contains 'match_type', 'suggested_extension', 'match_rate'

    Examples:
        >>> is_valid, msg, details = check_id_integrity(
        ...     ['PC1_123', 'PC1_456'], audio_dir
        ... )
        >>> if not is_valid and details.get('suggested_extension'):
        ...     print(f"Add {details['suggested_extension']} to IDs when loading")
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".aiff", ".aif", ".jpg", ".png", ".jpeg"]

    data_dir = Path(data_dir)

    if not data_dir.exists():
        return False, f"Directory does not exist: {data_dir}", {"match_type": "dir_missing"}

    # Filter to first 20 samples
    sample_ids = [str(x).strip() for x in sample_ids[:20] if str(x).strip()]

    if not sample_ids:
        return False, "No sample IDs provided", {"match_type": "no_ids"}

    # Phase 1: Try direct match (IDs already include extension)
    direct_matches = sum(1 for rid in sample_ids if (data_dir / rid).exists())

    if direct_matches == len(sample_ids):
        return True, "All IDs match files directly", {"match_type": "direct", "match_rate": 1.0}

    if direct_matches > 0:
        # Partial direct match - some IDs work, but most likely need extensions
        # Return False to surface the warning - partial matches often indicate
        # inconsistent data that will cause failures downstream
        match_rate = direct_matches / len(sample_ids)
        return False, (
            f"WARNING: Only {direct_matches}/{len(sample_ids)} IDs match files directly. "
            f"Remaining IDs may need file extensions appended."
        ), {
            "match_type": "partial_direct",
            "match_rate": match_rate,
        }

    # Phase 2: Try with extensions
    for ext in extensions:
        ext_matches = sum(1 for rid in sample_ids if (data_dir / f"{rid}{ext}").exists())
        match_rate = ext_matches / len(sample_ids) if sample_ids else 0

        if match_rate >= 0.8:  # 80% threshold
            return False, (
                f"CRITICAL: IDs lack extensions. "
                f"Found files ending in '{ext}'. "
                f"Add '{ext}' to IDs when loading audio files."
            ), {
                "match_type": "extension_required",
                "suggested_extension": ext,
                "match_rate": match_rate,
            }

    # Phase 3: No match found - provide debugging info
    try:
        actual_files = [f.name for f in list(data_dir.iterdir())[:5] if f.is_file()]
        actual_exts = list(set(f.suffix for f in data_dir.iterdir() if f.is_file()))[:5]
    except PermissionError:
        actual_files = ["<permission denied>"]
        actual_exts = []

    return False, (
        f"CRITICAL: IDs do not match any files in {data_dir}. "
        f"Sample IDs: {sample_ids[:3]}. "
        f"Actual files: {actual_files}. "
        f"Extensions found: {actual_exts}"
    ), {
        "match_type": "no_match",
        "sample_ids": sample_ids[:5],
        "actual_files": actual_files,
        "actual_extensions": actual_exts,
    }


def validate_label_file(file_path: Path) -> tuple[bool, str, list[str]]:
    """
    Validate that a label file is parseable.

    Args:
        file_path: Path to label file

    Returns:
        Tuple of (is_valid, error_message, preview_lines)
    """
    if not file_path.exists():
        return False, f"Label file not found: {file_path}", []

    try:
        format_info = detect_label_format(file_path)
        delimiter = format_info["delimiter"]

        # Try to parse first few lines
        content = file_path.read_text(encoding=format_info["encoding"], errors="ignore")
        lines = content.strip().split("\n")[:10]

        preview = []
        for line in lines[:5]:
            fields = line.split(delimiter)
            preview.append(f"[{len(fields)} fields] {line[:80]}...")

        # Basic validation: at least 2 columns (id + label/value)
        if format_info["num_columns"] < 2:
            return (
                False,
                f"Expected at least 2 columns, found {format_info['num_columns']}",
                preview,
            )

        return True, "", preview

    except Exception as e:
        return False, f"Failed to parse: {e}", []


def audit_audio_competition(
    working_dir: Path,
    audio_source_dir: Path | None = None,
    label_files: list[Path] | None = None,
    train_path: Path | None = None,
    test_path: Path | None = None,
    min_audio_files: int = 10,
    strict: bool = True,
) -> AudioAuditResult:
    """
    Comprehensive audit of audio competition data.

    Validates that audio files exist, labels are parseable, and data is sufficient.
    FAIL-FAST: Raises AuditFailedError if critical issues found (when strict=True).

    Args:
        working_dir: Competition working directory
        audio_source_dir: Directory containing audio files (if known)
        label_files: List of label file paths to validate
        train_path: Path to training data directory
        test_path: Path to test data directory
        min_audio_files: Minimum number of audio files required
        strict: If True, raise AuditFailedError on failure. If False, just return result.

    Returns:
        AudioAuditResult with audit details

    Raises:
        AuditFailedError: If strict=True and audit fails
    """
    # Check if audit should be skipped
    if os.getenv("KAGGLE_AGENTS_SKIP_AUDIT", "0").lower() in ("1", "true"):
        return AudioAuditResult(
            audio_files_found=0,
            audio_extensions=set(),
            label_files_found=[],
            label_format_detected=None,
            sample_labels_preview=["AUDIT SKIPPED via KAGGLE_AGENTS_SKIP_AUDIT=1"],
            warnings=["Audit skipped - data may be invalid"],
            is_valid=True,  # Assume valid when skipped
            failure_reason=None,
        )

    warnings: list[str] = []
    audio_files: list[Path] = []
    extensions_found: set[str] = set()

    # Find audio source directory
    search_dirs = []
    if audio_source_dir and audio_source_dir.exists():
        # Only add if it's a directory
        if audio_source_dir.is_dir():
            search_dirs.append(audio_source_dir)
    if train_path and train_path.exists():
        if train_path.is_dir():
            search_dirs.append(train_path)
        else:
            # train_path is a file (e.g., train.csv) - search parent directory instead
            # Audio files are often in sibling directories like train_audio/
            search_dirs.append(train_path.parent)
    if working_dir.exists() and working_dir.is_dir():
        search_dirs.append(working_dir)

    # Common audio directory patterns to search
    audio_dir_patterns = [
        "src_wavs",
        "wavs",
        "audio",
        "audio_files",
        "raw_audio",
        "train_audio",
        "train",
        "train2",  # for whale competition (non-standard naming)
        "test2",   # for whale competition (non-standard naming)
        "essential_data",
    ]

    # Search for audio files
    actual_audio_source = None
    for search_dir in search_dirs:
        files = find_audio_files(search_dir)
        if files:
            audio_files.extend(files)
            extensions_found.update(f.suffix.lower() for f in files)
            actual_audio_source = search_dir
            break

        # Try common subdirectories
        for pattern in audio_dir_patterns:
            subdir = search_dir / pattern
            if subdir.exists():
                files = find_audio_files(subdir)
                if files:
                    audio_files.extend(files)
                    extensions_found.update(f.suffix.lower() for f in files)
                    actual_audio_source = subdir
                    break

        if audio_files:
            break

    # Count train vs test samples if possible (only for directories)
    train_samples = 0
    test_samples = 0
    if train_path and train_path.exists() and train_path.is_dir():
        train_samples = len(find_audio_files(train_path))
    if test_path and test_path.exists() and test_path.is_dir():
        test_samples = len(find_audio_files(test_path))

    # Validate label files
    valid_label_files = []
    label_format = None
    label_preview: list[str] = []

    if label_files:
        for lf in label_files:
            if lf.exists():
                is_valid, error, preview = validate_label_file(lf)
                if is_valid:
                    valid_label_files.append(lf)
                    format_info = detect_label_format(lf)
                    label_format = format_info.get("format_type")
                    label_preview.extend(preview)
                else:
                    warnings.append(f"Label file {lf.name}: {error}")
            else:
                warnings.append(f"Label file not found: {lf}")

    # Determine if audit passed
    failure_reason = None
    is_valid = True

    if len(audio_files) < min_audio_files:
        failure_reason = (
            f"INSUFFICIENT AUDIO DATA: Only {len(audio_files)} audio files found "
            f"(minimum required: {min_audio_files}). "
            f"Searched directories: {[str(d) for d in search_dirs]}. "
            "Check that data is downloaded and paths are correct."
        )
        is_valid = False
    elif not valid_label_files and label_files:
        failure_reason = (
            f"NO VALID LABEL FILES: None of the {len(label_files)} label files could be parsed. "
            f"Warnings: {warnings}"
        )
        is_valid = False

    result = AudioAuditResult(
        audio_files_found=len(audio_files),
        audio_extensions=extensions_found,
        label_files_found=valid_label_files,
        label_format_detected=label_format,
        sample_labels_preview=label_preview,
        warnings=warnings,
        is_valid=is_valid,
        failure_reason=failure_reason,
        audio_source_dir=actual_audio_source,
        train_samples=train_samples,
        test_samples=test_samples,
    )

    # Fail-fast if strict mode and audit failed
    if strict and not is_valid:
        raise AuditFailedError(failure_reason or "Audio competition audit failed", result)

    return result


def print_audit_report(result: AudioAuditResult) -> None:
    """Print a formatted audit report."""
    print("\n" + "=" * 60)
    print("=== AUDIO COMPETITION DATA AUDIT ===")
    print("=" * 60)

    status = "PASSED" if result.is_valid else "FAILED"
    print(f"Status: {status}")
    print(f"Audio files found: {result.audio_files_found}")
    print(f"Extensions: {set(result.audio_extensions)}")
    print(f"Audio source dir: {result.audio_source_dir}")
    print(f"Train samples: {result.train_samples}")
    print(f"Test samples: {result.test_samples}")
    print(f"Label files found: {len(result.label_files_found)}")
    print(f"Label format: {result.label_format_detected}")

    if result.sample_labels_preview:
        print("\nLabel preview:")
        for line in result.sample_labels_preview[:3]:
            print(f"  {line}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.failure_reason:
        print(f"\nFAILURE REASON: {result.failure_reason}")

    print("=" * 60 + "\n")
