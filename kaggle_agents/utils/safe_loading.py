"""Smart file path resolution for audio/image files with missing extensions.

This module provides robust file location utilities that handle common issues
like missing file extensions and case sensitivity mismatches.

Usage:
    from kaggle_agents.utils.safe_loading import smart_locate_file, build_id_to_path_map

    # Find a single file
    path = smart_locate_file(audio_dir, "PC1_123")  # Returns '/data/audio/PC1_123.wav'

    # Build a mapping for multiple IDs
    id_to_path, unresolved = build_id_to_path_map(id_list, audio_dir)
"""

from __future__ import annotations

import glob as glob_module
from pathlib import Path
from typing import Any


# Common extensions by data type
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]


def smart_locate_file(
    base_dir: Path | str,
    file_id: str,
    likely_extensions: list[str] | None = None,
    case_variants: bool = True,
) -> str | None:
    """
    Robustly locate a file, handling missing extensions and case sensitivity.

    This function probes multiple extension variants to find a file when the
    exact path doesn't exist. Useful for competitions where ID-to-filename
    mappings don't include file extensions.

    Args:
        base_dir: Directory to search in
        file_id: ID or partial filename (may lack extension)
        likely_extensions: Extensions to try ['.wav', '.mp3'], or None for auto-detect
        case_variants: Try uppercase/lowercase extension variants

    Returns:
        Full path as string if found, None if not found

    Examples:
        >>> smart_locate_file(Path("audio"), "PC1_123")
        '/data/audio/PC1_123.wav'  # Found with .wav extension

        >>> smart_locate_file(Path("images"), "img_001", ['.jpg', '.png'])
        '/data/images/img_001.PNG'  # Found with uppercase extension
    """
    base_dir = Path(base_dir)
    file_id = str(file_id).strip()

    # Handle empty or invalid inputs
    if not file_id or not base_dir.exists():
        return None

    # 1. Direct exact match (ID already has extension)
    direct_path = base_dir / file_id
    if direct_path.exists():
        return str(direct_path)

    # 2. Try with extensions
    if likely_extensions is None:
        # Auto-detect from directory contents
        likely_extensions = _detect_extensions_in_dir(base_dir)

    for ext in likely_extensions:
        ext = f".{ext.lstrip('.')}"  # Normalize: ensure starts with dot

        # Try exact case
        candidate = base_dir / f"{file_id}{ext}"
        if candidate.exists():
            return str(candidate)

        if case_variants:
            # Try lowercase
            candidate_lower = base_dir / f"{file_id}{ext.lower()}"
            if candidate_lower.exists():
                return str(candidate_lower)

            # Try uppercase
            candidate_upper = base_dir / f"{file_id}{ext.upper()}"
            if candidate_upper.exists():
                return str(candidate_upper)

    # 3. Glob fallback (more expensive, handles unusual extensions)
    # Escape glob special characters in file_id to prevent pattern injection
    escaped_id = glob_module.escape(file_id)
    matches = list(base_dir.glob(f"{escaped_id}.*"))
    if matches:
        return str(matches[0])

    # 4. Case-insensitive stem match (last resort for case-mismatched filenames)
    try:
        for f in base_dir.iterdir():
            if f.is_file() and f.stem.lower() == file_id.lower():
                return str(f)
    except PermissionError:
        pass

    return None


def _detect_extensions_in_dir(directory: Path, sample_size: int = 20) -> list[str]:
    """
    Detect common extensions in a directory by sampling files.

    Args:
        directory: Directory to sample
        sample_size: Number of files to sample

    Returns:
        List of extensions found, prioritizing common audio/image types
    """
    extensions: set[str] = set()
    count = 0

    try:
        for f in directory.iterdir():
            if f.is_file() and f.suffix:
                extensions.add(f.suffix.lower())
                count += 1
                if count >= sample_size:
                    break
    except PermissionError:
        pass

    # Return detected extensions, prioritizing common audio/image types
    result = []
    for ext in AUDIO_EXTENSIONS + IMAGE_EXTENSIONS:
        if ext in extensions:
            result.append(ext)

    # Add any other detected extensions
    for ext in extensions:
        if ext not in result:
            result.append(ext)

    # Default to audio extensions if nothing detected
    return result if result else AUDIO_EXTENSIONS


def build_id_to_path_map(
    id_list: list[str],
    base_dir: Path | str,
    extensions: list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """
    Build a mapping from IDs to resolved file paths.

    This is useful for preprocessing a list of file IDs to their actual paths,
    handling missing extensions in the process.

    Args:
        id_list: List of file IDs (potentially without extensions)
        base_dir: Directory containing files
        extensions: Extensions to try (None = auto-detect)
        verbose: Print warnings for unresolved IDs

    Returns:
        Tuple of (id_to_path_map, unresolved_ids)

    Examples:
        >>> ids = ['PC1_123', 'PC1_456', 'PC1_789']
        >>> id_map, missing = build_id_to_path_map(ids, audio_dir)
        >>> print(f"Resolved {len(id_map)}, missing {len(missing)}")
        Resolved 3, missing 0
    """
    base_dir = Path(base_dir)
    id_to_path: dict[str, str] = {}
    unresolved: list[str] = []

    # Cache detected extensions for efficiency
    if extensions is None:
        extensions = _detect_extensions_in_dir(base_dir)

    for file_id in id_list:
        path = smart_locate_file(base_dir, str(file_id), extensions)
        if path:
            id_to_path[str(file_id)] = path
        else:
            unresolved.append(str(file_id))

    if verbose and unresolved:
        print(f"[WARNING] Could not resolve {len(unresolved)}/{len(id_list)} file IDs")
        print(f"[WARNING] Sample unresolved: {unresolved[:5]}")

    return id_to_path, unresolved


def detect_extension_requirement(
    sample_ids: list[str],
    data_dir: Path | str,
    extensions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Detect if IDs require extensions to be appended.

    This function checks if sample IDs need an extension added to find
    corresponding files. Useful for early validation in pipelines.

    Args:
        sample_ids: Sample of file IDs to check
        data_dir: Directory where files should exist
        extensions: Extensions to probe (None = common audio/image types)

    Returns:
        Dictionary with detection results:
        {
            'direct_match': bool - True if IDs match files directly
            'extension_required': str | None - Extension needed (e.g., '.wav')
            'match_rate': float - Percentage of IDs that matched
            'message': str - Human-readable summary
        }

    Examples:
        >>> result = detect_extension_requirement(['PC1_123', 'PC1_456'], audio_dir)
        >>> if result['extension_required']:
        ...     print(f"Add {result['extension_required']} to IDs")
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".aiff", ".aif", ".jpg", ".png"]

    data_dir = Path(data_dir)
    sample_ids = [str(x) for x in sample_ids[:20]]

    if not data_dir.exists():
        return {
            "direct_match": False,
            "extension_required": None,
            "match_rate": 0.0,
            "message": f"Directory does not exist: {data_dir}",
        }

    # Phase 1: Check direct match
    direct_matches = sum(1 for rid in sample_ids if (data_dir / rid).exists())

    if direct_matches == len(sample_ids):
        return {
            "direct_match": True,
            "extension_required": None,
            "match_rate": 1.0,
            "message": "All IDs match files directly (no extension needed)",
        }

    # Phase 2: Check with extensions
    for ext in extensions:
        ext_matches = sum(1 for rid in sample_ids if (data_dir / f"{rid}{ext}").exists())
        match_rate = ext_matches / len(sample_ids) if sample_ids else 0

        if match_rate >= 0.8:  # 80% threshold
            return {
                "direct_match": False,
                "extension_required": ext,
                "match_rate": match_rate,
                "message": f"IDs require '{ext}' extension ({match_rate:.0%} match rate)",
            }

    # Phase 3: No good match found
    return {
        "direct_match": False,
        "extension_required": None,
        "match_rate": 0.0,
        "message": f"Could not match IDs to files. Sample: {sample_ids[:3]}",
    }


# Export for module interface
__all__ = [
    "smart_locate_file",
    "build_id_to_path_map",
    "detect_extension_requirement",
    "AUDIO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
]
