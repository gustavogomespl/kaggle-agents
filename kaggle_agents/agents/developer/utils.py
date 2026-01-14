"""
Utility functions for developer agent.

Provides helper methods for dataset info extraction and code parsing.
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...core.state import KaggleState


class DeveloperUtilsMixin:
    """Mixin providing utility methods."""

    def _get_dataset_info(self, working_dir: Path, state: "KaggleState" = None) -> str:
        """
        Read dataset columns and file system structure to provide to LLM.

        Args:
            working_dir: Working directory containing data files
            state: Current state (optional)

        Returns:
            Formatted string with dataset information including file system structure
        """
        import pandas as pd

        info_parts = []
        data_files = {}
        if state:
            data_files = state.get("data_files", {}) if isinstance(state, dict) else {}

        # 1. FILE SYSTEM STRUCTURE (CRITICAL for non-tabular competitions)
        info_parts.append("**FILE SYSTEM STRUCTURE (Use this to build paths):**")
        data_dirs_to_check = [
            "train", "test", "audio", "images", "train_images", "test_images",
            "essential_data", "supplemental_data", "data", "train_audio", "test_audio"
        ]

        found_dirs = []
        for dirname in data_dirs_to_check:
            dir_path = working_dir / dirname
            if dir_path.exists() and dir_path.is_dir():
                try:
                    files = [f for f in dir_path.rglob("*") if f.is_file()]
                    count = len(files)
                    if count == 0:
                        continue

                    # Analyze extensions
                    extensions: dict[str, int] = {}
                    for f in files[:200]:
                        ext = f.suffix.lower()
                        if ext:
                            extensions[ext] = extensions.get(ext, 0) + 1

                    # Get sample filenames to show naming patterns
                    sample_files = [f.name for f in files[:5]]

                    # Extract ID patterns from filenames (stem without extension)
                    sample_ids = [f.stem for f in files[:5]]

                    dir_info = f"- Directory `{dirname}/`: {count} files found"
                    info_parts.append(dir_info)

                    if extensions:
                        dominant_ext = max(extensions, key=extensions.get)
                        info_parts.append(f"  - Dominant extension: `{dominant_ext}` ({extensions[dominant_ext]} files)")
                        if len(extensions) > 1:
                            other_exts = [f"{k}({v})" for k, v in extensions.items() if k != dominant_ext][:3]
                            info_parts.append(f"  - Other extensions: {', '.join(other_exts)}")

                    info_parts.append(f"  - Sample files: {sample_files}")
                    info_parts.append(f"  - Sample IDs (stems): {sample_ids}")
                    found_dirs.append((dirname, count, extensions))
                except (PermissionError, OSError):
                    continue

        if not found_dirs:
            info_parts.append("  - No standard data directories found. Check working directory structure.")

        # 2. CSV FILE ANALYSIS
        info_parts.append("\n**CSV FILES:**")

        def _analyze_csv(csv_path: Path) -> bool:
            """Analyze a CSV file and append info. Returns True if successful."""
            if not csv_path.exists():
                return False
            try:
                df = pd.read_csv(csv_path, nrows=5)
                columns = df.columns.tolist()

                info_parts.append(f"- `{csv_path.name}` (at {csv_path.parent.name}/): {len(columns)} columns")
                info_parts.append(f"  - Columns: {', '.join(columns)}")

                # Detect target column
                target_col = "UNKNOWN"
                if state and state.get("target_col"):
                    target_col = state["target_col"]
                else:
                    target_candidates = [
                        c for c in columns
                        if c.lower() in ["target", "label", "y", "class", "species", "category"]
                    ]
                    target_col = target_candidates[0] if target_candidates else columns[-1] if len(columns) > 1 else "UNKNOWN"

                info_parts.append(f"  - Likely target column: `{target_col}`")

                # Show sample values for ID column (usually first column)
                if columns:
                    id_col = columns[0]
                    sample_ids = df[id_col].astype(str).tolist()
                    info_parts.append(f"  - Sample IDs from `{id_col}`: {sample_ids}")

                return True
            except Exception as e:
                info_parts.append(f"- `{csv_path.name}`: Error reading ({e})")
                return False

        csv_found = False

        # First: Check train_csv from data_files (any filename)
        if data_files.get("train_csv"):
            train_csv_path = Path(data_files["train_csv"])
            if _analyze_csv(train_csv_path):
                csv_found = True

        # Second: Check standard CSV names in working directory root
        if not csv_found:
            for csv_name in ["train.csv", "train_labels.csv", "metadata.csv"]:
                csv_path = working_dir / csv_name
                if _analyze_csv(csv_path):
                    csv_found = True
                    break

        # Third: Scan for CSVs inside discovered data directories (CRITICAL for MLSP-2013-Birds)
        # The training labels might be inside essential_data/, supplemental_data/, etc.
        if not csv_found and found_dirs:
            info_parts.append("  - No CSV at root, scanning inside data directories...")
            for dirname, count, extensions in found_dirs:
                if ".csv" in extensions:
                    dir_path = working_dir / dirname
                    for csv_file in sorted(dir_path.glob("*.csv"))[:3]:  # Limit to first 3
                        if _analyze_csv(csv_file):
                            csv_found = True

        if not csv_found:
            info_parts.append("  - No CSV files found at standard locations or inside data directories")
            info_parts.append("  - You may need to scan directories directly for data files")

        # 3. CRITICAL PATH BUILDING GUIDANCE
        if found_dirs:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

            has_audio = any(
                ext in audio_exts
                for _, _, exts in found_dirs
                for ext in exts
            )
            has_images = any(
                ext in image_exts
                for _, _, exts in found_dirs
                for ext in exts
            )

            if has_audio or has_images:
                info_parts.append("\n**PATH BUILDING GUIDANCE (CRITICAL):**")
                info_parts.append("  - DO NOT assume `path = dir / f'{id}.ext'` - IDs often don't match filenames")
                info_parts.append("  - INSTEAD: Scan directory first, build id_to_path mapping:")
                info_parts.append("    ```python")
                info_parts.append("    from pathlib import Path")
                info_parts.append("    data_dir = Path('...')  # Use directory from FILE SYSTEM STRUCTURE above")
                info_parts.append("    all_files = list(data_dir.rglob('*.*'))  # Get all files recursively")
                info_parts.append("    id_to_path = {f.stem: f for f in all_files if f.is_file()}")
                info_parts.append("    # Then: df['path'] = df['id_col'].map(id_to_path)")
                info_parts.append("    # Filter: df = df[df['path'].notna()]")
                info_parts.append("    ```")

        # 4. CUSTOM DATA FORMAT (from data_format_discovery_node)
        if state and state.get("parsing_info"):
            parsing_info = state["parsing_info"]
            info_parts.append("\n**CUSTOM DATA FORMAT (discovered from competition page):**")
            info_parts.append(f"  - Format type: `{parsing_info.get('format_type', 'unknown')}`")
            info_parts.append(f"  - ID column: `{parsing_info.get('id_column', 'unknown')}`")
            info_parts.append(f"  - Target column: `{parsing_info.get('target_column', 'unknown')}`")

            if parsing_info.get("train_file"):
                info_parts.append(f"  - Train file: `{parsing_info.get('train_file')}`")
            if parsing_info.get("test_file"):
                info_parts.append(f"  - Test file: `{parsing_info.get('test_file')}`")
            if parsing_info.get("train_test_split_method"):
                info_parts.append(f"  - Split method: `{parsing_info.get('train_test_split_method')}`")
            if parsing_info.get("multi_label"):
                info_parts.append("  - **Multi-label**: Yes (one sample can have multiple labels)")

            if parsing_info.get("column_mapping"):
                col_map = parsing_info["column_mapping"]
                info_parts.append(f"  - Column mapping: {col_map}")

            if parsing_info.get("notes"):
                info_parts.append(f"  - **Notes**: {parsing_info.get('notes')}")

            # Include loading code if available
            loading_code = parsing_info.get("loading_code") or state.get("data_loading_code")
            if loading_code:
                info_parts.append("\n**DATA LOADING CODE (use this to load the data):**")
                info_parts.append("```python")
                info_parts.append(loading_code)
                info_parts.append("```")

        return "\n".join(info_parts)

    def _get_domain_template(self, domain: str, component_type: str) -> str:
        """Get domain-specific code template.

        Args:
            domain: Competition domain (e.g., 'image_classification', 'text_classification')
            component_type: Component type (e.g., 'model', 'preprocessing')

        Returns:
            Domain-specific code template string for audio, otherwise empty
        """
        # Audio domain: Provide essential templates for audio processing
        # These are critical because audio competitions often have non-standard formats
        if domain == "audio" and component_type in ("model", "preprocessing"):
            try:
                from ...prompts.templates.audio_template import (
                    AUDIO_CONFIG_TEMPLATE,
                    AUDIO_CONSTRAINTS,
                    AUDIO_LOAD_TEMPLATE,
                    AUDIO_MELSPEC_TEMPLATE,
                )
                return f"""
## Audio Domain Guidelines

{AUDIO_CONSTRAINTS}

## Recommended Audio Configuration

{AUDIO_CONFIG_TEMPLATE}

## Audio Loading Function (use this pattern)

{AUDIO_LOAD_TEMPLATE}

## Mel Spectrogram Conversion (use this pattern)

{AUDIO_MELSPEC_TEMPLATE}
"""
            except ImportError:
                pass

        # Other domains: use agentic approach with SOTA solutions and feedback
        return ""

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response

        return code.strip()
