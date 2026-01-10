"""
Robust label file parser for non-standard competition formats.

Handles various delimiters, encodings, and multi-label formats like MLSP 2013 Birds.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LabelFormatInfo:
    """Detected format information for a label file."""

    delimiter: str
    has_header: bool
    num_columns: int
    format_type: str  # 'csv', 'txt_space', 'txt_tab', 'txt_comma', etc.
    encoding: str
    column_names: list[str] | None = None
    error: str | None = None


class RobustLabelParser:
    """
    Robust parser for non-standard label files.

    Handles:
    - csv.Sniffer for automatic delimiter detection
    - Multiple fallback patterns (space, tab, comma, semicolon)
    - Header detection
    - Multi-label formats (MLSP 2013 Birds style: rec_id, label pairs)
    - Various encodings (utf-8, latin-1, cp1252)
    """

    DELIMITER_PRIORITY = [",", "\t", " ", ";", "|"]
    ENCODINGS = ["utf-8", "latin-1", "cp1252", "utf-16"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LabelParser] {msg}")

    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding by trying multiple options."""
        for encoding in self.ENCODINGS:
            try:
                file_path.read_text(encoding=encoding, errors="strict")
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        return "utf-8"  # Fallback with error handling

    def detect_format(self, file_path: Path, sample_lines: int = 50) -> LabelFormatInfo:
        """
        Auto-detect label file format.

        Args:
            file_path: Path to the label file
            sample_lines: Number of lines to sample for detection

        Returns:
            LabelFormatInfo with detected format details
        """
        encoding = self.detect_encoding(file_path)

        try:
            content = file_path.read_text(encoding=encoding, errors="ignore")
            lines = content.strip().split("\n")[:sample_lines]

            if not lines:
                return LabelFormatInfo(
                    delimiter=",",
                    has_header=False,
                    num_columns=0,
                    format_type="empty",
                    encoding=encoding,
                    error="Empty file",
                )

            sample = "\n".join(lines)

            # Try csv.Sniffer first
            delimiter = ","
            has_header = False

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;|")
                delimiter = dialect.delimiter
                has_header = csv.Sniffer().has_header(sample)
                self._log(f"Sniffer detected delimiter={repr(delimiter)}, header={has_header}")
            except csv.Error:
                # Fallback: count delimiter occurrences in first line
                self._log("csv.Sniffer failed, using fallback detection")
                max_count = 0
                for delim in self.DELIMITER_PRIORITY:
                    count = lines[0].count(delim)
                    if count > max_count:
                        max_count = count
                        delimiter = delim

            # Determine format type
            if delimiter == ",":
                format_type = "csv"
            elif delimiter == "\t":
                format_type = "txt_tab"
            elif delimiter == " ":
                format_type = "txt_space"
            elif delimiter == ";":
                format_type = "csv_semicolon"
            else:
                format_type = f"delimited_{repr(delimiter)}"

            # Count columns and detect header
            first_fields = self._split_line(lines[0], delimiter)
            num_columns = len(first_fields)

            # Better header detection
            if len(lines) > 1:
                second_fields = self._split_line(lines[1], delimiter)

                # Check if first line looks like header
                first_all_text = all(not self._is_numeric(f) for f in first_fields if f)
                second_has_numeric = any(self._is_numeric(f) for f in second_fields if f)

                if first_all_text and second_has_numeric:
                    has_header = True
                    self._log("Detected header based on text vs numeric analysis")

            column_names = first_fields if has_header else None

            return LabelFormatInfo(
                delimiter=delimiter,
                has_header=has_header,
                num_columns=num_columns,
                format_type=format_type,
                encoding=encoding,
                column_names=column_names,
            )

        except Exception as e:
            return LabelFormatInfo(
                delimiter=",",
                has_header=False,
                num_columns=0,
                format_type="error",
                encoding=encoding,
                error=str(e),
            )

    def _split_line(self, line: str, delimiter: str) -> list[str]:
        """Split a line by delimiter, handling quoted fields."""
        if delimiter == " ":
            # For space delimiter, split by any whitespace
            return [f.strip() for f in re.split(r"\s+", line.strip()) if f.strip()]
        else:
            return [f.strip() for f in line.split(delimiter)]

    def _is_numeric(self, s: str) -> bool:
        """Check if string is numeric (int or float)."""
        s = s.strip()
        if not s:
            return False
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def parse(
        self,
        file_path: Path,
        format_hint: LabelFormatInfo | None = None,
        column_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Parse label file with automatic format detection + fallbacks.

        Args:
            file_path: Path to the label file
            format_hint: Optional pre-detected format info
            column_names: Optional column names to use

        Returns:
            Parsed DataFrame
        """
        if format_hint is None:
            format_hint = self.detect_format(file_path)

        self._log(f"Parsing {file_path.name} with format: {format_hint.format_type}")

        # Build pandas read_csv arguments
        kwargs: dict[str, Any] = {
            "filepath_or_buffer": file_path,
            "encoding": format_hint.encoding,
            "engine": "python",  # More flexible parser
        }

        # Handle delimiter
        if format_hint.format_type == "txt_space":
            kwargs["sep"] = r"\s+"
            kwargs["skipinitialspace"] = True
        else:
            kwargs["sep"] = format_hint.delimiter

        # Handle header
        if format_hint.has_header:
            kwargs["header"] = 0
        else:
            kwargs["header"] = None

        # Apply column names if provided
        if column_names:
            kwargs["names"] = column_names
            kwargs["header"] = 0 if format_hint.has_header else None

        # Try parsing with detected format
        try:
            df = pd.read_csv(**kwargs)
            self._log(f"Parsed successfully: {df.shape}")
            return df
        except Exception as e:
            self._log(f"Primary parse failed: {e}, trying fallbacks")

        # Fallback 1: Try with error_bad_lines=False (for inconsistent row lengths)
        try:
            kwargs["on_bad_lines"] = "warn"
            df = pd.read_csv(**kwargs)
            self._log(f"Parsed with bad_lines handling: {df.shape}")
            return df
        except Exception:
            pass

        # Fallback 2: Read raw and split manually
        try:
            content = file_path.read_text(encoding=format_hint.encoding, errors="ignore")
            lines = content.strip().split("\n")

            # Skip header if present
            start_idx = 1 if format_hint.has_header else 0
            header_line = lines[0] if format_hint.has_header else None

            data = []
            for line in lines[start_idx:]:
                fields = self._split_line(line, format_hint.delimiter)
                if fields:
                    data.append(fields)

            # Create DataFrame
            if header_line:
                cols = self._split_line(header_line, format_hint.delimiter)
                df = pd.DataFrame(data, columns=cols[: len(data[0])] if data else cols)
            else:
                df = pd.DataFrame(data)

            self._log(f"Parsed with manual fallback: {df.shape}")
            return df

        except Exception as e:
            raise ValueError(f"Failed to parse {file_path}: {e}") from e

    def parse_multi_label(
        self,
        file_path: Path,
        id_column: str = "rec_id",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """
        Parse multi-label format (e.g., MLSP 2013 Birds style).

        Handles format where each row is (id, label) and same id can have multiple labels.

        Args:
            file_path: Path to label file
            id_column: Name of the ID column
            label_column: Name of the label column

        Returns:
            DataFrame pivoted to (id, label1, label2, ...) format with 0/1 values
        """
        df = self.parse(file_path)

        # Ensure we have at least 2 columns
        if len(df.columns) < 2:
            raise ValueError(f"Expected at least 2 columns, got {len(df.columns)}")

        # Rename columns if numeric
        if df.columns[0] != id_column:
            df.columns = [id_column, label_column] + list(df.columns[2:])

        # Create pivot table
        df["value"] = 1
        pivot = df.pivot_table(
            index=id_column,
            columns=label_column,
            values="value",
            fill_value=0,
            aggfunc="max",
        ).reset_index()

        return pivot


def sniff_and_read(file_path: Path) -> pd.DataFrame:
    """
    Convenience function to parse any label file with automatic format detection.

    Args:
        file_path: Path to label file

    Returns:
        Parsed DataFrame
    """
    parser = RobustLabelParser()
    return parser.parse(file_path)


def read_id_mapping(
    file_path: Path,
    id_col: str = "rec_id",
    filename_col: str = "filename",
    audio_dir: Path | None = None,
    extensions: list[str] | None = None,
    resolve_extensions: bool = True,
) -> pd.DataFrame:
    """
    Read an ID-to-filename mapping file (common in audio competitions).

    Supports automatic extension resolution: if filenames don't include extensions,
    tries to find matching audio files with common extensions (.wav, .mp3, .flac).

    Args:
        file_path: Path to mapping file
        id_col: Name for ID column
        filename_col: Name for filename column
        audio_dir: Directory containing audio files (for extension resolution)
        extensions: List of extensions to try (default: ['.wav', '.mp3', '.flac', '.ogg'])
        resolve_extensions: Whether to automatically resolve missing extensions

    Returns:
        DataFrame with id and filename columns (with resolved extensions if applicable)

    Example:
        Mapping file without extensions:
        ```
        rec_id,filename
        0,PC1_20100705_050000_0010
        1,PC1_20100705_050000_0020
        ```

        After resolution (if audio_dir contains .wav files):
        ```
        rec_id,filename
        0,PC1_20100705_050000_0010.wav
        1,PC1_20100705_050000_0020.wav
        ```
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"]

    parser = RobustLabelParser()
    df = parser.parse(file_path)

    # Ensure proper column names
    if len(df.columns) >= 2:
        if df.columns[0] != id_col:
            new_cols = list(df.columns)
            new_cols[0] = id_col
            new_cols[1] = filename_col
            df.columns = new_cols

    # Resolve missing extensions if audio_dir is provided
    if resolve_extensions and audio_dir and filename_col in df.columns:
        audio_dir = Path(audio_dir)
        if audio_dir.exists() and audio_dir.is_dir():
            df = _resolve_filename_extensions(df, filename_col, audio_dir, extensions)

    return df


def _resolve_filename_extensions(
    df: pd.DataFrame,
    filename_col: str,
    audio_dir: Path,
    extensions: list[str],
) -> pd.DataFrame:
    """
    Resolve missing file extensions by checking which files exist.

    Args:
        df: DataFrame with filename column
        filename_col: Name of the filename column
        audio_dir: Directory containing audio files
        extensions: List of extensions to try

    Returns:
        DataFrame with resolved extensions
    """
    resolved_count = 0
    missing_count = 0

    for idx, row in df.iterrows():
        filename = str(row[filename_col])

        # Skip if filename already has an extension
        if any(filename.lower().endswith(ext.lower()) for ext in extensions):
            continue

        # Try each extension
        found = False
        for ext in extensions:
            # Try exact case
            candidate = audio_dir / f"{filename}{ext}"
            if candidate.exists():
                df.at[idx, filename_col] = f"{filename}{ext}"
                resolved_count += 1
                found = True
                break

            # Try uppercase extension
            candidate_upper = audio_dir / f"{filename}{ext.upper()}"
            if candidate_upper.exists():
                df.at[idx, filename_col] = f"{filename}{ext.upper()}"
                resolved_count += 1
                found = True
                break

        if not found:
            missing_count += 1

    if resolved_count > 0 or missing_count > 0:
        print(f"[LabelParser] Extension resolution: {resolved_count} resolved, {missing_count} not found")

    return df


def parse_mlsp_multilabel(
    file_path: Path | str,
    outer_delimiter: str = ";",
    inner_delimiter: str = ",",
    num_classes: int | None = None,
    hidden_marker: str = "?",
) -> tuple:
    """Parse MLSP 2013 Birds style multi-label format.

    This function handles the two-level delimiter format used by MLSP 2013 Birds
    and similar competitions:
    - Outer delimiter (semicolon) separates rec_id from labels
    - Inner delimiter (comma) separates individual label indices

    Format examples:
        rec_id;label1,label2,label3   (semicolon outer)
        0,3,7,12                      (comma-only: first is ID, rest are labels)
        42,?                          (hidden test labels marked with ?)

    Args:
        file_path: Path to the label file
        outer_delimiter: Delimiter between rec_id and label section (default: ";")
        inner_delimiter: Delimiter between individual labels (default: ",")
        num_classes: Number of classes (auto-detected if None)
        hidden_marker: Marker for hidden test labels (default: "?")

    Returns:
        Tuple of (rec_ids, labels):
            - rec_ids: numpy array of record IDs (int)
            - labels: numpy array of shape (n_samples, num_classes) with binary indicators

    Example:
        >>> import numpy as np
        >>> rec_ids, labels = parse_mlsp_multilabel(
        ...     "rec_labels_test_hidden.txt",
        ...     num_classes=19,
        ... )
        >>> print(labels.shape)  # (645, 19)
        >>> print(f"Detected {labels.shape[1]} target columns")  # 19

    Note:
        This function automatically handles the case where the file uses comma-only
        format (e.g., "0,3,7,12" where first element is rec_id and rest are labels).
    """
    import numpy as np

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Skip header if present (first char is not a digit)
    if lines and lines[0].strip() and not lines[0].strip()[0].isdigit():
        lines = lines[1:]

    rec_ids = []
    all_labels = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split by outer delimiter first
        parts = line.split(outer_delimiter)
        if len(parts) == 1:
            # Fallback: might be comma-only format (e.g., "0,3,7,12")
            parts = line.split(inner_delimiter)

        # First element is rec_id
        rec_id_str = parts[0].strip()
        if not rec_id_str or rec_id_str == hidden_marker:
            continue

        try:
            rec_id = int(rec_id_str)
        except ValueError:
            continue

        rec_ids.append(rec_id)

        # Remaining elements are labels
        row_labels = []
        for label_str in parts[1:]:
            # Handle inner delimiter (e.g., "3,7,12" when outer delimiter is semicolon)
            if inner_delimiter in label_str:
                sub_labels = label_str.split(inner_delimiter)
            else:
                sub_labels = [label_str]

            for sub in sub_labels:
                sub = sub.strip()
                if sub and sub != hidden_marker:
                    try:
                        row_labels.append(int(sub))
                    except ValueError:
                        continue

        all_labels.append(row_labels)

    # Auto-detect num_classes if not provided
    if num_classes is None:
        max_label = 0
        for labels in all_labels:
            if labels:
                max_label = max(max_label, max(labels))
        num_classes = max_label + 1

    # Create binary indicator matrix
    label_matrix = np.zeros((len(rec_ids), num_classes), dtype=np.float32)
    for i, row_labels in enumerate(all_labels):
        for label in row_labels:
            if 0 <= label < num_classes:
                label_matrix[i, label] = 1.0

    return np.array(rec_ids), label_matrix
