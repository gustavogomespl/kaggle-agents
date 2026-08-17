"""
Robust label file parser for non-standard competition formats.

Handles various delimiters, encodings, and sparse multi-label formats.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


TargetScalarType = Literal["string", "integer"]

# --- Bounded, quote-aware label-layout inspection -------------------------
#
# This section is independent of ``RobustLabelParser`` below: it never
# guesses at a format, reads a bounded sample only, and its verdicts are
# safe to gate parsing decisions on (see `inspect_label_layout`).

_MAX_INSPECT_BYTES = 64 * 1024

_NUMERIC_HEADER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_INTEGER_TARGET_RE = re.compile(r"^[+-]?\d+$")

# Generic, domain-agnostic semantic vocabularies only - never a competition,
# dataset, or language identifier. See the Task 0 audit findings in the
# project's implementation plan for the evidence behind each entry.
_ID_LIKE_STANDALONE_TOKENS = frozenset({"id", "uid"})
_TARGET_LIKE_TOKENS = frozenset(
    {
        "target",
        "targets",
        "label",
        "labels",
        "class",
        "classes",
        "annotation",
        "annotations",
        "category",
        "categories",
    }
)
_FILE_LIKE_TOKENS = frozenset({"file", "filename", "path", "filepath"})


def split_semantic_tokens(text: str) -> tuple[str, ...]:
    """Split ``text`` into lowercase semantic tokens.

    Splits on camelCase transitions first - a lowercase/digit followed by an
    uppercase letter, and an uppercase run followed by an uppercase+lowercase
    pair - then lowercases, then splits on runs of non-alphanumeric
    characters. This is a pure text-in/tokens-out helper with no
    label-specific logic, so it is reused both for header-token
    normalization here and for filename-stem tokenization elsewhere.

    Examples:
        >>> split_semantic_tokens("sampleSubmission")
        ('sample', 'submission')
        >>> split_semantic_tokens("contest")
        ('contest',)
        >>> split_semantic_tokens("StudyInstanceUID")
        ('study', 'instance', 'uid')
    """
    if not text:
        return ()
    boundary = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    boundary = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", boundary)
    lowered = boundary.lower()
    return tuple(token for token in re.split(r"[^a-z0-9]+", lowered) if token)


@dataclass(frozen=True)
class LabelLayoutInspection:
    """Result of bounded, quote-aware inspection of a label-like file."""

    layout: Literal[
        "sparse_labels",
        "id_mapping",
        "rectangular_table",
        "unknown",
    ]
    delimiter: str
    has_header: bool
    evidence: tuple[str, ...]


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
    - Sparse multi-label formats with record IDs and variable-width label lists
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
                self._log(f"Sniffer detected delimiter={delimiter!r}, header={has_header}")
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
                format_type = f"delimited_{delimiter!r}"

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
        id_column: str = "record_id",
        label_column: str = "target",
    ) -> pd.DataFrame:
        """
        Parse a sparse, variable-width multi-label format.

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
    id_col: str = "record_id",
    filename_col: str = "file_path",
    audio_dir: Path | None = None,
    extensions: list[str] | None = None,
    resolve_extensions: bool = True,
) -> pd.DataFrame:
    """
    Read a generic record-ID-to-file mapping.

    Supports automatic extension resolution: if filenames don't include extensions,
    tries to find matching audio files with common extensions (.wav, .mp3, .flac).

    Args:
        file_path: Path to mapping file
        id_col: Canonical name for the record identifier column
        filename_col: Canonical name for the file path/name column
        audio_dir: Directory containing audio files (for extension resolution)
        extensions: List of extensions to try (default: ['.wav', '.mp3', '.flac', '.ogg'])
        resolve_extensions: Whether to automatically resolve missing extensions

    Returns:
        DataFrame with record ID and file path columns

    Example:
        Mapping file without extensions:
        ```
        record_id,file_path
        a,clip_alpha
        b,clip_beta
        ```

        After resolution (if audio_dir contains .wav files):
        ```
        record_id,file_path
        a,clip_alpha.wav
        b,clip_beta.wav
        ```
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"]

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


def infer_filename_label_table(
    file_paths: list[Path] | tuple[Path, ...],
    *,
    explicit_pattern: str | None = None,
) -> pd.DataFrame:
    """Build a canonical label table from filenames using auditable evidence.

    An explicit regular expression is accepted only when it contains exactly
    one capture group (or a named ``target`` group) and matches every file.
    Without an explicit pattern, inference is deliberately conservative: the
    immediate parent directory and the final delimiter-separated stem token
    are considered, and exactly one repeated class partition must remain.

    This avoids assigning targets from a benchmark-shaped filename regex. If
    the filenames do not provide a unique structural interpretation, callers
    must use a public annotation artifact or an explicit parsing contract.
    """
    paths = sorted(
        (Path(path) for path in file_paths if Path(path).is_file()),
        key=lambda path: str(path),
    )
    if not paths:
        raise ValueError("No existing files were supplied for filename-label inference")

    if explicit_pattern:
        try:
            pattern = re.compile(explicit_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid explicit filename-label pattern: {exc}") from exc

        if "target" in pattern.groupindex:
            target_group: str | int = "target"
        elif pattern.groups == 1:
            target_group = 1
        else:
            raise ValueError(
                "Explicit filename-label pattern must define one capture group "
                "or a named 'target' group"
            )

        targets: list[str] = []
        for path in paths:
            match = pattern.search(path.name)
            if match is None:
                raise ValueError(
                    "Explicit filename-label pattern does not match every file; "
                    f"first mismatch: {path.name}"
                )
            value = str(match.group(target_group)).strip()
            if not value:
                raise ValueError(
                    "Explicit filename-label pattern produced an empty target "
                    f"for {path.name}"
                )
            targets.append(value)
        evidence = f"explicit_pattern:{explicit_pattern}"
        mode = "explicit_pattern"
    else:
        candidates: dict[str, list[str]] = {}

        parent_targets = [path.parent.name for path in paths]
        if len(set(parent_targets)) > 1:
            candidates["immediate_parent_directory"] = parent_targets

        # Both ends of the stem are considered, and symmetrically: a class can
        # prefix a record counter (``<class>.<n>``) exactly as often as it can
        # follow one. Neither end is trusted on its own - the viability and
        # uniqueness rules below still have to accept the partition.
        leading_targets: list[str] = []
        terminal_targets: list[str] = []
        has_stem_tokens = True
        for path in paths:
            tokens = [
                token
                for token in re.split(r"[^A-Za-z0-9]+", path.stem)
                if token
            ]
            if len(tokens) < 2:
                has_stem_tokens = False
                break
            leading_targets.append(tokens[0])
            terminal_targets.append(tokens[-1])
        if has_stem_tokens:
            candidates["leading_delimited_stem_token"] = leading_targets
            candidates["terminal_delimited_stem_token"] = terminal_targets

        viable: dict[str, list[str]] = {}
        for source, values in candidates.items():
            counts = Counter(values)
            # Every inferred class must repeat. Unique suffixes are much more
            # likely record identifiers than target evidence.
            if len(counts) >= 2 and min(counts.values()) >= 2:
                viable[source] = values

        # Collapse duplicate evidence paths that produce exactly the same
        # labels, but fail closed if distinct interpretations remain.
        unique_partitions: dict[tuple[str, ...], list[str]] = {}
        for source, values in viable.items():
            unique_partitions.setdefault(tuple(values), []).append(source)

        if len(unique_partitions) != 1:
            sources = sorted(viable)
            reason = "none" if not sources else ", ".join(sources)
            raise ValueError(
                "Filename targets are not uniquely supported by structure "
                f"(viable interpretations: {reason}). Use a public annotation "
                "artifact or explicit filename-label pattern."
            )

        target_tuple, evidence_sources = next(iter(unique_partitions.items()))
        targets = list(target_tuple)
        evidence = "+".join(sorted(evidence_sources))
        mode = "unique_filename_structure"

    record_ids = [path.stem for path in paths]
    if len(set(record_ids)) != len(record_ids):
        record_ids = [str(path) for path in paths]

    table = pd.DataFrame(
        {
            "record_id": record_ids,
            "file_path": [str(path) for path in paths],
            "target": targets,
        }
    )
    table.attrs["target_inference"] = {
        "mode": mode,
        "evidence": evidence,
        "files_evaluated": len(paths),
    }
    return table


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
        file_ref = str(row[filename_col]).strip()
        supplied_path = Path(file_ref).expanduser()
        direct_path = supplied_path if supplied_path.is_absolute() else audio_dir / supplied_path
        if direct_path.is_file():
            df.at[idx, filename_col] = str(direct_path)
            resolved_count += 1
            continue

        # A supplied extension that does not exist must not be accepted as a
        # resolved file path.
        if any(file_ref.lower().endswith(ext.lower()) for ext in extensions):
            missing_count += 1
            continue

        found = False
        for ext in extensions:
            candidate = audio_dir / f"{file_ref}{ext}"
            if candidate.exists():
                df.at[idx, filename_col] = str(candidate)
                resolved_count += 1
                found = True
                break

            candidate_upper = audio_dir / f"{file_ref}{ext.upper()}"
            if candidate_upper.exists():
                df.at[idx, filename_col] = str(candidate_upper)
                resolved_count += 1
                found = True
                break

        if not found:
            missing_count += 1

    if resolved_count > 0 or missing_count > 0:
        print(f"[LabelParser] Extension resolution: {resolved_count} resolved, {missing_count} not found")

    return df


def parse_sparse_multilabel(
    file_path: Path | str,
    outer_delimiter: str = ";",
    inner_delimiter: str = ",",
    num_classes: int | None = None,
    hidden_marker: str = "?",
) -> tuple:
    """Parse a two-level-delimiter sparse multi-label format.

    Parsing is entirely driven by the supplied file and arguments:
    - Outer delimiter (semicolon) separates record ID from labels
    - Inner delimiter (comma) separates individual label indices

    Format examples:
        record_id;label1,label2,label3   (semicolon outer)
        0,3,7,12                      (comma-only: first is ID, rest are labels)
        42,?                          (hidden test labels marked with ?)

    Args:
        file_path: Path to the label file
        outer_delimiter: Delimiter between record ID and label section
        inner_delimiter: Delimiter between individual labels (default: ",")
        num_classes: Number of classes (auto-detected if None)
        hidden_marker: Marker for hidden test labels (default: "?")

    Returns:
        Tuple of (record_ids, labels):
            - record_ids: numpy array of record IDs (int)
            - labels: numpy array of shape (n_samples, num_classes) with binary indicators

    Example:
        >>> import numpy as np
        >>> record_ids, labels = parse_sparse_multilabel("labels.txt", num_classes=None)
        >>> print(f"Detected {labels.shape[1]} target columns")

    Note:
        This function automatically handles the case where the file uses comma-only
        format (e.g., "0,3,7,12" where first element is the record ID).
    """
    import numpy as np

    file_path = Path(file_path)

    with open(file_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Skip header if present (first char is not a digit)
    if lines and lines[0].strip() and not lines[0].strip()[0].isdigit():
        lines = lines[1:]

    record_ids = []
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

        # First element is the record ID.
        record_id_str = parts[0].strip()
        if not record_id_str or record_id_str == hidden_marker:
            continue

        try:
            record_id = int(record_id_str)
        except ValueError:
            continue

        record_ids.append(record_id)

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
    label_matrix = np.zeros((len(record_ids), num_classes), dtype=np.float32)
    for i, row_labels in enumerate(all_labels):
        for label in row_labels:
            if 0 <= label < num_classes:
                label_matrix[i, label] = 1.0

    return np.array(record_ids), label_matrix


# --- Header semantics --------------------------------------------------


def _is_id_like_header(name: str) -> bool:
    """A header column is ID-like (generic, symmetric, camelCase-aware).

    True when the tokenized column contains a standalone ``id``/``uid``
    token, or the raw normalized name is exactly ``id``/``record_id``, or
    it ends in ``_id``, or it starts with ``id_``.
    """
    tokens = split_semantic_tokens(name)
    if _ID_LIKE_STANDALONE_TOKENS.intersection(tokens):
        return True
    normalized = name.strip().lower()
    if normalized in ("id", "record_id"):
        return True
    if normalized.endswith("_id"):
        return True
    return normalized.startswith("id_")


def _is_target_like_header(name: str) -> bool:
    """A header column explicitly names a generic target/label concept."""
    tokens = split_semantic_tokens(name)
    return bool(_TARGET_LIKE_TOKENS.intersection(tokens))


def _is_file_like_header(name: str) -> bool:
    """A header column names a generic file/path concept."""
    tokens = split_semantic_tokens(name)
    return bool(_FILE_LIKE_TOKENS.intersection(tokens))


def _looks_like_header(first_row: list[str]) -> bool:
    """The first sampled row reads as column names, not data.

    A row is treated as a header candidate when every non-empty field fails
    to look like a bare number. This mirrors the signal a human would use
    and stays deterministic on the tiny synthetic samples where
    ``csv.Sniffer``'s statistical ``has_header`` heuristic is unreliable.

    This signal alone is not sufficient evidence - see
    ``_header_row_is_corroborated`` - because a genuinely headerless file
    whose first record's fields all happen to be non-numeric strings (for
    example ``"abc,alpha,beta"``) satisfies it just as well as a real
    header row does.
    """
    fields = [field.strip() for field in first_row if field.strip() != ""]
    if not fields:
        return False
    return all(_NUMERIC_HEADER_RE.match(field) is None for field in fields)


def _header_row_is_corroborated(header_row: list[str]) -> bool:
    """A header candidate is only trustworthy with explicit semantic
    evidence: at least one column must be id-like, target-like, or
    file-like.

    Without this, ``_looks_like_header``'s all-non-numeric guess is the
    only signal gating whether ``parse_sparse_label_rows`` skips row 0 -
    and an all-string headerless data row (an ID plus non-numeric labels)
    satisfies that guess just as well as a real header, silently dropping
    the first record and its labels. Requiring corroboration keeps every
    fixture with a genuine semantic header (``record_id,target``,
    ``rec_id,labels``, ``id,feature``, ...) classified exactly as before,
    since real headers always carry at least one such token.
    """
    return any(
        _is_id_like_header(column)
        or _is_target_like_header(column)
        or _is_file_like_header(column)
        for column in header_row
    )


# --- Bounded, quote-aware reading ---------------------------------------


def _sniff_delimiter(text: str) -> str:
    """Detect the delimiter with ``csv.Sniffer``, never ``str.split``."""
    try:
        return csv.Sniffer().sniff(text, delimiters=",\t;|").delimiter
    except csv.Error:
        return ","


def _read_bounded_text(path: Path) -> str | None:
    """Read at most ``_MAX_INSPECT_BYTES`` from ``path``.

    When the file is larger than the cap, the trailing partial line (which
    may have been cut mid-record) is dropped so a truncated final field
    never masquerades as a short logical row.
    """
    try:
        with path.open("rb") as fh:
            raw = fh.read(_MAX_INSPECT_BYTES)
            has_more = bool(fh.read(1))
    except OSError:
        return None

    text = raw.decode("utf-8", errors="replace")
    if has_more:
        last_newline = text.rfind("\n")
        if last_newline != -1:
            text = text[: last_newline + 1]
    return text


def _read_bounded_rows(text: str, delimiter: str, sample_rows: int) -> list[list[str]]:
    """Parse ``text`` with ``csv.reader`` and stop after ``sample_rows`` rows."""
    rows: list[list[str]] = []
    try:
        for row in csv.reader(io.StringIO(text), delimiter=delimiter):
            if not row:
                continue
            rows.append(row)
            if len(rows) >= sample_rows:
                break
    except csv.Error:
        pass
    return rows


# --- Layout classification ----------------------------------------------


def _is_variable_width(data_rows: list[list[str]]) -> bool:
    """Genuinely variable-width ID-plus-label records (audit ruling M3).

    Every sampled row must have a non-empty first field (the record ID),
    and at least two sampled rows must have different widths. A width-1
    row (a bare ID with zero labels and no delimiter) is a legitimately
    different width from any wider row, so it needs no separate floor.
    """
    widths: list[int] = []
    for row in data_rows:
        if not row:
            continue
        if not row[0].strip():
            return False
        widths.append(len(row))
    return len(set(widths)) >= 2


def _classify_header_row(
    header_row: list[str], data_rows: list[list[str]]
) -> tuple[str, tuple[str, ...]] | None:
    """Header-driven classification; ``None`` defers to structural rules."""
    id_cols = [column for column in header_row if _is_id_like_header(column)]
    file_cols = [column for column in header_row if _is_file_like_header(column)]
    target_cols = [column for column in header_row if _is_target_like_header(column)]

    if id_cols and file_cols:
        return "id_mapping", ("id_like_header_column", "file_like_header_column")

    widths = {len(row) for row in data_rows if row}
    fixed_width_matches_header = bool(data_rows) and widths == {len(header_row)}
    if not fixed_width_matches_header:
        return None

    if len(header_row) == 2 and id_cols and target_cols:
        return "sparse_labels", (
            "two_column_header",
            "id_like_header_column",
            "target_like_header_column",
        )
    if len(header_row) > 2:
        return "rectangular_table", (f"fixed_width_columns:{len(header_row)}",)

    return None


def _classify_rows(
    header_row: list[str] | None, data_rows: list[list[str]]
) -> tuple[str, tuple[str, ...]]:
    if header_row is not None:
        header_layout = _classify_header_row(header_row, data_rows)
        if header_layout is not None:
            return header_layout

    if _is_variable_width(data_rows):
        return "sparse_labels", ("variable_width_data_rows",)

    return "unknown", ("ambiguous_layout",)


def inspect_label_layout(path: str | Path, sample_rows: int = 50) -> LabelLayoutInspection:
    """Bounded, quote-aware inspection of a sparse-label-like file.

    Reads at most 64 KiB and stops after ``sample_rows`` logical CSV
    records (via ``csv.Sniffer``/``csv.reader``, never ``str.split``), then
    classifies the file as ``id_mapping``, ``sparse_labels``,
    ``rectangular_table``, or ``unknown`` with the reason in ``evidence``.

    This function must be called - and must return ``sparse_labels`` -
    before ``parse_sparse_label_rows`` will parse anything.
    """
    path = Path(path)
    text = _read_bounded_text(path)
    if text is None or not text.strip():
        return LabelLayoutInspection(
            layout="unknown",
            delimiter="",
            has_header=False,
            evidence=("empty_or_unreadable_file",),
        )

    delimiter = _sniff_delimiter(text)
    rows = _read_bounded_rows(text, delimiter, sample_rows)
    if not rows:
        return LabelLayoutInspection(
            layout="unknown",
            delimiter=delimiter,
            has_header=False,
            evidence=("no_parsable_rows",),
        )

    has_header = _looks_like_header(rows[0]) and _header_row_is_corroborated(rows[0])
    header_row = rows[0] if has_header else None
    data_rows = rows[1:] if has_header else rows

    layout, evidence = _classify_rows(header_row, data_rows)
    return LabelLayoutInspection(
        layout=layout,
        delimiter=delimiter,
        has_header=has_header,
        evidence=evidence,
    )


def parse_sparse_label_rows(
    path: str | Path,
    hidden_marker: str = "?",
    target_scalar_type: TargetScalarType = "string",
) -> pd.DataFrame:
    """Parse a verified sparse-label file into long-format ``(record_id, target)`` rows.

    Calls :func:`inspect_label_layout` first and rejects every layout
    except ``sparse_labels``. Record IDs are always preserved as strings.
    Targets are streamed through ``csv.reader`` (quote-aware, never
    ``str.split``), the ``hidden_marker`` is filtered out, and every
    remaining target is kept as its raw lexical string - ``"001"``, ``"1"``,
    and ``"+1"`` stay distinct - unless the caller explicitly passes
    ``target_scalar_type="integer"``, in which case every remaining lexical
    value is validated against ``^[+-]?\\d+$`` and the whole column is
    converted uniformly; a single non-matching value fails the whole call.
    A record with an ID but zero non-hidden label values (for example a
    bare width-1 row) contributes zero rows to the output.
    """
    path = Path(path)
    inspection = inspect_label_layout(path)
    if inspection.layout != "sparse_labels":
        raise ValueError(
            f"{path} is not a sparse-label layout: detected "
            f"{inspection.layout!r} (evidence={inspection.evidence!r})"
        )

    record_ids: list[str] = []
    raw_targets: list[str] = []

    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows_iter = iter(csv.reader(fh, delimiter=inspection.delimiter))
        if inspection.has_header:
            next(rows_iter, None)

        for row in rows_iter:
            if not row:
                continue
            record_id = row[0].strip()
            if not record_id:
                continue
            for raw_value in row[1:]:
                value = raw_value.strip()
                if value in ("", hidden_marker):
                    continue
                record_ids.append(record_id)
                raw_targets.append(value)

    if target_scalar_type == "integer":
        for value in raw_targets:
            if _INTEGER_TARGET_RE.match(value) is None:
                raise ValueError(
                    f"Cannot convert target value {value!r} to integer: "
                    "does not match ^[+-]?\\d+$"
                )
        target_values: list[int] | list[str] = [int(value) for value in raw_targets]
    else:
        target_values = raw_targets

    return pd.DataFrame({"record_id": record_ids, "target": target_values})
