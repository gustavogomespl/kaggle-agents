"""Centralized CSV parsing with automatic delimiter detection.

This module provides utilities for reading CSV files with automatic
delimiter detection, handling non-standard formats like semicolon-delimited
files supplied by some datasets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


DELIMITER_PRIORITY = [",", "\t", ";", " ", "|"]
CSV_COUNT_CHUNK_ROWS = 100_000


def detect_delimiter(file_path: Path | str, sample_lines: int = 5) -> str:
    """Auto-detect CSV delimiter using csv.Sniffer with fallbacks.

    Args:
        file_path: Path to the CSV file
        sample_lines: Number of lines to sample for detection

    Returns:
        Detected delimiter character

    Example:
        >>> delim = detect_delimiter("labels_semicolon.txt")
        >>> print(delim)  # ";" for semicolon-delimited files
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8", errors="replace") as f:
        sample = "".join(f.readline() for _ in range(sample_lines))

    # Try csv.Sniffer first (most reliable)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;|")
        return dialect.delimiter
    except csv.Error:
        pass

    # Fallback: count delimiter occurrences in first line
    lines = sample.split("\n")
    if not lines:
        return ","

    first_line = lines[0]
    max_count = 0
    best_delim = ","

    for delim in DELIMITER_PRIORITY:
        count = first_line.count(delim)
        if count > max_count:
            max_count = count
            best_delim = delim

    return best_delim


def read_csv_auto(file_path: Path | str, **kwargs) -> pd.DataFrame:
    """Read CSV with automatic delimiter detection.

    This function should be used EVERYWHERE instead of pd.read_csv()
    to handle non-standard delimiters (semicolon, tab, etc.).

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame with correctly parsed columns

    Example:
        >>> # Semicolon-delimited file
        >>> df = read_csv_auto("labels_semicolon.txt")
        >>> print(len(df.columns))  # Correct column count

        >>> # With explicit separator (skips detection)
        >>> df = read_csv_auto("file.csv", sep=",")
    """
    file_path = Path(file_path)

    # Auto-detect delimiter if not specified
    if "sep" not in kwargs and "delimiter" not in kwargs:
        kwargs["sep"] = detect_delimiter(file_path)

    return pd.read_csv(file_path, **kwargs)


def read_csv_preview_and_count(
    file_path: Path | str,
    *,
    preview_rows: int = 0,
    chunk_rows: int = CSV_COUNT_CHUNK_ROWS,
) -> tuple[pd.DataFrame, int]:
    """Read a bounded preview and count rows without materializing the CSV.

    Submission templates for pixel-level tasks can contain millions of rows.
    Callers that only need schema, a small ID sample, and cardinality must use
    this helper instead of loading the entire template into memory.
    """
    if preview_rows < 0:
        raise ValueError("preview_rows must be non-negative")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    path = Path(file_path)
    preview = read_csv_auto(
        path,
        nrows=preview_rows,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    row_count = sum(
        len(chunk)
        for chunk in read_csv_auto(
            path,
            chunksize=chunk_rows,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    )
    return preview, row_count


def detect_and_report(file_path: Path | str) -> dict:
    """Detect delimiter and report file structure.

    Useful for debugging delimiter issues.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary with delimiter, column count, and sample data
    """
    file_path = Path(file_path)
    delimiter = detect_delimiter(file_path)

    df = pd.read_csv(file_path, sep=delimiter, nrows=3)

    return {
        "delimiter": repr(delimiter),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "sample_rows": len(df),
    }
