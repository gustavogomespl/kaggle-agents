"""Memory-bounded helpers for validating large NumPy artifacts."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np


DEFAULT_CHUNK_ROWS = 100_000
ProgressCallback = Callable[[int, int], None]
ArraySequence = np.ndarray | Sequence[object]


def load_npy_readonly(
    path: str | Path,
    *,
    allow_pickle: bool = False,
) -> np.ndarray:
    """Load a trusted NPY read-only, preferring a memory map.

    NumPy cannot memory-map object arrays. Callers must opt into their legacy
    eager fallback with ``allow_pickle=True``; candidate artifacts never do.
    """
    try:
        loaded = np.load(
            Path(path),
            allow_pickle=allow_pickle,
            mmap_mode="r",
        )
    except ValueError as exc:
        object_array_error = any(
            marker in str(exc)
            for marker in ("Python objects", "Object arrays")
        )
        if not object_array_error:
            raise
        if not allow_pickle:
            raise ValueError(
                "Object arrays cannot be loaded with allow_pickle=False"
            ) from exc
        loaded = np.load(Path(path), allow_pickle=True)
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"Expected an NPY array at {path}")
    return loaded


def iter_chunk_slices(
    total_rows: int,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Iterator[slice]:
    """Yield contiguous slices whose size never exceeds ``chunk_rows``."""
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    for start in range(0, total_rows, chunk_rows):
        yield slice(start, min(start + chunk_rows, total_rows))


def contains_none(
    values: np.ndarray,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> bool:
    """Return whether an array contains the Python ``None`` singleton."""
    array = np.asarray(values)
    if array.dtype != object:
        return False
    flat = array.reshape(-1)
    for row_slice in iter_chunk_slices(len(flat), chunk_rows=chunk_rows):
        if any(value is None for value in flat[row_slice]):
            return True
    return False


def _sequence_size(values: ArraySequence) -> int:
    if isinstance(values, np.ndarray):
        return int(values.size)
    return len(values)


def _sequence_chunk(values: ArraySequence, row_slice: slice) -> object:
    if isinstance(values, np.ndarray):
        return values.reshape(-1)[row_slice]
    return values[row_slice]


def _normalize_string_chunk(values: object) -> list[str]:
    """Normalize one bounded chunk with the historical ``str(value)`` rule."""
    iterable = values.reshape(-1) if isinstance(values, np.ndarray) else values
    return [str(value) for value in iterable]


def string_arrays_equal(
    actual: ArraySequence,
    expected: ArraySequence,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    progress: ProgressCallback | None = None,
) -> bool:
    """Compare string-normalized values exactly without global conversion."""
    total = _sequence_size(actual)
    if total != _sequence_size(expected):
        return False
    for row_slice in iter_chunk_slices(total, chunk_rows=chunk_rows):
        actual_values = _normalize_string_chunk(
            _sequence_chunk(actual, row_slice)
        )
        expected_values = _normalize_string_chunk(
            _sequence_chunk(expected, row_slice)
        )
        processed = int(row_slice.stop)
        if progress is not None:
            progress(processed, total)
        if actual_values != expected_values:
            return False
    return True


def string_values_are_unique(
    values: ArraySequence,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    progress: ProgressCallback | None = None,
) -> bool:
    """Check global string uniqueness with bounded RAM and disk-backed state."""
    total = _sequence_size(values)
    with tempfile.TemporaryDirectory(prefix="kaggle-agent-id-check-") as tmp:
        database_path = Path(tmp) / "seen.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE seen (value BLOB PRIMARY KEY) WITHOUT ROWID"
            )
            for row_slice in iter_chunk_slices(total, chunk_rows=chunk_rows):
                normalized = _normalize_string_chunk(
                    _sequence_chunk(values, row_slice)
                )
                try:
                    connection.executemany(
                        "INSERT INTO seen(value) VALUES (?)",
                        (
                            (value.encode("utf-8", errors="surrogatepass"),)
                            for value in normalized
                        ),
                    )
                except sqlite3.IntegrityError:
                    return False
                if progress is not None:
                    progress(int(row_slice.stop), total)
    return True


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "ProgressCallback",
    "contains_none",
    "iter_chunk_slices",
    "load_npy_readonly",
    "string_arrays_equal",
    "string_values_are_unique",
]
