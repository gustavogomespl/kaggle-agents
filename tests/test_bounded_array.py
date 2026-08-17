"""Structural tests for memory-bounded NPY validation helpers."""

from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.utils.bounded_array import (
    contains_none,
    iter_chunk_slices,
    load_npy_readonly,
    string_arrays_equal,
    string_values_are_unique,
)


def test_fixed_width_npy_is_loaded_as_read_only_memmap(tmp_path: Path) -> None:
    path = tmp_path / "text.npy"
    np.save(path, np.array(["alpha", "beta", "gamma"], dtype=str))

    loaded = load_npy_readonly(path)

    assert isinstance(loaded, np.memmap)
    assert loaded.flags.writeable is False
    assert loaded.tolist() == ["alpha", "beta", "gamma"]


def test_trusted_object_npy_falls_back_when_memmap_is_impossible(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trusted-object.npy"
    np.save(path, np.array(["alpha", "beta"], dtype=object), allow_pickle=True)

    loaded = load_npy_readonly(path, allow_pickle=True)

    assert isinstance(loaded, np.ndarray)
    assert not isinstance(loaded, np.memmap)
    assert loaded.tolist() == ["alpha", "beta"]
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_npy_readonly(path, allow_pickle=False)


def test_iter_chunk_slices_never_exceeds_requested_bound() -> None:
    chunks = list(iter_chunk_slices(10, chunk_rows=4))

    assert chunks == [slice(0, 4), slice(4, 8), slice(8, 10)]
    assert max(chunk.stop - chunk.start for chunk in chunks) == 4


def test_contains_none_scans_object_values_in_bounded_chunks() -> None:
    values = np.array([f"value-{index}" for index in range(17)], dtype=object)
    values[9] = None

    assert contains_none(values, chunk_rows=4) is True
    assert contains_none(np.asarray(values, dtype=str), chunk_rows=4) is False


def test_string_comparison_normalizes_per_chunk_and_reports_progress() -> None:
    actual = np.arange(5)
    expected = np.array(["0", "1", "2", "3", "4"], dtype=str)
    progress: list[tuple[int, int]] = []

    equal = string_arrays_equal(
        actual,
        expected,
        chunk_rows=2,
        progress=lambda processed, total: progress.append((processed, total)),
    )

    assert equal is True
    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_string_comparison_detects_cross_chunk_order_mismatch() -> None:
    actual = np.array(["a", "b", "c", "d", "e"], dtype=str)
    expected = np.array(["a", "b", "d", "c", "e"], dtype=str)

    assert string_arrays_equal(actual, expected, chunk_rows=2) is False


def test_string_uniqueness_detects_nonadjacent_cross_chunk_duplicate() -> None:
    values = np.array(["a", "b", "c", "d", "a"], dtype=str)

    assert string_values_are_unique(values, chunk_rows=2) is False
    assert string_values_are_unique(
        np.array(["a", "b", "c", "d", "e"], dtype=str),
        chunk_rows=2,
    ) is True


def test_string_uniqueness_reports_cumulative_progress() -> None:
    progress: list[tuple[int, int]] = []

    assert string_values_are_unique(
        np.array(["a", "b", "c", "d", "e"], dtype=str),
        chunk_rows=2,
        progress=lambda processed, total: progress.append((processed, total)),
    )
    assert progress == [(2, 5), (4, 5), (5, 5)]
