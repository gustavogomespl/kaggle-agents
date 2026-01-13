"""Tests for data contract utilities - deterministic sampling."""


import numpy as np
import pandas as pd

from kaggle_agents.utils.data_contract import (
    _deterministic_hash,
    _ensure_id_column,
    _hash_based_sample,
    _remove_synthetic_id_from_features,
)


class TestDeterministicHash:
    """Tests for deterministic MD5-based hashing."""

    def test_hash_is_deterministic_same_seed(self):
        """Same value and seed should produce same hash."""
        value = "test_id_123"
        seed = 42

        hash1 = _deterministic_hash(value, seed)
        hash2 = _deterministic_hash(value, seed)

        assert hash1 == hash2

    def test_hash_differs_with_different_seeds(self):
        """Different seeds should produce different hashes."""
        value = "test_id_123"

        hash1 = _deterministic_hash(value, seed=42)
        hash2 = _deterministic_hash(value, seed=123)

        assert hash1 != hash2

    def test_hash_differs_with_different_values(self):
        """Different values should produce different hashes."""
        seed = 42

        hash1 = _deterministic_hash("value_a", seed)
        hash2 = _deterministic_hash("value_b", seed)

        assert hash1 != hash2

    def test_hash_returns_integer(self):
        """Hash should return an integer."""
        result = _deterministic_hash("test", seed=42)
        assert isinstance(result, int)

    def test_hash_is_positive(self):
        """Hash should be a positive integer."""
        result = _deterministic_hash("test", seed=42)
        assert result >= 0


class TestEnsureIdColumn:
    """Tests for ID column detection and synthetic ID creation."""

    def test_uses_existing_id_column(self):
        """Should use existing ID column when specified and present."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "feature": ["a", "b", "c"],
        })

        result_df, id_col, is_synthetic = _ensure_id_column(df, "id")

        assert id_col == "id"
        assert not is_synthetic
        assert "_row_id" not in result_df.columns

    def test_creates_synthetic_id_when_missing(self):
        """Should create synthetic _row_id when ID column is missing."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": ["a", "b", "c"],
        })

        result_df, id_col, is_synthetic = _ensure_id_column(df, None)

        assert id_col == "_row_id"
        assert is_synthetic
        assert "_row_id" in result_df.columns
        assert list(result_df["_row_id"]) == ["0", "1", "2"]

    def test_creates_synthetic_id_when_id_not_in_columns(self):
        """Should create synthetic ID when specified column doesn't exist."""
        df = pd.DataFrame({
            "feature": [1, 2, 3],
        })

        result_df, id_col, is_synthetic = _ensure_id_column(df, "nonexistent_id")

        assert id_col == "_row_id"
        assert is_synthetic

    def test_preserves_original_dataframe(self):
        """Should not modify the original dataframe."""
        df = pd.DataFrame({
            "feature": [1, 2, 3],
        })
        original_cols = list(df.columns)

        _ensure_id_column(df, None)

        assert list(df.columns) == original_cols


class TestHashBasedSample:
    """Tests for hash-based deterministic sampling."""

    def test_no_sampling_when_under_limit(self):
        """Should not sample when data is under max_rows."""
        df = pd.DataFrame({
            "id": range(100),
            "feature": range(100),
        })

        sampled_df, metadata = _hash_based_sample(df, "id", max_rows=200, seed=42)

        assert len(sampled_df) == 100
        assert not metadata["sampled"]

    def test_samples_when_over_limit(self):
        """Should sample when data exceeds max_rows."""
        df = pd.DataFrame({
            "id": range(1000),
            "feature": range(1000),
        })

        sampled_df, metadata = _hash_based_sample(df, "id", max_rows=100, seed=42)

        assert len(sampled_df) < 1000
        assert metadata["sampled"]
        assert metadata["original_rows"] == 1000
        assert metadata["sampled_rows"] == len(sampled_df)

    def test_sampling_is_deterministic_across_calls(self):
        """Same seed should produce same sample."""
        df = pd.DataFrame({
            "id": range(1000),
            "feature": range(1000),
        })

        sampled1, _ = _hash_based_sample(df, "id", max_rows=100, seed=42)
        sampled2, _ = _hash_based_sample(df, "id", max_rows=100, seed=42)

        pd.testing.assert_frame_equal(sampled1, sampled2)

    def test_different_seeds_produce_different_samples(self):
        """Different seeds should produce different samples."""
        df = pd.DataFrame({
            "id": range(1000),
            "feature": range(1000),
        })

        sampled1, _ = _hash_based_sample(df, "id", max_rows=100, seed=42)
        sampled2, _ = _hash_based_sample(df, "id", max_rows=100, seed=123)

        # Samples should differ (highly unlikely to be identical)
        assert not sampled1["id"].equals(sampled2["id"])

    def test_metadata_contains_hash_method(self):
        """Metadata should indicate MD5 hash method."""
        df = pd.DataFrame({
            "id": range(1000),
            "feature": range(1000),
        })

        _, metadata = _hash_based_sample(df, "id", max_rows=100, seed=42)

        assert metadata["hash_method"] == "md5"
        assert metadata["sampling_seed"] == 42

    def test_works_with_synthetic_id(self):
        """Should work when no ID column is provided (uses synthetic)."""
        df = pd.DataFrame({
            "feature1": range(1000),
            "feature2": range(1000),
        })

        sampled_df, metadata = _hash_based_sample(df, None, max_rows=100, seed=42)

        assert len(sampled_df) < 1000
        assert metadata["sampled"]
        assert metadata["id_is_synthetic"]
        assert "_row_id" in sampled_df.columns


class TestRemoveSyntheticId:
    """Tests for removing synthetic ID from features."""

    def test_removes_synthetic_id_when_flagged(self):
        """Should remove _row_id when is_synthetic is True."""
        df = pd.DataFrame({
            "_row_id": ["0", "1", "2"],
            "feature": [1, 2, 3],
        })

        result = _remove_synthetic_id_from_features(df, is_synthetic=True)

        assert "_row_id" not in result.columns
        assert "feature" in result.columns

    def test_keeps_id_when_not_synthetic(self):
        """Should keep _row_id when is_synthetic is False."""
        df = pd.DataFrame({
            "_row_id": ["0", "1", "2"],
            "feature": [1, 2, 3],
        })

        result = _remove_synthetic_id_from_features(df, is_synthetic=False)

        assert "_row_id" in result.columns

    def test_handles_missing_row_id_gracefully(self):
        """Should not fail when _row_id doesn't exist."""
        df = pd.DataFrame({
            "feature": [1, 2, 3],
        })

        result = _remove_synthetic_id_from_features(df, is_synthetic=True)

        assert "feature" in result.columns


class TestSamplingDeterminismAcrossProcesses:
    """Integration test for cross-process determinism."""

    def test_sampling_produces_consistent_results(self):
        """
        Verify that MD5-based sampling produces consistent results.
        This is a proxy for cross-process testing.
        """
        # Create a dataset
        np.random.seed(12345)
        df = pd.DataFrame({
            "id": [f"user_{i}" for i in range(10000)],
            "value": np.random.randn(10000),
        })

        # Sample multiple times with same seed
        results = []
        for _ in range(5):
            sampled, _ = _hash_based_sample(df.copy(), "id", max_rows=500, seed=42)
            results.append(set(sampled["id"].tolist()))

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Sampling should be deterministic"
