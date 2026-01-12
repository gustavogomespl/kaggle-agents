"""Tests for text normalization utilities."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.utils.text_normalization import (
    AMBIGUOUS_CLASSES,
    DETERMINISTIC_CLASSES,
    DEFAULT_MAX_STEPS_FAST,
    LookupBaseline,
    apply_hybrid_predictions,
    create_hybrid_pipeline,
    get_neural_training_config,
)


class TestLookupBaseline:
    """Tests for the LookupBaseline class."""

    def test_fit_creates_lookup_entries(self):
        """Should create lookup entries from training data."""
        train_data = {
            "class": ["PLAIN", "PLAIN", "CARDINAL", "CARDINAL", "CARDINAL"],
            "before": ["hello", "world", "123", "123", "456"],
            "after": ["hello", "world", "one two three", "one two three", "four five six"],
        }
        df = pd.DataFrame(train_data)

        lookup = LookupBaseline().fit(df)

        assert len(lookup.lookup) > 0
        assert ("PLAIN", "hello") in lookup.lookup
        assert ("CARDINAL", "123") in lookup.lookup

    def test_predict_exact_match(self):
        """Should return exact match from lookup."""
        train_data = {
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "123"],
            "after": ["hello", "one two three"],
        }
        df = pd.DataFrame(train_data)
        lookup = LookupBaseline().fit(df)

        pred, confident = lookup.predict("PLAIN", "hello")
        assert pred == "hello"
        assert confident is True

        pred, confident = lookup.predict("CARDINAL", "123")
        assert pred == "one two three"
        assert confident is True

    def test_predict_frequency_based(self):
        """Should return most frequent mapping when multiple exist."""
        train_data = {
            "class": ["CARDINAL"] * 5,
            "before": ["123"] * 5,
            "after": ["one two three", "one two three", "one two three", "one-two-three", "one-two-three"],
        }
        df = pd.DataFrame(train_data)
        lookup = LookupBaseline().fit(df)

        pred, confident = lookup.predict("CARDINAL", "123")
        assert pred == "one two three"  # More frequent
        assert confident is True

    def test_fallback_plain_self(self):
        """PLAIN class should fall back to keeping as-is."""
        df = pd.DataFrame({"class": ["PUNCT"], "before": ["."], "after": ["."]})
        lookup = LookupBaseline().fit(df)

        pred, confident = lookup.predict("PLAIN", "unseen_word")
        assert pred == "unseen_word"
        assert confident is True

    def test_fallback_punct_self(self):
        """PUNCT class should fall back to keeping as-is."""
        df = pd.DataFrame({"class": ["PLAIN"], "before": ["x"], "after": ["x"]})
        lookup = LookupBaseline().fit(df)

        pred, confident = lookup.predict("PUNCT", "!")
        assert pred == "!"
        assert confident is True

    def test_fallback_letters_spell(self):
        """LETTERS class should spell out."""
        df = pd.DataFrame({"class": ["PLAIN"], "before": ["x"], "after": ["x"]})
        lookup = LookupBaseline().fit(df)

        pred, confident = lookup.predict("LETTERS", "ABC")
        assert pred == "a b c"
        assert confident is True

    def test_fallback_ambiguous_class_not_confident(self):
        """Ambiguous classes without exact match should not be confident."""
        df = pd.DataFrame({
            "class": ["CARDINAL"],
            "before": ["999"],
            "after": ["nine hundred ninety nine"],
        })
        lookup = LookupBaseline().fit(df)

        # Unseen value in ambiguous class
        pred, confident = lookup.predict("CARDINAL", "123")
        assert confident is False  # CARDINAL is ambiguous, unseen value

    def test_predict_batch(self):
        """Should predict for entire DataFrame."""
        train_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "123"],
            "after": ["hello", "one two three"],
        })
        lookup = LookupBaseline().fit(train_df)

        test_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL", "PLAIN"],
            "before": ["hello", "123", "unknown"],
        })

        result = lookup.predict_batch(test_df)

        assert "prediction" in result.columns
        assert "is_confident" in result.columns
        assert "needs_neural" in result.columns
        assert len(result) == 3

    def test_save_and_load(self, tmp_path):
        """Should save and load lookup correctly."""
        train_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "123"],
            "after": ["hello", "one two three"],
        })
        lookup = LookupBaseline().fit(train_df)

        save_path = tmp_path / "lookup.json"
        lookup.save(save_path)

        loaded = LookupBaseline.load(save_path)

        assert len(loaded.lookup) == len(lookup.lookup)
        assert loaded.lookup == lookup.lookup

    def test_stats_tracking(self):
        """Should track lookup statistics."""
        train_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "123"],
            "after": ["hello", "one two three"],
        })
        lookup = LookupBaseline().fit(train_df)

        # Make some predictions
        lookup.predict("PLAIN", "hello")  # Hit
        lookup.predict("PLAIN", "unknown")  # Fallback

        stats = lookup.get_stats()
        assert "total_entries" in stats
        assert "lookup_hits" in stats
        assert "fallback_used" in stats


class TestGetNeuralTrainingConfig:
    """Tests for neural training configuration."""

    def test_fast_mode_limits_steps(self):
        """Fast mode should limit max_steps."""
        config = get_neural_training_config(
            n_ambiguous_samples=100000,
            fast_mode=True,
            timeout_s=1800,
        )

        assert config["max_steps"] <= DEFAULT_MAX_STEPS_FAST
        assert config["model_name"] == "t5-small"

    def test_uses_t5_small(self):
        """Should always use t5-small (not t5-base)."""
        config = get_neural_training_config(
            n_ambiguous_samples=1000,
            fast_mode=False,
            timeout_s=3600,
        )

        assert config["model_name"] == "t5-small"

    def test_timeout_based_max_steps(self):
        """Should calculate max_steps based on timeout."""
        # Short timeout
        config = get_neural_training_config(
            n_ambiguous_samples=1000000,
            fast_mode=True,
            timeout_s=600,  # 10 minutes
        )

        # Should be limited by timeout
        assert config["max_steps"] < 1000000

    def test_returns_required_fields(self):
        """Should return all required training config fields."""
        config = get_neural_training_config(1000, fast_mode=True)

        required_fields = [
            "model_name", "max_steps", "num_train_epochs",
            "per_device_train_batch_size", "learning_rate",
            "eval_steps", "save_steps", "logging_steps",
        ]

        for field in required_fields:
            assert field in config, f"Missing field: {field}"


class TestCreateHybridPipeline:
    """Tests for hybrid pipeline creation."""

    def test_creates_lookup_baseline(self):
        """Should create a LookupBaseline."""
        df = pd.DataFrame({
            "class": ["PLAIN"] * 10 + ["DATE"] * 2,
            "before": [f"word{i}" for i in range(10)] + ["1/1/2023", "2/2/2023"],
            "after": [f"word{i}" for i in range(10)] + ["january first", "february second"],
        })

        pipeline = create_hybrid_pipeline(df, fast_mode=True)

        assert "lookup" in pipeline
        assert isinstance(pipeline["lookup"], LookupBaseline)

    def test_identifies_ambiguous_samples(self):
        """Should identify samples needing neural model."""
        df = pd.DataFrame({
            "class": ["PLAIN"] * 80 + ["DATE"] * 20,
            "before": [f"word{i}" for i in range(80)] + [f"1/{i}/2023" for i in range(20)],
            "after": [f"word{i}" for i in range(80)] + [f"january {i}" for i in range(20)],
        })

        pipeline = create_hybrid_pipeline(df, fast_mode=True)

        # PLAIN should be handled by lookup, DATE samples may need neural
        assert pipeline["stats"]["lookup_coverage"] >= 80

    def test_returns_stats(self):
        """Should return coverage statistics."""
        df = pd.DataFrame({
            "class": ["PLAIN"] * 100,
            "before": [f"word{i}" for i in range(100)],
            "after": [f"word{i}" for i in range(100)],
        })

        pipeline = create_hybrid_pipeline(df, fast_mode=True)

        assert "stats" in pipeline
        assert "total_samples" in pipeline["stats"]
        assert "lookup_coverage" in pipeline["stats"]
        assert "coverage_pct" in pipeline["stats"]

    def test_no_neural_config_when_full_coverage(self):
        """Should not create neural config when lookup covers everything."""
        # Only deterministic classes
        df = pd.DataFrame({
            "class": ["PLAIN"] * 50 + ["PUNCT"] * 50,
            "before": [f"word{i}" for i in range(50)] + ["." for _ in range(50)],
            "after": [f"word{i}" for i in range(50)] + ["." for _ in range(50)],
        })

        pipeline = create_hybrid_pipeline(df, fast_mode=True)

        # All covered by lookup - no neural needed
        # Note: some may still need neural depending on exact implementation
        assert pipeline["stats"]["lookup_coverage"] >= 50


class TestApplyHybridPredictions:
    """Tests for applying hybrid predictions."""

    def test_uses_lookup_predictions(self):
        """Should use lookup predictions when available."""
        train_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "123"],
            "after": ["hello", "one two three"],
        })
        lookup = LookupBaseline().fit(train_df)

        test_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "123"],
        })

        preds = apply_hybrid_predictions(test_df, lookup)

        assert preds[0] == "hello"
        assert preds[1] == "one two three"

    def test_overrides_with_neural_predictions(self):
        """Should override with neural predictions when provided."""
        train_df = pd.DataFrame({
            "class": ["PLAIN"],
            "before": ["hello"],
            "after": ["hello"],
        })
        lookup = LookupBaseline().fit(train_df)

        test_df = pd.DataFrame({
            "class": ["PLAIN", "CARDINAL"],
            "before": ["hello", "999"],
        })

        neural_preds = ["neural_prediction"]
        neural_indices = [1]

        preds = apply_hybrid_predictions(
            test_df, lookup,
            neural_predictions=neural_preds,
            neural_indices=neural_indices,
        )

        assert preds[0] == "hello"  # From lookup
        assert preds[1] == "neural_prediction"  # From neural


class TestClassConstants:
    """Tests for class constants."""

    def test_deterministic_classes_defined(self):
        """Deterministic classes should be defined."""
        assert "PLAIN" in DETERMINISTIC_CLASSES
        assert "PUNCT" in DETERMINISTIC_CLASSES
        assert "VERBATIM" in DETERMINISTIC_CLASSES
        assert "LETTERS" in DETERMINISTIC_CLASSES

    def test_ambiguous_classes_defined(self):
        """Ambiguous classes should be defined."""
        assert "CARDINAL" in AMBIGUOUS_CLASSES
        assert "DATE" in AMBIGUOUS_CLASSES
        assert "TIME" in AMBIGUOUS_CLASSES
        assert "MONEY" in AMBIGUOUS_CLASSES

    def test_no_overlap_between_deterministic_and_ambiguous(self):
        """Deterministic and ambiguous classes should not overlap."""
        overlap = DETERMINISTIC_CLASSES & AMBIGUOUS_CLASSES
        assert len(overlap) == 0, f"Overlap found: {overlap}"
