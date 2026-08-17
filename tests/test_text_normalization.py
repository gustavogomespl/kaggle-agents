"""Tests for text normalization utilities."""


import pandas as pd
import pytest

from kaggle_agents.utils.text_normalization import (
    AMBIGUOUS_CLASSES,
    DEFAULT_MAX_STEPS_FAST,
    DETERMINISTIC_CLASSES,
    LookupBaseline,
    apply_hybrid_predictions,
    create_hybrid_pipeline,
    get_neural_training_config,
)


TEXT_SCHEMA = {
    "class_col": "class",
    "before_col": "before",
    "after_col": "after",
}


def _fit_lookup(df: pd.DataFrame) -> LookupBaseline:
    return LookupBaseline().fit(df, **TEXT_SCHEMA)


def _create_pipeline(df: pd.DataFrame, **kwargs):
    return create_hybrid_pipeline(df, **TEXT_SCHEMA, **kwargs)


class TestLookupBaseline:
    """Tests for the LookupBaseline class."""

    def test_fit_creates_lookup_entries(self):
        """Should create lookup entries from training data."""
        train_data = {
            "class": ["identity", "identity", "spoken", "spoken", "spoken"],
            "before": ["hello", "world", "123", "123", "456"],
            "after": ["hello", "world", "one two three", "one two three", "four five six"],
        }
        df = pd.DataFrame(train_data)

        lookup = _fit_lookup(df)

        assert len(lookup.lookup) > 0
        assert ("identity", "hello") in lookup.lookup
        assert ("spoken", "123") in lookup.lookup

    def test_derives_renamed_columns_from_public_schemas(self):
        train_df = pd.DataFrame(
            {
                "row_id": range(16),
                "group_tag": ["stable"] * 8 + ["variable"] * 8,
                "raw_text": [
                    "a", "a", "b", "b", "c", "c", "d", "d",
                    "w", "w", "x", "x", "y", "y", "z", "z",
                ],
                "result_text": [
                    "a", "a", "b", "b", "c", "c", "d", "d",
                    "one", "one", "two", "two", "three", "three", "four", "four",
                ],
            }
        )
        test_df = train_df.drop(columns=["result_text"])
        sample = pd.DataFrame(
            {"row_id": test_df["row_id"], "result_text": ""}
        )

        lookup = LookupBaseline().fit(
            train_df,
            test_df=test_df,
            sample_submission=sample,
        )

        assert lookup.class_col == "group_tag"
        assert lookup.before_col == "raw_text"
        assert lookup.after_col == "result_text"
        assert lookup.predict_batch(test_df)["is_confident"].all()

    def test_fit_fails_closed_without_enough_schema_evidence(self):
        ambiguous = pd.DataFrame(
            {
                "text_a": ["a", "b"],
                "text_b": ["x", "y"],
                "text_c": ["m", "n"],
            }
        )

        with pytest.raises(ValueError, match="Ambiguous seq2seq target column"):
            LookupBaseline().fit(ambiguous)

    def test_predict_exact_match(self):
        """Should return exact match from lookup."""
        train_data = {
            "class": ["identity", "identity", "spoken", "spoken"],
            "before": ["hello", "hello", "123", "123"],
            "after": ["hello", "hello", "one two three", "one two three"],
        }
        df = pd.DataFrame(train_data)
        lookup = _fit_lookup(df)

        pred, confident = lookup.predict("identity", "hello")
        assert pred == "hello"
        assert confident is True

        pred, confident = lookup.predict("spoken", "123")
        assert pred == "one two three"
        assert confident is True

    def test_predict_frequency_based(self):
        """Should return most frequent mapping when multiple exist."""
        train_data = {
            "class": ["variable"] * 5,
            "before": ["123"] * 5,
            "after": ["one two three", "one two three", "one two three", "one-two-three", "one-two-three"],
        }
        df = pd.DataFrame(train_data)
        lookup = _fit_lookup(df)

        pred, confident = lookup.predict("variable", "123")
        assert pred == "one two three"  # More frequent
        assert confident is False  # Majority is below the confidence threshold

    def test_learns_identity_fallback_from_data(self):
        """A high-purity identity rule should be learned without a class name."""
        df = pd.DataFrame({
            "class": ["group_a"] * 3,
            "before": ["alpha", "beta", "gamma"],
            "after": ["alpha", "beta", "gamma"],
        })
        lookup = _fit_lookup(df)

        pred, confident = lookup.predict("group_a", "unseen_word")
        assert pred == "unseen_word"
        assert confident is True

    def test_learns_constant_fallback_from_data(self):
        """A constant output is enabled only after repeated supporting rows."""
        df = pd.DataFrame({
            "class": ["group_b"] * 3,
            "before": ["x", "y", "z"],
            "after": ["pause", "pause", "pause"],
        })
        lookup = _fit_lookup(df)

        pred, confident = lookup.predict("group_b", "unseen")
        assert pred == "pause"
        assert confident is True

    def test_learns_character_spelling_fallback_from_data(self):
        """Character spelling must be supported by observed transformations."""
        df = pd.DataFrame({
            "class": ["group_c"] * 3,
            "before": ["AB", "CD", "EF"],
            "after": ["a b", "c d", "e f"],
        })
        lookup = _fit_lookup(df)

        pred, confident = lookup.predict("group_c", "ABC")
        assert pred == "a b c"
        assert confident is True

    def test_low_purity_class_not_confident(self):
        """A class without a supported rule must fail closed."""
        df = pd.DataFrame({
            "class": ["group_d"] * 3,
            "before": ["one", "two", "three"],
            "after": ["first", "second", "third"],
        })
        lookup = _fit_lookup(df)

        pred, confident = lookup.predict("group_d", "unseen")
        assert pred == "unseen"
        assert confident is False

    def test_predict_batch(self):
        """Should predict for entire DataFrame."""
        train_df = pd.DataFrame({
            "class": ["identity"] * 3 + ["spoken"] * 2,
            "before": ["hello", "world", "again", "123", "123"],
            "after": ["hello", "world", "again", "one two three", "one two three"],
        })
        lookup = _fit_lookup(train_df)

        test_df = pd.DataFrame({
            "class": ["identity", "spoken", "identity"],
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
            "class": ["identity"] * 3 + ["spoken"] * 2,
            "before": ["hello", "world", "again", "123", "123"],
            "after": ["hello", "world", "again", "one two three", "one two three"],
        })
        lookup = _fit_lookup(train_df)

        save_path = tmp_path / "lookup.json"
        lookup.save(save_path)

        loaded = LookupBaseline.load(save_path)

        assert len(loaded.lookup) == len(lookup.lookup)
        assert loaded.lookup == lookup.lookup
        assert loaded.confident_lookup_keys == lookup.confident_lookup_keys
        assert loaded.deterministic_classes == lookup.deterministic_classes

    def test_stats_tracking(self):
        """Should track lookup statistics."""
        train_df = pd.DataFrame({
            "class": ["identity"] * 3,
            "before": ["hello", "world", "again"],
            "after": ["hello", "world", "again"],
        })
        lookup = _fit_lookup(train_df)

        # Make some predictions
        lookup.predict("identity", "hello")  # Hit
        lookup.predict("identity", "unknown")  # Learned fallback

        stats = lookup.get_stats()
        assert "total_entries" in stats
        assert "lookup_hits" in stats
        assert "fallback_used" in stats
        assert stats["learned_deterministic_classes"] == ["identity"]


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
            "class": ["identity"] * 10 + ["variable"] * 2,
            "before": [f"word{i}" for i in range(10)] + ["input-a", "input-b"],
            "after": [f"word{i}" for i in range(10)] + ["output-a", "output-b"],
        })

        pipeline = _create_pipeline(df, fast_mode=True)

        assert "lookup" in pipeline
        assert isinstance(pipeline["lookup"], LookupBaseline)

    def test_identifies_ambiguous_samples(self):
        """Should identify samples needing neural model."""
        df = pd.DataFrame({
            "class": ["identity"] * 80 + ["variable"] * 20,
            "before": [f"word{i}" for i in range(80)] + [f"input-{i}" for i in range(20)],
            "after": [f"word{i}" for i in range(80)] + [f"output-{i}" for i in range(20)],
        })

        pipeline = _create_pipeline(df, fast_mode=True)

        # Identity behavior is learned OOF; arbitrary transformations fail closed.
        assert pipeline["stats"]["lookup_coverage"] >= 80
        assert pipeline["stats"]["neural_samples"] >= 20
        assert pipeline["stats"]["routing_evaluation"] == "out_of_fold"

    def test_returns_stats(self):
        """Should return coverage statistics."""
        df = pd.DataFrame({
            "class": ["identity"] * 100,
            "before": [f"word{i}" for i in range(100)],
            "after": [f"word{i}" for i in range(100)],
        })

        pipeline = _create_pipeline(df, fast_mode=True)

        assert "stats" in pipeline
        assert "total_samples" in pipeline["stats"]
        assert "lookup_coverage" in pipeline["stats"]
        assert "coverage_pct" in pipeline["stats"]

    def test_no_neural_config_when_full_coverage(self):
        """Should not create neural config when lookup covers everything."""
        df = pd.DataFrame({
            "class": ["identity_a"] * 50 + ["identity_b"] * 50,
            "before": [f"word{i}" for i in range(50)] + [f"mark{i}" for i in range(50)],
            "after": [f"word{i}" for i in range(50)] + [f"mark{i}" for i in range(50)],
        })

        pipeline = _create_pipeline(df, fast_mode=True)

        assert pipeline["stats"]["lookup_coverage"] == 100
        assert pipeline["neural_config"] is None


class TestApplyHybridPredictions:
    """Tests for applying hybrid predictions."""

    def test_uses_lookup_predictions(self):
        """Should use lookup predictions when available."""
        train_df = pd.DataFrame({
            "class": ["identity", "spoken"],
            "before": ["hello", "123"],
            "after": ["hello", "one two three"],
        })
        lookup = _fit_lookup(train_df)

        test_df = pd.DataFrame({
            "class": ["identity", "spoken"],
            "before": ["hello", "123"],
        })

        preds = apply_hybrid_predictions(test_df, lookup)

        assert preds[0] == "hello"
        assert preds[1] == "one two three"

    def test_overrides_with_neural_predictions(self):
        """Should override with neural predictions when provided."""
        train_df = pd.DataFrame({
            "class": ["identity"],
            "before": ["hello"],
            "after": ["hello"],
        })
        lookup = _fit_lookup(train_df)

        test_df = pd.DataFrame({
            "class": ["identity", "unseen_group"],
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


class TestLearnedClassRouting:
    """Tests that class behavior is inferred rather than predeclared."""

    def test_compatibility_constants_do_not_encode_a_taxonomy(self):
        assert DETERMINISTIC_CLASSES == frozenset()
        assert AMBIGUOUS_CLASSES == frozenset()

    def test_fit_learns_deterministic_and_ambiguous_groups(self):
        df = pd.DataFrame({
            "class": ["stable"] * 3 + ["variable"] * 3,
            "before": ["a", "b", "c", "x", "y", "z"],
            "after": ["a", "b", "c", "one", "two", "three"],
        })

        lookup = _fit_lookup(df)

        assert lookup.deterministic_classes == {"stable"}
        assert lookup.ambiguous_classes == {"variable"}
