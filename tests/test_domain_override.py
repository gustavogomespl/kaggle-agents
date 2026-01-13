"""Tests for domain detection override and related functionality."""

from __future__ import annotations

import pytest


# Domain override mapping (copied from workflow.py for isolated testing)
FORCE_TYPE_TO_DOMAIN = {
    # Seq2seq / text normalization
    "seq2seq": "seq_to_seq",
    "seq_to_seq": "seq_to_seq",
    "text_normalization": "seq_to_seq",
    # Image domains
    "image": "image_classification",
    "image_classification": "image_classification",
    "image_to_image": "image_to_image",
    # Audio domains
    "audio": "audio_classification",
    "audio_classification": "audio_classification",
    "audio_tagging": "audio_tagging",
    # Text domains
    "text": "text_classification",
    "text_classification": "text_classification",
    "nlp": "text_classification",
    # Tabular domains
    "tabular": "tabular_classification",
    "tabular_classification": "tabular_classification",
    "tabular_regression": "tabular_regression",
    "regression": "tabular_regression",
}


def get_forced_domain(force_type: str) -> str | None:
    """Simulate the domain override logic."""
    normalized = force_type.strip().lower()
    return FORCE_TYPE_TO_DOMAIN.get(normalized)


class TestDomainOverrideMapping:
    """Tests for domain override type mapping."""

    def test_seq2seq_maps_to_seq_to_seq(self):
        """seq2seq should map to seq_to_seq domain."""
        assert get_forced_domain("seq2seq") == "seq_to_seq"
        assert get_forced_domain("SEQ2SEQ") == "seq_to_seq"
        assert get_forced_domain("  seq2seq  ") == "seq_to_seq"

    def test_seq_to_seq_maps_to_seq_to_seq(self):
        """seq_to_seq should map to seq_to_seq domain."""
        assert get_forced_domain("seq_to_seq") == "seq_to_seq"

    def test_image_maps_to_image_classification(self):
        """image should map to image_classification domain."""
        assert get_forced_domain("image") == "image_classification"
        assert get_forced_domain("IMAGE") == "image_classification"

    def test_audio_maps_to_audio_classification(self):
        """audio should map to audio_classification domain."""
        assert get_forced_domain("audio") == "audio_classification"
        assert get_forced_domain("AUDIO") == "audio_classification"

    def test_text_maps_to_text_classification(self):
        """text should map to text_classification domain."""
        assert get_forced_domain("text") == "text_classification"
        assert get_forced_domain("nlp") == "text_classification"

    def test_regression_maps_to_tabular_regression(self):
        """regression should map to tabular_regression domain."""
        assert get_forced_domain("regression") == "tabular_regression"
        assert get_forced_domain("tabular_regression") == "tabular_regression"

    def test_tabular_maps_to_tabular_classification(self):
        """tabular should map to tabular_classification domain."""
        assert get_forced_domain("tabular") == "tabular_classification"

    def test_unknown_type_returns_none(self):
        """Unknown types should return None."""
        assert get_forced_domain("unknown") is None
        assert get_forced_domain("") is None
        assert get_forced_domain("foobar") is None


class TestSeq2SeqFallbackPlan:
    """Tests for seq2seq fallback plan creation."""

    @pytest.fixture
    def create_seq2seq_fallback_plan(self):
        """Load the seq2seq fallback plan function using importlib to avoid circular imports."""
        import importlib.util
        from pathlib import Path

        # Load the module directly without going through package __init__
        spec = importlib.util.spec_from_file_location(
            "seq2seq",
            Path(__file__).parent.parent / "kaggle_agents" / "agents" / "planner" / "fallback_plans" / "seq2seq.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_seq2seq_fallback_plan

    def test_creates_seq2seq_fallback_plan(self, create_seq2seq_fallback_plan):
        """Should create seq2seq fallback plan for seq_to_seq domain."""
        plan = create_seq2seq_fallback_plan(
            domain="seq_to_seq",
            sota_analysis={},
            competition_name="text-normalization-challenge-english-language",
        )

        # Text normalization uses 4-component hybrid approach:
        # 1. lookup_baseline (frequency-based lookup table)
        # 2. rule_based_normalizer (class-specific fallback rules)
        # 3. t5_small_ambiguous_only (neural for ambiguous cases)
        # 4. hybrid_ensemble (lookup-priority ensemble)
        assert len(plan) == 4
        assert plan[0]["name"] == "lookup_baseline"
        assert plan[1]["name"] == "rule_based_normalizer"
        assert plan[2]["name"] == "t5_small_ambiguous_only"
        assert plan[3]["name"] == "hybrid_ensemble"

    def test_creates_generic_seq2seq_plan_for_translation(self, create_seq2seq_fallback_plan):
        """Should create generic seq2seq plan for non-normalization tasks."""
        plan = create_seq2seq_fallback_plan(
            domain="seq_to_seq",
            sota_analysis={},
            competition_name="machine-translation-task",
        )

        assert len(plan) == 3
        assert plan[0]["name"] == "t5_base_seq2seq"
        assert plan[1]["name"] == "bart_seq2seq"
        assert plan[2]["name"] == "seq2seq_ensemble"

    def test_seq2seq_text_normalization_detection(self, create_seq2seq_fallback_plan):
        """Should detect text normalization from competition name."""
        # Test various text normalization competition names
        norm_names = [
            "text-normalization-challenge",
            "normalize-text-competition",
            "tts-preprocessing",
            "speech-synthesis-text-norm",
        ]

        for name in norm_names:
            plan = create_seq2seq_fallback_plan(
                domain="seq_to_seq",
                sota_analysis={},
                competition_name=name,
            )
            # Text normalization should use hybrid approach with lookup_baseline first
            assert plan[0]["name"] == "lookup_baseline", f"Failed for {name}"
            assert len(plan) == 4, f"Expected 4 components for {name}"

    def test_text_normalization_domain_overrides_competition_name(self, create_seq2seq_fallback_plan):
        """Should use text normalization plan when domain is explicitly set, regardless of competition name."""
        # Competition name has no normalization keywords, but domain is explicit
        plan = create_seq2seq_fallback_plan(
            domain="text_normalization",
            sota_analysis={},
            competition_name="some-generic-competition",
        )

        # Should still use the hybrid text normalization approach (4 components)
        assert len(plan) == 4
        assert plan[0]["name"] == "lookup_baseline"
        assert plan[1]["name"] == "rule_based_normalizer"
        assert plan[2]["name"] == "t5_small_ambiguous_only"
        assert plan[3]["name"] == "hybrid_ensemble"


class TestDataContractStrFloatComparison:
    """Tests for the str/float comparison fix in data_contract.py."""

    def test_validate_schema_parity_with_numeric_columns(self):
        """Should handle mixed str/float column names without error."""
        import tempfile
        from pathlib import Path

        import pandas as pd

        from kaggle_agents.utils.data_contract import validate_schema_parity

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            test_path = Path(tmpdir) / "test.csv"

            # Create CSVs with mixed column types (numeric and string headers)
            train_df = pd.DataFrame({
                "id": [1, 2, 3],
                "feature1": [1.0, 2.0, 3.0],
                1: [4.0, 5.0, 6.0],  # Numeric column name
                2: [7.0, 8.0, 9.0],  # Numeric column name
                "target": [0, 1, 0],
            })
            test_df = pd.DataFrame({
                "id": [4, 5],
                "feature1": [1.0, 2.0],
                1: [4.0, 5.0],  # Numeric column name
                2: [7.0, 8.0],  # Numeric column name
            })

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            # This should not raise TypeError: '<' not supported
            common, missing = validate_schema_parity(
                train_path=str(train_path),
                test_path=str(test_path),
                id_col="id",
                target_col="target",
            )

            # All columns should be strings now
            assert all(isinstance(c, str) for c in common)
            assert all(isinstance(c, str) for c in missing)

            # Verify the columns are present
            assert "feature1" in common
            assert "1" in common or 1 in common  # Converted to string
            assert "2" in common or 2 in common  # Converted to string
