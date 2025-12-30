"""
Tests for DPO-style Preference Learning.

Tests PreferenceCollector and PreferenceRewardModel functionality.
"""


from kaggle_agents.core.state import PreferencePair
from kaggle_agents.optimization.preference_learning import (
    PreferenceCollector,
    PreferenceRewardModel,
    create_preference_collector,
    create_preference_reward_model,
)


class TestPreferenceCollector:
    """Tests for PreferenceCollector class."""

    def test_create_collector(self):
        """Test collector creation."""
        collector = create_preference_collector()
        assert collector is not None
        assert isinstance(collector, PreferenceCollector)
        assert len(collector) == 0

    def test_mark_as_rejected(self):
        """Test marking code as rejected."""
        collector = PreferenceCollector()

        collector.mark_as_rejected(
            component_name="test_model",
            component_type="model",
            code="bad code here",
            context="Generate a model",
            error="SyntaxError: invalid syntax",
        )

        # Should have pending rejection but no pairs yet
        assert len(collector) == 0
        assert "test_model_model" in collector._pending_rejected

    def test_mark_as_chosen_creates_pair(self):
        """Test that marking as chosen creates a preference pair."""
        collector = PreferenceCollector()

        # First mark as rejected
        collector.mark_as_rejected(
            component_name="test_model",
            component_type="model",
            code="bad code",
            context="Generate a model",
            error="ValueError: invalid",
        )

        # Then mark as chosen
        pair = collector.mark_as_chosen(
            component_name="test_model",
            component_type="model",
            code="good code",
            score=0.85,
        )

        assert pair is not None
        assert pair.chosen == "good code"
        assert pair.rejected == "bad code"
        assert pair.margin > 0
        assert len(collector) == 1

    def test_mark_as_chosen_without_rejected_returns_none(self):
        """Test that chosen without rejected doesn't create pair."""
        collector = PreferenceCollector()

        pair = collector.mark_as_chosen(
            component_name="test_model",
            component_type="model",
            code="good code",
            score=0.9,
        )

        assert pair is None
        assert len(collector) == 0

    def test_collect_from_fix_cycle(self):
        """Test direct collection from fix cycle."""
        collector = PreferenceCollector()

        pair = collector.collect_from_fix_cycle(
            component_name="test_fe",
            component_type="feature_engineering",
            original_code="def bad(): pass",
            fixed_code="def good(): return True",
            context="Implement feature engineering",
            error="TypeError: missing return",
            cv_score=0.75,
        )

        assert pair is not None
        assert pair.chosen == "def good(): return True"
        assert pair.rejected == "def bad(): pass"
        assert pair.component_type == "feature_engineering"
        assert len(collector) == 1

    def test_get_pairs_for_state_clears_buffer(self):
        """Test that get_pairs_for_state clears internal buffer."""
        collector = PreferenceCollector()

        collector.collect_from_fix_cycle(
            component_name="comp1",
            component_type="model",
            original_code="bad",
            fixed_code="good",
            context="ctx",
            error="error",
        )

        pairs = collector.get_pairs_for_state()
        assert len(pairs) == 1
        assert len(collector) == 0  # Buffer cleared

    def test_margin_calculation_syntax_error(self):
        """Test that syntax errors get high margin."""
        collector = PreferenceCollector()

        pair = collector.collect_from_fix_cycle(
            component_name="comp",
            component_type="model",
            original_code="bad",
            fixed_code="good",
            context="ctx",
            error="SyntaxError: invalid syntax at line 5",
        )

        # Syntax errors should have high margin (0.5 base + 0.3 for syntax)
        assert pair.margin >= 0.8

    def test_margin_calculation_import_error(self):
        """Test that import errors get moderate margin."""
        collector = PreferenceCollector()

        pair = collector.collect_from_fix_cycle(
            component_name="comp",
            component_type="model",
            original_code="bad",
            fixed_code="good",
            context="ctx",
            error="ImportError: No module named 'nonexistent'",
        )

        # Import errors: 0.5 base + 0.2
        assert pair.margin >= 0.7

    def test_clear(self):
        """Test clearing collector."""
        collector = PreferenceCollector()

        collector.collect_from_fix_cycle(
            component_name="comp",
            component_type="model",
            original_code="bad",
            fixed_code="good",
            context="ctx",
            error="error",
        )
        collector.mark_as_rejected("comp2", "model", "code", "ctx")

        collector.clear()

        assert len(collector) == 0
        assert len(collector._pending_rejected) == 0


class TestPreferenceRewardModel:
    """Tests for PreferenceRewardModel class."""

    def test_create_reward_model(self):
        """Test reward model creation."""
        model = create_preference_reward_model(use_llm=False)
        assert model is not None
        assert isinstance(model, PreferenceRewardModel)

    def test_learn_from_pairs_empty(self):
        """Test learning from empty pairs list."""
        model = PreferenceRewardModel(use_llm=False)
        result = model.learn_from_pairs([])

        assert result["pairs_analyzed"] == 0
        assert len(result["patterns"]["good_patterns"]) == 0

    def test_learn_from_pairs_extracts_patterns(self):
        """Test that learning extracts patterns from code."""
        model = PreferenceRewardModel(use_llm=False)

        pairs = [
            PreferencePair(
                context="ctx",
                chosen="""
import pandas as pd
try:
    df = pd.read_csv('train.csv')
except:
    pass
gc.collect()
""",
                rejected="""
import pandas
# TODO: fix this
raise Exception()
""",
                margin=0.8,
                component_type="model",
            )
        ]

        result = model.learn_from_pairs(pairs)

        assert result["pairs_analyzed"] == 1
        # Should extract good patterns like error handling
        assert "has_error_handling" in result["patterns"]["good_patterns"]
        # Should extract bad patterns like unhandled raise
        assert "unhandled_raise" in result["patterns"]["bad_patterns"]

    def test_score_code_without_learning(self):
        """Test scoring code without prior learning."""
        model = PreferenceRewardModel(use_llm=False)

        scores = model.score_code(
            code="import pandas as pd\nprint('hello')",
            component_type="model",
        )

        assert "pattern_score" in scores
        assert "structure_score" in scores
        assert "combined" in scores
        assert 0.0 <= scores["combined"] <= 1.0

    def test_score_code_with_learning(self):
        """Test scoring code after learning."""
        model = PreferenceRewardModel(use_llm=False)

        # Learn from pairs
        pairs = [
            PreferencePair(
                context="ctx",
                chosen="try:\n    x = 1\nexcept:\n    pass",
                rejected="raise Exception()",
                margin=0.8,
                component_type="model",
            )
        ]
        model.learn_from_pairs(pairs)

        # Score good code (has error handling)
        good_scores = model.score_code(
            code="try:\n    process()\nexcept:\n    handle_error()",
            component_type="model",
        )

        # Score bad code (has unhandled raise)
        bad_scores = model.score_code(
            code="raise RuntimeError('fail')",
            component_type="model",
        )

        # Good code should score higher on pattern matching
        assert good_scores["pattern_score"] >= bad_scores["pattern_score"]

    def test_score_structure_detects_syntax_error(self):
        """Test that structure scoring detects syntax errors."""
        model = PreferenceRewardModel(use_llm=False)

        # Valid code
        valid_scores = model.score_code(
            code="def foo():\n    return 1",
            component_type="model",
        )

        # Invalid code (syntax error)
        invalid_scores = model.score_code(
            code="def foo(\n    return 1",
            component_type="model",
        )

        assert valid_scores["structure_score"] > invalid_scores["structure_score"]

    def test_get_guidance_for_prompt_empty(self):
        """Test guidance generation with no learning."""
        model = PreferenceRewardModel(use_llm=False)
        guidance = model.get_guidance_for_prompt()
        assert guidance == ""

    def test_get_guidance_for_prompt_with_patterns(self):
        """Test guidance generation after learning."""
        model = PreferenceRewardModel(use_llm=False)

        pairs = [
            PreferencePair(
                context="ctx",
                chosen="try:\n    x = 1\nexcept:\n    pass\ngc.collect()",
                rejected="raise Exception()",
                margin=0.8,
                component_type="model",
            )
        ]
        model.learn_from_pairs(pairs)

        guidance = model.get_guidance_for_prompt()

        assert "Learned Preferences" in guidance
        assert "INCLUDE" in guidance or "AVOID" in guidance


class TestIntegration:
    """Integration tests for preference learning system."""

    def test_collector_and_model_integration(self):
        """Test collector feeding into reward model."""
        collector = PreferenceCollector()
        model = PreferenceRewardModel(use_llm=False)

        # Simulate fix cycle
        collector.collect_from_fix_cycle(
            component_name="xgb_model",
            component_type="model",
            original_code="import xgb\nmodel.fit()",
            fixed_code="""
import xgboost as xgb
try:
    model = xgb.XGBClassifier()
    model.fit(X, y)
except Exception as e:
    print(f'Error: {e}')
gc.collect()
""",
            context="Train XGBoost model",
            error="ImportError: No module named 'xgb'",
            cv_score=0.87,
        )

        # Get pairs and learn
        pairs = collector.get_pairs_for_state()
        result = model.learn_from_pairs(pairs)

        assert result["pairs_analyzed"] == 1
        assert model.pairs_analyzed == 1

        # Generate guidance
        guidance = model.get_guidance_for_prompt()
        assert len(guidance) > 0
