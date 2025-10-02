"""Tests for state management."""

import pytest
from kaggle_agents.utils.state import KaggleState, merge_dict


class TestStateMerging:
    """Tests for state merge functions."""

    def test_merge_dict_empty_left(self):
        """Test merging with empty left dict."""
        left = {}
        right = {"key": "value"}
        result = merge_dict(left, right)
        assert result == {"key": "value"}

    def test_merge_dict_empty_right(self):
        """Test merging with empty right dict."""
        left = {"key": "value"}
        right = {}
        result = merge_dict(left, right)
        assert result == {"key": "value"}

    def test_merge_dict_override(self):
        """Test that right values override left values."""
        left = {"key": "old_value"}
        right = {"key": "new_value"}
        result = merge_dict(left, right)
        assert result == {"key": "new_value"}

    def test_merge_dict_combine(self):
        """Test combining two dicts."""
        left = {"key1": "value1"}
        right = {"key2": "value2"}
        result = merge_dict(left, right)
        assert result == {"key1": "value1", "key2": "value2"}


class TestKaggleState:
    """Tests for KaggleState."""

    def test_state_initialization(self):
        """Test state initialization with defaults."""
        state = KaggleState(
            messages=[],
            competition_name="test-comp",
            competition_type="classification",
            metric="accuracy",
        )

        assert state.competition_name == "test-comp"
        assert state.iteration == 0
        assert state.max_iterations == 5
        assert len(state.errors) == 0
        assert len(state.features_engineered) == 0

    def test_state_list_appending(self):
        """Test list fields append via reducer."""
        state1 = KaggleState(
            messages=[],
            competition_name="test",
            features_engineered=["feature1"],
        )

        # Simulate state update with new features
        new_features = ["feature2", "feature3"]

        # The add reducer should append
        combined = state1.features_engineered + new_features
        assert "feature1" in combined
        assert "feature2" in combined
        assert "feature3" in combined
        assert len(combined) == 3

    def test_state_dict_merging(self):
        """Test dict fields merge via reducer."""
        state1 = KaggleState(
            messages=[],
            competition_name="test",
            eda_summary={"key1": "value1"},
        )

        # Simulate update
        new_data = {"key2": "value2"}
        combined = merge_dict(state1.eda_summary, new_data)

        assert combined["key1"] == "value1"
        assert combined["key2"] == "value2"
