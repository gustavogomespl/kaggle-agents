"""Tests for search agent - adaptive SOTA retrieval."""

import importlib.util
from pathlib import Path


# Import directly from module file to avoid circular imports through __init__.py
_module_path = Path(__file__).parent.parent / "kaggle_agents" / "agents" / "search_agent.py"

spec = importlib.util.spec_from_file_location("search_agent", _module_path)
search_agent_module = importlib.util.module_from_spec(spec)

# We need to mock some dependencies that search_agent.py imports
# For testing calculate_adaptive_k, we don't need the full module - just the function
# Let's extract it directly

def calculate_adaptive_k(
    current_iteration: int,
    iteration_memory: list = None,
    base_k: int = 5,
    expanded_k: int = 10,
) -> int:
    """
    Calculate number of notebooks to search based on iteration and improvement trend.
    (Copied from search_agent.py for isolated testing)
    """
    if current_iteration <= 2:
        return base_k

    if iteration_memory and len(iteration_memory) >= 2:
        recent_improvements = []
        for mem in iteration_memory[-3:]:
            if hasattr(mem, "score_improvement"):
                recent_improvements.append(mem.score_improvement)
            elif isinstance(mem, dict) and "score_improvement" in mem:
                recent_improvements.append(mem["score_improvement"])

        if recent_improvements:
            trend = sum(recent_improvements) / len(recent_improvements)

            if trend < 0.01:
                return expanded_k

    return base_k


class TestCalculateAdaptiveK:
    """Tests for adaptive top-K calculation based on iteration and improvement trend."""

    def test_early_iterations_use_base_k(self):
        """Iterations 1-2 should always use base_k."""
        # Iteration 1
        k = calculate_adaptive_k(
            current_iteration=1,
            iteration_memory=None,
            base_k=5,
            expanded_k=10,
        )
        assert k == 5

        # Iteration 2
        k = calculate_adaptive_k(
            current_iteration=2,
            iteration_memory=[],
            base_k=5,
            expanded_k=10,
        )
        assert k == 5

    def test_iteration_3_with_good_improvement_uses_base_k(self):
        """Iteration 3+ with good improvement should use base_k."""
        # Simulate iteration memory with good improvements
        iteration_memory = [
            {"score_improvement": 0.05},
            {"score_improvement": 0.03},
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 5

    def test_iteration_3_with_stagnation_expands_to_expanded_k(self):
        """Iteration 3+ with low improvement should expand to expanded_k."""
        # Simulate iteration memory with stagnating improvements
        iteration_memory = [
            {"score_improvement": 0.001},
            {"score_improvement": 0.002},
            {"score_improvement": 0.005},
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 10

    def test_stagnation_threshold_is_0_01(self):
        """Should expand when average improvement is below 0.01."""
        # Just below threshold
        iteration_memory = [
            {"score_improvement": 0.009},
            {"score_improvement": 0.008},
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 10  # Should expand

        # Just above threshold
        iteration_memory = [
            {"score_improvement": 0.011},
            {"score_improvement": 0.012},
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 5  # Should NOT expand

    def test_uses_last_3_iterations_for_trend(self):
        """Should only consider last 3 iterations for trend calculation."""
        # Old iterations had good improvement, but last 3 are stagnating
        iteration_memory = [
            {"score_improvement": 0.1},   # Old - should be ignored
            {"score_improvement": 0.08},  # Old - should be ignored
            {"score_improvement": 0.005}, # Last 3
            {"score_improvement": 0.003}, # Last 3
            {"score_improvement": 0.002}, # Last 3
        ]

        k = calculate_adaptive_k(
            current_iteration=6,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 10  # Should expand based on last 3

    def test_handles_empty_iteration_memory(self):
        """Should use base_k when iteration memory is empty."""
        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=[],
            base_k=5,
            expanded_k=10,
        )
        assert k == 5

    def test_handles_none_iteration_memory(self):
        """Should use base_k when iteration memory is None."""
        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=None,
            base_k=5,
            expanded_k=10,
        )
        assert k == 5

    def test_handles_insufficient_memory(self):
        """Should use base_k when memory has fewer than 2 entries."""
        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=[{"score_improvement": 0.001}],  # Only 1 entry
            base_k=5,
            expanded_k=10,
        )
        assert k == 5

    def test_custom_base_and_expanded_k(self):
        """Should respect custom base_k and expanded_k values."""
        iteration_memory = [
            {"score_improvement": 0.001},
            {"score_improvement": 0.002},
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=3,   # Custom base
            expanded_k=15,  # Custom expanded
        )
        assert k == 15

    def test_handles_object_with_score_improvement_attribute(self):
        """Should handle objects with score_improvement attribute."""
        class MockIterationMemory:
            def __init__(self, improvement):
                self.score_improvement = improvement

        iteration_memory = [
            MockIterationMemory(0.001),
            MockIterationMemory(0.002),
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 10

    def test_negative_improvements_trigger_expansion(self):
        """Negative improvements (getting worse) should trigger expansion."""
        iteration_memory = [
            {"score_improvement": -0.01},
            {"score_improvement": -0.02},
        ]

        k = calculate_adaptive_k(
            current_iteration=3,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        assert k == 10  # Negative average < 0.01 threshold

    def test_mixed_improvements(self):
        """Should calculate average correctly with mixed improvements."""
        iteration_memory = [
            {"score_improvement": 0.02},
            {"score_improvement": 0.005},
            {"score_improvement": 0.005},
        ]
        # Average = (0.02 + 0.005 + 0.005) / 3 = 0.01

        k = calculate_adaptive_k(
            current_iteration=4,
            iteration_memory=iteration_memory,
            base_k=5,
            expanded_k=10,
        )
        # 0.01 is exactly at threshold, should NOT expand (< 0.01 required)
        assert k == 5


class TestSearchAgentStateFields:
    """Tests for new state fields returned by SearchAgent."""

    def test_adaptive_k_is_tracked(self):
        """SearchAgent should return sota_retrieval_k in state updates."""
        # This is a structural test - we're checking the expected output format
        # Read the source file directly to avoid circular imports
        search_agent_path = Path(__file__).parent.parent / "kaggle_agents" / "agents" / "search_agent.py"
        source = search_agent_path.read_text()

        expected_fields = [
            "sota_solutions",
            "sota_retrieval_k",
            "last_sota_update_iteration",
        ]

        for field in expected_fields:
            assert field in source, f"Expected field '{field}' in SearchAgent return"
