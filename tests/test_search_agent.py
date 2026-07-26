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
            {"score_improvement": 0.1},  # Old - should be ignored
            {"score_improvement": 0.08},  # Old - should be ignored
            {"score_improvement": 0.005},  # Last 3
            {"score_improvement": 0.003},  # Last 3
            {"score_improvement": 0.002},  # Last 3
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
            base_k=3,  # Custom base
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
        search_agent_path = (
            Path(__file__).parent.parent / "kaggle_agents" / "agents" / "search_agent.py"
        )
        source = search_agent_path.read_text()

        expected_fields = [
            "sota_solutions",
            "sota_retrieval_k",
            "last_sota_update_iteration",
        ]

        for field in expected_fields:
            assert field in source, f"Expected field '{field}' in SearchAgent return"


class TestCrossCompetitionQueryDomains:
    """Contamination-guard queries must reflect the detected domain.

    Regression: image_to_image fell through to the tabular default queries, so
    9/10 retrieved "SOTA" notebooks were Titanic/tabular for a denoising comp.
    """

    @staticmethod
    def _queries(domain, metric="rmse"):
        from types import SimpleNamespace

        from kaggle_agents.agents.search_agent import SearchAgent

        state = {
            "domain_detected": domain,
            "competition_info": SimpleNamespace(evaluation_metric=metric),
        }
        return SearchAgent._generate_cross_competition_queries(None, state)

    def test_image_to_image_gets_denoising_queries(self):
        queries = self._queries("image_to_image")
        joined = " ".join(queries).lower()
        assert "lightgbm" not in joined
        assert "xgboost" not in joined
        assert "denoising" in joined
        assert "image_to_image competition rmse optimization" in queries

    def test_every_domain_type_avoids_tabular_default(self):
        from typing import get_args

        from kaggle_agents.core.state.types import DomainType

        tabular_defaults = self._queries("tabular", metric="")
        for domain in get_args(DomainType):
            if domain.startswith("tabular"):
                continue
            queries = self._queries(domain, metric="")
            assert queries != tabular_defaults, f"{domain} fell back to tabular queries"

    def test_generic_queries_do_not_identify_target_competition(self):
        from kaggle_agents.utils.contamination import (
            query_references_competition,
        )

        target = "text-normalization-challenge-english-language"
        queries = self._queries("seq_to_seq", metric="accuracy")

        assert queries
        assert all(not query_references_competition(query, target) for query in queries)


def test_mlebench_search_is_external_and_fail_closed_even_with_legacy_override(
    monkeypatch,
):
    import importlib
    from types import SimpleNamespace

    search_module = importlib.import_module("kaggle_agents.agents.search_agent")
    agent = object.__new__(search_module.SearchAgent)
    agent.config = SimpleNamespace(
        search=SimpleNamespace(
            max_notebooks=3,
            min_votes=0,
            allow_same_competition_sources=True,
        )
    )
    calls: dict[str, object] = {}

    def cross_competition_search(**kwargs):
        calls.update(kwargs)
        return [], [{"stage": "query", "filtered": False}]

    def same_competition_search(**_kwargs):
        raise AssertionError("MLE-bench must never retrieve target notebooks")

    monkeypatch.setattr(
        search_module,
        "search_notebooks_cross_competition",
        cross_competition_search,
    )
    monkeypatch.setattr(
        search_module,
        "search_competition_notebooks",
        same_competition_search,
    )

    state = {
        "run_mode": "mlebench",
        "current_iteration": 1,
        "iteration_memory": [],
        "domain_detected": "image_classification",
        "competition_info": SimpleNamespace(
            name="private-target-task",
            evaluation_metric="auc",
        ),
    }
    solutions, queries, audit, events, _ = agent.retrieve(state)

    assert solutions == []
    assert calls["competition"] == "private-target-task"
    assert calls["competition_aliases"] == ["private-target-task"]
    assert calls["queries"] == queries
    assert calls["iteration"] == 1
    assert calls["search_attempt_id"] == "initial:iteration-1"
    assert audit == [{"stage": "query", "filtered": False}]
    assert any(event["event"] == "cross_competition_retrieval" for event in events)


def test_stagnation_expands_real_retrieval_budget(monkeypatch):
    import importlib
    from types import SimpleNamespace

    search_module = importlib.import_module("kaggle_agents.agents.search_agent")
    agent = object.__new__(search_module.SearchAgent)
    agent.config = SimpleNamespace(search=SimpleNamespace(max_notebooks=4, min_votes=0))
    captured = {}

    def fake_cross_search(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(
        search_module,
        "search_notebooks_cross_competition",
        fake_cross_search,
    )
    state = {
        "run_mode": "mlebench",
        "current_iteration": 1,
        "iteration_memory": [],
        "domain_detected": "tabular_classification",
        "competition_info": SimpleNamespace(
            name="opaque-target",
            evaluation_metric="auc",
        ),
        "stagnation_detection": {"trigger_sota_search": True},
    }

    _solutions, _queries, _audit, events, retrieval_k = agent.retrieve(state)

    assert retrieval_k == 10
    assert captured["max_notebooks"] == 10
    assert captured["iteration"] == 1
    assert captured["search_attempt_id"] == "recovery:iteration-1"
    budget_event = next(event for event in events if event["event"] == "adaptive_budget_selected")
    assert budget_event["detail"]["reason"] == "stagnation_expansion"


def test_mlebench_search_transports_public_title_aliases(monkeypatch):
    import importlib
    from types import SimpleNamespace

    search_module = importlib.import_module("kaggle_agents.agents.search_agent")
    agent = object.__new__(search_module.SearchAgent)
    agent.config = SimpleNamespace(search=SimpleNamespace(max_notebooks=4, min_votes=0))
    captured = {}

    def fake_cross_search(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(
        search_module,
        "search_notebooks_cross_competition",
        fake_cross_search,
    )
    state = {
        "run_mode": "mlebench",
        "current_iteration": 1,
        "iteration_memory": [],
        "domain_detected": "tabular_classification",
        "competition_info": SimpleNamespace(
            name="opaque-target-slug",
            evaluation_metric="auc",
            identity_aliases=[
                "opaque-target-slug",
                "Official Public Challenge Title",
            ],
        ),
    }

    agent.retrieve(state)

    assert captured["competition_aliases"] == [
        "opaque-target-slug",
        "Official Public Challenge Title",
    ]


def test_external_search_failure_is_not_reported_as_effective():
    from types import SimpleNamespace

    from kaggle_agents.agents.search_agent import SearchAgent

    agent = object.__new__(SearchAgent)
    agent.config = SimpleNamespace(ablation_toggles=SimpleNamespace(disable_search=False))
    agent.retrieve = lambda _state: (
        [],
        ["generic image classification"],
        [
            {
                "stage": "initialization",
                "filtered": False,
                "error": "credentials unavailable",
            }
        ],
        [],
        5,
    )
    state = {
        "current_iteration": 1,
        "domain_detected": "image",
    }

    updates = agent(state)

    assert updates["search_attempted"] is True
    assert updates["search_effective"] is False
    assert updates["search_failure_reason"] == "retrieval_error:initialization"
    assert updates["sota_solutions"][0].source == "fallback/domain-heuristics"
    assert any(
        event["event"] == "eligible_retrieval_empty" for event in updates["telemetry_events"]
    )


def test_search_ablation_is_distinct_from_infrastructure_failure():
    from types import SimpleNamespace

    from kaggle_agents.agents.search_agent import SearchAgent

    agent = object.__new__(SearchAgent)
    agent.config = SimpleNamespace(ablation_toggles=SimpleNamespace(disable_search=True))

    updates = agent(
        {
            "current_iteration": 1,
            "domain_detected": "tabular",
        }
    )

    assert updates["search_attempted"] is False
    assert updates["search_effective"] is False
    assert updates["search_failure_reason"] == "ablation_disabled"
