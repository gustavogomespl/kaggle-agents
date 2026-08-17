"""Cross-run DSPy isolation for formal MLE-bench execution."""

from __future__ import annotations

from types import SimpleNamespace

from kaggle_agents.agents.developer import agent as developer_module
from kaggle_agents.agents.meta_evaluator import agent as meta_module
from kaggle_agents.agents.planner import agent as planner_module
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    DevelopmentResult,
)
from kaggle_agents.nodes import prompt_refinement as refinement_module


def _config(tmp_path):
    return SimpleNamespace(
        dspy=SimpleNamespace(enabled=True),
        llm=SimpleNamespace(
            provider="test",
            model="test-model",
            temperature=0.2,
            max_tokens=512,
        ),
        ablation=SimpleNamespace(testing_timeout=60),
        paths=SimpleNamespace(base_dir=tmp_path),
    )


def test_sequential_mle_nodes_never_load_optimized_prompts(tmp_path, monkeypatch):
    """Every MLE task gets fresh direct-LLM planner/developer prompts."""
    config = _config(tmp_path)
    optimizer_calls: list[str] = []

    def _forbidden_optimizer():
        optimizer_calls.append("loaded")
        raise AssertionError("MLE-bench must not access global optimized prompts")

    monkeypatch.setattr(planner_module, "get_config", lambda: config)
    monkeypatch.setattr(planner_module, "get_llm_for_role", lambda **_kwargs: object())
    monkeypatch.setattr(planner_module, "create_optimizer", _forbidden_optimizer)
    monkeypatch.setattr(
        planner_module.PlannerAgent,
        "__call__",
        lambda self, _state: {"use_dspy": self.use_dspy},
    )

    monkeypatch.setattr(developer_module, "get_config", lambda: config)
    monkeypatch.setattr(developer_module, "get_llm_for_role", lambda **_kwargs: object())
    monkeypatch.setattr(developer_module, "create_optimizer", _forbidden_optimizer)
    monkeypatch.setattr(developer_module, "CodeExecutor", lambda **_kwargs: object())
    monkeypatch.setattr(developer_module, "ArtifactValidator", lambda: object())
    monkeypatch.setattr(
        developer_module,
        "create_preference_collector",
        lambda: object(),
    )
    monkeypatch.setattr(
        developer_module.DeveloperAgent,
        "__call__",
        lambda self, _state: {"use_dspy": self.use_dspy},
    )

    states = [
        {"run_mode": "mlebench", "competition_name": "opaque-task-a"},
        {"run_mode": " MLEBENCH ", "competition_name": "opaque-task-b"},
    ]
    planner_results = [
        planner_module.planner_agent_node(state) for state in states
    ]
    developer_results = [
        developer_module.developer_agent_node(state) for state in states
    ]

    assert planner_results == [{"use_dspy": False}, {"use_dspy": False}]
    assert developer_results == [{"use_dspy": False}, {"use_dspy": False}]
    assert optimizer_calls == []


def test_sequential_mle_meta_nodes_do_not_create_training_collector(
    tmp_path,
    monkeypatch,
):
    """MLE evaluation never creates or writes the global training-data store."""
    config = _config(tmp_path)
    collector_calls: list[str] = []

    def _forbidden_collector():
        collector_calls.append("created")
        raise AssertionError("MLE-bench must not instantiate a training collector")

    monkeypatch.setattr(meta_module, "get_config", lambda: config)
    monkeypatch.setattr(meta_module, "get_llm_for_role", lambda **_kwargs: object())
    monkeypatch.setattr(
        "kaggle_agents.optimization.create_training_collector",
        _forbidden_collector,
    )

    component = AblationComponent("model", "model", "train")
    development = DevelopmentResult(code="print('trained')", success=True)

    def _exercise_collection(self, state):
        self._collect_training_data(
            {
                **state,
                "ablation_plan": [component],
                "development_results": [development],
            },
            {},
            {"r_combined": 1.0},
        )
        return {
            "enabled": self._training_collection_enabled,
            "collector": self.training_collector,
        }

    monkeypatch.setattr(
        meta_module.MetaEvaluatorAgent,
        "__call__",
        _exercise_collection,
    )

    results = [
        meta_module.meta_evaluator_node(
            {"run_mode": "mlebench", "competition_name": name}
        )
        for name in ("opaque-task-a", "opaque-task-b")
    ]

    assert results == [
        {"enabled": False, "collector": None},
        {"enabled": False, "collector": None},
    ]
    assert collector_calls == []
    assert not (tmp_path / "training_data").exists()


def test_kaggle_training_collector_remains_lazy_and_reusable(tmp_path, monkeypatch):
    """Normal Kaggle runs still collect examples, but only when data exists."""
    config = _config(tmp_path)
    collectors = []

    class _Collector:
        def __init__(self):
            self.examples = []

        def add_example(self, **example):
            self.examples.append(example)

    def _create_collector():
        collector = _Collector()
        collectors.append(collector)
        return collector

    monkeypatch.setattr(meta_module, "get_config", lambda: config)
    monkeypatch.setattr(meta_module, "get_llm_for_role", lambda **_kwargs: object())
    monkeypatch.setattr(
        "kaggle_agents.optimization.create_training_collector",
        _create_collector,
    )

    agent = meta_module.MetaEvaluatorAgent()
    assert agent.training_collector is None
    assert collectors == []

    component = AblationComponent("model", "model", "train")
    state = {
        "run_mode": "kaggle",
        "competition_info": CompetitionInfo(
            "ordinary-task",
            "",
            "auc",
            "classification",
        ),
        "domain_detected": "tabular_classification",
        "ablation_plan": [component],
        "development_results": [
            DevelopmentResult(code="print('trained')", success=True)
        ],
    }
    agent._collect_training_data(state, {}, {"r_combined": 1.0})
    agent._collect_training_data(state, {}, {"r_combined": 1.0})

    assert len(collectors) == 1
    assert agent.training_collector is collectors[0]
    assert len(collectors[0].examples) == 4


def test_mle_prompt_refinement_skips_before_global_objects_are_created(
    tmp_path,
    monkeypatch,
):
    """The downstream optimizer path is unreachable in every MLE task."""
    construction_calls: list[str] = []

    def _forbidden(name):
        def _construct(*_args, **_kwargs):
            construction_calls.append(name)
            raise AssertionError(f"MLE-bench constructed global {name}")

        return _construct

    monkeypatch.setattr(
        refinement_module,
        "PromptRefinementDecider",
        _forbidden("decider"),
    )
    monkeypatch.setattr(
        refinement_module,
        "PromptOptimizer",
        _forbidden("optimizer"),
    )
    monkeypatch.setattr(
        refinement_module,
        "create_training_collector",
        _forbidden("collector"),
    )
    monkeypatch.setattr(
        refinement_module,
        "create_optimizer",
        _forbidden("DSPy optimizer"),
    )

    results = [
        refinement_module.prompt_refinement_node(
            {
                "run_mode": "mlebench",
                "competition_name": name,
                "current_iteration": iteration,
            }
        )
        for iteration, name in enumerate(("opaque-task-a", "opaque-task-b"), 1)
    ]

    assert construction_calls == []
    assert not (tmp_path / "training_data").exists()
    for result in results:
        [event] = result["telemetry_events"]
        assert event["category"] == "protocol"
        assert event["event"] == "prompt_refinement_skipped"
        assert event["detail"]["reason"] == "mlebench_cross_run_isolation"
