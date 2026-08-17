"""Regression tests for iteration-to-plan cursor handling."""

from types import SimpleNamespace

from kaggle_agents.agents.planner.agent import PlannerAgent
from kaggle_agents.core.state import AblationComponent
from kaggle_agents.workflow.nodes.iteration import iteration_control_node


def test_first_refinement_cycle_resets_component_cursor() -> None:
    updates = iteration_control_node(
        {
            "current_iteration": 0,
            "max_iterations": 2,
            "current_component_index": 3,
            "skip_remaining_components": True,
        }
    )

    assert updates["should_continue"] is True
    assert updates["current_iteration"] == 1
    assert updates["current_component_index"] == 0
    assert updates["skip_remaining_components"] is False


def test_terminal_iteration_does_not_open_another_plan() -> None:
    updates = iteration_control_node(
        {
            "current_iteration": 1,
            "max_iterations": 2,
            "current_component_index": 2,
        }
    )

    assert updates["should_continue"] is False
    assert "current_component_index" not in updates


def test_planner_always_starts_new_plan_at_component_zero(monkeypatch) -> None:
    agent = object.__new__(PlannerAgent)
    agent.config = SimpleNamespace(
        ablation_toggles=SimpleNamespace(disable_ensemble=False)
    )
    component = AblationComponent("fresh_model", "model", "train")

    monkeypatch.setattr(
        agent,
        "_analyze_sota_solutions",
        lambda _state: {},
    )
    monkeypatch.setattr(
        agent,
        "_refine_ablation_plan",
        lambda _state, _sota: [component],
    )
    monkeypatch.setattr(
        agent,
        "_validate_plan",
        lambda plan, state=None: plan,
    )
    monkeypatch.setattr(
        agent,
        "_finalize_plan",
        lambda plan, _state: (plan, [123]),
    )
    monkeypatch.setattr(agent, "_print_summary", lambda _plan: None)

    updates = agent(
        {
            "current_iteration": 1,
            "current_component_index": 8,
            "force_refinement": False,
            "force_eureka_planning": False,
            "crossover_guidance": {},
        }
    )

    assert updates["current_component_index"] == 0
    assert updates["ablation_plan"] == [component]
