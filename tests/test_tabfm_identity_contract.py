"""Regression tests for honest model-family attribution in the TabFM arm."""

from kaggle_agents.agents.planner.fallback_plans.tabular import (
    create_tabular_fallback_plan,
)
from kaggle_agents.prompts.templates.constraints.tabular import TABULAR_CONSTRAINTS


def _tabfm_component(plan: list[dict]) -> dict:
    return next(component for component in plan if component["name"] == "tabfm_zero_shot")


def test_tabfm_plans_fail_instead_of_aliasing_a_tree_model() -> None:
    """Both compact and full plans must preserve the selected model identity."""
    compact = create_tabular_fallback_plan(
        "tabular_classification",
        {},
        fast_mode=True,
        stagnation_iteration=2,
    )
    full = create_tabular_fallback_plan(
        "tabular_classification",
        {},
        fast_mode=False,
    )

    for component in (_tabfm_component(compact), _tabfm_component(full)):
        outline = component["code_outline"].lower()
        assert "runtimeerror" in outline
        assert "never substitute lightgbm" in outline
        assert "fallback to lightgbm" not in outline
        assert "fall back to lightgbm" not in outline


def test_tabfm_prompt_contract_forbids_substitute_artifacts() -> None:
    """Generated code is told to fail/prune without writing mislabeled OOF."""
    tabfm_section = TABULAR_CONSTRAINTS.split("### 9. TabFM:", 1)[1]
    lowered = tabfm_section.lower()

    assert "mandatory model-identity contract" in lowered
    assert "raise runtimeerror" in lowered
    assert "`runtimeerror`" in lowered
    assert "never train lightgbm" in lowered
    assert "only after" in lowered
    assert "genuine tabfm inference succeeds" in lowered
    assert "fall back to lightgbm" not in lowered
    assert "fallback to lightgbm" not in lowered


def test_failed_tabfm_arm_is_pruned_from_later_rotation() -> None:
    """The existing failed-component memory must prevent repeat selection."""
    plan = create_tabular_fallback_plan(
        "tabular_classification",
        {},
        fast_mode=True,
        state={"failed_component_names": ["tabfm_zero_shot"]},
        stagnation_iteration=2,
    )

    assert all(component["name"] != "tabfm_zero_shot" for component in plan)
