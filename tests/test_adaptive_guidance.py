"""Focused tests for direction-aware guidance and adaptive image fallbacks."""

from __future__ import annotations

import json

import pytest

from kaggle_agents.agents.meta_evaluator.guidance import GuidanceMixin
from kaggle_agents.agents.planner.fallback_plans.image import (
    create_image_fallback_plan,
    create_image_to_image_fallback_plan,
)
from kaggle_agents.prompts.templates.builders.model import (
    _build_regression_postprocessing_instructions,
    _detect_is_classification,
    _infer_from_sample_submission,
)
from kaggle_agents.core.state import CompetitionInfo
from kaggle_agents.workflow.nodes.iteration import performance_evaluation_node
from kaggle_agents.workflow.routing import route_after_iteration_control


def _failure_analysis() -> dict:
    return {
        "success_components": [],
        "failed_components": [],
        "success_patterns": [],
        "error_patterns": [],
    }


def _evaluation_context(
    *,
    metric: str,
    current_score: float,
    target_score=None,
    metric_contract=None,
) -> str:
    state = {
        "competition_info": CompetitionInfo(
            name="fixture",
            description="",
            evaluation_metric=metric,
            problem_type="classification",
        ),
        "current_iteration": 2,
        "current_performance_score": current_score,
        "target_score": target_score,
        "metric_contract": metric_contract,
        "development_results": [],
    }
    return GuidanceMixin()._build_evaluation_context(
        state,
        _failure_analysis(),
        {},
    )


def test_missing_target_remains_unconfigured() -> None:
    context = _evaluation_context(
        metric="auc",
        current_score=0.73,
        target_score=None,
    )

    assert "- Target: not configured" in context
    assert "- Gap to target (positive means not yet reached): not available" in context
    assert "- Target: 1.0000" not in context


@pytest.mark.parametrize(
    ("metric", "current_score", "target_score", "direction", "expected_gap"),
    [
        ("rmse", 0.42, 0.30, "minimize (lower is better)", 0.12),
        ("auc", 0.72, 0.80, "maximize (higher is better)", 0.08),
    ],
)
def test_gap_respects_metric_direction(
    metric: str,
    current_score: float,
    target_score: float,
    direction: str,
    expected_gap: float,
) -> None:
    context = _evaluation_context(
        metric=metric,
        current_score=current_score,
        target_score=target_score,
    )

    assert f"- Direction: {direction}" in context
    assert (
        "- Gap to target (positive means not yet reached): "
        f"{expected_gap:.4f}"
    ) in context


def test_metric_contract_supplies_direction_and_optional_target() -> None:
    context = _evaluation_context(
        metric="custom_score",
        current_score=0.40,
        target_score=None,
        metric_contract={
            "metric_name": "contract_metric",
            "is_lower_better": True,
            "target_score": 0.25,
        },
    )

    assert "- Metric: contract_metric" in context
    assert "- Direction: minimize (lower is better)" in context
    assert "- Target: 0.2500" in context
    assert "- Gap to target (positive means not yet reached): 0.1500" in context


def test_fast_image_fallback_is_name_independent_and_contract_driven() -> None:
    first = create_image_fallback_plan(
        "image_classification",
        {},
        fast_mode=True,
        competition_name="opaque-a",
    )
    second = create_image_fallback_plan(
        "image_classification",
        {},
        fast_mode=True,
        competition_name="opaque-b",
    )
    rendered = json.dumps(first).lower()

    assert first == second
    for required_signal in (
        "sample_submission",
        "canonical_metadata",
        "throughput",
        "memory",
        "deadline",
        "oof",
    ):
        assert required_signal in rendered

    for fixed_recipe in (
        "384",
        "patientid",
        "unfreeze",
        "10-15",
        "2-3 epochs",
    ):
        assert fixed_recipe not in rendered


def test_normal_image_fallback_gates_capacity_and_ensemble_on_oof() -> None:
    plan = create_image_fallback_plan(
        "image_regression",
        {},
        fast_mode=False,
        competition_name="opaque",
    )
    rendered = json.dumps(plan).lower()

    assert len(plan) == 3
    assert "pilot throughput" in rendered
    assert "identical oof" in rendered
    assert "declared metric" in rendered
    assert "fixed architecture pairing" in rendered


def test_dense_image_fallback_derives_shape_and_training_budget() -> None:
    plan = create_image_to_image_fallback_plan(
        "image_to_image",
        {},
        fast_mode=False,
    )
    rendered = json.dumps(plan).lower()

    assert "derive the number of downsampling blocks" in rendered
    assert "paired targets" in rendered
    assert "measured component budget" in rendered
    assert "epochs" not in rendered


def test_target_name_does_not_choose_classification() -> None:
    state = {
        "target_col": "class",
        "submission_contract": {},
    }

    assert _detect_is_classification(state) is None
    assert _infer_from_sample_submission(state) is None


def test_placeholder_submission_values_do_not_choose_task(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("record_id,price\nrow-1,0\n", encoding="utf-8")

    assert (
        _infer_from_sample_submission(
            {
                "sample_submission_path": str(sample),
                "submission_contract": {},
            }
        )
        is None
    )


def test_explicit_public_contract_can_choose_task() -> None:
    assert (
        _infer_from_sample_submission(
            {"submission_contract": {"problem_type": "multiclass classification"}}
        )
        is True
    )
    assert (
        _infer_from_sample_submission(
            {"submission_contract": {"problem_type": "continuous regression"}}
        )
        is False
    )


def test_regression_guidance_has_no_name_based_bounds() -> None:
    guidance = "\n".join(
        _build_regression_postprocessing_instructions(
            {
                "competition_info": CompetitionInfo(
                    name="fixture",
                    description="",
                    evaluation_metric="rmse",
                    problem_type="regression",
                ),
                "target_col": "age",
            }
        )
    ).lower()

    assert "0, 120" not in guidance
    assert "target-column name" in guidance
    assert "identical held-out folds" in guidance


def _mle_iteration_state(
    *,
    current_score: float,
    baseline_score: float,
    current_iteration: int = 1,
    max_iterations: int = 3,
    target_score=None,
    metric: str = "auc",
) -> dict:
    return {
        "run_mode": "mlebench",
        "competition_info": CompetitionInfo(
            name="fixture",
            description="",
            evaluation_metric=metric,
            problem_type="classification",
        ),
        "best_score": 0.0,
        "best_single_model_score": current_score,
        "baseline_cv_score": baseline_score,
        "target_score": target_score,
        "current_iteration": current_iteration,
        "max_iterations": max_iterations,
        "development_results": [],
        "submissions": [],
    }


def test_mle_iteration_without_target_uses_cv_and_remaining_budget(
    capsys,
) -> None:
    updates = performance_evaluation_node(
        _mle_iteration_state(
            current_score=0.64,
            baseline_score=0.60,
        )
    )
    output = capsys.readouterr().out

    assert updates["needs_refinement"] is True
    assert (
        updates["refinement_reason"]
        == "mlebench_cv_improved_budget_remaining"
    )
    assert "Target Score:  not configured" in output
    assert "Gap:           not available" in output
    assert "1.0000" not in output


def test_mle_iteration_without_target_stops_at_budget() -> None:
    updates = performance_evaluation_node(
        _mle_iteration_state(
            current_score=0.58,
            baseline_score=0.60,
            current_iteration=3,
            max_iterations=3,
        )
    )

    assert updates["needs_refinement"] is False
    assert updates["refinement_reason"] is None


def test_mle_iteration_preserves_optional_directional_target() -> None:
    updates = performance_evaluation_node(
        _mle_iteration_state(
            current_score=0.28,
            baseline_score=0.35,
            target_score=0.30,
            metric="rmse",
        )
    )

    assert updates["needs_refinement"] is False
    assert updates["refinement_reason"] is None


def test_mle_routing_without_target_refines_until_budget() -> None:
    state = {
        "run_mode": "mlebench",
        "current_iteration": 1,
        "max_iterations": 2,
        "target_score": None,
        "current_performance_score": 0.99,
        "needs_refinement": False,
    }

    assert route_after_iteration_control(state) == "refine"
    state["current_iteration"] = 2
    assert route_after_iteration_control(state) == "end"
