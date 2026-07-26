"""Direction-aware iteration-memory regression tests."""

from types import SimpleNamespace

import pytest

from kaggle_agents.agents.meta_evaluator.memory import MemoryMixin


def _failure_analysis() -> dict:
    return {
        "success_patterns": [],
        "error_patterns": [],
    }


@pytest.mark.parametrize(
    ("metric", "baseline", "current", "expected"),
    [
        ("rmse", 0.50, 0.40, 0.10),
        ("accuracy", 0.70, 0.75, 0.05),
    ],
)
def test_first_memory_uses_metric_direction(
    metric,
    baseline,
    current,
    expected,
):
    memory = MemoryMixin()._create_iteration_memory(
        {
            "current_iteration": 1,
            "current_performance_score": current,
            "baseline_cv_score": baseline,
            "competition_info": SimpleNamespace(evaluation_metric=metric),
        },
        _failure_analysis(),
        {},
    )

    assert memory.score_improvement == pytest.approx(expected)
    assert memory.results["previous_score"] == baseline


def test_later_memory_uses_previous_iteration_not_original_baseline():
    mixin = MemoryMixin()
    first = mixin._create_iteration_memory(
        {
            "current_iteration": 1,
            "current_performance_score": 0.40,
            "baseline_cv_score": 0.50,
            "competition_info": SimpleNamespace(evaluation_metric="rmse"),
        },
        _failure_analysis(),
        {},
    )
    second = mixin._create_iteration_memory(
        {
            "current_iteration": 2,
            "current_performance_score": 0.38,
            "baseline_cv_score": 0.50,
            "iteration_memory": [first],
            "competition_info": SimpleNamespace(evaluation_metric="rmse"),
        },
        _failure_analysis(),
        {},
    )

    assert second.score_improvement == pytest.approx(0.02)
    assert second.results["previous_score"] == pytest.approx(0.40)
