"""Metric-direction regressions for evaluation contracts."""

from kaggle_agents.core.state import create_metric_contract


def test_rmsle_is_a_minimization_metric() -> None:
    contract = create_metric_contract("rmsle")

    assert contract.is_lower_better is True
