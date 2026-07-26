"""Contract-driven EDA regression tests."""

from __future__ import annotations

import pandas as pd

from kaggle_agents.workflow.nodes.data_exploration import data_exploration_node


def test_eda_uses_arbitrary_canonical_target_and_id_roles(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "record_key": ["a", "b", "c", "d"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "outcome_value": [0, 1, 0, 1],
        }
    )
    train_path = tmp_path / "observed_train.csv"
    train.to_csv(train_path, index=False)

    updates = data_exploration_node(
        {
            "working_directory": str(tmp_path),
            "data_files": {"train": str(train_path)},
            "target_col": "outcome_value",
            "canonical_contract": {
                "target_col": "outcome_value",
                "id_col": "record_key",
                "is_classification": True,
            },
        }
    )

    insights = updates["data_insights"]
    assert insights.n_features == 1
    assert insights.n_classes == 2
    assert insights.numeric_features == ["feature"]
    assert "record_key" not in insights.categorical_features


def test_eda_does_not_treat_declared_regression_values_as_classes(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "row_key": ["a", "b", "c", "d"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "response": [0.0, 0.1, 0.2, 0.3],
        }
    )
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)

    updates = data_exploration_node(
        {
            "working_directory": str(tmp_path),
            "data_files": {"train": str(train_path)},
            "target_col": "response",
            "canonical_contract": {
                "target_col": "response",
                "id_col": "row_key",
                "is_classification": False,
            },
        }
    )

    insights = updates["data_insights"]
    assert insights.n_classes is None
    assert insights.target_distribution == {}


def test_eda_does_not_guess_last_column_as_target_without_contract(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 20.0, 30.0],
        }
    )
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)

    updates = data_exploration_node(
        {
            "working_directory": str(tmp_path),
            "data_files": {"train": str(train_path)},
        }
    )

    insights = updates["data_insights"]
    assert insights.n_classes is None
    assert insights.numeric_features == ["feature_a", "feature_b"]
