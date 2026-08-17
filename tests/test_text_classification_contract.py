"""Regression coverage for the text-classification data contract."""

from __future__ import annotations

import pandas as pd

from kaggle_agents.agents.planner.fallback_plans.text import (
    create_text_fallback_plan,
)
from kaggle_agents.prompts.templates.constraints.nlp import NLP_CONSTRAINTS
from kaggle_agents.utils.data_contract import prepare_canonical_data
from kaggle_agents.workflow.nodes.data_exploration import data_exploration_node


def test_canonical_contract_marks_comment_as_text_and_date_as_datetime(tmp_path) -> None:
    """Text columns remain raw features but receive safe modelling roles."""
    train = pd.DataFrame(
        {
            "row_id": [f"train-{index}" for index in range(12)],
            "Comment": [f"comment {index}" for index in range(12)],
            "Date": [f"2024-01-{index + 1:02d}" for index in range(12)],
            "score": [float(index) for index in range(12)],
            "label": [index % 2 for index in range(12)],
        }
    )
    test = pd.DataFrame(
        {
            "row_id": [f"test-{index}" for index in range(4)],
            "Comment": [f"held out comment {index}" for index in range(4)],
            "Date": [f"2024-02-{index + 1:02d}" for index in range(4)],
            "score": [float(index) for index in range(4)],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="label",
        output_dir=tmp_path,
        id_col="row_id",
        n_folds=2,
        task_type="text_classification",
    )

    metadata = result["metadata"]
    assert metadata["text_feature_cols"] == ["Comment"]
    assert metadata["feature_roles"] == {
        "numeric": ["score"],
        "categorical": [],
        "datetime": ["Date"],
        "text": ["Comment"],
    }
    assert metadata["class_order"] == ["0", "1"]


def test_canonical_multiclass_contract_records_probability_column_order(
    tmp_path,
) -> None:
    train = pd.DataFrame(
        {
            "row_id": [f"train-{index}" for index in range(18)],
            "Comment": [f"comment {index}" for index in range(18)],
            "label": ["zebra", "ant", "moose"] * 6,
        }
    )
    test = pd.DataFrame(
        {
            "row_id": ["test-0", "test-1"],
            "Comment": ["held out one", "held out two"],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="label",
        output_dir=tmp_path,
        id_col="row_id",
        n_folds=2,
        task_type="text_classification",
    )

    assert result["metadata"]["class_order"] == ["ant", "moose", "zebra"]


def test_eda_uses_declared_text_feature_roles(tmp_path) -> None:
    """EDA must expose canonical text roles instead of treating prose as categorical."""
    train = pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "Comment": ["kind words", "rude words", "neutral", "another note"],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "label": [0, 1, 0, 1],
        }
    )
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)

    updates = data_exploration_node(
        {
            "working_directory": str(tmp_path),
            "data_files": {"train": str(train_path)},
            "canonical_contract": {
                "id_col": "row_id",
                "target_col": "label",
                "is_classification": True,
            },
            "canonical_metadata": {"text_feature_cols": ["Comment"]},
        }
    )

    insights = updates["data_insights"]
    assert insights.text_features == ["Comment"]
    assert "Comment" not in insights.categorical_features
    assert insights.datetime_features == ["Date"]


def test_nlp_constraints_resolve_the_declared_text_role_not_a_named_column() -> None:
    """Generated NLP code must resolve the text role from canonical metadata.

    The guidance must stay task-neutral: naming the literal text/timestamp
    columns of a development competition is memorized schema, and it lets a run
    look correct without ever resolving roles from public evidence.
    """
    assert "text_feature_cols" in NLP_CONSTRAINTS
    assert "CANONICAL_METADATA" in NLP_CONSTRAINTS
    assert "declares no text feature columns" in NLP_CONSTRAINTS
    assert "`comment`" not in NLP_CONSTRAINTS.lower()
    assert "`date`" not in NLP_CONSTRAINTS.lower()
    assert "TARGET_COLS" in NLP_CONSTRAINTS
    assert "train_idx" in NLP_CONSTRAINTS
    assert ".fit_transform(train_texts)" not in NLP_CONSTRAINTS
    assert "max_features=5000" in NLP_CONSTRAINTS
    assert "min_df=2" in NLP_CONSTRAINTS


def test_text_fallback_is_one_fold_local_word_and_char_tfidf_model() -> None:
    """The simple fallback is reproducible and cannot fit a corpus containing test rows."""
    plan = create_text_fallback_plan("text_classification", {})

    assert len(plan) == 1
    component = plan[0]
    outline = component["code_outline"]
    assert component["name"] == "word_char_tfidf_logreg"
    assert "TfidfVectorizer" in outline
    assert "analyzer='char_wb'" in outline
    assert "LogisticRegression" in outline
    assert "train_idx" in outline
    assert "fit_transform" in outline
    assert "transform" in outline
    assert "align_train_to_canonical" in outline
    assert "TARGET_COLS" in outline
    assert "save_component_artifacts" in outline
    assert "write_submission" in outline
    assert "all_text" not in outline
    assert "max_features=5000" in outline
    assert "min_df=2" in outline
    assert "OneVsRestClassifier" in outline
    assert "N_TARGETS" in outline
    assert "multiclass" in outline.lower()
    assert "SUBMISSION_TARGET_COLS" in outline
    assert "class_order" in outline
    assert "artifact probabilities" in outline
    assert "submission labels" in outline
    assert "argmax" in outline


def test_text_regression_fallback_uses_a_regressor() -> None:
    """A recognized regression domain must not emit class probabilities."""
    plan = create_text_fallback_plan("text_regression", {})

    assert len(plan) == 1
    component = plan[0]
    outline = component["code_outline"]
    assert component["name"] == "word_char_tfidf_ridge"
    assert "Ridge" in outline
    assert "LogisticRegression" not in outline
    assert "predict_proba" not in outline
