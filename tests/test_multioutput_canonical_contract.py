"""Regression coverage for canonical multi-output targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, mean_squared_error

from kaggle_agents.agents.ensemble.scoring import (
    compute_oof_score,
    score_predictions,
)
from kaggle_agents.agents.ensemble.stacking import stack_from_prediction_pairs
from kaggle_agents.utils.data_contract import prepare_canonical_data
from kaggle_agents.utils.oof_validation import validate_oof_stack


JIGSAW_TARGETS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def test_jigsaw_like_contract_uses_ordered_real_multilabel_targets(
    tmp_path,
) -> None:
    """All-zero template values must not collapse six real labels to one."""
    n_train = 24
    row = np.arange(n_train)
    train = pd.DataFrame(
        {
            "id": [f"train-{index}" for index in row],
            "comment_text": [f"public text {index}" for index in row],
            # Deliberately use a different CSV order than the submission.
            "identity_hate": (row % 8 == 0).astype(int),
            "insult": np.isin(row % 3, [0, 1]).astype(int),
            "threat": (row % 6 == 0).astype(int),
            "obscene": (row % 3 == 0).astype(int),
            "severe_toxic": (row % 4 == 0).astype(int),
            "toxic": (row % 2 == 0).astype(int),
        }
    )
    test = pd.DataFrame(
        {
            "id": [f"test-{index}" for index in range(6)],
            "comment_text": [f"held-out text {index}" for index in range(6)],
        }
    )
    sample = pd.DataFrame(
        {
            "id": test["id"],
            **{target: np.zeros(len(test)) for target in JIGSAW_TARGETS},
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col=JIGSAW_TARGETS[0],
        target_cols=JIGSAW_TARGETS,
        output_dir=tmp_path,
        n_folds=3,
        task_type="text_classification",
        sample_submission=sample_path,
    )

    y = np.load(result["y_path"])
    folds = np.load(result["folds_path"])
    metadata = result["metadata"]
    with Path(result["feature_cols_path"]).open(encoding="utf-8") as handle:
        feature_cols = json.load(handle)

    assert y.shape == (n_train, len(JIGSAW_TARGETS))
    np.testing.assert_array_equal(
        y,
        train.loc[:, JIGSAW_TARGETS].to_numpy(),
    )
    assert metadata["target_col"] == JIGSAW_TARGETS[0]
    assert metadata["target_cols"] == JIGSAW_TARGETS
    assert metadata["target_type"] == "multi_label"
    assert metadata["n_targets"] == len(JIGSAW_TARGETS)
    assert metadata["is_classification"] is True
    assert metadata["cv_strategy"] == "multilabel_stratified_kfold"
    assert feature_cols == ["comment_text"]
    assert set(np.unique(folds)) == {0, 1, 2}
    for column in range(y.shape[1]):
        positive_counts = [
            int(y[folds == fold, column].sum())
            for fold in range(3)
        ]
        assert max(positive_counts) - min(positive_counts) <= 1


def test_multi_target_regression_preserves_template_order_and_scores_columns(
    tmp_path,
) -> None:
    n_train = 12
    row = np.arange(n_train, dtype=float)
    train = pd.DataFrame(
        {
            "record_id": [f"row-{index}" for index in range(n_train)],
            "feature": row / 10,
            "y_primary": 0.25 + row * 0.7,
            "y_secondary": 100.0 + row**2,
        }
    )
    test = pd.DataFrame(
        {
            "record_id": [f"test-{index}" for index in range(4)],
            "feature": np.arange(4, dtype=float),
        }
    )
    ordered_targets = ["y_secondary", "y_primary"]
    sample = pd.DataFrame(
        {
            "record_id": test["record_id"],
            "y_secondary": np.zeros(len(test)),
            "y_primary": np.zeros(len(test)),
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="y_secondary",
        target_cols=ordered_targets,
        output_dir=tmp_path,
        n_folds=3,
        task_type="tabular_regression",
        sample_submission=sample_path,
    )

    y = np.load(result["y_path"])
    metadata = result["metadata"]
    np.testing.assert_allclose(
        y,
        train.loc[:, ordered_targets].to_numpy(),
    )
    assert y.shape == (n_train, 2)
    assert metadata["target_cols"] == ordered_targets
    assert metadata["target_type"] == "multi_target"
    assert metadata["is_classification"] is False
    assert metadata["cv_strategy"] == "kfold"

    predictions = y.copy()
    predictions[:, 0] += np.linspace(0.0, 11.0, n_train)
    predictions[:, 1] += 0.5
    expected = np.mean(
        [
            np.sqrt(mean_squared_error(y[:, column], predictions[:, column]))
            for column in range(y.shape[1])
        ]
    )
    assert score_predictions(
        predictions,
        y,
        "multi_target_regression",
        "rmse",
    ) == pytest.approx(expected)


def test_multilabel_host_scoring_is_column_wise_and_rejects_wrong_shape(
    tmp_path,
) -> None:
    y = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ]
    )
    predictions = np.array(
        [
            [0.1, 0.8, 0.2],
            [0.7, 0.3, 0.9],
            [0.4, 0.2, 0.8],
            [0.9, 0.7, 0.1],
        ]
    )
    expected = np.mean(
        [
            log_loss(y[:, column], predictions[:, column], labels=[0, 1])
            for column in range(y.shape[1])
        ]
    )

    assert score_predictions(
        predictions,
        y,
        "multi_label_classification",
        "log_loss",
    ) == pytest.approx(expected)

    wrong_shape_path = tmp_path / "oof_wrong_shape.npy"
    np.save(wrong_shape_path, predictions[:, :1])
    assert compute_oof_score(
        wrong_shape_path,
        y,
        "log_loss",
    ) == float("inf")
    with pytest.raises(ValueError, match="shape mismatch"):
        score_predictions(
            predictions[:, :1],
            y,
            "multi_label_classification",
            "log_loss",
        )


def test_multilabel_oof_contract_uses_canonical_shape_not_peer_shape(
    tmp_path,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    malformed = np.full((8, 2), 0.5)
    np.save(models_dir / "oof_bad.npy", malformed)
    np.save(models_dir / "test_bad.npy", np.full((3, 2), 0.5))

    valid, results = validate_oof_stack(
        {
            "bad": (
                models_dir / "oof_bad.npy",
                models_dir / "test_bad.npy",
            )
        },
        models_dir,
        expected_shape=(8, 3),
        problem_type="multi_label",
    )

    assert valid == {}
    assert results[0].shape_match is False
    assert any("Shape mismatch" in error for error in results[0].errors)


def test_multilabel_stacking_preserves_independent_probability_columns(
    tmp_path,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    y = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ]
        * 3
    )
    model_oofs = {
        "a": y * 0.70 + 0.15,
        "b": y * 0.50 + 0.25,
    }
    model_tests = {
        "a": np.array(
            [
                [0.80, 0.70, 0.60],
                [0.20, 0.30, 0.90],
                [0.90, 0.90, 0.10],
            ]
        ),
        "b": np.array(
            [
                [0.70, 0.65, 0.55],
                [0.25, 0.35, 0.80],
                [0.85, 0.75, 0.20],
            ]
        ),
    }
    pairs = {}
    for name, oof_predictions in model_oofs.items():
        oof_path = models_dir / f"oof_{name}.npy"
        test_path = models_dir / f"test_{name}.npy"
        np.save(oof_path, oof_predictions)
        np.save(test_path, model_tests[name])
        np.save(
            models_dir / f"test_ids_{name}.npy",
            np.array(["test-0", "test-1", "test-2"]),
        )
        pairs[name] = (oof_path, test_path)

    ensemble, test_predictions = stack_from_prediction_pairs(
        prediction_pairs=pairs,
        y=y,
        problem_type="classification",
        metric_name="log_loss",
        models_dir=models_dir,
        expected_class_order=["label_a", "label_b", "label_c"],
        train_ids=None,
        folds_path=None,
        enable_calibration=True,
        enable_post_calibration=True,
        n_targets=3,
    )

    assert ensemble is not None
    assert test_predictions is not None
    assert test_predictions.shape == (3, 3)
    assert np.all((test_predictions > 0) & (test_predictions < 1))
    assert not np.allclose(test_predictions.sum(axis=1), 1.0)
