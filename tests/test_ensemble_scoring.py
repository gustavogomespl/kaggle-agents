"""Regression tests for metric-aware ensemble scoring."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, mean_squared_log_error

from kaggle_agents.agents.ensemble.scoring import compute_oof_score, score_predictions


@pytest.mark.parametrize("metric", ["f1", "precision", "recall", "mcc"])
def test_binary_label_metrics_prefer_good_predictions(metric):
    y = np.array([0, 1, 0, 1, 0, 1])
    good = np.array([0.05, 0.95, 0.10, 0.90, 0.20, 0.80])
    bad = 1.0 - good

    good_score = score_predictions(good, y, "classification", metric)
    bad_score = score_predictions(bad, y, "classification", metric)

    assert good_score < bad_score
    assert good_score == pytest.approx(-1.0)


def test_qwk_prefers_correct_multiclass_probabilities():
    y = np.array([0, 1, 2, 0, 1, 2])
    good = np.eye(3)[y] * 0.9 + 0.1 / 3
    bad = np.roll(good, shift=1, axis=1)

    good_score = score_predictions(good, y, "classification", "quadratic_weighted_kappa")
    bad_score = score_predictions(bad, y, "classification", "quadratic_weighted_kappa")

    assert good_score == pytest.approx(-1.0)
    assert good_score < bad_score


def test_qwk_continuous_predictions_keep_original_numeric_scale():
    y = np.array([1, 2, 3, 4, 5])
    predictions = np.array([1.1, 1.9, 3.2, 3.8, 5.0])

    score = score_predictions(
        predictions,
        y,
        "classification",
        "quadratic_weighted_kappa",
    )

    assert score == pytest.approx(-1.0)


def test_qwk_binary_probabilities_support_nonzero_original_labels():
    y = np.array([1, 2, 1, 2])
    predictions = np.array([0.1, 0.9, 0.2, 0.8])

    assert score_predictions(
        predictions, y, "classification", "quadratic_weighted_kappa"
    ) == pytest.approx(-1.0)


def test_qwk_continuous_predictions_map_to_noncontiguous_classes():
    y = np.array([1, 2, 3, 5])
    predictions = np.array([1.1, 1.9, 3.1, 4.8])

    assert score_predictions(
        predictions, y, "classification", "quadratic_weighted_kappa"
    ) == pytest.approx(-1.0)


def test_explicit_macro_f1_is_honored_for_binary_targets():
    y = np.array([0, 0, 0, 1])
    predictions = np.array([0.1, 0.1, 0.1, 0.1])
    expected = f1_score(y, np.zeros_like(y), average="macro", zero_division=0)

    assert score_predictions(
        predictions, y, "classification", "f1_macro"
    ) == pytest.approx(-expected)


def test_regression_metrics_are_not_conflated():
    y = np.array([1.0, 4.0, 9.0])
    predictions = np.array([2.0, 2.0, 12.0])
    mse = np.mean((y - predictions) ** 2)

    assert score_predictions(predictions, y, "regression", "mse") == pytest.approx(mse)
    assert score_predictions(predictions, y, "regression", "rmse") == pytest.approx(
        np.sqrt(mse)
    )
    assert score_predictions(predictions, y, "regression", "rmsle") == pytest.approx(
        np.sqrt(mean_squared_log_error(y, predictions))
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("f1", 0.0),
        ("precision", 0.0),
        ("recall", 0.0),
        ("mcc", 0.0),
        ("quadratic_weighted_kappa", 0.0),
    ],
)
def test_compute_oof_score_returns_non_negative_loss(temp_data_dir, metric, expected):
    y = np.array([0, 1, 0, 1])
    oof_path = temp_data_dir / f"{metric}.npy"
    np.save(oof_path, np.array([0.05, 0.95, 0.10, 0.90]))

    assert compute_oof_score(oof_path, y, metric) == pytest.approx(expected)


def test_compute_oof_score_uses_rmsle(temp_data_dir):
    y = np.array([1.0, 4.0, 9.0])
    predictions = np.array([2.0, 2.0, 12.0])
    oof_path = temp_data_dir / "rmsle.npy"
    np.save(oof_path, predictions)

    expected = np.sqrt(mean_squared_log_error(y, predictions))
    assert compute_oof_score(oof_path, y, "rmsle") == pytest.approx(expected)
