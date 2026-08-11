"""Regression tests for metric-aware ensemble scoring."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, mean_squared_log_error, roc_auc_score

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


def test_multilabel_auc_is_mean_column_wise_without_row_normalization():
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
    expected = roc_auc_score(y, predictions, average="macro")

    assert score_predictions(
        predictions,
        y,
        "classification",
        "auc",
    ) == pytest.approx(-expected)


def test_multioutput_rmsle_is_mean_column_wise():
    y = np.array([[1.0, 100.0], [4.0, 400.0], [9.0, 900.0]])
    predictions = np.array([[2.0, 120.0], [2.0, 350.0], [12.0, 1000.0]])
    expected = np.mean(
        [
            np.sqrt(mean_squared_log_error(y[:, 0], predictions[:, 0])),
            np.sqrt(mean_squared_log_error(y[:, 1], predictions[:, 1])),
        ]
    )

    assert score_predictions(
        predictions,
        y,
        "regression",
        "rmsle",
    ) == pytest.approx(expected)


def test_score_predictions_supports_seq2seq_exact_match():
    score = score_predictions(
        np.array(["one", "wrong", "three"], dtype=object),
        np.array(["one", "two", "three"], dtype=object),
        "seq2seq",
        "accuracy",
    )

    assert score == pytest.approx(-(2 / 3))


def test_seq2seq_scoring_is_chunked_and_honors_row_eligibility():
    predictions = np.array(
        ["zero", "wrong", "two", "wrong", "four"], dtype=object
    )
    targets = np.array(
        ["zero", "one", "two", "three", "four"], dtype=object
    )
    eligible = np.array([True, False, True, True, True])
    progress: list[tuple[int, int]] = []

    score = score_predictions(
        predictions,
        targets,
        "seq2seq",
        "accuracy",
        row_mask=eligible,
        chunk_rows=2,
        progress=lambda processed, total: progress.append((processed, total)),
    )

    assert score == pytest.approx(-(3 / 4))
    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_seq2seq_scoring_rejects_an_empty_eligibility_mask():
    with pytest.raises(ValueError, match="selects no seq2seq rows"):
        score_predictions(
            np.array(["one", "two"], dtype=str),
            np.array(["one", "two"], dtype=str),
            "seq2seq",
            "accuracy",
            row_mask=np.array([False, False]),
            chunk_rows=1,
        )
