"""Regression tests for fail-closed model artifact validation."""

from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.utils.image_to_image_contract import save_packed_images
from kaggle_agents.utils.strict_validation import (
    StrictValidationConfig,
    validate_model_artifacts,
    validate_prediction_quality,
)


def _write_pickle_sentinel(path: str) -> None:
    Path(path).write_text("PICKLE_EXECUTED", encoding="utf-8")


class _PicklePayload:
    """Payload that writes only if a candidate artifact is unpickled."""

    def __init__(self, sentinel: Path):
        self.sentinel = str(sentinel)

    def __reduce__(self):
        return _write_pickle_sentinel, (self.sentinel,)


def _save_artifacts(
    working_dir: Path,
    component_name: str,
    oof: np.ndarray,
    test: np.ndarray,
) -> None:
    models_dir = working_dir / "models"
    models_dir.mkdir()
    np.save(models_dir / f"oof_{component_name}.npy", oof)
    np.save(models_dir / f"test_{component_name}.npy", test)


@pytest.mark.parametrize(
    ("problem_type", "oof", "test"),
    [
        (
            "binary_classification",
            np.array([0.2, np.nan, 0.8]),
            np.array([0.3, 0.7]),
        ),
        (
            "multiclass_classification",
            np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]),
            np.array([[0.7, 0.3], [np.inf, 0.2]]),
        ),
        (
            "multilabel_classification",
            np.array([[0.8, 0.1], [0.4, 0.6], [0.1, 0.9]]),
            np.array([[0.7, 0.3], [0.2, np.inf]]),
        ),
        (
            "tabular_regression",
            np.array([1.0, 2.0, np.inf]),
            np.array([3.0, 4.0]),
        ),
    ],
)
def test_concrete_problem_types_reject_non_finite_predictions(
    tmp_path,
    problem_type,
    oof,
    test,
):
    _save_artifacts(tmp_path, "model", oof, test)

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        problem_type=problem_type,
        config=StrictValidationConfig(strict_mode=True),
    )

    assert result.is_valid is False
    assert any("NaN or Inf" in error for error in result.errors)


def test_rejects_prediction_width_mismatch(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]),
        np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]),
    )

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        problem_type="multiclass_classification",
        config=StrictValidationConfig(strict_mode=True),
    )

    assert result.is_valid is False
    assert any("width mismatch" in error for error in result.errors)


def test_rejects_multiclass_probability_rows_that_do_not_sum_to_one(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array([[0.8, 0.3], [0.4, 0.6], [0.1, 0.9]]),
        np.array([[0.7, 0.3], [0.2, 0.8]]),
    )

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        problem_type="multiclass_classification",
        config=StrictValidationConfig(strict_mode=True),
    )

    assert result.is_valid is False
    assert any("probability-sum contract" in error for error in result.errors)


def test_accepts_valid_concrete_multiclass_artifacts(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]),
        np.array([[0.7, 0.3], [0.2, 0.8]]),
    )
    np.save(
        tmp_path / "models" / "class_order_model.npy",
        np.array(["negative", "positive"], dtype=str),
    )

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        expected_class_order=["negative", "positive"],
        problem_type="multiclass_classification",
        config=StrictValidationConfig(
            strict_mode=True,
            require_class_order=True,
            require_component_class_order=True,
        ),
    )

    assert result.is_valid is True
    assert result.errors == []


def test_rejects_multiclass_probabilities_without_expected_class_order(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        ),
        np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]),
    )

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        problem_type="multiclass_classification",
        config=StrictValidationConfig(strict_mode=True),
    )

    assert result.is_valid is False
    assert any("expected class order contract" in error for error in result.errors)


def test_mle_contract_rejects_global_class_order_fallback(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        ),
        np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]),
    )
    models = tmp_path / "models"
    np.save(models / "class_order.npy", np.array(["a", "b", "c"], dtype=str))

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        expected_class_order=["a", "b", "c"],
        problem_type="multiclass_classification",
        config=StrictValidationConfig(
            strict_mode=True,
            require_class_order=True,
            require_component_class_order=True,
        ),
    )

    assert result.is_valid is False
    assert any("component-specific class_order" in error for error in result.errors)


def test_quality_gate_rejects_non_finite_predictions():
    valid, issues = validate_prediction_quality(
        np.array([0.1, np.nan, 0.8]),
        problem_type="binary_classification",
    )

    assert valid is False
    assert issues == ["Predictions contain NaN or Inf values"]


def test_balanced_multiclass_mean_is_not_treated_as_broken():
    predictions = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    valid, issues = validate_prediction_quality(
        predictions,
        problem_type="multiclass_classification",
    )

    assert valid is True
    assert issues == []


def test_strict_id_contract_rejects_permuted_oof_rows(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array([0.1, 0.8, 0.3]),
        np.array([0.2, 0.7]),
    )
    models = tmp_path / "models"
    np.save(models / "train_ids_model.npy", np.array(["b", "a", "c"]))
    np.save(models / "test_ids_model.npy", np.array(["x", "y"]))

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        expected_train_ids=["a", "b", "c"],
        expected_test_ids=["x", "y"],
        problem_type="binary_classification",
        config=StrictValidationConfig(
            strict_mode=True,
            require_train_ids=True,
            require_test_ids=True,
        ),
    )

    assert result.is_valid is False
    assert any("exact row order" in error for error in result.errors)


def test_seq2seq_accepts_text_oof_with_exact_id_contract(tmp_path):
    _save_artifacts(
        tmp_path,
        "model",
        np.array(["one", "two", "three"], dtype=str),
        np.array(["four", "five"], dtype=str),
    )
    models = tmp_path / "models"
    np.save(models / "train_ids_model.npy", np.array(["a", "b", "c"]))
    np.save(models / "test_ids_model.npy", np.array(["x", "y"]))

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=3,
        expected_n_test=2,
        expected_train_ids=["a", "b", "c"],
        expected_test_ids=["x", "y"],
        problem_type="seq2seq",
        config=StrictValidationConfig(
            strict_mode=True,
            require_train_ids=True,
            require_test_ids=True,
        ),
    )

    assert result.is_valid is True
    assert result.errors == []


def test_candidate_pickle_in_id_artifact_is_rejected_without_execution(
    tmp_path: Path,
) -> None:
    _save_artifacts(
        tmp_path,
        "model",
        np.array([0.2, 0.8]),
        np.array([0.4]),
    )
    sentinel = tmp_path / "id-pickle-executed"
    np.save(
        tmp_path / "models" / "train_ids_model.npy",
        np.array([_PicklePayload(sentinel)], dtype=object),
    )
    assert not sentinel.exists()

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=2,
        expected_n_test=1,
        expected_train_ids=["a", "b"],
        problem_type="binary_classification",
        config=StrictValidationConfig(
            strict_mode=True,
            require_train_ids=True,
        ),
    )

    assert result.is_valid is False
    assert any("Failed to load Train IDs" in error for error in result.errors)
    assert not sentinel.exists()


def test_candidate_pickle_in_seq2seq_predictions_is_rejected_without_execution(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    sentinel = tmp_path / "oof-pickle-executed"
    np.save(
        models / "oof_model.npy",
        np.array([_PicklePayload(sentinel)], dtype=object),
    )
    np.save(models / "test_model.npy", np.array(["safe"], dtype=str))
    assert not sentinel.exists()

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=1,
        expected_n_test=1,
        problem_type="seq2seq",
        config=StrictValidationConfig(strict_mode=True),
    )

    assert result.is_valid is False
    assert any("Failed to load OOF file" in error for error in result.errors)
    assert not sentinel.exists()


def test_image_to_image_accepts_variable_sized_packed_component_artifacts(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    save_packed_images(
        models / "oof_model.npz",
        [
            np.zeros((2, 3, 3), dtype=np.float32),
            np.ones((4, 2, 3), dtype=np.float32),
        ],
        image_ids=["a.png", "nested/b.png"],
    )
    save_packed_images(
        models / "test_model.npz",
        [np.full((3, 5, 3), 0.5, dtype=np.float32)],
        image_ids=["test.png"],
    )

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_n_train=2,
        # Pixel-level CSV rows are not the number of packed test images.
        expected_n_test=100_001,
        expected_train_ids=["a.png", "nested/b.png"],
        expected_test_ids=["test.png"],
        problem_type="image_to_image",
        config=StrictValidationConfig(
            strict_mode=True,
            require_train_ids=True,
            require_test_ids=True,
        ),
    )

    assert result.is_valid is True
    assert result.errors == []
    assert result.files_verified == ["oof_model.npz", "test_model.npz"]


def test_image_to_image_rejects_packed_oof_with_wrong_id_order(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    save_packed_images(
        models / "oof_model.npz",
        [
            np.zeros((2, 2, 3), dtype=np.float32),
            np.ones((2, 2, 3), dtype=np.float32),
        ],
        image_ids=["b.png", "a.png"],
    )
    save_packed_images(
        models / "test_model.npz",
        [np.zeros((2, 2, 3), dtype=np.float32)],
        image_ids=["test.png"],
    )

    result = validate_model_artifacts(
        tmp_path,
        "model",
        expected_train_ids=["a.png", "b.png"],
        expected_test_ids=["test.png"],
        problem_type="image_to_image",
        config=StrictValidationConfig(strict_mode=True),
    )

    assert result.is_valid is False
    assert any("OOF image IDs do not match" in error for error in result.errors)
