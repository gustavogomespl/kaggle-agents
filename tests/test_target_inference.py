"""Tests for target inference utilities (multi-label support)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.utils.target_inference import (
    TargetInfo,
    get_target_type_constraints,
    infer_target_columns,
    validate_predictions_shape,
)


class TestInferTargetColumns:
    """Tests for target column inference from sample_submission."""

    def test_single_target_classification(self):
        """Should detect single target classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_sub = pd.DataFrame({
                "id": range(10),
                "target": [0] * 10,
            })
            path = Path(tmpdir) / "sample_submission.csv"
            sample_sub.to_csv(path, index=False)

            result = infer_target_columns(path)

            assert result.target_type == "single"
            assert result.target_cols == ["target"]
            assert result.id_col == "id"
            assert not result.is_multi_output
            assert result.n_targets == 1

    def test_multi_label_binary_targets(self):
        """Should detect multi-label classification (binary values)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_sub = pd.DataFrame({
                "id": range(10),
                "class_A": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "class_B": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "class_C": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            })
            path = Path(tmpdir) / "sample_submission.csv"
            sample_sub.to_csv(path, index=False)

            result = infer_target_columns(path)

            assert result.target_type == "multi_label"
            assert result.target_cols == ["class_A", "class_B", "class_C"]
            assert result.id_col == "id"
            assert result.is_multi_output
            assert result.n_targets == 3

    def test_multi_target_continuous(self):
        """Should detect multi-target regression (continuous values)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_sub = pd.DataFrame({
                "id": range(10),
                "target_1": np.random.randn(10),
                "target_2": np.random.randn(10),
            })
            path = Path(tmpdir) / "sample_submission.csv"
            sample_sub.to_csv(path, index=False)

            result = infer_target_columns(path)

            assert result.target_type == "multi_target"
            assert result.target_cols == ["target_1", "target_2"]
            assert result.id_col == "id"
            assert result.is_multi_output
            assert result.n_targets == 2

    def test_multi_label_with_float_binary(self):
        """Should detect multi-label when values are 0.0 and 1.0 floats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_sub = pd.DataFrame({
                "id": range(10),
                "class_A": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "class_B": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            })
            path = Path(tmpdir) / "sample_submission.csv"
            sample_sub.to_csv(path, index=False)

            result = infer_target_columns(path)

            assert result.target_type == "multi_label"

    def test_raises_error_for_single_column(self):
        """Should raise error when only ID column exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_sub = pd.DataFrame({
                "id": range(10),
            })
            path = Path(tmpdir) / "sample_submission.csv"
            sample_sub.to_csv(path, index=False)

            with pytest.raises(ValueError, match="at least 2 columns"):
                infer_target_columns(path)


class TestGetTargetTypeConstraints:
    """Tests for constraint generation based on target type."""

    def test_multi_label_constraints_mention_sigmoid(self):
        """Multi-label constraints should mention sigmoid, not softmax."""
        constraints = get_target_type_constraints("multi_label")

        assert "sigmoid" in constraints.lower()
        assert "softmax" in constraints.lower()  # Should mention NOT to use softmax
        assert "BCEWithLogitsLoss" in constraints or "log_loss" in constraints.lower()

    def test_multi_target_constraints_mention_rmse(self):
        """Multi-target constraints should mention RMSE."""
        constraints = get_target_type_constraints("multi_target")

        assert "rmse" in constraints.lower() or "mean" in constraints.lower()

    def test_single_constraints_mention_softmax(self):
        """Single target constraints should mention softmax for multiclass."""
        constraints = get_target_type_constraints("single")

        assert "softmax" in constraints.lower() or "sigmoid" in constraints.lower()


class TestValidatePredictionsShape:
    """Tests for prediction shape validation."""

    def test_valid_single_target_1d(self):
        """Should pass for 1D predictions with single target."""
        target_info = TargetInfo(
            target_cols=["target"],
            target_type="single",
            id_col="id",
        )
        predictions = np.random.rand(100)

        is_valid, error = validate_predictions_shape(predictions, target_info)

        assert is_valid
        assert error == ""

    def test_valid_multi_label_2d(self):
        """Should pass for 2D predictions with multi-label."""
        target_info = TargetInfo(
            target_cols=["class_A", "class_B", "class_C"],
            target_type="multi_label",
            id_col="id",
        )
        # Sigmoid-style predictions (independent per class, don't sum to 1)
        predictions = np.random.rand(100, 3) * 0.5  # Values between 0 and 0.5

        is_valid, error = validate_predictions_shape(predictions, target_info)

        assert is_valid
        assert error == ""

    def test_fails_1d_for_multi_target(self):
        """Should fail for 1D predictions when multi-target expected."""
        target_info = TargetInfo(
            target_cols=["target_1", "target_2"],
            target_type="multi_target",
            id_col="id",
        )
        predictions = np.random.rand(100)  # 1D, should be 2D

        is_valid, error = validate_predictions_shape(predictions, target_info)

        assert not is_valid
        assert "2D" in error

    def test_fails_wrong_column_count(self):
        """Should fail when column count doesn't match target count."""
        target_info = TargetInfo(
            target_cols=["class_A", "class_B", "class_C"],
            target_type="multi_label",
            id_col="id",
        )
        predictions = np.random.rand(100, 5)  # 5 columns, should be 3

        is_valid, error = validate_predictions_shape(predictions, target_info)

        assert not is_valid
        assert "3" in error
        assert "5" in error

    def test_fails_multi_label_out_of_range(self):
        """Should fail for multi-label predictions outside [0, 1]."""
        target_info = TargetInfo(
            target_cols=["class_A", "class_B"],
            target_type="multi_label",
            id_col="id",
        )
        predictions = np.random.rand(100, 2) * 2 - 0.5  # Range [-0.5, 1.5]

        is_valid, error = validate_predictions_shape(predictions, target_info)

        assert not is_valid
        assert "[0, 1]" in error

    def test_fails_multi_label_softmax_detected(self):
        """Should fail if multi-label predictions appear to be softmax (sum to 1)."""
        target_info = TargetInfo(
            target_cols=["class_A", "class_B", "class_C"],
            target_type="multi_label",
            id_col="id",
        )
        # Softmax-style predictions (sum to 1 per row)
        predictions = np.array([
            [0.3, 0.5, 0.2],
            [0.1, 0.6, 0.3],
            [0.4, 0.4, 0.2],
        ])

        is_valid, error = validate_predictions_shape(predictions, target_info)

        assert not is_valid
        assert "softmax" in error.lower() or "sum to 1" in error.lower()


class TestTargetInfoProperties:
    """Tests for TargetInfo dataclass properties."""

    def test_is_multi_output_for_multi_label(self):
        """is_multi_output should be True for multi-label."""
        info = TargetInfo(
            target_cols=["a", "b"],
            target_type="multi_label",
            id_col="id",
        )
        assert info.is_multi_output

    def test_is_multi_output_for_multi_target(self):
        """is_multi_output should be True for multi-target."""
        info = TargetInfo(
            target_cols=["a", "b"],
            target_type="multi_target",
            id_col="id",
        )
        assert info.is_multi_output

    def test_is_multi_output_false_for_single(self):
        """is_multi_output should be False for single target."""
        info = TargetInfo(
            target_cols=["target"],
            target_type="single",
            id_col="id",
        )
        assert not info.is_multi_output

    def test_n_targets_count(self):
        """n_targets should return correct count."""
        info = TargetInfo(
            target_cols=["a", "b", "c", "d"],
            target_type="multi_label",
            id_col="id",
        )
        assert info.n_targets == 4
