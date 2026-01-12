"""Tests for cross-validation utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from kaggle_agents.utils.cross_validation import (
    AdaptiveCrossValidator,
    _detect_group_column,
    _detect_temporal_column,
    _validate_temporal_folds,
    generate_folds,
)


class TestAdaptiveCrossValidator:
    """Tests for AdaptiveCrossValidator."""

    def test_get_stratified_kfold(self):
        """Test StratifiedKFold selection."""
        cv = AdaptiveCrossValidator.get_cv_strategy("StratifiedKFold", n_splits=5)
        assert isinstance(cv, StratifiedKFold)
        assert cv.n_splits == 5

    def test_get_time_series_split(self):
        """Test TimeSeriesSplit selection."""
        cv = AdaptiveCrossValidator.get_cv_strategy("TimeSeriesSplit", n_splits=5)
        assert isinstance(cv, TimeSeriesSplit)
        assert cv.n_splits == 5

    def test_get_kfold_default(self):
        """Test default KFold selection."""
        cv = AdaptiveCrossValidator.get_cv_strategy("KFold", n_splits=5)
        assert isinstance(cv, KFold)
        assert cv.n_splits == 5

    def test_determine_best_strategy_temporal(self):
        """Test strategy determination for temporal data."""
        X = pd.DataFrame({"feature": range(100)})
        y = pd.Series(np.random.choice([0, 1], 100))

        data_chars = {
            "temporal": {"has_date_columns": True},
            "target": {"type": "categorical"},
        }

        strategy = AdaptiveCrossValidator.determine_best_strategy(X, y, data_chars)
        assert strategy == "TimeSeriesSplit"

    def test_determine_best_strategy_imbalanced(self):
        """Test strategy determination for imbalanced data."""
        X = pd.DataFrame({"feature": range(100)})
        y = pd.Series(np.random.choice([0, 1], 100))

        data_chars = {
            "temporal": {"has_date_columns": False},
            "target": {"type": "categorical", "imbalance_ratio": 5.0},
        }

        strategy = AdaptiveCrossValidator.determine_best_strategy(X, y, data_chars)
        assert strategy == "StratifiedKFold"

    def test_get_scoring_metric_classification(self):
        """Test scoring metric for classification."""
        metric = AdaptiveCrossValidator.get_scoring_metric("classification")
        assert metric == "accuracy"

    def test_get_scoring_metric_regression(self):
        """Test scoring metric for regression."""
        metric = AdaptiveCrossValidator.get_scoring_metric("regression")
        assert metric == "neg_mean_squared_error"

    def test_get_scoring_metric_custom(self):
        """Test custom metric mapping."""
        metric = AdaptiveCrossValidator.get_scoring_metric("classification", "AUC")
        assert metric == "roc_auc"

        metric = AdaptiveCrossValidator.get_scoring_metric("regression", "RMSE")
        assert metric == "neg_root_mean_squared_error"


class TestTemporalColumnDetection:
    """Tests for automatic temporal column detection."""

    def test_detects_date_column(self):
        """Should detect column named 'date'."""
        df = pd.DataFrame({
            "id": range(10),
            "date": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = _detect_temporal_column(df)
        assert result == "date"

    def test_detects_datetime_column(self):
        """Should detect column named 'datetime'."""
        df = pd.DataFrame({
            "id": range(10),
            "datetime": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = _detect_temporal_column(df)
        assert result == "datetime"

    def test_detects_timestamp_column(self):
        """Should detect column named 'timestamp'."""
        df = pd.DataFrame({
            "id": range(10),
            "timestamp": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = _detect_temporal_column(df)
        assert result == "timestamp"

    def test_detects_transaction_date(self):
        """Should detect column named 'transaction_date'."""
        df = pd.DataFrame({
            "id": range(10),
            "transaction_date": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = _detect_temporal_column(df)
        assert result == "transaction_date"

    def test_detects_string_dates(self):
        """Should detect string columns that parse as dates."""
        df = pd.DataFrame({
            "id": range(10),
            "event_date": ["2023-01-01", "2023-01-02", "2023-01-03"] * 3 + ["2023-01-04"],
            "value": range(10),
        })

        result = _detect_temporal_column(df)
        assert result == "event_date"

    def test_returns_none_for_non_temporal_data(self):
        """Should return None when no temporal column exists."""
        df = pd.DataFrame({
            "id": range(10),
            "feature": range(10),
            "value": range(10),
        })

        result = _detect_temporal_column(df)
        assert result is None

    def test_uses_explicit_temporal_col(self):
        """Should use explicitly specified temporal column."""
        df = pd.DataFrame({
            "id": range(10),
            "my_time": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = _detect_temporal_column(df, temporal_col="my_time")
        assert result == "my_time"


class TestGroupColumnDetection:
    """Tests for automatic group column detection."""

    def test_detects_patient_id(self):
        """Should detect PatientID column."""
        df = pd.DataFrame({
            "PatientID": ["P1", "P1", "P2", "P2", "P3"],
            "feature": range(5),
        })

        result = _detect_group_column(df)
        assert result == "PatientID"

    def test_detects_user_id(self):
        """Should detect user_id column."""
        df = pd.DataFrame({
            "user_id": ["U1", "U1", "U2", "U2", "U3"],
            "feature": range(5),
        })

        result = _detect_group_column(df)
        assert result == "user_id"

    def test_detects_subject_id(self):
        """Should detect subject_id column."""
        df = pd.DataFrame({
            "subject_id": ["S1", "S1", "S2", "S2", "S3"],
            "feature": range(5),
        })

        result = _detect_group_column(df)
        assert result == "subject_id"

    def test_ignores_unique_id_column(self):
        """Should ignore columns where all values are unique (not a real group)."""
        df = pd.DataFrame({
            "PatientID": range(100),  # All unique - not a valid group
            "feature": range(100),
        })

        result = _detect_group_column(df)
        assert result is None

    def test_uses_explicit_group_col(self):
        """Should use explicitly specified group column."""
        df = pd.DataFrame({
            "my_group": ["G1", "G1", "G2"],
            "feature": range(3),
        })

        result = _detect_group_column(df, group_col="my_group")
        assert result == "my_group"


class TestValidateTemporalFolds:
    """Tests for temporal fold validation."""

    def test_passes_with_correct_ordering(self):
        """Should pass when folds are in non-decreasing temporal order."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })
        folds = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        # Should not raise
        _validate_temporal_folds(df, folds, "date", n_folds=5)

    def test_passes_with_equal_timestamps(self):
        """Should pass when some folds have equal max timestamps."""
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2023-01-01", "2023-01-01",  # fold 0
                "2023-01-02", "2023-01-02",  # fold 1
                "2023-01-02", "2023-01-02",  # fold 2 - same as fold 1
            ]),
            "value": range(6),
        })
        folds = np.array([0, 0, 1, 1, 2, 2])

        # Should not raise (allows equal timestamps)
        _validate_temporal_folds(df, folds, "date", n_folds=3)

    def test_fails_with_wrong_ordering(self):
        """Should fail when later fold has earlier timestamps."""
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2023-01-03", "2023-01-03",  # fold 0 - later dates!
                "2023-01-01", "2023-01-01",  # fold 1 - earlier dates!
            ]),
            "value": range(4),
        })
        folds = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="Temporal ordering violation"):
            _validate_temporal_folds(df, folds, "date", n_folds=2)


class TestGenerateFolds:
    """Tests for fold generation with temporal support."""

    def test_generates_stratified_folds_for_classification(self):
        """Should use StratifiedKFold for classification tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            output_path = Path(tmpdir) / "folds.csv"

            # Create classification data
            df = pd.DataFrame({
                "id": range(100),
                "feature": range(100),
                "target": [0] * 50 + [1] * 50,
            })
            df.to_csv(train_path, index=False)

            result_path = generate_folds(
                train_path=str(train_path),
                target_col="target",
                output_path=str(output_path),
                n_folds=5,
                seed=42,
            )

            # Verify file was created
            assert Path(result_path).exists()

            # Verify folds are balanced (stratified)
            result_df = pd.read_csv(result_path)
            for fold in range(5):
                fold_df = result_df[result_df["fold"] == fold]
                # Each fold should have roughly equal class distribution
                class_counts = fold_df["target"].value_counts()
                assert len(class_counts) == 2  # Both classes present

    def test_generates_timeseries_folds_for_temporal_data(self):
        """Should use TimeSeriesSplit when temporal column is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            output_path = Path(tmpdir) / "folds.csv"

            # Create temporal data
            df = pd.DataFrame({
                "id": range(100),
                "date": pd.date_range("2023-01-01", periods=100),
                "target": range(100),
            })
            df.to_csv(train_path, index=False)

            result_path = generate_folds(
                train_path=str(train_path),
                target_col="target",
                output_path=str(output_path),
                n_folds=5,
                seed=42,
            )

            # Verify file was created
            assert Path(result_path).exists()

            # Verify folds are in temporal order
            result_df = pd.read_csv(result_path)
            result_df["date"] = pd.to_datetime(result_df["date"])

            for fold in range(1, 5):
                current_fold_max = result_df[result_df["fold"] == fold]["date"].max()
                prev_fold_max = result_df[result_df["fold"] == fold - 1]["date"].max()
                assert current_fold_max >= prev_fold_max

    def test_respects_force_strategy(self):
        """Should use forced strategy when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            output_path = Path(tmpdir) / "folds.csv"

            # Create data with temporal column but force KFold
            df = pd.DataFrame({
                "id": range(100),
                "date": pd.date_range("2023-01-01", periods=100),
                "target": [0] * 50 + [1] * 50,
            })
            df.to_csv(train_path, index=False)

            result_path = generate_folds(
                train_path=str(train_path),
                target_col="target",
                output_path=str(output_path),
                n_folds=5,
                seed=42,
                force_strategy="kfold",  # Force KFold despite temporal column
            )

            # Verify file was created
            result_df = pd.read_csv(result_path)
            assert "fold" in result_df.columns

    def test_handles_missing_target_column(self):
        """Should handle case where target column is not in training data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            output_path = Path(tmpdir) / "folds.csv"

            # Create data without target column (multi-label case)
            df = pd.DataFrame({
                "id": range(100),
                "feature1": range(100),
                "feature2": range(100),
            })
            df.to_csv(train_path, index=False)

            # Should not raise, should use KFold
            result_path = generate_folds(
                train_path=str(train_path),
                target_col="nonexistent_target",
                output_path=str(output_path),
                n_folds=5,
                seed=42,
            )

            result_df = pd.read_csv(result_path)
            assert "fold" in result_df.columns

    def test_force_strategy_overrides_group_detection(self):
        """force_strategy should override automatic group detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            output_path = Path(tmpdir) / "folds.csv"

            # Create data with a group column (PatientID) that would normally trigger GroupKFold
            df = pd.DataFrame({
                "id": range(100),
                "PatientID": [f"P{i // 5}" for i in range(100)],  # 20 patients, 5 rows each
                "feature": range(100),
                "target": [0] * 50 + [1] * 50,
            })
            df.to_csv(train_path, index=False)

            # Force KFold even though PatientID is detected
            result_path = generate_folds(
                train_path=str(train_path),
                target_col="target",
                output_path=str(output_path),
                n_folds=5,
                seed=42,
                force_strategy="kfold",  # Should override group detection
            )

            result_df = pd.read_csv(result_path)
            assert "fold" in result_df.columns

            # Verify that patients are split across folds (KFold behavior, not GroupKFold)
            # With GroupKFold, all rows for a patient would be in the same fold
            # With KFold, patients can be split across folds
            patient_folds = result_df.groupby("PatientID")["fold"].nunique()
            # At least some patients should have rows in multiple folds
            assert (patient_folds > 1).any(), "force_strategy=kfold should split patients across folds"

    def test_force_strategy_stratified_overrides_group_detection(self):
        """force_strategy=stratified should override automatic group detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.csv"
            output_path = Path(tmpdir) / "folds.csv"

            # Create data with group column
            df = pd.DataFrame({
                "id": range(100),
                "user_id": [f"U{i // 10}" for i in range(100)],  # 10 users
                "feature": range(100),
                "target": [0] * 50 + [1] * 50,
            })
            df.to_csv(train_path, index=False)

            # Force StratifiedKFold
            result_path = generate_folds(
                train_path=str(train_path),
                target_col="target",
                output_path=str(output_path),
                n_folds=5,
                seed=42,
                force_strategy="stratified",
            )

            result_df = pd.read_csv(result_path)

            # Verify stratification - each fold should have similar class distribution
            for fold in range(5):
                fold_df = result_df[result_df["fold"] == fold]
                class_ratio = fold_df["target"].mean()
                # Original ratio is 0.5, each fold should be close to that
                assert 0.4 <= class_ratio <= 0.6, f"Fold {fold} has unbalanced classes: {class_ratio}"
