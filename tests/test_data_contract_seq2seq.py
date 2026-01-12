"""Tests for seq2seq canonical data contract support."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.utils.data_contract import (
    _detect_seq2seq_group_column,
    prepare_canonical_data,
    load_canonical_data,
    SEQ2SEQ_GROUP_CANDIDATES,
    SEQ2SEQ_TASK_INDICATORS,
)


class TestSeq2seqGroupDetection:
    """Tests for seq2seq group column detection."""

    def test_detects_sentence_id(self):
        """Should detect sentence_id column."""
        df = pd.DataFrame({
            "sentence_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "token": range(10),
        })

        group_col = _detect_seq2seq_group_column(df)
        assert group_col == "sentence_id"

    def test_detects_utterance_id(self):
        """Should detect utterance_id column."""
        df = pd.DataFrame({
            "utterance_id": [1, 1, 2, 2, 3, 3],
            "text": ["a", "b", "c", "d", "e", "f"],
        })

        group_col = _detect_seq2seq_group_column(df)
        assert group_col == "utterance_id"

    def test_returns_none_when_no_group(self):
        """Should return None when no group column exists."""
        df = pd.DataFrame({
            "feature1": range(10),
            "feature2": range(10),
        })

        group_col = _detect_seq2seq_group_column(df)
        assert group_col is None

    def test_ignores_1to1_mapping(self):
        """Should ignore columns with 1:1 mapping (not a group)."""
        df = pd.DataFrame({
            "sentence_id": range(10),  # Each row has unique sentence_id
            "token": range(10),
        })

        group_col = _detect_seq2seq_group_column(df)
        assert group_col is None


class TestSeq2seqConstants:
    """Tests for seq2seq constants."""

    def test_seq2seq_group_candidates_includes_sentence_id(self):
        """Should include sentence_id in candidates."""
        assert "sentence_id" in SEQ2SEQ_GROUP_CANDIDATES

    def test_text_normalization_indicators(self):
        """Should have text_normalization task indicators."""
        assert "text_normalization" in SEQ2SEQ_TASK_INDICATORS

        tn_indicator = SEQ2SEQ_TASK_INDICATORS["text_normalization"]
        assert tn_indicator["source_col"] == "before"
        assert tn_indicator["target_col"] == "after"
        assert tn_indicator["class_col"] == "class"


class TestPrepareCanonicalDataSeq2seq:
    """Tests for prepare_canonical_data with seq2seq tasks."""

    def test_string_targets_saved_correctly(self, tmp_path):
        """Should save string targets with allow_pickle=True."""
        # Create mock text normalization data
        train_data = {
            "id": range(100),
            "sentence_id": [i // 5 for i in range(100)],
            "class": ["PLAIN"] * 50 + ["CARDINAL"] * 50,
            "before": ["hello"] * 50 + ["123"] * 50,
            "after": ["hello"] * 50 + ["one two three"] * 50,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        # Prepare canonical data
        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="text_normalization",
            class_col="class",
        )

        # Verify string targets were saved
        y = np.load(tmp_path / "canonical" / "y.npy", allow_pickle=True)
        assert y.dtype == object
        assert "hello" in y
        assert "one two three" in y

    def test_groupkfold_with_sentence_id(self, tmp_path):
        """Should use GroupKFold with sentence_id."""
        # Create data with sentence groups
        train_data = {
            "id": range(100),
            "sentence_id": [i // 10 for i in range(100)],  # 10 sentences
            "class": ["PLAIN"] * 100,
            "before": ["word"] * 100,
            "after": ["word"] * 100,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="text_normalization",
            n_folds=5,
        )

        # Load folds and verify no sentence is split across folds
        folds = np.load(tmp_path / "canonical" / "folds.npy")
        train_df["fold"] = folds

        for sentence_id in train_df["sentence_id"].unique():
            sentence_folds = train_df[train_df["sentence_id"] == sentence_id]["fold"].unique()
            assert len(sentence_folds) == 1, f"Sentence {sentence_id} split across folds: {sentence_folds}"

    def test_metadata_includes_seq2seq_fields(self, tmp_path):
        """Should include seq2seq-specific metadata fields."""
        train_data = {
            "id": range(50),
            "class": ["PLAIN"] * 50,
            "before": ["hello"] * 50,
            "after": ["hello"] * 50,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="text_normalization",
            source_col="before",
            class_col="class",
        )

        metadata = result["metadata"]

        assert metadata["task_type"] == "text_normalization"
        assert metadata["is_seq2seq"] is True
        assert metadata["source_col"] == "before"
        assert metadata["class_col"] == "class"
        assert metadata["target_dtype"] == "object"

    def test_canonical_version_bumped(self, tmp_path):
        """Should have canonical version 1.3 for seq2seq support."""
        train_data = {
            "id": range(50),
            "before": ["hello"] * 50,
            "after": ["hello"] * 50,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="text_normalization",
        )

        metadata = result["metadata"]
        assert metadata["canonical_version"] == "1.3"

    def test_canonical_version_correct_with_sampling(self, tmp_path):
        """Should have version 1.3 even when sampling is triggered."""
        # Create large dataset to trigger sampling
        train_data = {
            "id": range(1000),
            "before": [f"word_{i}" for i in range(1000)],
            "after": [f"word_{i}" for i in range(1000)],
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="text_normalization",
            max_rows=100,  # Trigger sampling
        )

        metadata = result["metadata"]
        # Version should be 1.3 even with sampling
        assert metadata["canonical_version"] == "1.3"
        # Verify sampling actually occurred
        assert metadata["sampled"] is True

    def test_no_stratification_for_string_targets(self, tmp_path):
        """Should not attempt stratification for string targets."""
        # Create data with many unique string targets
        train_data = {
            "id": range(100),
            "before": [f"word_{i}" for i in range(100)],
            "after": [f"result_{i}" for i in range(100)],  # 100 unique strings
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        # Should not raise error about stratification
        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="seq2seq",
        )

        assert result["metadata"]["is_classification"] is False


class TestLoadCanonicalDataSeq2seq:
    """Tests for loading seq2seq canonical data."""

    def test_loads_string_targets(self, tmp_path):
        """Should load string targets correctly."""
        # Prepare test data
        train_data = {
            "id": range(50),
            "before": ["hello"] * 50,
            "after": ["hello"] * 50,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["after"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="after",
            output_dir=tmp_path,
            task_type="text_normalization",
        )

        # Load and verify
        canonical = load_canonical_data(tmp_path)

        assert canonical["y"].dtype == object
        assert "hello" in canonical["y"]


class TestStringLabeledClassification:
    """Tests for string-labeled classification (NOT seq2seq)."""

    def test_string_classification_uses_stratified_cv(self, tmp_path):
        """String-labeled classification should still use stratified CV."""
        # Create imbalanced classification data with string labels
        train_data = {
            "id": range(100),
            "feature": range(100),
            # Imbalanced: 70 "cat", 20 "dog", 10 "bird"
            "label": ["cat"] * 70 + ["dog"] * 20 + ["bird"] * 10,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["label"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        # Use default task_type="tabular" (NOT seq2seq)
        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="label",
            output_dir=tmp_path,
            n_folds=5,
        )

        # Should be detected as classification
        assert result["metadata"]["is_classification"] is True
        assert result["metadata"]["n_classes"] == 3

        # Verify stratification: each fold should have roughly proportional classes
        folds = np.load(tmp_path / "canonical" / "folds.npy")
        train_df["fold"] = folds

        for fold_idx in range(5):
            fold_df = train_df[train_df["fold"] == fold_idx]
            # Each fold should have some of each class (stratified)
            fold_classes = set(fold_df["label"].unique())
            assert len(fold_classes) >= 2, f"Fold {fold_idx} missing classes: {fold_classes}"

    def test_binary_string_classification(self, tmp_path):
        """Binary classification with string labels should use stratified CV."""
        train_data = {
            "id": range(100),
            "text": [f"text_{i}" for i in range(100)],
            # Binary: "spam" vs "ham"
            "label": ["spam"] * 30 + ["ham"] * 70,
        }
        train_df = pd.DataFrame(train_data)
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        test_df = train_df.drop(columns=["label"])
        test_path = tmp_path / "test.csv"
        test_df.to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="label",
            output_dir=tmp_path,
        )

        # Should be classification with 2 classes
        assert result["metadata"]["is_classification"] is True
        assert result["metadata"]["n_classes"] == 2
