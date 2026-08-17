"""Tests for seq2seq canonical data contract support."""


import numpy as np
import pandas as pd
import pytest

from kaggle_agents.utils.data_contract import (
    SEQ2SEQ_GROUP_CANDIDATES,
    SEQ2SEQ_TASK_INDICATORS,
    _detect_seq2seq_group_column,
    infer_seq2seq_columns,
    load_canonical_data,
    prepare_canonical_data,
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

    def test_seq2seq_group_candidates_are_generic_identifier_conventions(self):
        """Group detection must not encode a task-specific column name."""
        assert "_id" in SEQ2SEQ_GROUP_CANDIDATES
        assert "sentence_id" not in SEQ2SEQ_GROUP_CANDIDATES

    def test_text_normalization_indicator_has_no_fixed_schema(self):
        """Task-family compatibility metadata must not prescribe column names."""
        assert "text_normalization" in SEQ2SEQ_TASK_INDICATORS
        assert SEQ2SEQ_TASK_INDICATORS["text_normalization"] == {}


class TestSeq2seqColumnInference:
    def test_derives_renamed_schema_from_train_test_and_sample(self):
        train_df = pd.DataFrame(
            {
                "record_id": range(30),
                "segment_id": [i // 3 for i in range(30)],
                "kind": ["stable"] * 15 + ["variable"] * 15,
                "raw_text": [f"input-{i}" for i in range(30)],
                "normalized_text": [f"output-{i}" for i in range(30)],
            }
        )
        test_df = train_df.drop(columns=["normalized_text"])
        sample = pd.DataFrame(
            {"record_id": test_df["record_id"], "normalized_text": ""}
        )

        resolved = infer_seq2seq_columns(
            train_df,
            test_df,
            sample_submission=sample,
        )

        assert resolved["target_col"] == "normalized_text"
        assert resolved["source_col"] == "raw_text"
        assert resolved["class_col"] == "kind"
        assert resolved["seq2seq_group_col"] == "segment_id"

    def test_fails_closed_when_source_roles_are_ambiguous(self):
        train_df = pd.DataFrame(
            {
                "input_a": ["a", "b", "c"],
                "input_b": ["x", "y", "z"],
                "output": ["m", "n", "o"],
            }
        )
        test_df = train_df.drop(columns=["output"])

        with pytest.raises(ValueError, match="Ambiguous seq2seq source column"):
            infer_seq2seq_columns(train_df, test_df)


class TestPrepareCanonicalDataSeq2seq:
    """Tests for prepare_canonical_data with seq2seq tasks."""

    def test_prepares_renamed_schema_from_public_artifacts(self, tmp_path):
        train_df = pd.DataFrame(
            {
                "record_id": range(40),
                "segment_id": [i // 4 for i in range(40)],
                "kind": ["stable"] * 20 + ["variable"] * 20,
                "raw_text": [f"input-{i}" for i in range(40)],
                "normalized_text": [f"output-{i}" for i in range(40)],
            }
        )
        test_df = train_df.drop(columns=["normalized_text"])
        sample = pd.DataFrame(
            {"record_id": test_df["record_id"], "normalized_text": ""}
        )
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        sample_path = tmp_path / "sample_submission.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        sample.to_csv(sample_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="unresolved_placeholder",
            output_dir=tmp_path,
            task_type="seq2seq",
            sample_submission=sample_path,
        )

        metadata = result["metadata"]
        assert metadata["target_col"] == "normalized_text"
        assert metadata["source_col"] == "raw_text"
        assert metadata["class_col"] == "kind"
        assert metadata["group_col"] == "segment_id"
        assert metadata["id_col"] == "record_id"

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
            source_col="before",
            class_col="class",
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
        """Should expose the current canonical contract version."""
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
        assert metadata["canonical_version"] == "1.5"

    def test_canonical_version_correct_with_sampling(self, tmp_path):
        """Should keep the current version when sampling is triggered."""
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
        assert metadata["canonical_version"] == "1.5"
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


class TestExplicitTaskContract:
    """Task metadata must override misleading target cardinality."""

    def test_low_cardinality_integer_regression_remains_regression(
        self,
        tmp_path,
    ):
        train_df = pd.DataFrame(
            {
                "id": range(90),
                "feature": range(90),
                "outcome": [0, 1, 2] * 30,
            }
        )
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        train_df.drop(columns=["outcome"]).to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="outcome",
            output_dir=tmp_path,
            n_folds=3,
            task_type="tabular_regression",
        )

        assert result["metadata"]["is_classification"] is False
        assert result["metadata"]["task_type_source"] == "explicit_task_contract"

    def test_more_than_twenty_classes_remains_classification(
        self,
        tmp_path,
    ):
        train_df = pd.DataFrame(
            {
                "id": range(150),
                "feature": range(150),
                "outcome": [class_id for class_id in range(30) for _ in range(5)],
            }
        )
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        train_df.drop(columns=["outcome"]).to_csv(test_path, index=False)

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="outcome",
            output_dir=tmp_path,
            n_folds=5,
            task_type="tabular_classification",
        )

        assert result["metadata"]["is_classification"] is True
        assert result["metadata"]["n_classes"] == 30
        assert result["metadata"]["task_type_source"] == "explicit_task_contract"
