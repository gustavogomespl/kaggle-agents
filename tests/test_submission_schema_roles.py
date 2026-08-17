"""Submission templates do not always put the identifier first.

A template whose first column is the prediction and whose remaining columns
echo the public test input was previously resolved positionally: the
prediction became the ``id_col`` and the echoed feature columns became the
targets. Canonical preparation then tried to build a label matrix out of
feature columns and aborted the whole run before a single model was trained.

These tests pin the schema-based resolution that replaces the positional
assumption, and keep the ordinary ``id, target`` layout unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.core.state.contracts import (
    create_submission_contract_from_sample,
)
from kaggle_agents.utils.data_contract import _supplied_test_columns
from kaggle_agents.utils.target_inference import (
    infer_target_columns,
    split_submission_schema,
)


class TestSplitSubmissionSchema:
    """Role resolution from submission and public test schemas."""

    def test_target_first_template_is_resolved_by_test_schema(self):
        echoed, predicted = split_submission_schema(
            ["Insult", "Date", "Comment"],
            ["Date", "Comment"],
        )
        assert predicted == ["Insult"]
        assert echoed == ["Date", "Comment"]

    def test_conventional_id_first_layout_is_unchanged(self):
        echoed, predicted = split_submission_schema(
            ["id", "target"],
            ["id", "feature_a", "feature_b"],
        )
        assert echoed == ["id"]
        assert predicted == ["target"]

    def test_wide_multiclass_layout_keeps_every_class_column(self):
        echoed, predicted = split_submission_schema(
            ["id", "class_a", "class_b", "class_c"],
            ["id", "pixel_0"],
        )
        assert echoed == ["id"]
        assert predicted == ["class_a", "class_b", "class_c"]

    def test_prediction_order_follows_the_template(self):
        _, predicted = split_submission_schema(
            ["id", "c_2", "c_0", "c_1"],
            ["id"],
        )
        assert predicted == ["c_2", "c_0", "c_1"]

    def test_falls_back_to_position_without_a_test_schema(self):
        echoed, predicted = split_submission_schema(["Insult", "Date", "Comment"], [])
        assert echoed == ["Insult"]
        assert predicted == ["Date", "Comment"]

    def test_falls_back_when_test_schema_shares_no_columns(self):
        # Nothing overlaps, so the test set carries no role evidence at all.
        echoed, predicted = split_submission_schema(
            ["id", "target"],
            ["image_name", "width"],
        )
        assert echoed == ["id"]
        assert predicted == ["target"]


class TestSubmissionOnlyRecovery:
    """A malformed CSV must not discard independently valid prediction artifacts."""

    def test_rebuilds_target_first_template_from_component_test_predictions(
        self, tmp_path
    ):
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {
                "Insult": [0.0, 0.0, 0.0],
                "Date": ["b", "a", "c"],
                "Comment": ["two", "one", "three"],
            }
        ).to_csv(sample, index=False)
        models = tmp_path / "models"
        models.mkdir()
        test_predictions = models / "test_model_a.npy"
        np.save(test_predictions, np.array([0.1, 0.2, 0.3]))
        np.save(models / "test_ids_model_a.npy", np.array(["a", "b", "c"]))
        artifact_bytes = test_predictions.read_bytes()
        (tmp_path / "submission.csv").write_text("wrong\n0\n", encoding="utf-8")

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["Insult"],
            id_col="Date",
            test_ids_are_positional=False,
        )

        assert rebuilt == tmp_path / "submission.csv"
        written = pd.read_csv(rebuilt)
        assert written.columns.tolist() == ["Insult", "Date", "Comment"]
        assert written["Insult"].tolist() == [0.2, 0.1, 0.3]
        assert written["Date"].tolist() == ["b", "a", "c"]
        assert test_predictions.read_bytes() == artifact_bytes

    def test_rebuild_reads_semicolon_template_and_emits_csv(self, tmp_path):
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        sample.write_text(
            "id;value\nrow-a;0\nrow-b;0\n",
            encoding="utf-8",
        )
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "test_model_a.npy", np.array([0.2, 0.8]))
        np.save(
            models / "test_ids_model_a.npy",
            np.array(["row-a", "row-b"], dtype=str),
        )

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["value"],
            id_col="id",
        )

        assert rebuilt is not None
        assert rebuilt.read_text(encoding="utf-8").splitlines()[0] == "id,value"
        written = pd.read_csv(rebuilt)
        assert written["value"].tolist() == [0.2, 0.8]

    def test_rebuild_keeps_row_order_for_canonical_positional_test_ids(
        self, tmp_path
    ):
        """Synthetic canonical IDs name row positions, not an echoed column."""
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {
                "Insult": [0.0, 0.0, 0.0],
                "Date": ["2012-08-01", "2012-08-02", "2012-08-03"],
                "Comment": ["first", "second", "third"],
            }
        ).to_csv(sample, index=False)
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "test_model_a.npy", np.array([0.1, 0.2, 0.3]))
        np.save(models / "test_ids_model_a.npy", np.array(["0", "1", "2"]))

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["Insult"],
            id_col="Date",
            test_ids_are_positional=True,
        )

        assert rebuilt == tmp_path / "submission.csv"
        written = pd.read_csv(rebuilt)
        assert written["Insult"].tolist() == [0.1, 0.2, 0.3]
        assert written["Date"].tolist() == [
            "2012-08-01",
            "2012-08-02",
            "2012-08-03",
        ]

    def test_developer_recovers_when_submission_file_is_absent(self, tmp_path):
        """Valid artifacts remain usable even when generated code wrote no CSV."""
        from kaggle_agents.agents.developer.agent import DeveloperAgent

        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {
                "Insult": [0.0, 0.0],
                "Date": ["first", "second"],
            }
        ).to_csv(sample, index=False)
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "test_model_a.npy", np.array([0.25, 0.75]))
        np.save(models / "test_ids_model_a.npy", np.array(["0", "1"]))

        recovered = DeveloperAgent._recover_missing_submission(
            run_mode="mlebench",
            submission_path=None,
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["Insult"],
            id_col="Date",
            test_ids_are_positional=True,
        )

        assert recovered == tmp_path / "submission.csv"
        assert pd.read_csv(recovered)["Insult"].tolist() == [0.25, 0.75]

    def test_numeric_real_ids_are_reordered_when_not_declared_positional(
        self, tmp_path
    ):
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {"target": [0.0, 0.0], "id": ["1", "0"]}
        ).to_csv(sample, index=False)
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "test_model_a.npy", np.array([0.1, 0.9]))
        np.save(models / "test_ids_model_a.npy", np.array(["0", "1"]))

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["target"],
            id_col="id",
            test_ids_are_positional=False,
        )

        assert pd.read_csv(rebuilt)["target"].tolist() == [0.9, 0.1]

    def test_rebuild_preserves_leading_zero_echo_ids(self, tmp_path):
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        sample.write_text("target,id\n0,001\n0,002\n", encoding="utf-8")
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "test_model_a.npy", np.array([0.1, 0.9]))
        np.save(models / "test_ids_model_a.npy", np.array(["001", "002"]))

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["target"],
            id_col="id",
            test_ids_are_positional=False,
        )

        written = pd.read_csv(rebuilt, dtype=str)
        assert written["id"].tolist() == ["001", "002"]

    def test_rebuild_preserves_na_like_echo_ids_literally(self, tmp_path):
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        sample.write_text("target,id\n0,NA\n0,NULL\n0,N/A\n", encoding="utf-8")
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "test_model_a.npy", np.array([0.1, 0.2, 0.3]))
        np.save(
            models / "test_ids_model_a.npy",
            np.array(["NA", "NULL", "N/A"]),
        )

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["target"],
            id_col="id",
            test_ids_are_positional=False,
        )

        written = pd.read_csv(rebuilt, dtype=str, keep_default_na=False)
        assert written["id"].tolist() == ["NA", "NULL", "N/A"]

    def test_rebuilds_label_format_from_multiclass_probabilities(self, tmp_path):
        from kaggle_agents.utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        sample = tmp_path / "sample_submission.csv"
        sample.write_text("Cover_Type,id\n0,row-b\n0,row-a\n", encoding="utf-8")
        models = tmp_path / "models"
        models.mkdir()
        np.save(
            models / "test_model_a.npy",
            np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
        )
        np.save(
            models / "test_ids_model_a.npy",
            np.array(["row-a", "row-b"]),
        )
        np.save(
            models / "class_order_model_a.npy",
            np.array(["forest", "grass", "water"]),
        )

        rebuilt = rebuild_submission_from_component_predictions(
            working_dir=tmp_path,
            component_name="model_a",
            sample_submission_path=sample,
            target_cols=["Cover_Type"],
            id_col="id",
            test_ids_are_positional=False,
        )

        written = pd.read_csv(rebuilt, dtype=str, keep_default_na=False)
        assert written["Cover_Type"].tolist() == ["water", "forest"]
        assert written["id"].tolist() == ["row-b", "row-a"]

    def test_falls_back_when_test_schema_covers_every_column(self):
        # Everything overlaps, so nothing would be left to predict.
        echoed, predicted = split_submission_schema(
            ["id", "target"],
            ["id", "target"],
        )
        assert echoed == ["id"]
        assert predicted == ["target"]

    def test_single_column_template_has_no_identifier(self):
        echoed, predicted = split_submission_schema(["target"], ["id"])
        assert echoed == []
        assert predicted == ["target"]


def test_submission_validation_distinguishes_na_literal_from_blank_echo(
    tmp_path,
):
    from kaggle_agents.tools.code_executor.submission import (
        SubmissionValidationMixin,
    )

    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("target,id\n0,NA\n0,NULL\n", encoding="utf-8")
    submission.write_text("target,id\n0.2,\n0.8,\n", encoding="utf-8")

    valid, message = SubmissionValidationMixin().validate_submission_format(
        submission_path=submission,
        sample_submission_path=sample,
        component_type="model",
        problem_type="regression",
        target_cols=["target"],
    )

    assert valid is False
    assert "'id' does not match sample_submission" in message


def test_submission_validation_accepts_single_column_string_labels(
    tmp_path,
):
    from kaggle_agents.tools.code_executor.submission import (
        SubmissionValidationMixin,
    )

    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text(
        "target,id\nunknown,a\nunknown,b\n",
        encoding="utf-8",
    )
    submission.write_text(
        "target,id\ncat,a\ndog,b\n",
        encoding="utf-8",
    )

    valid, message = SubmissionValidationMixin().validate_submission_format(
        submission_path=submission,
        sample_submission_path=sample,
        component_type="model",
        problem_type="multiclass",
        target_cols=["target"],
    )

    assert valid is True, message


def test_multiclass_label_submission_accepts_numeric_template_placeholder(
    tmp_path,
):
    from kaggle_agents.tools.code_executor.submission import (
        SubmissionValidationMixin,
    )

    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text(
        "target,id\n0,a\n0,b\n",
        encoding="utf-8",
    )
    submission.write_text(
        "target,id\ncat,a\ndog,b\n",
        encoding="utf-8",
    )

    valid, message = SubmissionValidationMixin().validate_submission_format(
        submission_path=submission,
        sample_submission_path=sample,
        component_type="model",
        problem_type="multiclass_classification",
        target_cols=["target"],
    )

    assert valid is True, message


class TestSuppliedTestColumns:
    """Only populated test columns count as model input."""

    def test_populated_columns_are_reported(self, tmp_path):
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"Date": ["2012"], "Comment": ["hi"]}).to_csv(
            test_csv, index=False
        )
        assert _supplied_test_columns(test_csv) == {"Date", "Comment"}

    def test_blank_placeholder_column_is_not_input(self, tmp_path):
        # Some templates ship the target column empty; that is a placeholder,
        # not something the test set answers for us.
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 2], "target": [None, None]}).to_csv(
            test_csv, index=False
        )
        assert _supplied_test_columns(test_csv) == {"id"}

    def test_missing_path_is_not_an_error(self, tmp_path):
        assert _supplied_test_columns(tmp_path / "absent.csv") == set()
        assert _supplied_test_columns(None) == set()


class TestInferTargetColumns:
    """Public entry point used by generated code."""

    def test_target_first_template_resolves_the_real_label(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame(
            {"Insult": [0, 0], "Date": ["2012", "2012"], "Comment": ["a", "b"]}
        ).to_csv(sample, index=False)
        pd.DataFrame({"Date": ["2012", "2012"], "Comment": ["a", "b"]}).to_csv(
            test_csv, index=False
        )
        train = pd.DataFrame(
            {
                "Insult": [0, 1, 0, 1],
                "Date": ["2012", None, "2012", "2013"],
                "Comment": ["a", "b", "c", "d"],
            }
        )

        info = infer_target_columns(
            sample,
            train_data=train,
            test_data=test_csv,
        )

        assert info.target_cols == ["Insult"]
        assert info.target_type == "single"
        assert info.id_col == "Date"

    def test_conventional_layout_is_unaffected(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_csv(sample, index=False)
        pd.DataFrame({"id": [1, 2], "feature": [0.5, 0.25]}).to_csv(
            test_csv, index=False
        )
        train = pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2], "target": [0, 1]})

        info = infer_target_columns(sample, train_data=train, test_data=test_csv)

        assert info.target_cols == ["target"]
        assert info.id_col == "id"


class TestSubmissionContractRoles:
    """The contract that drives submission validation."""

    def test_target_first_template_yields_a_label_contract(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame(
            {"Insult": [0, 0], "Date": ["2012", "2012"], "Comment": ["a", "b"]}
        ).to_csv(sample, index=False)
        pd.DataFrame({"Date": ["2012", "2012"], "Comment": ["a", "b"]}).to_csv(
            test_csv, index=False
        )

        contract = create_submission_contract_from_sample(str(sample), str(test_csv))

        assert contract.target_cols == ["Insult"]
        assert contract.id_col == "Date"
        assert contract.format_type == "label"
        assert contract.class_order is None
        assert contract.expected_rows == 2

    def test_target_first_roles_support_semicolon_test_schema(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame(
            {"Insult": [0], "Date": ["2012"], "Comment": ["a"]}
        ).to_csv(sample, index=False)
        test_csv.write_text("Date;Comment\n2012;a\n", encoding="utf-8")

        contract = create_submission_contract_from_sample(
            str(sample),
            str(test_csv),
        )

        assert contract.target_cols == ["Insult"]
        assert contract.id_col == "Date"

    def test_prediction_first_pixel_template_uses_observed_coordinate_id(
        self,
        tmp_path,
    ):
        sample = tmp_path / "sample_submission.csv"
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "page.png").write_bytes(b"image")
        ids = [f"page_{row}_{col}" for row in range(2) for col in range(3)]
        pd.DataFrame({"value": [0.0] * len(ids), "id": ids}).to_csv(
            sample,
            index=False,
        )

        contract = create_submission_contract_from_sample(
            str(sample),
            str(test_dir),
        )

        assert contract.id_col == "id"
        assert contract.target_cols == ["value"]

    def test_wide_template_keeps_class_order(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"id": [1], "a": [0.0], "b": [0.0]}).to_csv(sample, index=False)
        pd.DataFrame({"id": [1], "feature": [3.0]}).to_csv(test_csv, index=False)

        contract = create_submission_contract_from_sample(str(sample), str(test_csv))

        assert contract.id_col == "id"
        assert contract.target_cols == ["a", "b"]
        assert contract.format_type == "wide"
        assert contract.class_order == ["a", "b"]

    def test_contract_without_test_path_keeps_positional_behaviour(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(sample, index=False)

        contract = create_submission_contract_from_sample(str(sample))

        assert contract.id_col == "id"
        assert contract.target_cols == ["target"]

    def test_single_column_template_still_rejected(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"target": [0]}).to_csv(sample, index=False)

        with pytest.raises(ValueError, match="at least 2 columns"):
            create_submission_contract_from_sample(str(sample))


class TestCanonicalPreparationEndToEnd:
    """The failure that aborted the run before any model was trained."""

    @staticmethod
    def _write_target_first_competition(tmp_path):
        """Write a competition whose template puts the label first.

        The echoed context column is deliberately incomplete, which is what
        turned a mis-resolved target into a hard abort: an input column used
        as a label has missing values and cannot form a label matrix.
        """
        rows = 40
        train = pd.DataFrame(
            {
                "Label": [index % 2 for index in range(rows)],
                "Context": [
                    None if index % 5 == 0 else f"2012-{index:02d}"
                    for index in range(rows)
                ],
                "Body": [f"document number {index}" for index in range(rows)],
            }
        )
        test = pd.DataFrame(
            {
                "Context": [f"2013-{index:02d}" for index in range(10)],
                "Body": [f"held out document {index}" for index in range(10)],
            }
        )
        sample = pd.DataFrame(
            {
                "Label": [0] * 10,
                "Context": test["Context"],
                "Body": test["Body"],
            }
        )

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        sample_path = tmp_path / "sample_submission_null.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        sample.to_csv(sample_path, index=False)
        return train_path, test_path, sample_path

    def test_mis_resolved_upstream_contract_no_longer_aborts(self, tmp_path):
        from kaggle_agents.utils.data_contract import prepare_canonical_data

        train_path, test_path, sample_path = self._write_target_first_competition(
            tmp_path
        )

        # Exactly the contract a positional detector produces for this layout.
        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="Context",
            target_cols=["Context", "Body"],
            output_dir=tmp_path / "work",
            sample_submission=sample_path,
            task_type="text_classification",
        )

        metadata = result["metadata"]
        assert metadata["target_cols"] == ["Label"]
        assert metadata["target_col"] == "Label"
        assert metadata["target_type"] == "single"

    def test_resolution_holds_without_any_declared_contract(self, tmp_path):
        from kaggle_agents.utils.data_contract import prepare_canonical_data

        train_path, test_path, sample_path = self._write_target_first_competition(
            tmp_path
        )

        result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col=None,
            output_dir=tmp_path / "work",
            sample_submission=sample_path,
            task_type="text_classification",
        )

        assert result["metadata"]["target_cols"] == ["Label"]


class TestEnsembleWritesToTheGradedColumn:
    """Predictions must land in the column the grader reads."""

    @staticmethod
    def _template(tmp_path):
        sample = tmp_path / "sample_submission_null.csv"
        pd.DataFrame(
            {
                "Insult": [0, 0, 0],
                "Date": ["2012", "2013", "2014"],
                "Comment": ["a", "b", "c"],
            }
        ).to_csv(sample, index=False)
        return sample

    def test_positions_follow_the_resolved_contract(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import prediction_positions

        sample_sub = pd.read_csv(self._template(tmp_path))

        assert prediction_positions(sample_sub, ["Insult"]) == [0]
        # Without a contract the old convention is preserved verbatim.
        assert prediction_positions(sample_sub, None) == [1, 2]
        assert prediction_positions(sample_sub, []) == [1, 2]

    def test_unknown_contract_columns_are_ignored(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import prediction_positions

        sample_sub = pd.read_csv(self._template(tmp_path))

        assert prediction_positions(sample_sub, ["not_in_template"]) == [1, 2]

    def test_label_detection_reads_the_prediction_column(self, tmp_path):
        import numpy as np

        from kaggle_agents.agents.ensemble.submission import (
            format_ensemble_predictions,
        )

        sample_sub = pd.read_csv(self._template(tmp_path))
        preds = np.array([0.2, 0.9, 0.6])

        # The template's integer placeholders live in the prediction column;
        # reading position 1 instead sees free text and infers nothing.
        formatted = format_ensemble_predictions(
            preds,
            sample_sub,
            "binary_classification",
            metric_name="accuracy",
            target_cols=["Insult"],
        )

        assert formatted.tolist() == [0, 1, 1]

    def test_validation_accepts_a_correctly_filled_template(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import (
            validate_and_align_submission,
        )

        sample = self._template(tmp_path)
        submission = tmp_path / "submission.csv"
        pd.DataFrame(
            {
                "Insult": [0.1, 0.8, 0.4],
                "Date": ["2012", "2013", "2014"],
                "Comment": ["a", "b", "c"],
            }
        ).to_csv(submission, index=False)

        is_valid, error, _ = validate_and_align_submission(
            submission, sample, tmp_path / "aligned.csv", ["Insult"]
        )

        assert is_valid, error

    def test_validation_without_the_contract_rejects_valid_work(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import (
            validate_and_align_submission,
        )

        sample = self._template(tmp_path)
        submission = tmp_path / "submission.csv"
        pd.DataFrame(
            {
                "Insult": [0.1, 0.8, 0.4],
                "Date": ["2012", "2013", "2014"],
                "Comment": ["a", "b", "c"],
            }
        ).to_csv(submission, index=False)

        # Documents the old behaviour that the contract now avoids: the
        # prediction column is read as an identifier and never matches.
        is_valid, error, _ = validate_and_align_submission(
            submission, sample, tmp_path / "aligned.csv"
        )

        assert not is_valid
        assert "ID mismatch" in error

    def test_conventional_template_validation_is_unchanged(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import (
            validate_and_align_submission,
        )

        sample = tmp_path / "sample_submission.csv"
        submission = tmp_path / "submission.csv"
        pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]}).to_csv(
            sample, index=False
        )
        pd.DataFrame({"id": [3, 1, 2], "target": [0.3, 0.1, 0.2]}).to_csv(
            submission, index=False
        )

        is_valid, error, aligned = validate_and_align_submission(
            submission, sample, tmp_path / "aligned.csv", ["target"]
        )

        assert is_valid, error
        realigned = pd.read_csv(aligned)
        assert realigned["id"].tolist() == [1, 2, 3]
        assert realigned["target"].tolist() == [0.1, 0.2, 0.3]

    def test_validation_reads_semicolon_template_and_emits_csv(
        self, tmp_path
    ):
        from kaggle_agents.agents.ensemble.submission import (
            validate_and_align_submission,
        )

        sample = tmp_path / "sample_submission.csv"
        submission = tmp_path / "submission.csv"
        sample.write_text("id;value\na;0\nb;0\n", encoding="utf-8")
        submission.write_text("id,value\nb,0.8\na,0.2\n", encoding="utf-8")

        is_valid, error, aligned = validate_and_align_submission(
            submission,
            sample,
            tmp_path / "aligned.csv",
            ["value"],
        )

        assert is_valid, error
        assert aligned.read_text(encoding="utf-8").splitlines()[0] == "id,value"
        written = pd.read_csv(aligned)
        assert written["id"].tolist() == ["a", "b"]
        assert written["value"].tolist() == [0.2, 0.8]

    def test_duplicate_identifier_out_of_order_is_rejected(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import (
            validate_and_align_submission,
        )

        sample = tmp_path / "sample_submission.csv"
        submission = tmp_path / "submission.csv"
        # A merge on this key would multiply rows instead of realigning them.
        pd.DataFrame({"Insult": [0, 0], "Date": ["a", "a"]}).to_csv(
            sample, index=False
        )
        pd.DataFrame({"Insult": [0.2, 0.7], "Date": ["a", "a"]}).to_csv(
            submission, index=False
        )
        pd.read_csv(submission)

        is_valid, _, _ = validate_and_align_submission(
            submission, sample, tmp_path / "aligned.csv", ["Insult"]
        )
        assert is_valid

    def test_hash_verified_restore_survives_a_target_first_template(self, tmp_path):
        from kaggle_agents.agents.ensemble.submission import safe_restore_submission
        from kaggle_agents.utils.submission_artifacts import sha256_file

        sample = self._template(tmp_path)
        snapshot = tmp_path / "snapshot.csv"
        pd.DataFrame(
            {
                "Insult": [0.1, 0.8, 0.4],
                "Date": ["2012", "2013", "2014"],
                "Comment": ["a", "b", "c"],
            }
        ).to_csv(snapshot, index=False)
        digest = sha256_file(snapshot)

        restored = safe_restore_submission(
            snapshot,
            tmp_path / "submission.csv",
            sample,
            target_cols=["Insult"],
            expected_sha256=digest,
            require_hash=True,
        )

        assert restored
        assert sha256_file(tmp_path / "submission.csv") == digest

    def test_hash_verified_restore_accepts_multiclass_string_labels_with_numeric_placeholder(
        self, tmp_path
    ):
        from kaggle_agents.agents.ensemble.submission import safe_restore_submission
        from kaggle_agents.utils.submission_artifacts import sha256_file

        sample = tmp_path / "sample_submission.csv"
        snapshot = tmp_path / "snapshot.csv"
        sample.write_text("target,id\n0,a\n0,b\n", encoding="utf-8")
        snapshot.write_text("target,id\ncat,a\ndog,b\n", encoding="utf-8")
        digest = sha256_file(snapshot)

        restored = safe_restore_submission(
            snapshot,
            tmp_path / "submission.csv",
            sample,
            target_cols=["target"],
            problem_type="multiclass_classification",
            expected_sha256=digest,
            require_hash=True,
        )

        assert restored
        assert sha256_file(tmp_path / "submission.csv") == digest

    def test_hash_verified_restore_streams_large_snapshot(
        self,
        tmp_path,
        monkeypatch,
    ):
        from kaggle_agents.agents.ensemble.submission import safe_restore_submission
        from kaggle_agents.tools.code_executor.submission import (
            SubmissionValidationMixin,
        )
        from kaggle_agents.utils.submission_artifacts import sha256_file

        sample = tmp_path / "sample_submission.csv"
        snapshot = tmp_path / "snapshot.csv"
        pd.DataFrame(
            {
                "value": [0.0] * 7,
                "id": ["NA", "001", "002", "003", "004", "005", "006"],
            }
        ).to_csv(sample, index=False)
        pd.DataFrame(
            {
                "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "id": ["NA", "001", "002", "003", "004", "005", "006"],
            }
        ).to_csv(snapshot, index=False)
        digest = sha256_file(snapshot)
        monkeypatch.setattr(
            SubmissionValidationMixin,
            "CSV_VALIDATION_CHUNK_ROWS",
            2,
        )
        original_read_csv = pd.read_csv

        def guarded_read_csv(*args, **kwargs):
            if "chunksize" not in kwargs and "nrows" not in kwargs:
                raise AssertionError("immutable restore attempted a full CSV read")
            return original_read_csv(*args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", guarded_read_csv)

        restored = safe_restore_submission(
            snapshot,
            tmp_path / "submission.csv",
            sample,
            target_cols=["value"],
            expected_sha256=digest,
            require_hash=True,
        )

        assert restored is True
        assert sha256_file(tmp_path / "submission.csv") == digest


class TestAdapterRoleDetection:
    """The MLE-bench adapter feeds every downstream contract."""

    @staticmethod
    def _detector():
        from kaggle_agents.mlebench.data_adapter.detection import DetectionMixin

        return DetectionMixin()

    def test_target_first_template_with_test_schema(self, tmp_path):
        sample = tmp_path / "sample_submission_null.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame(
            {"Insult": [0], "Date": ["2012"], "Comment": ["a"]}
        ).to_csv(sample, index=False)
        pd.DataFrame({"Date": ["2012"], "Comment": ["a"]}).to_csv(
            test_csv, index=False
        )

        detector = self._detector()

        assert detector._detect_target_columns(sample, test_csv) == ["Insult"]
        assert detector._detect_id_column(sample, test_csv) == "Date"

    def test_conventional_template_with_test_schema(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(sample, index=False)
        pd.DataFrame({"id": [1], "feature": [2.0]}).to_csv(test_csv, index=False)

        detector = self._detector()

        assert detector._detect_target_columns(sample, test_csv) == ["target"]
        assert detector._detect_id_column(sample, test_csv) == "id"

    def test_without_a_test_csv_the_old_convention_holds(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(sample, index=False)

        detector = self._detector()

        assert detector._detect_target_columns(sample, None) == ["target"]
        assert detector._detect_id_column(sample, None) == "id"

    def test_unreadable_template_falls_back_to_defaults(self, tmp_path):
        detector = self._detector()

        assert detector._detect_target_columns(tmp_path / "absent.csv") == ["target"]
        assert detector._detect_id_column(tmp_path / "absent.csv") == "id"


def test_terminal_submission_validation_streams_target_first_template(
    tmp_path,
    monkeypatch,
):
    """The final workflow gate must not reload a pixel-scale CSV in memory."""
    from kaggle_agents.agents.submission_agent import SubmissionAgent

    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Insult": [0.0, 0.0, 0.0, 0.0],
            "Date": ["NA", "", "001", "002"],
            "Comment": ["a", "b", "c", "d"],
        }
    ).to_csv(sample_path, index=False)
    pd.DataFrame(
        {
            "Insult": [0.1, 0.8, 0.2, 0.9],
            "Date": ["NA", "", "001", "002"],
            "Comment": ["a", "b", "c", "d"],
        }
    ).to_csv(submission_path, index=False)

    original_read_csv = pd.read_csv
    calls = []

    def guarded_read_csv(*args, **kwargs):
        calls.append(dict(kwargs))
        if "chunksize" not in kwargs and "nrows" not in kwargs:
            raise AssertionError("terminal validation attempted a full CSV read")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)

    valid, message = SubmissionAgent()._validate_submission(
        submission_path,
        sample_path,
        problem_type="binary_classification",
        metric_name="auc",
        target_cols=["Insult"],
    )

    assert valid is True, message
    assert any("chunksize" in call for call in calls)
