"""Components must be able to name their rows.

A competition whose public data has no identifier column gets a synthetic one
(`_row_id`) that exists only inside the canonical artifacts. The injected header
still advertised it as ``ID_COL``, so generated code indexed by a column no CSV
contains and died with KeyError before training anything. The same gap left test
rows unnamed: components invented their own test IDs, reached for a repeated
date or a placeholder target, and had their artifacts rejected as duplicates
however good the model was.

These tests pin the canonical row identity: an alignment helper that works with
or without a real ID column, and canonical test IDs that are always unique.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.utils.data_contract import (
    _resolve_canonical_test_ids,
    prepare_canonical_data,
)

HEADER_SOURCE = Path("kaggle_agents/agents/developer/code_generator.py").read_text()


def _extract_header_function(name: str) -> str:
    """Pull a helper out of the injected header template, as it ships.

    The header is an f-string, so its literal braces are doubled; undo that so
    the extracted source is the code components actually receive.
    """
    match = re.search(
        rf"^def {name}\(.*?(?=^\S|\Z)",
        HEADER_SOURCE,
        re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"{name} is not part of the injected header")
    return match.group(0).replace("{{", "{").replace("}}", "}")


def _align_namespace(train_ids, *, id_col: str, synthetic: bool) -> dict:
    namespace = {
        "np": np,
        "pd": pd,
        "ID_COL": id_col,
        "ID_IS_SYNTHETIC": synthetic,
        "CANONICAL_TRAIN_IDS": np.asarray([str(v) for v in train_ids]),
    }
    exec(  # noqa: S102 - executing our own shipped header text is the point
        compile(_extract_header_function("align_train_to_canonical"), "<hdr>", "exec"),
        namespace,
    )
    return namespace


class TestAlignTrainToCanonical:
    """The helper that replaces df.set_index(ID_COL)."""

    def test_synthetic_ids_select_rows_by_position(self):
        frame = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
        namespace = _align_namespace(["0", "2", "3"], id_col="_row_id", synthetic=True)

        aligned = namespace["align_train_to_canonical"](frame)

        assert aligned["text"].tolist() == ["a", "c", "d"]
        assert aligned["label"].tolist() == [0, 0, 1]

    def test_real_id_column_is_reordered_to_canonical(self):
        frame = pd.DataFrame({"id": [7, 3, 9], "label": [1, 2, 3]})
        namespace = _align_namespace(["3", "9", "7"], id_col="id", synthetic=False)

        aligned = namespace["align_train_to_canonical"](frame)

        assert aligned["id"].tolist() == ["3", "9", "7"]
        assert aligned["label"].tolist() == [2, 3, 1]

    def test_missing_canonical_rows_are_reported(self):
        frame = pd.DataFrame({"id": [1, 2], "label": [0, 1]})
        namespace = _align_namespace(["1", "2", "3"], id_col="id", synthetic=False)

        with pytest.raises(ValueError, match="missing 1 canonical rows"):
            namespace["align_train_to_canonical"](frame)

    def test_duplicate_ids_are_rejected(self):
        frame = pd.DataFrame({"id": [1, 1], "label": [0, 1]})
        namespace = _align_namespace(["1"], id_col="id", synthetic=False)

        with pytest.raises(ValueError, match="not unique"):
            namespace["align_train_to_canonical"](frame)

    def test_truncated_table_is_reported_not_silently_wrong(self):
        frame = pd.DataFrame({"text": ["a", "b"]})
        namespace = _align_namespace(["0", "5"], id_col="_row_id", synthetic=True)

        with pytest.raises(ValueError, match="beyond the 2 rows"):
            namespace["align_train_to_canonical"](frame)

    def test_absent_id_without_synthetic_naming_is_an_error(self):
        # Never guess row order when the contract says IDs are semantic.
        frame = pd.DataFrame({"text": ["a", "b"]})
        namespace = _align_namespace(["x", "y"], id_col="id", synthetic=False)

        with pytest.raises(ValueError, match="not positional"):
            namespace["align_train_to_canonical"](frame)


class TestCanonicalTestIds:
    """Every public test row gets exactly one name."""

    def test_declared_id_column_is_used(self, tmp_path):
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"id": [10, 11], "f": [0.1, 0.2]}).to_csv(test_csv, index=False)

        ids, positional = _resolve_canonical_test_ids(test_csv, "id")

        assert ids.tolist() == ["10", "11"]
        assert positional is False

    def test_conventional_id_name_is_found_without_a_declaration(self, tmp_path):
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"f": [0.1, 0.2], "Id": [5, 6]}).to_csv(test_csv, index=False)

        ids, positional = _resolve_canonical_test_ids(test_csv, None)

        assert ids.tolist() == ["5", "6"]
        assert positional is False

    def test_keyless_test_table_falls_back_to_row_position(self, tmp_path):
        test_csv = tmp_path / "test.csv"
        pd.DataFrame(
            {"Date": ["2012", "2012", "2012"], "Comment": ["a", "b", "c"]}
        ).to_csv(test_csv, index=False)

        ids, positional = _resolve_canonical_test_ids(test_csv, "_row_id")

        assert ids.tolist() == ["0", "1", "2"]
        assert positional is True

    def test_free_text_is_not_treated_as_an_identifier(self, tmp_path):
        # Incidentally unique text is not a key: two identical comments would
        # silently collapse two test rows into one.
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"Comment": ["unique a", "unique b"]}).to_csv(
            test_csv, index=False
        )

        ids, positional = _resolve_canonical_test_ids(test_csv, None)

        assert ids.tolist() == ["0", "1"]
        assert positional is True

    def test_duplicated_declared_id_falls_back_to_position(self, tmp_path):
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 1], "f": [0.1, 0.2]}).to_csv(test_csv, index=False)

        ids, positional = _resolve_canonical_test_ids(test_csv, "id")

        assert ids.tolist() == ["0", "1"]
        assert positional is True

    def test_unreadable_test_path_is_not_an_error(self, tmp_path):
        assert _resolve_canonical_test_ids(tmp_path / "absent.csv", "id") == (None, False)
        assert _resolve_canonical_test_ids(None, "id") == (None, False)


class TestCanonicalPreparationEmitsRowIdentity:
    """The artifacts components actually read."""

    @staticmethod
    def _prepare(tmp_path):
        rows = 40
        pd.DataFrame(
            {
                "Label": [index % 2 for index in range(rows)],
                "Context": ["2012" for _ in range(rows)],
                "Body": [f"document {index}" for index in range(rows)],
            }
        ).to_csv(tmp_path / "train.csv", index=False)
        pd.DataFrame(
            {
                "Context": ["2013" for _ in range(10)],
                "Body": [f"held out {index}" for index in range(10)],
            }
        ).to_csv(tmp_path / "test.csv", index=False)
        return prepare_canonical_data(
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
            target_col="Label",
            target_cols=["Label"],
            output_dir=tmp_path / "work",
            task_type="text_classification",
        )

    def test_synthetic_identity_is_declared_in_metadata(self, tmp_path):
        metadata = self._prepare(tmp_path)["metadata"]

        assert metadata["id_col"] == "_row_id"
        assert metadata["id_is_synthetic"] is True
        assert metadata["n_test"] == 10

    def test_test_ids_are_written_and_unique(self, tmp_path):
        self._prepare(tmp_path)

        test_ids = np.load(
            tmp_path / "work" / "canonical" / "test_ids.npy", allow_pickle=False
        )

        assert len(test_ids) == 10
        assert len(set(test_ids.tolist())) == 10

    def test_synthetic_id_is_materialized_into_the_staged_tables(self, tmp_path):
        # A name in the contract that no file carries is what components kept
        # crashing on; after preparation it must be loadable from the CSV.
        self._prepare(tmp_path)

        train = pd.read_csv(tmp_path / "train.csv")
        test = pd.read_csv(tmp_path / "test.csv")

        assert train["_row_id"].astype(str).tolist() == [str(i) for i in range(40)]
        assert test["_row_id"].astype(str).tolist() == [str(i) for i in range(10)]
        assert train["Label"].notna().all()

    def test_materialized_ids_agree_with_the_canonical_contract(self, tmp_path):
        self._prepare(tmp_path)

        train_ids = np.load(
            tmp_path / "work" / "canonical" / "train_ids.npy", allow_pickle=False
        )
        test_ids = np.load(
            tmp_path / "work" / "canonical" / "test_ids.npy", allow_pickle=False
        )
        train = pd.read_csv(tmp_path / "train.csv")
        test = pd.read_csv(tmp_path / "test.csv")

        assert set(train_ids.tolist()) <= set(train["_row_id"].astype(str))
        assert test_ids.tolist() == test["_row_id"].astype(str).tolist()

    def test_preparation_is_idempotent(self, tmp_path):
        self._prepare(tmp_path)
        columns_after_first = pd.read_csv(tmp_path / "train.csv").columns.tolist()

        second = prepare_canonical_data(
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
            target_col="Label",
            target_cols=["Label"],
            output_dir=tmp_path / "work2",
            task_type="text_classification",
        )

        assert pd.read_csv(tmp_path / "train.csv").columns.tolist() == columns_after_first
        # The column is real now, so it is no longer reported as synthetic.
        assert second["metadata"]["id_col"] == "_row_id"
        assert second["metadata"]["id_is_synthetic"] is False


class TestProblemTypeDetection:
    """Template introspection must survive echoed text columns."""

    @staticmethod
    def _detector():
        from kaggle_agents.tools.code_executor.submission import (
            SubmissionValidationMixin,
        )

        class _Detector(SubmissionValidationMixin):
            pass

        return _Detector()

    def test_target_first_template_no_longer_raises(self, tmp_path):
        # Summing the positional slice here means adding text to integers,
        # which aborted the whole run at the first submission check.
        sample = tmp_path / "sample_submission_null.csv"
        pd.DataFrame(
            {
                "Insult": [0, 0],
                "Date": ["20120618192155Z", "20120618192156Z"],
                "Comment": ["a", "b"],
            }
        ).to_csv(sample, index=False)

        assert self._detector()._detect_problem_type(sample) == "binary"

    def test_conventional_binary_template_is_unchanged(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_csv(sample, index=False)

        assert self._detector()._detect_problem_type(sample) == "binary"

    def test_wide_probability_template_is_unchanged(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {"id": [1, 2], "a": [0.5, 0.25], "b": [0.5, 0.75]}
        ).to_csv(sample, index=False)

        assert self._detector()._detect_problem_type(sample) == "multiclass"

    def test_wide_indicator_template_is_unchanged(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": [1, 2], "a": [1, 1], "b": [1, 1]}).to_csv(
            sample, index=False
        )

        assert self._detector()._detect_problem_type(sample) == "multilabel"


class TestSyntheticIdContractPropagation:
    """The robustness gate must not demand a column that cannot exist."""

    def test_contract_carries_the_synthetic_flag(self):
        from kaggle_agents.core.state.contracts import CanonicalDataContract

        contract = CanonicalDataContract(
            canonical_dir="d",
            train_ids_path="t",
            y_path="y",
            folds_path="f",
            feature_cols_path="fc",
            metadata_path="m",
            n_train=10,
            n_test=5,
            n_folds=5,
            id_col="_row_id",
            target_col="Label",
            is_classification=True,
            folds_hash="a",
            y_hash="b",
            train_ids_hash="c",
            train_schema_hash="d",
            id_is_synthetic=True,
        )

        assert contract.to_dict()["id_is_synthetic"] is True

    def test_flag_defaults_to_false_for_older_checkpoints(self):
        from kaggle_agents.core.state.contracts import CanonicalDataContract

        contract = CanonicalDataContract(
            canonical_dir="d",
            train_ids_path="t",
            y_path="y",
            folds_path="f",
            feature_cols_path="fc",
            metadata_path="m",
            n_train=10,
            n_test=5,
            n_folds=5,
            id_col="id",
            target_col="Label",
            is_classification=True,
            folds_hash="a",
            y_hash="b",
            train_ids_hash="c",
            train_schema_hash="d",
        )

        assert contract.id_is_synthetic is False


class TestSubmissionFormatValidation:
    """The check that aborted the run right after the first model succeeded."""

    @staticmethod
    def _validator():
        from kaggle_agents.tools.code_executor.submission import (
            SubmissionValidationMixin,
        )

        class _Validator(SubmissionValidationMixin):
            pass

        return _Validator()

    @staticmethod
    def _template(tmp_path):
        sample = tmp_path / "sample_submission_null.csv"
        pd.DataFrame(
            {
                "Insult": [0, 0, 0],
                "Date": ["20120618192155Z", "20120618192156Z", "20120618192157Z"],
                "Comment": ["a", "b", "c"],
            }
        ).to_csv(sample, index=False)
        return sample

    def test_target_first_submission_is_accepted(self, tmp_path):
        sample = self._template(tmp_path)
        submission = tmp_path / "submission.csv"
        filled = pd.read_csv(sample)
        filled["Insult"] = [0.1, 0.8, 0.4]
        filled.to_csv(submission, index=False)

        is_valid, message = self._validator().validate_submission_format(
            submission, sample, component_type="model", target_cols=["Insult"]
        )

        assert is_valid, message

    def test_echoed_columns_must_be_returned_unchanged(self, tmp_path):
        sample = self._template(tmp_path)
        submission = tmp_path / "submission.csv"
        filled = pd.read_csv(sample)
        filled["Insult"] = [0.1, 0.8, 0.4]
        filled["Date"] = filled["Date"][::-1].to_numpy()
        filled.to_csv(submission, index=False)

        is_valid, message = self._validator().validate_submission_format(
            submission, sample, component_type="model", target_cols=["Insult"]
        )

        assert not is_valid
        assert "WRONG ORDER" in message

    def test_overwritten_echo_column_names_the_real_mistake(self, tmp_path):
        # Predictions written into an input column: the message must point at
        # the graded column, not read as an ID alignment bug.
        sample = self._template(tmp_path)
        submission = tmp_path / "submission.csv"
        filled = pd.read_csv(sample)
        filled[filled.columns[1]] = [0.1, 0.8, 0.4]
        filled.to_csv(submission, index=False)

        is_valid, message = self._validator().validate_submission_format(
            submission, sample, component_type="model", target_cols=["Insult"]
        )

        assert not is_valid
        assert "'Date'" in message
        assert "['Insult']" in message
        assert "write_submission" in message

    def test_nan_predictions_are_still_rejected(self, tmp_path):
        sample = self._template(tmp_path)
        submission = tmp_path / "submission.csv"
        filled = pd.read_csv(sample)
        filled["Insult"] = [0.1, np.nan, 0.4]
        filled.to_csv(submission, index=False)

        is_valid, message = self._validator().validate_submission_format(
            submission, sample, component_type="model", target_cols=["Insult"]
        )

        assert not is_valid
        assert "NaN" in message

    def test_conventional_submission_is_unaffected(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        submission = tmp_path / "submission.csv"
        pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_csv(sample, index=False)
        pd.DataFrame({"id": [1, 2], "target": [0.3, 0.7]}).to_csv(
            submission, index=False
        )

        is_valid, message = self._validator().validate_submission_format(
            submission, sample, component_type="model", target_cols=["target"]
        )

        assert is_valid, message

    def test_wrong_id_order_is_still_caught_conventionally(self, tmp_path):
        sample = tmp_path / "sample_submission.csv"
        submission = tmp_path / "submission.csv"
        pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_csv(sample, index=False)
        pd.DataFrame({"id": [2, 1], "target": [0.3, 0.7]}).to_csv(
            submission, index=False
        )

        is_valid, message = self._validator().validate_submission_format(
            submission, sample, component_type="model", target_cols=["target"]
        )

        assert not is_valid
        assert "WRONG ORDER" in message


class TestWriteSubmissionHelper:
    """The last place a component had to guess column semantics."""

    @staticmethod
    def _namespace(tmp_path, target_cols):
        from kaggle_agents.agents.developer.code_generator import _SUBMISSION_HELPER

        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {
                "Insult": [0, 0, 0],
                "Date": ["20120618192155Z", "20120618192156Z", "20120618192157Z"],
                "Comment": ["a", "b", "c"],
            }
        ).to_csv(sample, index=False)
        namespace = {
            "SAMPLE_SUBMISSION_PATH": sample,
            "SUBMISSION_PATH": tmp_path / "submission.csv",
            "SUBMISSION_TARGET_COLS": target_cols,
        }
        exec(  # noqa: S102 - executing our own shipped header text is the point
            compile(_SUBMISSION_HELPER, "<helper>", "exec"), namespace
        )
        return namespace

    def test_predictions_land_in_the_graded_column(self, tmp_path):
        namespace = self._namespace(tmp_path, ["Insult"])

        namespace["write_submission"](np.array([0.1, 0.8, 0.4]))

        written = pd.read_csv(tmp_path / "submission.csv")
        assert written["Insult"].tolist() == [0.1, 0.8, 0.4]

    def test_echoed_columns_are_returned_untouched(self, tmp_path):
        namespace = self._namespace(tmp_path, ["Insult"])

        namespace["write_submission"](np.array([0.1, 0.8, 0.4]))

        written = pd.read_csv(tmp_path / "submission.csv")
        sample = pd.read_csv(namespace["SAMPLE_SUBMISSION_PATH"])
        assert written.columns.tolist() == sample.columns.tolist()
        assert written["Date"].tolist() == sample["Date"].tolist()
        assert written["Comment"].tolist() == sample["Comment"].tolist()

    def test_echoed_ids_with_leading_zeros_are_preserved(self, tmp_path):
        from kaggle_agents.agents.developer.code_generator import _SUBMISSION_HELPER

        sample = tmp_path / "sample_submission.csv"
        sample.write_text("target,id\n0,001\n0,002\n", encoding="utf-8")
        namespace = {
            "SAMPLE_SUBMISSION_PATH": sample,
            "SUBMISSION_PATH": tmp_path / "submission.csv",
            "SUBMISSION_TARGET_COLS": ["target"],
        }
        exec(compile(_SUBMISSION_HELPER, "<helper>", "exec"), namespace)  # noqa: S102

        namespace["write_submission"](np.array([0.3, 0.7]))

        written = pd.read_csv(tmp_path / "submission.csv", dtype=str)
        assert written["id"].tolist() == ["001", "002"]

    def test_echoed_na_like_ids_are_preserved_literally(self, tmp_path):
        from kaggle_agents.agents.developer.code_generator import _SUBMISSION_HELPER

        sample = tmp_path / "sample_submission.csv"
        sample.write_text("target,id\n0,NA\n0,NULL\n0,N/A\n", encoding="utf-8")
        namespace = {
            "SAMPLE_SUBMISSION_PATH": sample,
            "SUBMISSION_PATH": tmp_path / "submission.csv",
            "SUBMISSION_TARGET_COLS": ["target"],
        }
        exec(compile(_SUBMISSION_HELPER, "<helper>", "exec"), namespace)  # noqa: S102

        namespace["write_submission"](np.array([0.1, 0.2, 0.3]))

        written = pd.read_csv(
            tmp_path / "submission.csv",
            dtype=str,
            keep_default_na=False,
        )
        assert written["id"].tolist() == ["NA", "NULL", "N/A"]

    def test_test_ids_reorder_predictions_to_template_order(self, tmp_path):
        namespace = self._namespace(tmp_path, ["Insult"])

        namespace["write_submission"](
            np.array([0.4, 0.1, 0.8]),
            test_ids=["20120618192157Z", "20120618192155Z", "20120618192156Z"],
        )

        written = pd.read_csv(tmp_path / "submission.csv")
        assert written["Insult"].tolist() == [0.1, 0.8, 0.4]

    def test_row_count_mismatch_is_rejected(self, tmp_path):
        namespace = self._namespace(tmp_path, ["Insult"])

        with pytest.raises(ValueError, match="2 rows but the template has 3"):
            namespace["write_submission"](np.array([0.1, 0.8]))

    def test_column_count_mismatch_is_rejected(self, tmp_path):
        namespace = self._namespace(tmp_path, ["Insult"])

        with pytest.raises(ValueError, match="expects 1 prediction column"):
            namespace["write_submission"](np.zeros((3, 2)))

    def test_without_a_contract_it_refuses_to_guess_prediction_columns(
        self, tmp_path
    ):
        from kaggle_agents.agents.developer.code_generator import _SUBMISSION_HELPER

        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": [1, 2], "target": [0, 0]}).to_csv(sample, index=False)
        namespace = {
            "SAMPLE_SUBMISSION_PATH": sample,
            "SUBMISSION_PATH": tmp_path / "submission.csv",
            "SUBMISSION_TARGET_COLS": [],
        }
        exec(compile(_SUBMISSION_HELPER, "<helper>", "exec"), namespace)  # noqa: S102

        with pytest.raises(ValueError, match="could not be resolved"):
            namespace["write_submission"](np.array([0.3, 0.7]))
        assert not (tmp_path / "submission.csv").exists()


class TestHandwrittenSubmissionCheck:
    """The contract check that keeps the guess from reaching training."""

    @staticmethod
    def _check(code):
        from kaggle_agents.agents.developer.agent import handwritten_submission_write

        return handwritten_submission_write(code)

    def test_positional_assignment_is_reported(self):
        code = (
            "sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)\n"
            "target_col = sample_sub.columns[1]\n"
            "sample_sub[target_col] = test_preds\n"
            "sample_sub.to_csv(SUBMISSION_PATH, index=False)\n"
        )

        assert self._check(code) == "SUBMISSION_PATH"

    def test_literal_submission_path_is_reported(self):
        code = "sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)\n"

        assert self._check(code) is not None

    def test_calling_the_helper_satisfies_the_contract(self):
        code = "write_submission(test_preds)\n"

        assert self._check(code) is None

    def test_model_without_an_external_helper_call_is_reported(self):
        from kaggle_agents.agents.developer.code_contracts import (
            missing_submission_helper_call,
        )

        assert missing_submission_helper_call(
            "def write_submission(test_preds):\n    pass\n"
            "train_model()\n"
        ) is True

    def test_helper_call_does_not_excuse_a_second_manual_submission_write(self):
        code = (
            "write_submission(test_preds)\n"
            "submission.to_csv(SUBMISSION_PATH, index=False)\n"
        )

        assert self._check(code) == "SUBMISSION_PATH"

    def test_helper_definition_alone_does_not_satisfy_it(self):
        # The helper body is injected into every model script, so its own
        # to_csv proves nothing about the generated program.
        code = (
            "def write_submission(test_preds, test_ids=None):\n"
            "    _sample.to_csv(SUBMISSION_PATH, index=False)\n"
            "\n"
            "sub.to_csv(SUBMISSION_PATH, index=False)\n"
        )

        assert self._check(code) == "SUBMISSION_PATH"

    def test_writing_no_submission_is_not_a_finding(self):
        code = "np.save(MODELS_DIR / 'oof_x.npy', oof)\n"

        assert self._check(code) is None

    def test_unrelated_csv_writes_are_ignored(self):
        code = "features.to_csv(OUTPUT_DIR / 'train_engineered.csv', index=False)\n"

        assert self._check(code) is None

    def test_unparseable_code_is_not_a_finding(self):
        assert self._check("def broken(:\n") is None

    def test_import_that_shadows_an_injected_helper_is_rejected(self):
        from kaggle_agents.agents.developer.code_contracts import (
            untrusted_contract_helper_import,
        )

        assert untrusted_contract_helper_import(
            "from submission_utils import write_submission, save_component_artifacts\n"
            "write_submission(test_preds)\n"
            "save_component_artifacts(oof, test_preds)\n"
        ) == "from submission_utils import write_submission, save_component_artifacts"

    def test_assignment_that_shadows_an_injected_helper_is_rejected(self):
        from kaggle_agents.agents.developer.code_contracts import (
            untrusted_contract_helper_import,
        )

        code = (
            "def write_submission(test_preds):\n    pass\n"
            "def save_component_artifacts(oof, test_preds):\n    pass\n"
            "write_submission = arbitrary_writer\n"
            "write_submission(test_preds)\n"
        )

        assert untrusted_contract_helper_import(code) == (
            "write_submission = arbitrary_writer"
        )

    def test_second_helper_definition_is_rejected(self):
        from kaggle_agents.agents.developer.code_contracts import (
            untrusted_contract_helper_import,
        )

        code = (
            "def write_submission(test_preds):\n    pass\n"
            "def save_component_artifacts(oof, test_preds):\n    pass\n"
            "def write_submission(test_preds):\n    arbitrary_writer(test_preds)\n"
            "write_submission(test_preds)\n"
        )

        assert untrusted_contract_helper_import(code) == (
            "def write_submission(test_preds):\n"
            "    arbitrary_writer(test_preds)"
        )

    def test_globals_assignment_cannot_shadow_submission_helper(self):
        from kaggle_agents.agents.developer.code_contracts import (
            untrusted_contract_helper_import,
        )

        code = (
            "globals()['write_submission'] = arbitrary_writer\n"
            "write_submission(test_preds)\n"
        )

        assert untrusted_contract_helper_import(code) == (
            "globals()['write_submission'] = arbitrary_writer"
        )

    def test_keyword_submission_destination_is_a_handwritten_write(self):
        code = "submission.to_csv(path_or_buf=SUBMISSION_PATH, index=False)\n"

        assert self._check(code) == "SUBMISSION_PATH"

    def test_submission_helper_contract_applies_to_models_and_ensembles(self):
        from kaggle_agents.agents.developer.code_contracts import (
            requires_submission_helper,
        )

        assert requires_submission_helper("model") is True
        assert requires_submission_helper("ensemble") is True
        assert requires_submission_helper("preprocessing") is False


class TestFormatGateCountsOnlyPredictions:
    """The robustness gate that blocked grading on an accepted submission."""

    @staticmethod
    def _validate(tmp_path, submission, state):
        from types import SimpleNamespace

        from kaggle_agents.agents.robustness_agent import RobustnessAgent

        submission.to_csv(tmp_path / "submission.csv", index=False)
        agent = RobustnessAgent.__new__(RobustnessAgent)
        result = SimpleNamespace(artifacts_created=["submission.csv"])
        return RobustnessAgent._validate_format(agent, result, tmp_path, state)

    def test_blank_echoed_column_does_not_fail_the_gate(self, tmp_path):
        # The template itself ships blanks in Date; returning them unchanged is
        # correct, and there is no way to "fill" them without corrupting the
        # column the grader expects back verbatim.
        submission = pd.DataFrame(
            {
                "Insult": [0.1, 0.8, 0.4],
                "Date": ["20120618192155Z", None, "20120618192157Z"],
                "Comment": ["a", "b", "c"],
            }
        )

        result = self._validate(
            tmp_path, submission, {"submission_contract": {"target_cols": ["Insult"]}}
        )

        assert result.passed, result.issues
        assert result.score == 1.0

    def test_missing_predictions_still_fail_the_gate(self, tmp_path):
        submission = pd.DataFrame(
            {
                "Insult": [0.1, np.nan, 0.4],
                "Date": ["a", "b", "c"],
                "Comment": ["x", "y", "z"],
            }
        )

        result = self._validate(
            tmp_path, submission, {"submission_contract": {"target_cols": ["Insult"]}}
        )

        assert not result.passed
        assert any("missing predictions" in issue for issue in result.issues)

    def test_conventional_template_is_unaffected(self, tmp_path):
        submission = pd.DataFrame({"id": [1, 2], "target": [0.3, 0.7]})

        result = self._validate(tmp_path, submission, {})

        assert result.passed, result.issues

    def test_duplicate_ids_are_still_caught(self, tmp_path):
        submission = pd.DataFrame({"id": [1, 1], "target": [0.3, 0.7]})

        result = self._validate(tmp_path, submission, {})

        assert any("Duplicate IDs" in issue for issue in result.issues)

    def test_prediction_column_is_never_read_as_an_id(self, tmp_path):
        # 'Insult' holds probabilities; duplicates there are not duplicate IDs.
        submission = pd.DataFrame({"Insult": [0.5, 0.5], "id": [1, 2]})

        result = self._validate(
            tmp_path, submission, {"submission_contract": {"target_cols": ["Insult"]}}
        )

        assert not any("Duplicate IDs" in issue for issue in result.issues)


class TestOofArtifactDigest:
    """Telling new evidence apart from the previous program's evidence."""

    @staticmethod
    def _digest(tmp_path, name):
        from kaggle_agents.agents.developer.agent import _oof_artifact_digest

        return _oof_artifact_digest(tmp_path, name)

    def test_absent_artifact_has_no_digest(self, tmp_path):
        assert self._digest(tmp_path, "demo") is None

    def test_digest_changes_when_the_file_changes(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "oof_demo.npy", np.zeros(4))
        before = self._digest(tmp_path, "demo")

        np.save(models / "oof_demo.npy", np.ones(4))

        assert before is not None
        assert self._digest(tmp_path, "demo") != before

    def test_digest_is_stable_for_identical_content(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "oof_demo.npy", np.zeros(4))

        assert self._digest(tmp_path, "demo") == self._digest(tmp_path, "demo")


class TestDebugLoopEnforcesTheSubmissionContract:
    """The debug path is where the contract used to be lost."""

    def test_debug_rejects_handwritten_submissions_before_executing(self):
        import inspect

        from kaggle_agents.agents.developer import retry

        source = inspect.getsource(retry.RetryMixin)
        marker = source.index("handwritten_submission_write(debugged_code)")
        execution = source.index("self.executor.execute(\n                debugged_code")

        # The check must precede the execution it is meant to prevent.
        assert marker < execution

    def test_submission_hint_is_injected_from_the_contract_error(self):
        from kaggle_agents.agents.developer.code_contracts import (
            SUBMISSION_CONTRACT_ERROR,
        )
        from kaggle_agents.agents.developer.retry import (
            _SUBMISSION_CONTRACT_PATTERN,
        )

        assert _SUBMISSION_CONTRACT_PATTERN in SUBMISSION_CONTRACT_ERROR.lower()

    def test_debugged_body_keeps_the_original_injected_header(self):
        from kaggle_agents.agents.developer.retry import preserve_injected_header

        original = (
            "# === PATH CONSTANTS ===\n"
            "SUBMISSION_PATH = OUTPUT_DIR / 'submission.csv'\n"
            "def write_submission(preds):\n    pass\n"
            "# === END PATH CONSTANTS ===\n"
            "old_model_body()\n"
        )

        fixed = preserve_injected_header(original, "new_model_body()\n")

        assert "SUBMISSION_PATH" in fixed
        assert "def write_submission" in fixed
        assert fixed.endswith("new_model_body()\n")

    def test_debugged_body_cannot_redefine_injected_paths(self):
        from kaggle_agents.agents.developer.retry import preserve_injected_header

        original = (
            "# === PATH CONSTANTS ===\n"
            "MODELS_DIR = Path('/trusted/models')\n"
            "OUTPUT_DIR = Path('/trusted/output')\n"
            "TRAIN_PATH = Path('/trusted/train.csv')\n"
            "TEST_PATH = Path('/trusted/test.csv')\n"
            "# === END PATH CONSTANTS ===\n"
            "old_model_body()\n"
        )

        fixed = preserve_injected_header(
            original,
            "MODELS_DIR = Path('/tmp/untrusted') / 'models'\n"
            "OUTPUT_DIR = pathlib.Path('/tmp/untrusted')\n"
            "TRAIN_PATH = Path(os.getenv('TRAIN_PATH', '/tmp/train.csv'))\n"
            "globals()['TEST_PATH'] = Path('/tmp/test.csv')\n"
            "new_model_body()\n",
        )

        active_body = fixed.split("# === END PATH CONSTANTS ===", 1)[1]
        assert "\nMODELS_DIR =" not in active_body
        assert "\nOUTPUT_DIR =" not in active_body
        assert "\nTRAIN_PATH =" not in active_body
        assert "globals()['TEST_PATH'] =" not in active_body
        assert "new_model_body()" in active_body


def test_string_conversion_retry_hint_preserves_text_semantics():
    from kaggle_agents.agents.developer.retry import _maybe_add_encoding_hint

    hint = _maybe_add_encoding_hint(
        "ValueError: could not convert string to float: 'hello world'"
    )

    assert "CANONICAL_METADATA" in hint
    assert "text_feature_cols" in hint
    assert "TF-IDF" in hint
    assert "fit only on the fold training partition" in hint
    assert "unknown" in hint
    assert ".cat.codes" not in hint
