"""Target resolution when the only "test table" is the submission template.

Image competitions ship a labels CSV plus image directories and no test table,
so the canonical node passes ``sample_submission.csv`` as the test schema: it
does list the test rows in graded order, which is what row identity needs.

That substitution must not be read as evidence about column roles. The
template's prediction column is a placeholder, and counting it as a "column the
public test set supplies" removes the only target the training table has,
crashing the whole run before a single component is generated.
"""

from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.utils.data_contract import (
    _is_independent_test_schema,
    _resolve_supervised_target_contract,
)
from kaggle_agents.utils.target_inference import TargetInferenceError
from kaggle_agents.workflow.nodes.canonical_data import (
    canonical_data_preparation_node,
)


def _image_workspace(tmp_path: Path, rows: int = 40) -> dict:
    train_ids = [f"img{index:04d}" for index in range(rows)]
    test_ids = [f"t{index:04d}" for index in range(10)]
    pd.DataFrame(
        {"id": train_ids, "label": [index % 2 for index in range(rows)]}
    ).to_csv(tmp_path / "train.csv", index=False)
    # Placeholder predictions are filled, not blank: this is the case a
    # "column is entirely empty" heuristic cannot catch.
    pd.DataFrame({"id": test_ids, "label": [0] * len(test_ids)}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    for directory, names in (("train", train_ids), ("test", test_ids)):
        (tmp_path / directory).mkdir()
        for name in names:
            (tmp_path / directory / f"{name}.tif").write_bytes(b"\x00")
    return {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "domain_detected": "image_classification",
        "target_col": "label",
        "submission_contract": {
            "id_col": "id",
            "target_cols": ["label"],
            "expected_rows": len(test_ids),
            "format_type": "label",
        },
        "data_files": {
            "data_type": "image",
            "train": str(tmp_path / "train"),
            "train_csv": str(tmp_path / "train.csv"),
            "test": str(tmp_path / "test"),
            "sample_submission": str(tmp_path / "sample_submission.csv"),
        },
        "timeout_per_component": 2800,
    }


def _sparse_multiclass_image_workspace(tmp_path: Path, rows: int = 42) -> dict:
    train_ids = [f"img{index:04d}" for index in range(rows)]
    test_ids = [f"t{index:04d}" for index in range(10)]
    class_order = ["zebra", "ant", "moose"]
    pd.DataFrame(
        {
            "id": train_ids,
            "breed": ["ant", "moose", "zebra"] * (rows // 3),
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": test_ids,
            **{label: [0.0] * len(test_ids) for label in class_order},
        }
    ).to_csv(tmp_path / "sample_submission.csv", index=False)
    for directory, names in (("train", train_ids), ("test", test_ids)):
        (tmp_path / directory).mkdir()
        for name in names:
            (tmp_path / directory / f"{name}.jpg").write_bytes(b"\x00")
    return {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "domain_detected": "image_classification",
        # This is the upstream submission-role interpretation seen in the
        # failing run: the first probability column is not the training target.
        "target_col": class_order[0],
        "submission_contract": {
            "id_col": "id",
            "target_cols": class_order,
            "class_order": class_order,
            "expected_rows": len(test_ids),
            "format_type": "wide",
        },
        "data_files": {
            "data_type": "image",
            "train": str(tmp_path / "train"),
            "train_csv": str(tmp_path / "train.csv"),
            "test": str(tmp_path / "test"),
            "sample_submission": str(tmp_path / "sample_submission.csv"),
        },
        "timeout_per_component": 2800,
    }


class TestIndependentTestSchemaDetection:
    def test_the_submission_template_is_not_test_evidence(
        self, tmp_path: Path
    ) -> None:
        template = tmp_path / "sample_submission.csv"
        template.write_text("id,label\na,0\n", encoding="utf-8")

        assert not _is_independent_test_schema(
            template, tmp_path / "train.csv", template
        )

    def test_the_train_table_is_not_test_evidence(self, tmp_path: Path) -> None:
        train = tmp_path / "train.csv"
        train.write_text("id,label\na,0\n", encoding="utf-8")

        assert not _is_independent_test_schema(
            train, train, tmp_path / "sample_submission.csv"
        )

    def test_a_real_test_table_still_counts(self, tmp_path: Path) -> None:
        test = tmp_path / "test.csv"
        test.write_text("id,feature\na,1\n", encoding="utf-8")

        assert _is_independent_test_schema(
            test, tmp_path / "train.csv", tmp_path / "sample_submission.csv"
        )

    def test_a_dataframe_template_is_tolerated(self, tmp_path: Path) -> None:
        test = tmp_path / "test.csv"
        test.write_text("id,feature\na,1\n", encoding="utf-8")

        assert _is_independent_test_schema(
            test, tmp_path / "train.csv", pd.DataFrame({"id": ["a"]})
        )


class TestTargetResolutionWithTemplateAsTestSchema:
    def test_sparse_multiclass_label_values_resolve_wide_submission(
        self, tmp_path: Path
    ) -> None:
        train_df = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e", "f"],
                "breed": ["ant", "moose", "zebra"] * 2,
            }
        )
        template = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {
                "id": ["test-a"],
                "zebra": [0.0],
                "ant": [0.0],
                "moose": [0.0],
            }
        ).to_csv(template, index=False)

        targets, target_type, _ = _resolve_supervised_target_contract(
            train_df,
            template,
            train_path=tmp_path / "train.csv",
            target_col="zebra",
            target_cols=["zebra", "ant", "moose"],
            target_type=None,
            task_type="image_classification",
            sample_submission=template,
            column_contract=None,
        )

        assert targets == ["breed"]
        assert target_type == "single"

    @pytest.mark.parametrize(
        "train_df",
        [
            pd.DataFrame(
                {
                    "id": ["a", "b", "c"],
                    "breed": ["ant", "moose", "unknown"],
                }
            ),
            pd.DataFrame(
                {
                    "id": ["a", "b", "c"],
                    "breed": ["ant", "moose", "zebra"],
                    "duplicate_labels": ["zebra", "ant", "moose"],
                }
            ),
        ],
        ids=["class-set-mismatch", "ambiguous-candidates"],
    )
    def test_sparse_multiclass_resolution_stays_fail_closed(
        self, tmp_path: Path, train_df: pd.DataFrame
    ) -> None:
        template = tmp_path / "sample_submission.csv"
        template.write_text(
            "id,zebra,ant,moose\ntest-a,0,0,0\n",
            encoding="utf-8",
        )

        with pytest.raises(TargetInferenceError):
            _resolve_supervised_target_contract(
                train_df,
                template,
                train_path=tmp_path / "train.csv",
                target_col="zebra",
                target_cols=["zebra", "ant", "moose"],
                target_type=None,
                task_type="image_classification",
                sample_submission=template,
                column_contract=None,
            )

    def test_declared_target_survives_a_filled_placeholder_column(
        self, tmp_path: Path
    ) -> None:
        train_df = pd.DataFrame({"id": ["a", "b"], "label": [0, 1]})
        template = tmp_path / "sample_submission.csv"
        template.write_text("id,label\na,0\nb,0\n", encoding="utf-8")
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        targets, target_type, _ = _resolve_supervised_target_contract(
            train_df,
            template,
            train_path=train_path,
            target_col="label",
            target_cols=["label"],
            target_type=None,
            task_type="image_classification",
            sample_submission=template,
            column_contract=None,
        )

        assert targets == ["label"]
        assert target_type == "single"

    def test_a_supplied_test_column_is_still_rejected_as_a_target(
        self, tmp_path: Path
    ) -> None:
        """The original protection must stay: real test inputs are not targets."""
        train_df = pd.DataFrame(
            {"id": ["a", "b"], "feature": [1, 2], "label": [0, 1]}
        )
        test = tmp_path / "test.csv"
        test.write_text("id,feature\na,1\nb,2\n", encoding="utf-8")
        template = tmp_path / "sample_submission.csv"
        template.write_text("id,label\na,0\nb,0\n", encoding="utf-8")

        targets, _, _ = _resolve_supervised_target_contract(
            train_df,
            test,
            train_path=tmp_path / "train.csv",
            # An upstream contract that resolved "feature" positionally.
            target_col="feature",
            target_cols=["feature"],
            target_type=None,
            task_type="tabular_classification",
            sample_submission=template,
            column_contract=None,
        )

        assert targets == ["label"]

    def test_unresolvable_targets_report_the_missing_test_table(
        self, tmp_path: Path
    ) -> None:
        train_df = pd.DataFrame({"id": ["a", "b"], "other": [0, 1]})
        template = tmp_path / "sample_submission.csv"
        template.write_text("id,label\na,0\nb,0\n", encoding="utf-8")

        with pytest.raises(TargetInferenceError, match="no independent public"):
            _resolve_supervised_target_contract(
                train_df,
                template,
                train_path=tmp_path / "train.csv",
                target_col=None,
                target_cols=None,
                target_type=None,
                task_type="image_classification",
                sample_submission=template,
                column_contract=None,
            )


class TestCanonicalNodeForImageLabelsCsv:
    def test_sparse_multiclass_contract_preserves_submission_class_order(
        self, tmp_path: Path
    ) -> None:
        state = _sparse_multiclass_image_workspace(tmp_path)

        result = canonical_data_preparation_node(state)

        assert result["canonical_data_prepared"] is True
        assert result["target_col"] == "breed"
        assert result["target_cols"] == ["breed"]
        assert result["target_type"] == "single"
        assert result["canonical_metadata"]["class_order"] == [
            "zebra",
            "ant",
            "moose",
        ]

    def test_image_competition_with_labels_csv_prepares_canonical_data(
        self, tmp_path: Path
    ) -> None:
        state = _image_workspace(tmp_path)

        result = canonical_data_preparation_node(state)

        assert result["canonical_data_prepared"] is True
        assert result["target_cols"] == ["label"]
        assert result["expected_train_rows"] == 40
        canonical = tmp_path / "canonical"
        assert (canonical / "y.npy").is_file()
        assert (canonical / "folds.npy").is_file()
        assert (canonical / "train_ids.npy").is_file()

    def test_the_graded_test_rows_stay_in_template_order(
        self, tmp_path: Path
    ) -> None:
        import numpy as np

        state = _image_workspace(tmp_path)

        canonical_data_preparation_node(state)

        test_ids = np.load(
            tmp_path / "canonical" / "test_ids.npy", allow_pickle=True
        )
        template = pd.read_csv(tmp_path / "sample_submission.csv", dtype=str)
        assert [str(value) for value in test_ids] == template["id"].tolist()
