"""Regressions for the four defects that produced a graded-nothing image run.

The observed run trained a ResNet to 0.039 log_loss, wrote a valid submission,
and finished with `Valid Submission: No`. The chain was:

1. An image competition without train.csv skipped canonical prep entirely, so
   no CANONICAL_* header was injected and no trusted OOF score was possible.
2. Unscored candidates are preserved as a fallback snapshot, but the rollback
   path required a finite trusted score to restore it - so the fallback could
   never be used, and rejecting a later candidate deleted the live submission.
3. Ensemble components were told to call save_component_artifacts, which was
   injected only for models; they imported it and the shadowing guard rejected
   every attempt.
4. The failure message blamed the score threshold when a hard-failing module
   was the binding constraint.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from kaggle_agents.core.state import KaggleState
from kaggle_agents.core.state.contracts import CanonicalDataContract
from kaggle_agents.mlebench.data_adapter.detection import DetectionMixin
from kaggle_agents.utils.label_parser import infer_filename_label_table
from kaggle_agents.workflow.nodes.canonical_data import (
    canonical_data_preparation_node,
)


class TestClassPrefixFilenamesResolve:
    """A class prefixing a record counter is as structural as one following it."""

    def test_leading_stem_token_is_accepted_as_evidence(
        self, tmp_path: Path
    ) -> None:
        paths = []
        counter = 0
        for label in ("alpha", "beta"):
            for _ in range(4):
                # Record counters are globally unique, so the terminal token is
                # not a viable class partition and only the prefix remains.
                path = tmp_path / f"{label}.{counter}.jpg"
                path.write_bytes(b"\x00")
                paths.append(path)
                counter += 1

        table = infer_filename_label_table(paths)

        assert sorted(set(table["target"])) == ["alpha", "beta"]
        assert (
            "leading_delimited_stem_token"
            in table.attrs["target_inference"]["evidence"]
        )

    def test_a_unique_leading_token_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """Unique prefixes are record identifiers, not targets."""
        for index in range(6):
            (tmp_path / f"{index}.sample.jpg").write_bytes(b"\x00")

        with pytest.raises(ValueError, match="not uniquely supported"):
            infer_filename_label_table(sorted(tmp_path.iterdir()))

    def test_conflicting_ends_fail_closed(self, tmp_path: Path) -> None:
        """Two viable but different partitions must not silently pick one."""
        for left in ("a", "b"):
            for right in ("x", "y"):
                for index in range(2):
                    (tmp_path / f"{left}.{index}.{right}.jpg").write_bytes(b"\x00")

        with pytest.raises(ValueError, match="not uniquely supported"):
            infer_filename_label_table(sorted(tmp_path.iterdir()))


class TestImageFilenameCanonicalContract:
    """The artifacts must satisfy the contract the injected header loads."""

    @staticmethod
    def _image_workspace(tmp_path: Path) -> dict:
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()
        counter = 0
        for label in ("alpha", "beta"):
            for _ in range(5):
                (train_dir / f"{label}.{counter}.jpg").write_bytes(b"\x00")
                counter += 1
        test_ids = [f"t{index}" for index in range(4)]
        for name in test_ids:
            (test_dir / f"{name}.jpg").write_bytes(b"\x00")
        pd.DataFrame({"id": test_ids, "label": [0] * len(test_ids)}).to_csv(
            tmp_path / "sample_submission.csv", index=False
        )
        return {
            "working_directory": str(tmp_path),
            "run_mode": "mlebench",
            "domain_detected": "image_classification",
            "target_col": "label",
            "test_rec_ids": test_ids,
            "submission_contract": {
                "id_col": "id",
                "target_cols": ["label"],
                "expected_rows": len(test_ids),
            },
            "data_files": {
                "data_type": "image",
                "train": str(train_dir),
                "test": str(test_dir),
                "sample_submission": str(tmp_path / "sample_submission.csv"),
            },
            "timeout_per_component": 2800,
        }

    def test_canonical_data_is_prepared_without_a_labels_csv(
        self, tmp_path: Path
    ) -> None:
        state = self._image_workspace(tmp_path)

        result = canonical_data_preparation_node(state)

        assert result["canonical_data_prepared"] is True
        assert result["expected_train_rows"] == 10

    def test_the_injected_header_file_set_is_complete(
        self, tmp_path: Path
    ) -> None:
        """Missing feature_cols.json silently suppressed the whole header."""
        state = self._image_workspace(tmp_path)

        canonical_data_preparation_node(state)

        canonical = tmp_path / "canonical"
        for name in (
            "train_ids.npy",
            "y.npy",
            "folds.npy",
            "feature_cols.json",
            "metadata.json",
        ):
            assert (canonical / name).is_file(), name

    def test_metadata_carries_every_field_the_header_validates(
        self, tmp_path: Path
    ) -> None:
        import json

        state = self._image_workspace(tmp_path)

        canonical_data_preparation_node(state)

        metadata = json.loads(
            (tmp_path / "canonical" / "metadata.json").read_text(encoding="utf-8")
        )
        for field in (
            "n_folds",
            "id_col",
            "target_col",
            "target_cols",
            "target_type",
            "n_targets",
            "is_classification",
        ):
            assert field in metadata, field

    def test_graded_test_ids_are_recorded_in_template_order(
        self, tmp_path: Path
    ) -> None:
        state = self._image_workspace(tmp_path)

        canonical_data_preparation_node(state)

        test_ids = np.load(
            tmp_path / "canonical" / "test_ids.npy", allow_pickle=False
        )
        assert [str(value) for value in test_ids] == state["test_rec_ids"]

    def test_template_ids_complete_the_filename_canonical_contract(
        self, tmp_path: Path
    ) -> None:
        """The public template, not optional state, defines graded test order."""
        state = self._image_workspace(tmp_path)
        template_ids = ["t3", "t1", "t0", "t2"]
        pd.DataFrame({"id": template_ids, "label": [0] * len(template_ids)}).to_csv(
            tmp_path / "sample_submission.csv", index=False
        )
        state.pop("test_rec_ids")

        result = canonical_data_preparation_node(state)

        canonical = tmp_path / "canonical"
        assert result["test_rec_ids"] == template_ids
        assert np.load(canonical / "test_ids.npy", allow_pickle=False).tolist() == template_ids
        assert result["canonical_metadata"]["n_test"] == len(template_ids)
        assert CanonicalDataContract.from_dict(result["canonical_contract"]).validate()[0]

    def test_missing_test_image_fails_closed_without_a_canonical_contract(
        self, tmp_path: Path
    ) -> None:
        """Every submitted template ID must resolve to exactly one test image."""
        state = self._image_workspace(tmp_path)
        template_ids = ["t0", "t1", "missing", "t3"]
        pd.DataFrame({"id": template_ids, "label": [0] * len(template_ids)}).to_csv(
            tmp_path / "sample_submission.csv", index=False
        )
        state.pop("test_rec_ids")

        result = canonical_data_preparation_node(state)

        assert result["canonical_data_prepared"] is False
        assert "canonical_contract" not in result

    def test_template_ids_preserve_leading_zeroes_as_text(
        self, tmp_path: Path
    ) -> None:
        """Numeric-looking submission IDs are identifiers, never integers."""
        state = self._image_workspace(tmp_path)
        template_ids = ["0003", "0001", "0000", "0002"]
        test_dir = Path(state["data_files"]["test"])
        for index, path in enumerate(sorted(test_dir.glob("*.jpg"))):
            path.rename(test_dir / f"{template_ids[index]}.jpg")
        pd.DataFrame({"id": template_ids, "label": [0] * len(template_ids)}).to_csv(
            tmp_path / "sample_submission.csv", index=False
        )
        state.pop("test_rec_ids")

        result = canonical_data_preparation_node(state)

        assert result["test_rec_ids"] == template_ids

    def test_ambiguous_test_image_alias_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """A basename shared by two test images cannot select either image."""
        state = self._image_workspace(tmp_path)
        test_dir = Path(state["data_files"]["test"])
        (test_dir / "t0.jpg").unlink()
        for folder in ("left", "right"):
            duplicate = test_dir / folder
            duplicate.mkdir()
            (duplicate / "duplicate.jpg").write_bytes(b"\x00")
        template_ids = ["duplicate", "t1", "t2", "t3", "unused"]
        pd.DataFrame({"id": template_ids, "label": [0] * len(template_ids)}).to_csv(
            tmp_path / "sample_submission.csv", index=False
        )
        state.pop("test_rec_ids")

        result = canonical_data_preparation_node(state)

        assert result["canonical_data_prepared"] is False
        assert "resolves to 2 test images" in result["canonical_data_skipped_reason"]

    def test_stategraph_keeps_the_canonical_test_ids_path(self) -> None:
        """The graph schema must retain the canonical test-ID artifact path."""
        workflow = StateGraph(KaggleState)
        workflow.add_node(
            "record_test_ids",
            lambda _: {"canonical_test_ids_path": "/tmp/canonical/test_ids.npy"},
        )
        workflow.set_entry_point("record_test_ids")
        workflow.add_edge("record_test_ids", END)

        result = workflow.compile().invoke({})

        assert result["canonical_test_ids_path"] == "/tmp/canonical/test_ids.npy"

    def test_trusted_scoring_inputs_exist(self, tmp_path: Path) -> None:
        """y.npy plus train_ids.npy is what an independent OOF score needs."""
        state = self._image_workspace(tmp_path)

        result = canonical_data_preparation_node(state)

        y = np.load(result["canonical_y_path"], allow_pickle=True)
        train_ids = np.load(result["canonical_train_ids_path"], allow_pickle=False)
        assert len(y) == len(train_ids) == 10

    def test_ambiguous_filenames_still_fail_closed(self, tmp_path: Path) -> None:
        state = self._image_workspace(tmp_path)
        for path in Path(state["data_files"]["train"]).iterdir():
            path.unlink()
        for index in range(6):
            (Path(state["data_files"]["train"]) / f"{index}.jpg").write_bytes(
                b"\x00"
            )

        result = canonical_data_preparation_node(state)

        assert result["canonical_data_prepared"] is False
        assert "filename labels" in result["canonical_data_skipped_reason"]


class TestUnscoredFallbackIsRestorable:
    """The preserved fallback must be usable, or preserving it is theatre."""

    @staticmethod
    def _rejection_state(tmp_path: Path) -> dict:
        """A workspace whose only good submission is an unscored fallback."""
        from kaggle_agents.utils.submission_artifacts import (
            snapshot_best_candidate_submission,
        )

        good = tmp_path / "submission.csv"
        pd.DataFrame({"id": ["a", "b"], "label": [0.25, 0.75]}).to_csv(
            good, index=False
        )
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            good,
            run_id="run1",
            iteration=0,
        )
        # The later candidate overwrites the live file with its own output.
        pd.DataFrame({"id": ["a", "b"], "label": [0.9, 0.9]}).to_csv(
            good, index=False
        )
        return {
            "working_directory": str(tmp_path),
            "run_id": "run1",
            "run_mode": "mlebench",
            "best_candidate_submission_snapshot_path": str(snapshot),
            "best_candidate_submission_sha256": digest,
            "best_candidate_submission_component_name": "first_model",
            # Deliberately empty: this domain could not produce trusted scores.
            "trusted_component_scores": {},
            "oof_availability": {},
            "component_results": {},
        }

    def test_unscored_fallback_is_restored_when_a_later_candidate_is_rejected(
        self, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        from kaggle_agents.agents.developer.agent import DeveloperAgent

        state = self._rejection_state(tmp_path)

        DeveloperAgent._reject_model_candidate(
            None,
            state=state,
            component=SimpleNamespace(
                name="second_model", component_type="model"
            ),
            working_dir=tmp_path,
            current_index=1,
            attempt_records=[],
            reason="unscored candidate cannot replace the preserved one",
            retry_invalid=False,
        )

        restored = pd.read_csv(tmp_path / "submission.csv")
        assert restored["label"].tolist() == [0.25, 0.75]

    def test_the_rejected_component_cannot_restore_its_own_snapshot(
        self, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        from kaggle_agents.agents.developer.agent import DeveloperAgent

        state = self._rejection_state(tmp_path)

        DeveloperAgent._reject_model_candidate(
            None,
            state=state,
            component=SimpleNamespace(
                name="first_model", component_type="model"
            ),
            working_dir=tmp_path,
            current_index=1,
            attempt_records=[],
            reason="owner of the snapshot was itself rejected",
            retry_invalid=False,
        )

        assert not (tmp_path / "submission.csv").exists()

    def test_media_builder_reports_every_path_the_node_consumes(
        self, tmp_path: Path
    ) -> None:
        media = tmp_path / "train"
        media.mkdir()
        counter = 0
        for label in ("alpha", "beta"):
            for _ in range(3):
                (media / f"{label}.{counter}.png").write_bytes(b"\x00")
                counter += 1

        result = DetectionMixin().create_canonical_from_image_filenames(
            image_dir=media,
            canonical_dir=tmp_path / "canonical",
            n_folds=3,
            test_ids=["t0", "t1"],
        )

        assert result["success"] is True
        for key in (
            "canonical_dir",
            "train_ids_path",
            "y_path",
            "folds_path",
            "feature_cols_path",
            "metadata_path",
            "test_ids_path",
        ):
            assert result[key], key
        assert result["metadata"]["source"] == "image_filenames"

    def test_audio_entry_point_still_works(self, tmp_path: Path) -> None:
        media = tmp_path / "train"
        media.mkdir()
        counter = 0
        for label in ("alpha", "beta"):
            for _ in range(3):
                (media / f"{label}.{counter}.wav").write_bytes(b"\x00")
                counter += 1

        result = DetectionMixin().create_canonical_from_audio_filenames(
            media,
            tmp_path / "canonical",
            n_folds=3,
        )

        assert result["success"] is True
        assert result["metadata"]["source"] == "audio_filenames"


class TestContractGuardMessage:
    """The message must not send the fixer after an import it never wrote."""

    def test_message_covers_redefinition_and_reassignment(self) -> None:
        from kaggle_agents.agents.developer.code_contracts import (
            HELPER_IMPORT_CONTRACT_ERROR,
        )

        lowered = HELPER_IMPORT_CONTRACT_ERROR.lower()
        assert "already defined" in lowered
        assert "do not import" in lowered
        assert "define your own" in lowered
        assert "assign over" in lowered
        assert not HELPER_IMPORT_CONTRACT_ERROR.startswith("Do not import")


class TestRobustnessFailureMessage:
    """An 85% aggregate with a hard-failing module is not a threshold problem."""

    def test_failure_message_names_the_binding_constraint(self) -> None:
        import inspect

        from kaggle_agents.agents.robustness_agent import RobustnessAgent

        source = inspect.getsource(RobustnessAgent)

        assert 'print(f"L Validation FAILED ({\'; \'.join(reasons)})")' in source
        assert "failed modules: " in source
        assert "below threshold" in source
