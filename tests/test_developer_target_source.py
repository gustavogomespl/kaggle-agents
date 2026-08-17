"""One canonical-target decision per generated component.

Every Developer path (code header, prompts, audio rewrites) must consume the
same immutable :class:`DeveloperTargetSource`. A complete canonical claim
always wins; a partial or contradictory one fails closed BEFORE any LLM call;
sparse preloading happens only for an inspector-verified sparse-label file.

The validity fixtures below are all produced by the four REAL canonical
producers (dense tabular, media-filename, packed image-to-image, audio
fallback) through ``canonical_data_preparation_node``. Their metadata is
heterogeneous by design, so validation is keyed by representation kind — a
universal schema would refuse legitimate contracts.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import kaggle_agents.agents.developer.code_generator as code_generator_module
from kaggle_agents.agents.developer import agent as developer_agent_module
from kaggle_agents.agents.developer.code_generator import CodeGeneratorMixin
from kaggle_agents.agents.developer.target_source import (
    CanonicalTargetContractError,
    DeveloperTargetSource,
    ProtectedInput,
    reset_target_source_caches,
    resolve_developer_target_source,
)
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    KaggleState,
    create_initial_state,
)
from kaggle_agents.core.state.contracts import CanonicalDataContract
from kaggle_agents.mlebench.data_adapter.artifact_roles import (
    resolve_public_artifacts,
)
from kaggle_agents.mlebench.data_adapter.detection import DetectionMixin
from kaggle_agents.prompts.templates.builders.context import DynamicContext
from kaggle_agents.prompts.templates.developer.prompt_composition import (
    compose_generate_prompt,
)
from kaggle_agents.utils.data_contract import prepare_canonical_data
from kaggle_agents.workflow.nodes.canonical_data import (
    _assert_contract_rows_and_semantics,
    _media_fallback_state_updates,
    canonical_data_preparation_node,
)


@pytest.fixture(autouse=True)
def _clean_caches():
    reset_target_source_caches()
    yield
    reset_target_source_caches()


# ---------------------------------------------------------------------------
# Real-producer fixtures
# ---------------------------------------------------------------------------


def _resolve(state: dict, component_type: str = "model") -> DeveloperTargetSource:
    return resolve_developer_target_source(
        working_dir=Path(state["working_directory"]),
        state=state,
        data_files=state.get("data_files", {}),
        precomputed_info=state.get("precomputed_features_info", {}),
        component_type=component_type,
    )


def _apply(state: dict, updates: dict) -> dict:
    state.update(updates)
    return state


def dense_tabular_state(tmp_path: Path, **extra_data_files) -> dict:
    """Real ``prepare_canonical_data`` contract (dense tabular)."""
    rows = 40
    train = pd.DataFrame(
        {
            "id": np.arange(rows),
            "feature": np.linspace(0.0, 1.0, rows),
            "target": [index % 2 for index in range(rows)],
        }
    )
    test = train.drop(columns=["target"]).iloc[:10].copy()
    (tmp_path / "train.csv").write_text(train.to_csv(index=False), encoding="utf-8")
    (tmp_path / "test.csv").write_text(test.to_csv(index=False), encoding="utf-8")
    state = {
        "working_directory": str(tmp_path),
        "target_col": "target",
        "data_files": {
            "data_type": "tabular",
            "train_csv": str(tmp_path / "train.csv"),
            "test_csv": str(tmp_path / "test.csv"),
            **extra_data_files,
        },
    }
    return _apply(state, canonical_data_preparation_node(state))


def temporal_state(tmp_path: Path) -> dict:
    """Legitimate temporal contract: warm-up rows stay out of OOF."""
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D").repeat(2)
    train = pd.DataFrame(
        {
            "id": np.arange(len(timestamps)),
            "date": timestamps[::-1].astype(str),
            "feature": np.linspace(0.0, 1.0, len(timestamps)),
            "target": np.linspace(10.0, 20.0, len(timestamps)),
        }
    )
    test = train.drop(columns=["target"]).iloc[:8].copy()
    (tmp_path / "train.csv").write_text(train.to_csv(index=False), encoding="utf-8")
    (tmp_path / "test.csv").write_text(test.to_csv(index=False), encoding="utf-8")
    state = {
        "working_directory": str(tmp_path),
        "target_col": "target",
        "domain_detected": "time_series_forecasting",
        "temporal_col": "date",
        "data_files": {
            "data_type": "tabular",
            "train_csv": str(tmp_path / "train.csv"),
            "test_csv": str(tmp_path / "test.csv"),
        },
    }
    return _apply(state, canonical_data_preparation_node(state))


def audio_fallback_state(tmp_path: Path, **extra_data_files) -> dict:
    """Real media-filename fallback through the audio branch of the node."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    for name in ("clip_a_red.wav", "clip_b_red.wav", "clip_c_blue.wav", "clip_d_blue.wav"):
        (train_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (train_dir / name).touch()
    for name in ("t_1.wav", "t_2.wav"):
        (test_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (test_dir / name).touch()
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("clip,probability\nt_1,0\nt_2,0\n", encoding="utf-8")
    state = {
        "working_directory": str(tmp_path),
        "sample_submission_path": str(sample),
        "submission_contract": {"id_col": "clip"},
        "data_files": {
            "data_type": "audio",
            "train": str(train_dir),
            "test": str(test_dir),
            "sample_submission": str(sample),
            "audio_source": str(train_dir),
            **extra_data_files,
        },
    }
    return _apply(state, canonical_data_preparation_node(state))


def image_filename_state(tmp_path: Path) -> dict:
    """Real ``create_canonical_from_media_filenames`` contract (image branch)."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    for name in ("pic_a_red.png", "pic_b_red.png", "pic_c_blue.png", "pic_d_blue.png"):
        path = train_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(path)
    for name in ("t_1.png", "t_2.png"):
        path = test_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(path)
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("image,label\nt_1,0\nt_2,0\n", encoding="utf-8")
    state = {
        "working_directory": str(tmp_path),
        "sample_submission_path": str(sample),
        "submission_contract": {"id_col": "image"},
        "data_files": {
            "data_type": "image",
            "train": str(train_dir),
            "test": str(test_dir),
            "sample_submission": str(sample),
        },
    }
    return _apply(state, canonical_data_preparation_node(state))


def no_test_ids_media_state(tmp_path: Path) -> dict:
    """Real media-filename contract that declares NO test identity (n_test == 0)."""
    audio_dir = tmp_path / "train"
    for name in ("clip_a_red.wav", "clip_b_red.wav", "clip_c_blue.wav", "clip_d_blue.wav"):
        path = audio_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    result = DetectionMixin().create_canonical_from_audio_filenames(
        audio_dir,
        tmp_path / "canonical",
        n_folds=2,
    )
    assert result["success"] is True
    assert result["test_ids_path"] is None
    state = {
        "working_directory": str(tmp_path),
        "data_files": {
            "data_type": "audio",
            "train": str(audio_dir),
            "audio_source": str(audio_dir),
        },
    }
    return _apply(state, _media_fallback_state_updates(result, []))


def packed_image_state(tmp_path: Path) -> dict:
    """Real ``prepare_image_to_image_canonical_data`` packed contract."""
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    test_dir = tmp_path / "test"
    for index, name in enumerate(("a.png", "b.png", "c.png", "d.png")):
        base = np.full((2, 2, 3), 10 * (index + 1), dtype=np.uint8)
        for directory, array in ((noisy, base + 1), (clean, base)):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array, mode="RGB").save(path)
    for name in ("t1.png", "t2.png"):
        path = test_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((2, 2, 3), 7, dtype=np.uint8), mode="RGB").save(path)
    state = {
        "working_directory": str(tmp_path),
        "domain_detected": "image_to_image",
        "data_files": {
            "data_type": "image",
            "train": str(noisy),
            "clean_train": str(clean),
            "test": str(test_dir),
        },
    }
    return _apply(state, canonical_data_preparation_node(state))


ALL_PRODUCERS = {
    "dense_tabular": dense_tabular_state,
    "audio_fallback": audio_fallback_state,
    "image_filename": image_filename_state,
    "packed_image": packed_image_state,
}


def _canonical_dir(state: dict) -> Path:
    return Path(state["working_directory"]) / "canonical"


def _read_metadata(state: dict) -> dict:
    return json.loads((_canonical_dir(state) / "metadata.json").read_text(encoding="utf-8"))


def _write_metadata(state: dict, metadata: dict) -> None:
    (_canonical_dir(state) / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    state["canonical_metadata"] = metadata


def _drop_marker(state: dict) -> dict:
    """Simulate a legacy checkpoint written before validation markers existed."""
    state["canonical_contract_validation"] = None
    return state


def _write_sparse_labels(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "rec_1,3,7,11\nrec_2,4\nrec_3,3,4,9,12\nrec_4,7\n",
        encoding="utf-8",
    )
    return path


def _typed_artifact(path: Path, *, role: str, layout: str, evidence=("x",)) -> dict:
    return {
        "path": str(path),
        "role": role,
        "layout": layout,
        "source_archive": "",
        "evidence": list(evidence),
        "fingerprint": f"stat:{path.name}",
    }


# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------


class TestSelectionTruthTable:
    def test_complete_dense_canonical_beats_stale_sparse_label_path(self, tmp_path):
        stale = _write_sparse_labels(tmp_path / "train_labels.txt")
        state = dense_tabular_state(
            tmp_path,
            label_files=[str(stale)],
            public_artifacts=[
                _typed_artifact(stale, role="auxiliary", layout="sparse_labels")
            ],
        )

        source = _resolve(state)

        assert source.mode == "canonical"
        assert source.canonical_authoritative is True
        # Stale sparse paths are known (so they can be stripped) but never rendered.
        assert source.label_files == ()
        assert source.sparse_label_files == (str(stale),)

    def test_complete_packed_canonical_does_not_assume_y_npy(self, tmp_path):
        state = packed_image_state(tmp_path)

        source = _resolve(state)

        assert source.mode == "canonical"
        assert source.packed_image_contract is True
        assert source.canonical_target_path is not None
        assert source.canonical_target_path.name == "image_targets.npz"
        assert not (_canonical_dir(state) / "y.npy").exists()

    @pytest.mark.parametrize("producer", sorted(ALL_PRODUCERS))
    def test_every_real_producer_shape_is_canonical(self, tmp_path, producer):
        state = ALL_PRODUCERS[producer](tmp_path)

        source = _resolve(state)

        assert source.mode == "canonical"
        assert source.required_canonical_paths
        assert source.protected_inputs
        assert all(isinstance(item, ProtectedInput) for item in source.protected_inputs)

    def test_legitimate_temporal_contract_is_accepted(self, tmp_path):
        state = temporal_state(tmp_path)
        assert state["canonical_metadata"]["cv_strategy"] == "temporal_forward_chaining"

        source = _resolve(state)

        assert source.mode == "canonical"
        marker_files = {Path(path).name for path in source.required_canonical_paths}
        assert "oof_eligible_mask.npy" in marker_files
        assert "temporal_splits.npz" in marker_files

    def test_no_canonical_plus_typed_sparse_artifact_preloads(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "train_labels.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "label_files": [str(labels)],
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels")
                ],
            },
        }

        source = _resolve(state)

        assert source.mode == "sparse_preload"
        assert source.label_files == (str(labels),)
        assert source.sparse_label_files == (str(labels),)
        assert source.canonical_target_path is None

    def test_explicit_empty_typed_artifacts_ignores_legacy_label_files(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "train_labels.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "label_files": [str(labels)],
                # The typed key is PRESENT and empty: the adapter proved there
                # is no public target artifact. Legacy fallback is forbidden.
                "public_artifacts": [],
            },
        }

        source = _resolve(state)

        assert source.mode == "none"
        assert source.label_files == ()

    def test_rectangular_auxiliary_without_canonical_is_none(self, tmp_path):
        table = tmp_path / "extra_metadata.csv"
        table.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "tabular",
                "public_artifacts": [
                    _typed_artifact(table, role="auxiliary", layout="rectangular_table")
                ],
            },
        }

        assert _resolve(state).mode == "none"

    def test_real_adapter_auxiliaries_without_canonical_are_none(self, tmp_path):
        """Ordinary public metadata must not hard-stop a canonical-less run.

        The layout strings above are hand-written, so they cannot catch this:
        the REAL adapter types every unassigned delimited table as
        ``auxiliary`` with whatever the REAL inspector says, and the inspector
        answers ``unknown``/``ambiguous_layout`` for plenty of perfectly
        ordinary tables. Treating that as "a target candidate we rejected"
        turns every image-without-train.csv or audio-without-labels run that
        ships one lookup table into a hard failure for every component.
        """
        public_dir = tmp_path / "public"
        public_dir.mkdir()
        (public_dir / "train.csv").write_text(
            "id,feature,target\n1,0.1,0\n2,0.2,1\n3,0.3,0\n", encoding="utf-8"
        )
        (public_dir / "test.csv").write_text(
            "id,feature\n4,0.4\n5,0.5\n", encoding="utf-8"
        )
        (public_dir / "sample_submission.csv").write_text(
            "id,target\n4,0\n5,0\n", encoding="utf-8"
        )
        # Ordinary auxiliary tables, no target claim anywhere in sight.
        (public_dir / "region_lookup.csv").write_text(
            "a,b\n1,2\n3,4\n", encoding="utf-8"
        )
        (public_dir / "image_index.csv").write_text(
            "image_id\npic_a\npic_b\npic_c\n", encoding="utf-8"
        )

        artifacts = resolve_public_artifacts(public_dir, [], "tabular")
        typed = [artifact.to_state() for artifact in artifacts]
        auxiliary_layouts = {
            record["layout"]
            for record in typed
            if record["role"] == "auxiliary"
        }
        # Guard against a vacuous fixture: at least one ordinary table really
        # is typed auxiliary/unknown by the real inspector.
        assert "unknown" in auxiliary_layouts

        state = {
            "working_directory": str(tmp_path),
            "data_files": {"data_type": "tabular", "public_artifacts": typed},
        }

        source = _resolve(state)

        assert source.mode == "none"
        assert source.label_files == ()

    def test_typed_unknown_that_is_also_a_declared_label_file_still_fails_closed(
        self, tmp_path
    ):
        """An artifact something CLAIMED is a label file keeps failing closed."""
        claimed = tmp_path / "train_labels.txt"
        claimed.write_text("", encoding="utf-8")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "label_files": [str(claimed)],
                "public_artifacts": [
                    _typed_artifact(
                        claimed,
                        role="auxiliary",
                        layout="unknown",
                        evidence=("empty_or_unreadable_file",),
                    )
                ],
            },
        }

        with pytest.raises(ValueError, match="empty_or_unreadable_file"):
            _resolve(state)

    def test_legacy_state_without_typed_key_inspects_instead_of_trusting_names(
        self, tmp_path
    ):
        verified = _write_sparse_labels(tmp_path / "opaque_annotations.txt")
        # Named like labels, but it is a rectangular table: filename trust would
        # preload it as targets.
        impostor = tmp_path / "train_labels.csv"
        impostor.write_text("id,a,b\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "label_files": [str(impostor), str(verified)],
            },
        }

        source = _resolve(state)

        assert source.mode == "sparse_preload"
        assert source.label_files == (str(verified),)

    def test_multiple_verified_sparse_targets_fail_closed_with_candidates(self, tmp_path):
        first = _write_sparse_labels(tmp_path / "labels_a.txt")
        second = _write_sparse_labels(tmp_path / "labels_b.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(first, role="auxiliary", layout="sparse_labels"),
                    _typed_artifact(second, role="auxiliary", layout="sparse_labels"),
                ],
            },
        }

        with pytest.raises(ValueError) as excinfo:
            _resolve(state)

        message = str(excinfo.value)
        assert str(first) in message and str(second) in message

    def test_multiple_id_mappings_fail_closed_with_candidates(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "labels.txt")
        first = tmp_path / "map_a.csv"
        second = tmp_path / "map_b.csv"
        for path in (first, second):
            path.write_text("record_id,file_name\n1,a.wav\n2,b.wav\n", encoding="utf-8")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels"),
                    _typed_artifact(first, role="auxiliary", layout="id_mapping"),
                    _typed_artifact(second, role="auxiliary", layout="id_mapping"),
                ],
            },
        }

        with pytest.raises(ValueError) as excinfo:
            _resolve(state)

        message = str(excinfo.value)
        assert str(first) in message and str(second) in message

    def test_explicit_mapping_manifest_disambiguates(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "labels.txt")
        first = tmp_path / "map_a.csv"
        second = tmp_path / "map_b.csv"
        for path in (first, second):
            path.write_text("record_id,file_name\n1,a.wav\n2,b.wav\n", encoding="utf-8")
        state = {
            "working_directory": str(tmp_path),
            "precomputed_features_info": {"features_found": {"id_mapping": str(first)}},
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels"),
                    _typed_artifact(first, role="auxiliary", layout="id_mapping"),
                    _typed_artifact(second, role="auxiliary", layout="id_mapping"),
                ],
            },
        }

        source = _resolve(state)

        assert source.id_mapping_path == first
        assert source.mode == "sparse_preload"

    def test_claimed_sparse_target_that_fails_reinspection_reports_evidence(
        self, tmp_path
    ):
        """A record that CLAIMS sparse labels is a target claim; it fails closed."""
        impostor = tmp_path / "train_labels.csv"
        impostor.write_text("id,a,b\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(
                        impostor,
                        role="auxiliary",
                        layout="sparse_labels",
                    )
                ],
            },
        }

        with pytest.raises(ValueError) as excinfo:
            _resolve(state)

        message = str(excinfo.value)
        assert str(impostor) in message
        assert "rectangular_table" in message

    def test_no_evidence_at_all_is_none(self, tmp_path):
        state = {"working_directory": str(tmp_path), "data_files": {}}

        source = _resolve(state)

        assert source.mode == "none"
        assert source.label_files == ()
        assert source.id_mapping_path is None


class TestSelectionIsIndependentOfRunModeAndComponent:
    @pytest.mark.parametrize("run_mode", ["", "mlebench", "kaggle", "local"])
    @pytest.mark.parametrize(
        "component_type",
        ["model", "ensemble", "preprocessing", "feature_engineering"],
    )
    def test_canonical_always_wins(self, tmp_path, run_mode, component_type):
        state = dense_tabular_state(tmp_path)
        state["run_mode"] = run_mode

        source = _resolve(state, component_type=component_type)

        assert source.mode == "canonical"

    @pytest.mark.parametrize("run_mode", ["", "mlebench"])
    @pytest.mark.parametrize("component_type", ["model", "preprocessing"])
    def test_partial_canonical_always_fails(self, tmp_path, run_mode, component_type):
        state = dense_tabular_state(tmp_path)
        state["run_mode"] = run_mode
        (_canonical_dir(state) / "folds.npy").unlink()

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state, component_type=component_type)

    def test_fingerprint_does_not_depend_on_component_type(self, tmp_path):
        state = dense_tabular_state(tmp_path)

        first = _resolve(state, component_type="model")
        second = _resolve(state, component_type="preprocessing")

        assert first.target_source_fingerprint == second.target_source_fingerprint


# ---------------------------------------------------------------------------
# Fail-closed canonical corruption
# ---------------------------------------------------------------------------


class TestCanonicalCorruptionFailsClosed:
    @pytest.mark.parametrize("producer", sorted(ALL_PRODUCERS))
    def test_missing_declared_target_file_per_producer(self, tmp_path, producer):
        state = ALL_PRODUCERS[producer](tmp_path)
        Path(state["canonical_contract"]["y_path"]).unlink()

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert excinfo.value.violations

    @pytest.mark.parametrize("producer", sorted(ALL_PRODUCERS))
    def test_truncated_folds_per_producer(self, tmp_path, producer):
        state = ALL_PRODUCERS[producer](tmp_path)
        folds_path = Path(state["canonical_contract"]["folds_path"])
        folds = np.load(folds_path)
        np.save(folds_path, folds[:-1])

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_contradictory_claim_without_prepared_flag(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        state["canonical_data_prepared"] = False

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert "canonical_data_prepared" in str(excinfo.value)

    def test_missing_canonical_file(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        (_canonical_dir(state) / "feature_cols.json").unlink()

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_invalid_metadata_json(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        (_canonical_dir(state) / "metadata.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_metadata_missing_required_semantic_field(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        del metadata["n_targets"]
        _write_metadata(state, metadata)

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert "n_targets" in str(excinfo.value)

    def test_feature_metadata_inconsistent_with_contract(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        (_canonical_dir(state) / "feature_cols.json").write_text(
            json.dumps(["a", "b", "c", "d", "e", "f"]), encoding="utf-8"
        )

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_invalid_feature_cols_json(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        (_canonical_dir(state) / "feature_cols.json").write_text("nope", encoding="utf-8")

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_extra_contract_key_is_a_corruption_error_not_a_typeerror(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        state["canonical_contract"] = {**state["canonical_contract"], "bogus": 1}

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert "bogus" in str(excinfo.value)

    def test_missing_contract_key_is_a_corruption_error_not_a_keyerror(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        contract = dict(state["canonical_contract"])
        del contract["y_path"]
        state["canonical_contract"] = contract

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert "y_path" in str(excinfo.value)

    def test_malformed_contract_value_is_a_corruption_error(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        state["canonical_contract"] = {
            **state["canonical_contract"],
            "n_train": "not-an-int",
        }

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_unequal_train_id_and_target_row_counts(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        train_ids_path = Path(state["canonical_contract"]["train_ids_path"])
        ids = np.load(train_ids_path, allow_pickle=False)
        np.save(train_ids_path, ids[:-2], allow_pickle=False)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_dense_target_shape_inconsistent_with_target_type(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        y_path = Path(state["canonical_contract"]["y_path"])
        y = np.load(y_path, allow_pickle=True)
        np.save(y_path, np.column_stack([y, y]))

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_duplicate_train_ids(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        train_ids_path = Path(state["canonical_contract"]["train_ids_path"])
        ids = np.load(train_ids_path, allow_pickle=False)
        ids = ids.copy()
        ids[-1] = ids[0]
        np.save(train_ids_path, ids, allow_pickle=False)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_non_scalar_train_ids(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        train_ids_path = Path(state["canonical_contract"]["train_ids_path"])
        ids = np.load(train_ids_path, allow_pickle=False)
        np.save(train_ids_path, np.column_stack([ids, ids]), allow_pickle=False)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_negative_folds_outside_a_temporal_contract(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        folds_path = Path(state["canonical_contract"]["folds_path"])
        folds = np.load(folds_path)
        folds = folds.copy()
        folds[0] = -1
        np.save(folds_path, folds)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_folds_out_of_declared_range(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        folds_path = Path(state["canonical_contract"]["folds_path"])
        folds = np.load(folds_path)
        folds = folds.copy()
        folds[0] = 99
        np.save(folds_path, folds)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_empty_folds(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        folds_path = Path(state["canonical_contract"]["folds_path"])
        np.save(folds_path, np.asarray([], dtype=np.int64))

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_metadata_fold_count_disagrees_with_folds_array(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        metadata["n_folds"] = 2
        _write_metadata(state, metadata)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_class_order_absent_for_single_target_classification(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        metadata["class_order"] = None
        _write_metadata(state, metadata)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_duplicate_class_order(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        metadata["class_order"] = ["0", "0"]
        _write_metadata(state, metadata)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_class_order_incompatible_with_labels(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        metadata["class_order"] = ["cat", "dog"]
        _write_metadata(state, metadata)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_temporal_arrays_that_disagree(self, tmp_path):
        state = temporal_state(tmp_path)
        mask_path = Path(state["canonical_contract"]["oof_eligible_mask_path"])
        mask = np.load(mask_path)
        np.save(mask_path, ~mask)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_temporal_claim_without_temporal_artifacts(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        metadata["cv_strategy"] = "temporal_forward_chaining"
        _write_metadata(state, metadata)
        state["canonical_contract"] = {
            **state["canonical_contract"],
            "cv_strategy": "temporal_forward_chaining",
        }

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_incomplete_packed_target_manifest(self, tmp_path):
        state = packed_image_state(tmp_path)
        Path(state["canonical_contract"]["image_test_input_paths_path"]).unlink()

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_packed_target_ids_out_of_sync_with_train_ids(self, tmp_path):
        state = packed_image_state(tmp_path)
        train_ids_path = Path(state["canonical_contract"]["train_ids_path"])
        ids = np.load(train_ids_path, allow_pickle=False)
        ids = ids.copy()
        ids[0] = "not-an-image"
        np.save(train_ids_path, ids, allow_pickle=False)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_canonical_path_escaping_the_canonical_directory(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        outside = tmp_path / "outside_y.npy"
        np.save(outside, np.load(state["canonical_contract"]["y_path"], allow_pickle=True))
        state["canonical_contract"] = {
            **state["canonical_contract"],
            "y_path": str(outside),
        }

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert "canonical" in str(excinfo.value)

    def test_symlinked_canonical_artifact(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        folds_path = Path(state["canonical_contract"]["folds_path"])
        real = tmp_path / "real_folds.npy"
        folds_path.rename(real)
        folds_path.symlink_to(real)

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        assert "symlink" in str(excinfo.value).lower()

    def test_state_metadata_contradicts_metadata_json(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        state["canonical_metadata"] = {
            **state["canonical_metadata"],
            "target_col": "something_else",
        }

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_contract_contradicts_metadata_json(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        state["canonical_contract"] = {
            **state["canonical_contract"],
            "n_train": 7,
        }

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_declared_test_ids_are_required_when_n_test_is_positive(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        assert state["canonical_contract"]["n_test"] > 0
        Path(state["canonical_contract"]["test_ids_path"]).unlink()

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_duplicate_test_ids_fail_closed(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        test_ids_path = Path(state["canonical_contract"]["test_ids_path"])
        ids = np.load(test_ids_path, allow_pickle=False)
        ids = ids.copy()
        ids[-1] = ids[0]
        np.save(test_ids_path, ids, allow_pickle=False)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_positional_metadata_inconsistent_with_test_ids(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        metadata = _read_metadata(state)
        metadata["test_ids_are_positional"] = True
        _write_metadata(state, metadata)
        test_ids_path = Path(state["canonical_contract"]["test_ids_path"])
        np.save(
            test_ids_path,
            np.asarray([f"row-{index}" for index in range(metadata["n_test"])], dtype=str),
            allow_pickle=False,
        )

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_violations_are_json_safe(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        (_canonical_dir(state) / "metadata.json").write_text("{oops", encoding="utf-8")

        with pytest.raises(CanonicalTargetContractError) as excinfo:
            _resolve(state)

        json.dumps(excinfo.value.violations)


# ---------------------------------------------------------------------------
# Validation marker, caching and fingerprints
# ---------------------------------------------------------------------------


class TestValidationMarkerAndCaching:
    def test_node_records_a_validation_marker(self, tmp_path):
        state = dense_tabular_state(tmp_path)

        marker = state["canonical_contract_validation"]

        assert isinstance(marker, dict)
        assert marker["representation_kind"] == "dense_tabular"
        assert marker["fingerprint"]
        assert marker["metadata_sha256"]
        assert marker["files"]
        json.dumps(marker)

    def test_marker_fast_path_skips_full_checksum_validation(self, tmp_path, monkeypatch):
        state = dense_tabular_state(tmp_path)
        calls = {"n": 0}
        original = CanonicalDataContract.validate

        def counted(self):
            calls["n"] += 1
            return original(self)

        monkeypatch.setattr(CanonicalDataContract, "validate", counted)

        assert _resolve(state).mode == "canonical"
        assert _resolve(state).mode == "canonical"

        assert calls["n"] == 0

    def test_legacy_contract_without_marker_validates_exactly_once(
        self, tmp_path, monkeypatch
    ):
        state = _drop_marker(dense_tabular_state(tmp_path))
        calls = {"n": 0}
        original = CanonicalDataContract.validate

        def counted(self):
            calls["n"] += 1
            return original(self)

        monkeypatch.setattr(CanonicalDataContract, "validate", counted)

        _resolve(state)
        _resolve(state)

        assert calls["n"] == 1

    def test_changed_file_stat_invalidates_the_cached_validation(
        self, tmp_path, monkeypatch
    ):
        state = _drop_marker(dense_tabular_state(tmp_path))
        calls = {"n": 0}
        original = CanonicalDataContract.validate

        def counted(self):
            calls["n"] += 1
            return original(self)

        monkeypatch.setattr(CanonicalDataContract, "validate", counted)

        _resolve(state)
        folds_path = Path(state["canonical_contract"]["folds_path"])
        np.save(folds_path, np.load(folds_path))
        _resolve(state)

        assert calls["n"] == 2

    def test_changed_metadata_invalidates_the_marker_fast_path(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        _resolve(state)
        metadata = _read_metadata(state)
        metadata["n_folds"] = 3
        _write_metadata(state, metadata)

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_corruption_still_fails_with_a_warm_cache(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        assert _resolve(state).mode == "canonical"
        y_path = Path(state["canonical_contract"]["y_path"])
        y = np.load(y_path, allow_pickle=True)
        np.save(y_path, np.concatenate([y, y]))

        with pytest.raises(CanonicalTargetContractError):
            _resolve(state)

    def test_protected_inputs_are_sorted_and_hashed(self, tmp_path):
        state = dense_tabular_state(tmp_path)

        source = _resolve(state)

        relative = [item.relative_path for item in source.protected_inputs]
        assert relative == sorted(relative)
        assert all(len(item.sha256) == 64 for item in source.protected_inputs)
        assert all(item.size > 0 for item in source.protected_inputs)
        # Lazily-read body inputs (train/test tables) are NOT protected inputs.
        assert not any("train.csv" in item.relative_path for item in source.protected_inputs)

    def test_fingerprint_changes_when_bytes_change_at_the_same_paths(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        before = _resolve(state).target_source_fingerprint

        folds_path = Path(state["canonical_contract"]["folds_path"])
        folds = np.load(folds_path).copy()
        folds[0] = (folds[0] + 1) % int(state["canonical_metadata"]["n_folds"])
        np.save(folds_path, folds)
        contract = dict(state["canonical_contract"])
        contract["folds_hash"] = CanonicalDataContract.compute_array_hash(folds)
        state["canonical_contract"] = contract
        _drop_marker(state)

        after = _resolve(state).target_source_fingerprint

        assert before != after

    def test_sparse_fingerprint_changes_when_label_bytes_change(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "labels.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels")
                ],
            },
        }
        before = _resolve(state).target_source_fingerprint

        labels.write_text(
            "rec_1,3,7,11\nrec_2,4\nrec_3,3,4,9,12\nrec_4,8\n", encoding="utf-8"
        )
        reset_target_source_caches()

        assert _resolve(state).target_source_fingerprint != before


# ---------------------------------------------------------------------------
# Test identity is a declared contract artifact
# ---------------------------------------------------------------------------


class TestDeclaredTestIdentity:
    def test_preparation_returns_and_declares_test_ids_path(self, tmp_path):
        state = dense_tabular_state(tmp_path)

        assert state["canonical_test_ids_path"]
        assert Path(state["canonical_test_ids_path"]).is_file()
        assert state["canonical_contract"]["test_ids_path"] == state["canonical_test_ids_path"]

    def test_materialized_synthetic_id_still_yields_positional_test_ids(self, tmp_path):
        rows = 40
        pd.DataFrame(
            {
                "Label": [index % 2 for index in range(rows)],
                "Body": [f"document {index}" for index in range(rows)],
            }
        ).to_csv(tmp_path / "train.csv", index=False)
        pd.DataFrame(
            {"Body": [f"held out {index}" for index in range(10)]}
        ).to_csv(tmp_path / "test.csv", index=False)
        result = prepare_canonical_data(
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
            target_col="Label",
            target_cols=["Label"],
            output_dir=tmp_path / "work",
            task_type="text_classification",
        )

        assert result["metadata"]["id_is_synthetic"] is True
        assert result["metadata"]["test_ids_are_positional"] is True
        assert result["test_ids_path"]
        test_ids = np.load(result["test_ids_path"], allow_pickle=False)
        assert [str(value) for value in test_ids] == [str(index) for index in range(10)]

    def test_genuine_public_identifier_remains_non_positional(self, tmp_path):
        state = dense_tabular_state(tmp_path)

        assert state["canonical_metadata"]["id_is_synthetic"] is False
        assert state["canonical_metadata"]["test_ids_are_positional"] is False


# ---------------------------------------------------------------------------
# Rendering and prompt consumption
# ---------------------------------------------------------------------------


class TestRenderedHeader:
    @staticmethod
    def _header(state: dict, *, data_type: str = "tabular") -> str:
        module = code_generator_module

        class _Generator(CodeGeneratorMixin):
            use_dspy = False
            config = SimpleNamespace()
            llm = SimpleNamespace(invoke=lambda _m: SimpleNamespace(content="pass"))

            @staticmethod
            def _get_dataset_info(*_args, **_kwargs) -> str:
                return ""

            @staticmethod
            def _get_domain_template(*_args, **_kwargs) -> str:
                return ""

            @staticmethod
            def _extract_code_from_response(response: str) -> str:
                return response.strip()

        original_instructions = module.build_dynamic_instructions
        original_context = module.build_context
        original_prompt = module.compose_generate_prompt
        module.build_dynamic_instructions = lambda **_k: ""
        module.build_context = lambda *_a, **_k: SimpleNamespace()
        module.compose_generate_prompt = lambda **_k: ""
        try:
            state.setdefault("data_files", {})["data_type"] = data_type
            generated = _Generator()._generate_code(
                AblationComponent("candidate", "model", "train"),
                CompetitionInfo("demo", "", "auc", "classification"),
                Path(state["working_directory"]),
                "tabular",
                state,
            )
        finally:
            module.build_dynamic_instructions = original_instructions
            module.build_context = original_context
            module.compose_generate_prompt = original_prompt
        return generated.split("# === END PATH CONSTANTS ===", 1)[0]

    def test_canonical_mode_never_preloads_auxiliary_targets(self, tmp_path):
        stale = _write_sparse_labels(tmp_path / "train_labels.txt")
        state = dense_tabular_state(
            tmp_path,
            label_files=[str(stale)],
            public_artifacts=[
                _typed_artifact(stale, role="auxiliary", layout="sparse_labels")
            ],
        )

        header = self._header(state)

        assert "_PRELOADED_TARGETS_DF" not in header
        assert "LABEL_FILES" not in header
        assert "_load_targets_from_files" not in header
        assert str(stale) not in header
        assert "CANONICAL_Y" in header

    def test_sparse_mode_renders_delegating_parser(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "labels.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels")
                ],
            },
        }

        header = self._header(state, data_type="audio")

        assert "LABEL_FILES" in header
        assert "_PRELOADED_TARGETS_DF" in header
        assert "parse_sparse_label_rows" in header
        assert "csv.Sniffer" not in header

    def test_id_mapping_helper_is_rendered_in_canonical_mode(self, tmp_path):
        mapping = tmp_path / "mapping.csv"
        mapping.write_text("record_id,file_name\nclip_a_red,clip_a_red.wav\n", encoding="utf-8")
        state = audio_fallback_state(tmp_path)
        state["precomputed_features_info"] = {
            "features_found": {"id_mapping": str(mapping)}
        }

        header = self._header(state, data_type="audio")

        assert "def parse_id_mapping_file" in header
        assert "_PRELOADED_RECORD_IDS" not in header
        assert "RECORD_ID_TO_INPUT_PATH" in header
        assert "CANONICAL_TRAIN_IDS" in header

    def test_canonical_audio_without_sparse_labels_defines_every_helper(self, tmp_path):
        state = audio_fallback_state(tmp_path)

        header = self._header(state, data_type="audio")

        namespace: dict = {}
        exec(compile(header, "<generated-audio-header>", "exec"), namespace)
        assert "RECORD_ID_TO_INPUT_PATH" in namespace
        assert "_PRELOADED_RECORD_ID_TO_PATH" not in namespace

    def test_undeclared_test_identity_ignores_a_stale_test_ids_file(self, tmp_path):
        """An undeclared test identity is never rediscovered from disk.

        A contract with no ``test_ids_path`` (n_test == 0) next to a leftover
        ``canonical/test_ids.npy`` from an earlier prep used to load the stale
        file, which is exactly the rediscovery-by-existence the contract is
        supposed to replace.
        """
        state = no_test_ids_media_state(tmp_path)
        assert state["canonical_contract"]["test_ids_path"] is None
        stale = _canonical_dir(state) / "test_ids.npy"
        np.save(stale, np.asarray(["stale-a", "stale-b"], dtype=str), allow_pickle=False)

        header = self._header(state, data_type="audio")

        assert "test_ids.npy" not in header
        namespace: dict = {}
        exec(compile(header, "<generated-audio-header>", "exec"), namespace)
        assert namespace["CANONICAL_TEST_IDS"] is None


class TestPromptBranches:
    @staticmethod
    def _prompt(source: DeveloperTargetSource, paths_extra: dict | None = None) -> str:
        paths = {"output_dir": ".", "target_source": source}
        paths.update(paths_extra or {})
        return compose_generate_prompt(
            component=AblationComponent("candidate", "model", "train"),
            competition_info=CompetitionInfo("demo", "", "auc", "classification"),
            paths=paths,
            context=DynamicContext(),
        )

    def test_canonical_prompt_describes_auxiliary_artifacts_not_targets(self, tmp_path):
        stale = _write_sparse_labels(tmp_path / "train_labels.txt")
        extra = tmp_path / "extra_metadata.csv"
        extra.write_text("a,b\n1,2\n", encoding="utf-8")
        state = dense_tabular_state(
            tmp_path,
            label_files=[str(stale)],
            public_artifacts=[
                _typed_artifact(stale, role="auxiliary", layout="sparse_labels"),
                _typed_artifact(extra, role="auxiliary", layout="rectangular_table"),
            ],
        )
        source = _resolve(state)

        prompt = self._prompt(
            source,
            {"public_artifacts": state["data_files"]["public_artifacts"]},
        )

        assert "AUXILIARY PUBLIC ARTIFACTS" in prompt
        assert "NON-STANDARD LABEL FILES" not in prompt
        assert str(extra) in prompt
        assert str(stale) not in prompt

    def test_packed_canonical_prompt_is_not_dense(self, tmp_path):
        state = packed_image_state(tmp_path)
        source = _resolve(state)

        prompt = self._prompt(source)

        assert "CANONICAL_TARGET_IMAGE_IDS" in prompt
        assert 'np.load(canonical_dir / "y.npy"' not in prompt

    def test_sparse_prompt_keeps_verified_label_instructions(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "labels.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels")
                ],
            },
        }
        source = _resolve(state)

        prompt = self._prompt(source)

        assert "NON-STANDARD LABEL FILES" in prompt
        assert str(labels) in prompt

    def test_none_mode_has_no_mandatory_label_section(self, tmp_path):
        state = {"working_directory": str(tmp_path), "data_files": {}}
        source = _resolve(state)

        prompt = self._prompt(source)

        assert "NON-STANDARD LABEL FILES" not in prompt
        assert "AUXILIARY PUBLIC ARTIFACTS" not in prompt

    def test_prompt_does_not_probe_the_canonical_directory(self, tmp_path):
        # A bare canonical/ directory on disk must not turn into a canonical
        # prompt: authority comes from the validated decision only.
        (tmp_path / "canonical").mkdir()
        (tmp_path / "canonical" / "metadata.json").write_text("{}", encoding="utf-8")
        state = {"working_directory": str(tmp_path), "data_files": {}}
        source = _resolve(state)

        prompt = self._prompt(source, {"output_dir": str(tmp_path)})

        assert "MANDATORY: Canonical Data Contract" not in prompt


class TestAudioRewritesConsumeTheDecision:
    @staticmethod
    def _mixin():
        return CodeGeneratorMixin

    def test_canonical_mode_strips_hidden_sparse_target_references(self, tmp_path):
        stale = _write_sparse_labels(tmp_path / "train_labels.txt")
        state = dense_tabular_state(
            tmp_path,
            label_files=[str(stale)],
            public_artifacts=[
                _typed_artifact(stale, role="auxiliary", layout="sparse_labels")
            ],
        )
        source = _resolve(state)
        code = f'''# === END PATH CONSTANTS ===
labels = pd.read_csv("{stale}")
'''

        warnings = self._mixin()._validate_audio_label_usage(
            None, code, "audio", target_source=source
        )
        rewritten, count = self._mixin()._strip_label_reparsing(
            None, code, target_source=source
        )

        assert warnings
        assert "_PRELOADED_TARGETS_DF" not in "\n".join(warnings)
        assert count == 1
        assert "_PRELOADED_TARGETS_DF" not in rewritten
        assert "CANONICAL_Y" in rewritten

    def test_sparse_mode_preserves_existing_enforcement(self, tmp_path):
        labels = _write_sparse_labels(tmp_path / "labels.txt")
        state = {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "public_artifacts": [
                    _typed_artifact(labels, role="auxiliary", layout="sparse_labels")
                ],
            },
        }
        source = _resolve(state)
        code = f'''# === END PATH CONSTANTS ===
labels = pd.read_csv("{labels}")
'''

        rewritten, count = self._mixin()._strip_label_reparsing(
            None, code, target_source=source
        )

        assert count == 1
        assert "labels = _PRELOADED_TARGETS_DF.copy()" in rewritten


class TestProducerAndConsumerShareTheRules:
    """The node and the selector must not fork the canonical validation rules."""

    def test_same_corruption_yields_the_same_violation_code(self, tmp_path):
        state = dense_tabular_state(tmp_path)
        contract = state["canonical_contract"]
        metadata = state["canonical_metadata"]
        train_ids = np.load(contract["train_ids_path"], allow_pickle=True)
        folds = np.load(contract["folds_path"], allow_pickle=True)
        targets = np.load(contract["y_path"], allow_pickle=True)

        # Producer lane: the arrays it holds disagree with the contract.
        with pytest.raises(ValueError) as producer:
            _assert_contract_rows_and_semantics(
                contract,
                metadata,
                train_ids=train_ids[:-2],
                folds=folds[:-2],
                y=targets[:-2],
            )

        # Consumer lane: the same disagreement, discovered from disk.
        np.save(contract["train_ids_path"], train_ids[:-2], allow_pickle=False)
        _drop_marker(state)
        with pytest.raises(CanonicalTargetContractError) as consumer:
            _resolve(state)

        assert "row_count_disagreement" in str(producer.value)
        assert any(
            item["code"] == "row_count_disagreement" for item in consumer.value.violations
        )

    def test_node_rejects_an_incomplete_packed_manifest(self, tmp_path):
        state = packed_image_state(tmp_path)
        contract = dict(state["canonical_contract"])
        Path(contract["image_test_input_paths_path"]).unlink()

        with pytest.raises(ValueError, match="packed_image_contract_violation"):
            _assert_contract_rows_and_semantics(contract, state["canonical_metadata"])

    def test_packed_multi_column_target_names_are_not_a_landmine(self, tmp_path):
        """A packed contract has no dense y, so target_cols length is not a shape."""
        state = packed_image_state(tmp_path)
        contract = dict(state["canonical_contract"])
        metadata = dict(state["canonical_metadata"])
        contract["target_cols"] = ["image_pixels", "image_alpha"]
        metadata["target_cols"] = ["image_pixels", "image_alpha"]
        metadata["n_targets"] = 2

        _assert_contract_rows_and_semantics(contract, metadata)


class TestExecutionMetadataReachesState:
    """The fingerprint and protected-input manifest must survive the node."""

    def test_target_source_record_is_a_declared_state_key(self):
        assert "target_source_record" in KaggleState.__annotations__
        # LangGraph silently drops undeclared keys, so it must also be seeded.
        initial = create_initial_state(competition_name="demo", working_dir=".")
        assert "target_source_record" in initial

    def test_developer_drains_the_decision_into_state_updates(self):
        source = inspect.getsource(developer_agent_module)

        assert 'state_updates["target_source_record"]' in source
        assert "self._last_target_source_metadata = None" in source

    def test_execution_metadata_is_json_safe_and_complete(self, tmp_path):
        state = dense_tabular_state(tmp_path)

        metadata = _resolve(state).execution_metadata()

        json.dumps(metadata)
        assert metadata["mode"] == "canonical"
        assert metadata["representation_kind"] == "dense_tabular"
        assert len(metadata["target_source_fingerprint"]) == 64
        assert metadata["protected_inputs"]
        assert all(
            len(item["sha256"]) == 64 for item in metadata["protected_inputs"]
        )
