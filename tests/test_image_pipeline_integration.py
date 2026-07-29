"""Generated image code must use the same packed contract as host validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from kaggle_agents.agents.developer.agent import DeveloperAgent
from kaggle_agents.agents.developer.code_generator import (
    _IMAGE_EVIDENCE_ARTIFACT_HELPER,
    _build_image_canonical_header,
)
from kaggle_agents.utils.image_to_image_contract import (
    load_packed_images,
    prepare_image_to_image_canonical_data,
    save_packed_images,
)
from kaggle_agents.workflow.nodes.robustness_gate import _mle_evidence_failures


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((2, 3), value, dtype=np.uint8), mode="L").save(path)


def test_generated_image_artifact_helper_writes_safe_npz_with_embedded_ids(
    tmp_path: Path,
) -> None:
    namespace = {
        "MODELS_DIR": tmp_path / "models",
        "COMPONENT_NAME": "candidate",
        "CANONICAL_TRAIN_IDS": np.array(["a.png", "b.png"], dtype=str),
        "CANONICAL_TEST_IDS": np.array(["test.png"], dtype=str),
    }
    exec(_IMAGE_EVIDENCE_ARTIFACT_HELPER, namespace)

    namespace["save_component_artifacts"](
        [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
        ],
        [np.full((1, 4), 0.5, dtype=np.float32)],
    )

    oof = load_packed_images(tmp_path / "models" / "oof_candidate.npz")
    test = load_packed_images(tmp_path / "models" / "test_candidate.npz")
    assert oof.image_ids.tolist() == ["a.png", "b.png"]
    assert test.image_ids.tolist() == ["test.png"]
    assert not list((tmp_path / "models").glob("*_ids_candidate.npy"))


def test_generated_image_canonical_header_loads_npz_without_tabular_shape_logic(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    test = tmp_path / "test"
    _write_gray(train / "a.png", 1)
    _write_gray(clean / "a.png", 0)
    _write_gray(test / "b.png", 2)
    canonical = prepare_image_to_image_canonical_data(
        noisy_dir=train,
        clean_dir=clean,
        test_dir=test,
        output_dir=tmp_path,
    )

    header = _build_image_canonical_header(
        Path(canonical["canonical_dir"]),
        Path(canonical["y_path"]),
    )
    namespace: dict[str, object] = {}
    exec(header, namespace)

    assert namespace["CANONICAL_TRAIN_IDS"].tolist() == ["a.png"]
    assert namespace["CANONICAL_TEST_IDS"].tolist() == ["b.png"]
    assert namespace["CANONICAL_TARGET_IMAGE_IDS"].tolist() == ["a.png"]
    assert namespace["canonical_target_image"](0).shape == (2, 3)
    assert "CANONICAL_Y = np.load" not in header
    assert "_expected_y_shape" not in header


def test_candidate_transaction_restores_packed_image_evidence(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    canonical = tmp_path / "canonical"
    models.mkdir()
    canonical.mkdir()
    (canonical / "metadata.json").write_text(
        '{"task_type": "image_to_image", "packed_image_contract": true}',
        encoding="utf-8",
    )
    original_oof = b"original-oof"
    original_test = b"original-test"
    (models / "oof_candidate.npz").write_bytes(original_oof)
    (models / "test_candidate.npz").write_bytes(original_test)

    transaction = DeveloperAgent._begin_candidate_transaction(
        tmp_path,
        "candidate",
    )
    (models / "oof_candidate.npz").write_bytes(b"broken-oof")
    (models / "test_candidate.npz").write_bytes(b"broken-test")
    DeveloperAgent._finish_candidate_transaction(
        tmp_path,
        transaction,
        commit=False,
    )

    assert (models / "oof_candidate.npz").read_bytes() == original_oof
    assert (models / "test_candidate.npz").read_bytes() == original_test


def test_approved_component_snapshot_restores_packed_evidence(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    original_oof = b"approved-oof"
    original_test = b"approved-test"
    (models / "oof_approved.npz").write_bytes(original_oof)
    (models / "test_approved.npz").write_bytes(original_test)
    state = {
        "robustness_approved_components": {"approved": True},
        "oof_availability": {"approved": True},
        "trusted_component_scores": {"approved": 0.1},
    }

    snapshot = DeveloperAgent._snapshot_approved_component_artifacts(
        state,
        tmp_path,
        active_component_name="candidate",
    )
    (models / "oof_approved.npz").write_bytes(b"tampered-oof")
    (models / "test_approved.npz").write_bytes(b"tampered-test")
    changed, unrecovered = (
        DeveloperAgent._verify_and_restore_approved_component_artifacts(
            snapshot,
            tmp_path,
            active_component_name="candidate",
        )
    )

    assert changed == ["approved"]
    assert unrecovered == []
    assert (models / "oof_approved.npz").read_bytes() == original_oof
    assert (models / "test_approved.npz").read_bytes() == original_test


def test_mle_robustness_gate_accepts_canonical_packed_image_evidence(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    models = tmp_path / "models"
    canonical.mkdir()
    models.mkdir()
    train_ids = np.array(["train/a.png", "train/b.png"], dtype=str)
    test_ids = np.array(["test/c.png"], dtype=str)
    np.save(canonical / "train_ids.npy", train_ids, allow_pickle=False)
    np.save(canonical / "test_ids.npy", test_ids, allow_pickle=False)
    save_packed_images(
        models / "oof_candidate.npz",
        [np.zeros((2, 2)), np.ones((3, 1))],
        image_ids=train_ids,
    )
    save_packed_images(
        models / "test_candidate.npz",
        [np.full((2, 3), 0.5)],
        image_ids=test_ids,
    )

    failures = _mle_evidence_failures(
        {
            "run_mode": "mlebench",
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "oof_availability": {"candidate": True},
            "trusted_component_scores": {"candidate": 0.1},
            "canonical_contract": {
                "packed_image_contract": True,
                "train_ids_path": str(canonical / "train_ids.npy"),
                "test_ids_path": str(canonical / "test_ids.npy"),
            },
        }
    )

    assert failures == {}


def test_mle_robustness_gate_rejects_misaligned_packed_image_ids(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    models = tmp_path / "models"
    canonical.mkdir()
    models.mkdir()
    train_ids = np.array(["train/a.png", "train/b.png"], dtype=str)
    test_ids = np.array(["test/c.png"], dtype=str)
    np.save(canonical / "train_ids.npy", train_ids, allow_pickle=False)
    np.save(canonical / "test_ids.npy", test_ids, allow_pickle=False)
    save_packed_images(
        models / "oof_candidate.npz",
        [np.zeros((2, 2)), np.ones((3, 1))],
        image_ids=train_ids[::-1],
    )
    save_packed_images(
        models / "test_candidate.npz",
        [np.full((2, 3), 0.5)],
        image_ids=test_ids,
    )

    failures = _mle_evidence_failures(
        {
            "run_mode": "mlebench",
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "oof_availability": {"candidate": True},
            "trusted_component_scores": {"candidate": 0.1},
            "canonical_contract": {
                "packed_image_contract": True,
                "train_ids_path": str(canonical / "train_ids.npy"),
                "test_ids_path": str(canonical / "test_ids.npy"),
            },
        }
    )

    assert "candidate" in failures
    assert any("canonical OOF image order" in issue for issue in failures["candidate"])
