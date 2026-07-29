"""Promotion in MLE-bench must use recomputed OOF metrics, not stdout."""

from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.agents.developer.agent import DeveloperAgent
from kaggle_agents.agents.developer.validation import ValidationMixin
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.tools.code_executor import ExecutionResult
from kaggle_agents.utils.image_to_image_contract import save_packed_images
from kaggle_agents.utils.submission_artifacts import (
    snapshot_best_candidate_submission,
)


class _Validator(ValidationMixin):
    pass


def _execution(stdout: str) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        stdout=stdout,
        stderr="",
        execution_time=0.1,
        exit_code=0,
        artifacts_created=[],
        errors=[],
    )


def test_mlebench_ignores_fabricated_stdout_score(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "canonical").mkdir()
    np.save(tmp_path / "canonical" / "y.npy", np.array([0, 1, 0, 1]))
    np.save(tmp_path / "canonical" / "train_ids.npy", np.arange(4))
    np.save(
        tmp_path / "models" / "oof_candidate.npy",
        np.array([0.9, 0.1, 0.8, 0.2]),
    )
    np.save(tmp_path / "models" / "train_ids_candidate.npy", np.arange(4))
    component = AblationComponent("candidate", "model", "train")

    keep, score = _Validator()._validate_component_improvement(
        component,
        _execution("CV Score: 0.999999"),
        {
            "working_directory": str(tmp_path),
            "run_mode": "mlebench",
            "canonical_contract": {
                "y_path": str(tmp_path / "canonical" / "y.npy"),
                "train_ids_path": str(tmp_path / "canonical" / "train_ids.npy"),
            },
            "competition_info": CompetitionInfo(
                "demo", "", "auc", "binary_classification"
            ),
        },
    )

    assert keep is True
    assert score == 0.0


def test_mlebench_rejects_when_canonical_exists_but_evidence_is_missing(
    tmp_path: Path,
) -> None:
    # The canonical contract exists, so a missing trusted score is the
    # component's fault: fail closed.
    (tmp_path / "canonical").mkdir()
    np.save(tmp_path / "canonical" / "y.npy", np.array([0, 1, 0, 1]))
    component = AblationComponent("candidate", "model", "train")

    keep, score = _Validator()._validate_component_improvement(
        component,
        _execution("CV Score: 0.999999"),
        {
            "working_directory": str(tmp_path),
            "run_mode": "mlebench",
            "competition_info": CompetitionInfo(
                "demo", "", "auc", "binary_classification"
            ),
        },
    )

    assert keep is False
    assert score is None


def test_mlebench_keeps_unscored_candidate_without_canonical_contract(
    tmp_path: Path,
) -> None:
    # Canonical prep legitimately skips for some domains (image without
    # train.csv, audio without labels). The candidate must survive unscored
    # so the deterministic unscored fallback can preserve a gradable
    # artifact — while never being promoted (score stays None).
    component = AblationComponent("candidate", "model", "train")

    keep, score = _Validator()._validate_component_improvement(
        component,
        _execution("CV Score: 0.999999"),
        {
            "working_directory": str(tmp_path),
            "run_mode": "mlebench",
            "competition_info": CompetitionInfo(
                "demo", "", "auc", "binary_classification"
            ),
        },
    )

    assert keep is True
    assert score is None


def test_trusted_score_respects_submission_contract_class_order(
    tmp_path: Path,
) -> None:
    # score_predictions encodes targets in sorted-label order, while mlebench
    # forces OOF columns to follow the submission-contract order. With classes
    # Type_1/Type_2/Type_10 the lexicographic sort permutes columns; the
    # trusted recompute must realign or good candidates get mis-scored.
    (tmp_path / "models").mkdir()
    (tmp_path / "canonical").mkdir()
    y = np.array(["Type_1", "Type_10", "Type_2", "Type_1"])
    np.save(tmp_path / "canonical" / "y.npy", y)
    np.save(tmp_path / "canonical" / "train_ids.npy", np.arange(4))
    contract_order = np.array(["Type_1", "Type_2", "Type_10"])
    # Near-perfect predictions expressed in CONTRACT column order.
    oof_contract = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.9, 0.05],
            [0.9, 0.05, 0.05],
        ]
    )
    np.save(tmp_path / "models" / "oof_candidate.npy", oof_contract)
    np.save(tmp_path / "models" / "train_ids_candidate.npy", np.arange(4))
    np.save(tmp_path / "models" / "class_order_candidate.npy", contract_order)
    component = AblationComponent("candidate", "model", "train")

    score = _Validator()._compute_trusted_oof_score(
        component,
        {
            "working_directory": str(tmp_path),
            "canonical_contract": {
                "y_path": str(tmp_path / "canonical" / "y.npy"),
                "train_ids_path": str(tmp_path / "canonical" / "train_ids.npy"),
            },
            "competition_info": CompetitionInfo(
                "demo", "", "log_loss", "multiclass_classification"
            ),
        },
    )

    # Correct alignment scores the near-perfect predictions well; the
    # permuted-column bug produced a log loss above 1.5 here.
    assert score is not None
    assert score < 0.5


def test_mlebench_does_not_infer_missing_metric_from_candidate_stdout(
    tmp_path: Path,
) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "canonical").mkdir()
    np.save(tmp_path / "canonical" / "y.npy", np.array([0, 1, 0, 1]))
    np.save(tmp_path / "canonical" / "train_ids.npy", np.arange(4))
    np.save(
        tmp_path / "models" / "oof_candidate.npy",
        np.array([0.1, 0.9, 0.2, 0.8]),
    )
    np.save(tmp_path / "models" / "train_ids_candidate.npy", np.arange(4))
    component = AblationComponent("candidate", "model", "train")

    keep, score = _Validator()._validate_component_improvement(
        component,
        _execution("ROC-AUC: 0.999999\nCV Score: 0.999999"),
        {
            "working_directory": str(tmp_path),
            "run_mode": "mlebench",
            "canonical_contract": {
                "y_path": str(tmp_path / "canonical" / "y.npy"),
                "train_ids_path": str(tmp_path / "canonical" / "train_ids.npy"),
            },
            "competition_info": CompetitionInfo(
                "opaque", "", "unknown", "binary_classification"
            ),
        },
    )

    assert keep is False
    assert score is None


def test_bounded_metric_rejects_out_of_domain_score() -> None:
    assert _Validator._is_score_implausible(999.0, "auc") is True


def test_image_to_image_trusted_oof_score_uses_packed_canonical_pixels(
    tmp_path: Path,
) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "canonical").mkdir()
    target_path = save_packed_images(
        tmp_path / "canonical" / "image_targets.npz",
        [
            np.array([[0.0, 2.0]], dtype=np.float32),
            np.array([[4.0]], dtype=np.float32),
        ],
        image_ids=["nested/a.png", "b.png"],
    )
    save_packed_images(
        tmp_path / "models" / "oof_candidate.npz",
        [
            np.array([[1.0, 4.0]], dtype=np.float32),
            np.array([[2.0]], dtype=np.float32),
        ],
        image_ids=["nested/a.png", "b.png"],
    )
    component = AblationComponent("candidate", "model", "train")

    score = _Validator()._compute_trusted_oof_score(
        component,
        {
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "canonical_contract": {"y_path": str(target_path)},
            "competition_info": CompetitionInfo(
                "demo", "", "rmse", "image_to_image"
            ),
        },
    )

    assert score == pytest.approx(np.sqrt(3.0), rel=1e-7)


def test_rejected_candidate_restores_verified_previous_best(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    previous = tmp_path / "submission.csv"
    previous.write_text("id,target\n1,0.8\n", encoding="utf-8")
    snapshot, digest = snapshot_best_candidate_submission(
        tmp_path,
        previous,
        run_id="run-42",
        iteration=0,
    )
    previous.write_text("id,target\n1,0.1\n", encoding="utf-8")
    np.save(tmp_path / "models" / "oof_candidate.npy", np.array([0.1]))

    updates = object.__new__(DeveloperAgent)._reject_model_candidate(
        state={
            "working_directory": str(tmp_path),
            "run_id": "run-42",
            "best_candidate_submission_snapshot_path": str(snapshot),
            "best_candidate_submission_sha256": digest,
            "best_candidate_submission_component_name": "accepted",
            "oof_availability": {"candidate": True},
            "trusted_component_scores": {"candidate": 0.99, "accepted": 0.75},
        },
        component=AblationComponent("candidate", "model", "train"),
        working_dir=tmp_path,
        current_index=1,
        attempt_records=[],
        reason="worse trusted OOF",
        retry_invalid=False,
    )

    assert previous.read_text(encoding="utf-8") == "id,target\n1,0.8\n"
    assert updates["current_component_index"] == 2
    assert updates["oof_availability"]["candidate"] is False
    assert updates["trusted_component_scores"] == {"accepted": 0.75}
    assert not (tmp_path / "models" / "oof_candidate.npy").exists()
