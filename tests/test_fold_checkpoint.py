"""Tests for checkpoint metadata validation during CV resume."""

import json

import numpy as np

from kaggle_agents.utils.fold_checkpoint import FoldCheckpointManager


def test_manager_reloads_compatible_fold_checkpoint(tmp_path):
    manager = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )
    manager.save_fold(
        0,
        model={"kind": "dummy"},
        oof_predictions=np.array([0.2, 0.8]),
        val_indices=np.array([0, 2]),
        score=0.7,
    )

    resumed = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )

    assert resumed.completed_folds == [0]
    recovered, folds = resumed.recover_partial_ensemble()
    assert folds == [0]
    assert recovered[[0, 2]].tolist() == [0.2, 0.8]


def test_manager_rejects_incompatible_metadata(tmp_path):
    manager = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )
    manager.save_fold(
        0,
        model={"kind": "dummy"},
        oof_predictions=np.array([0.2, 0.8]),
        val_indices=np.array([0, 2]),
        score=0.7,
    )
    state_path = tmp_path / "model_checkpoint_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["n_classes"] = 2
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )

    assert resumed.completed_folds == []


def test_manager_rejects_out_of_range_validation_indices(tmp_path):
    manager = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )
    manager.save_fold(
        0,
        model={"kind": "dummy"},
        oof_predictions=np.array([0.2, 0.8]),
        val_indices=np.array([0, 2]),
        score=0.7,
    )
    np.save(tmp_path / "model_fold_0_val_idx.npy", np.array([0, 7]))

    resumed = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )

    assert resumed.completed_folds == []


def test_best_fold_respects_metric_direction(tmp_path):
    manager = FoldCheckpointManager(
        tmp_path,
        component_name="model",
        n_samples=4,
        n_classes=1,
        min_folds=1,
    )
    for fold_idx, score in enumerate((0.7, 0.9)):
        manager.save_fold(
            fold_idx,
            model={"kind": "dummy"},
            oof_predictions=np.array([0.2]),
            val_indices=np.array([fold_idx]),
            score=score,
        )

    assert manager.get_best_fold() == (0, 0.7)
    assert manager.get_best_fold(maximize=True) == (1, 0.9)
