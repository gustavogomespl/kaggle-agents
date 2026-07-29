"""Focused tests for the canonical temporal CV contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.prompts.templates.builders.cv import build_cv_instructions
from kaggle_agents.agents.ensemble.scoring import compute_oof_score
from kaggle_agents.utils.calibration import calibrate_oof_predictions
from kaggle_agents.utils.data_contract import (
    get_canonical_data_instructions,
    load_canonical_data,
    prepare_canonical_data,
    validate_oof_alignment,
)
from kaggle_agents.utils.strict_validation import (
    StrictValidationConfig,
    validate_model_artifacts,
)
from kaggle_agents.workflow.nodes.canonical_data import (
    canonical_data_preparation_node,
)


def _write_temporal_csvs(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D").repeat(2)
    # Deliberately reverse row order: the contract must derive chronological
    # splits from the time evidence, not trust CSV row order.
    train = pd.DataFrame(
        {
            "id": np.arange(len(timestamps)),
            "date": timestamps[::-1].astype(str),
            "feature": np.linspace(0.0, 1.0, len(timestamps)),
            "target": np.linspace(10.0, 20.0, len(timestamps)),
        }
    )
    test = train.drop(columns=["target"]).iloc[:8].copy()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path, train


def test_temporal_contract_is_strict_forward_chaining(tmp_path: Path) -> None:
    train_path, test_path, train = _write_temporal_csvs(tmp_path)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="target",
        output_dir=tmp_path,
        id_col="id",
        n_folds=5,
        task_type="time_series_forecasting",
    )
    canonical = load_canonical_data(tmp_path)
    folds = canonical["folds"]
    order = canonical["temporal_order"]
    eligible = canonical["oof_eligible_mask"]
    splits = canonical["temporal_splits"]
    metadata = result["metadata"]

    assert metadata["cv_strategy"] == "temporal_forward_chaining"
    assert metadata["is_temporal"] is True
    assert metadata["temporal_cv"]["temporal_col"] == "date"
    assert metadata["temporal_cv"]["strict_train_before_validation"] is True
    assert metadata["temporal_cv"]["warmup_rows"] > 0
    assert len(folds) == len(train)
    assert np.array_equal(folds >= 0, eligible)
    assert np.all(folds[~eligible] == -1)

    validation_counts = np.zeros(len(train), dtype=int)
    previous_train: set[int] = set()
    for fold_idx in range(5):
        train_idx = splits[f"train_{fold_idx}"]
        val_idx = splits[f"validation_{fold_idx}"]
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert order[train_idx].max() < order[val_idx].min()
        assert np.all(folds[val_idx] == fold_idx)
        assert previous_train.issubset(set(train_idx))
        previous_train = set(train_idx)
        validation_counts[val_idx] += 1

    assert np.all(validation_counts[eligible] == 1)
    assert np.all(validation_counts[~eligible] == 0)
    assert int(eligible.sum()) + int((~eligible).sum()) == len(train)

    metadata_on_disk = json.loads(
        (tmp_path / "canonical" / "metadata.json").read_text()
    )
    cutoffs = metadata_on_disk["temporal_cv"]["fold_cutoffs"]
    assert len(cutoffs) == 5
    assert all(
        cutoff["train_time_max"] < cutoff["validation_time_min"]
        for cutoff in cutoffs
    )


def test_temporal_contract_accepts_explicit_numeric_order(
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "id": np.arange(24),
            "sequence": np.repeat(np.arange(12), 2),
            "x": np.arange(24) * 0.1,
            "target": np.arange(24) * 0.5,
        }
    )
    test = train.drop(columns=["target"]).iloc[:4]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="target",
        output_dir=tmp_path,
        id_col="id",
        n_folds=3,
        task_type="forecasting",
        column_contract={"order_col": "sequence"},
    )

    temporal = result["metadata"]["temporal_cv"]
    assert temporal["temporal_col"] == "sequence"
    assert temporal["evidence_source"] == "public_column_contract"
    assert temporal["value_type"] == "numeric_order"


def test_temporal_contract_fails_closed_without_time_evidence(
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "id": np.arange(20),
            "feature": np.arange(20),
            "target": np.arange(20, dtype=float),
        }
    )
    test = train.drop(columns=["target"]).iloc[:5]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="Temporal task requires"):
        prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="target",
            output_dir=tmp_path,
            id_col="id",
            n_folds=3,
            task_type="time_series_forecasting",
        )

    assert not (tmp_path / "canonical" / "folds.npy").exists()


def test_mlebench_node_raises_when_temporal_contract_is_unprovable(
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "id": np.arange(20),
            "feature": np.arange(20),
            "target": np.arange(20, dtype=float),
        }
    )
    test = train.drop(columns=["target"]).iloc[:5]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    with pytest.raises(
        RuntimeError,
        match="cannot proceed without a trustworthy forward-chaining",
    ):
        canonical_data_preparation_node(
            {
                "working_directory": str(tmp_path),
                "data_files": {
                    "train": str(train_path),
                    "test": str(test_path),
                    "data_type": "tabular",
                },
                "target_col": "target",
                "domain_detected": "time_series_forecasting",
                "run_mode": "mlebench",
                "fast_mode": False,
            }
        )


def test_temporal_contract_fails_closed_on_ambiguous_time_axes(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    train = pd.DataFrame(
        {
            "id": np.arange(20),
            "date": dates.astype(str),
            "timestamp": (dates + pd.Timedelta(hours=1)).astype(str),
            "target": np.arange(20, dtype=float),
        }
    )
    test = train.drop(columns=["target"]).iloc[:5]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="multiple plausible time axes"):
        prepare_canonical_data(
            train_path=train_path,
            test_path=test_path,
            target_col="target",
            output_dir=tmp_path,
            id_col="id",
            n_folds=3,
            task_type="time_series_forecasting",
        )


def test_normal_tabular_regression_keeps_complete_kfold_contract(
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "id": np.arange(30),
            "x": np.arange(30),
            "target": np.tile([0.0, 1.0, 2.0], 10),
        }
    )
    test = train.drop(columns=["target"]).iloc[:5]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="target",
        output_dir=tmp_path,
        id_col="id",
        n_folds=3,
        task_type="tabular_regression",
    )

    folds = np.load(result["folds_path"])
    assert result["metadata"]["cv_strategy"] == "kfold"
    assert result["metadata"]["is_classification"] is False
    assert result["metadata"]["is_temporal"] is False
    assert set(folds.tolist()) == {0, 1, 2}
    assert not (tmp_path / "canonical" / "temporal_splits.npz").exists()


def test_temporal_oof_validation_scores_only_eligible_rows(
    tmp_path: Path,
) -> None:
    train_path, test_path, _ = _write_temporal_csvs(tmp_path)
    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="target",
        output_dir=tmp_path,
        id_col="id",
        n_folds=5,
        task_type="time_series_forecasting",
    )
    eligible = np.load(result["oof_eligible_mask_path"])
    train_ids = np.load(result["train_ids_path"])

    oof = np.full(len(eligible), np.nan)
    oof[eligible] = np.linspace(10.0, 20.0, int(eligible.sum()))
    valid, issues = validate_oof_alignment(
        oof,
        tmp_path,
        model_train_ids=train_ids,
    )
    assert valid, issues

    oof[~eligible] = 0.0
    valid, issues = validate_oof_alignment(
        oof,
        tmp_path,
        model_train_ids=train_ids,
    )
    assert not valid
    assert any("warm-up" in issue for issue in issues)


def test_strict_artifact_validation_allows_only_masked_warmup_nan(
    tmp_path: Path,
) -> None:
    train_path, test_path, _ = _write_temporal_csvs(tmp_path)
    result = prepare_canonical_data(
        train_path=train_path,
        test_path=test_path,
        target_col="target",
        output_dir=tmp_path,
        id_col="id",
        n_folds=5,
        task_type="time_series_forecasting",
    )
    eligible = np.load(result["oof_eligible_mask_path"])
    train_ids = np.load(result["train_ids_path"])
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    oof = np.full(len(eligible), np.nan)
    oof[eligible] = np.linspace(1.0, 2.0, int(eligible.sum()))
    np.save(models_dir / "oof_temporal.npy", oof)
    np.save(models_dir / "test_temporal.npy", np.linspace(2.0, 3.0, 8))
    np.save(models_dir / "train_ids_temporal.npy", train_ids)
    np.save(models_dir / "test_ids_temporal.npy", np.arange(8))

    validation = validate_model_artifacts(
        working_dir=tmp_path,
        component_name="temporal",
        expected_n_train=len(train_ids),
        expected_n_test=8,
        expected_train_ids=train_ids,
        expected_test_ids=np.arange(8),
        problem_type="time_series_forecasting",
        config=StrictValidationConfig(
            strict_mode=True,
            require_train_ids=True,
            require_test_ids=True,
        ),
    )
    assert validation.is_valid, validation.errors

    oof[~eligible] = 1.0
    np.save(models_dir / "oof_temporal.npy", oof)
    invalid = validate_model_artifacts(
        working_dir=tmp_path,
        component_name="temporal",
        expected_n_train=len(train_ids),
        expected_n_test=8,
        problem_type="time_series_forecasting",
        config=StrictValidationConfig(strict_mode=True),
    )
    assert not invalid.is_valid
    assert any("warm-up" in error for error in invalid.errors)


def test_cv_prompt_forbids_fold_complement_for_temporal_contract() -> None:
    prompt = "\n".join(build_cv_instructions("/tmp/run", "model"))
    assert "iter_canonical_cv_splits()" in prompt
    assert "folds != fold` leaks future rows" in prompt
    assert "CANONICAL_OOF_ELIGIBLE_MASK" in prompt


def test_legacy_canonical_prompt_uses_audited_splitter_and_artifact_helper(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "metadata.json").write_text(
        json.dumps(
            {
                "canonical_rows": 12,
                "n_folds": 3,
                "id_col": "id",
            }
        ),
        encoding="utf-8",
    )

    prompt = get_canonical_data_instructions(tmp_path)

    assert "iter_canonical_cv_splits()" in prompt
    assert "folds != fold_idx" not in prompt
    assert "save_component_artifacts(" in prompt
    assert 'np.save("models/oof_' not in prompt


def test_oof_scoring_masks_temporal_warmup(tmp_path: Path) -> None:
    y_true = np.linspace(0.0, 1.0, 12)
    eligible = np.array([False, False, True, True, True, True] * 2)
    oof = np.full(12, np.nan)
    oof[eligible] = y_true[eligible] + 0.1
    oof_path = tmp_path / "oof_temporal.npy"
    np.save(oof_path, oof)

    score = compute_oof_score(
        oof_path,
        y_true,
        metric_name="rmse",
        oof_eligible_mask=eligible,
    )
    assert score == pytest.approx(0.1)

    oof[~eligible] = y_true[~eligible]
    np.save(oof_path, oof)
    assert np.isinf(
        compute_oof_score(
            oof_path,
            y_true,
            metric_name="rmse",
            oof_eligible_mask=eligible,
        )
    )


def test_temporal_calibration_preserves_masked_warmup(tmp_path: Path) -> None:
    eligible = np.ones(60, dtype=bool)
    eligible[:10] = False
    folds = np.full(60, -1, dtype=int)
    for fold_idx in range(5):
        folds[10 + fold_idx * 10 : 20 + fold_idx * 10] = fold_idx
    y_true = np.tile([0, 1], 30)
    oof = np.full(60, np.nan)
    oof[eligible] = np.tile([0.25, 0.75], 25)
    oof_path = tmp_path / "oof_temporal.npy"
    np.save(oof_path, oof)

    result = calibrate_oof_predictions(
        oof_path,
        y_true,
        method="platt",
        cv_folds=folds,
        save_both=True,
        oof_eligible_mask=eligible,
    )

    calibrated = np.load(tmp_path / "oof_cal_temporal.npy")
    assert result.method == "platt"
    assert np.isnan(calibrated[~eligible]).all()
    assert np.isfinite(calibrated[eligible]).all()
