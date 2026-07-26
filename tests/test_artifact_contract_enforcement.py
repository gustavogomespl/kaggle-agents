"""Execution-time artifact contract must match the promotion-time contract.

The trusted-OOF gate needs ``train_ids_<name>.npy`` to verify row alignment;
if the executor does not enforce it, components train successfully and then
die silently at promotion with no retry loop (aerial-cactus smoke run).
"""

from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.agents.developer.agent import (
    _expected_model_artifacts,
    _has_combinable_model_predictions,
)
from kaggle_agents.agents.developer.retry import (
    RetryMixin,
    _maybe_add_artifact_hint,
)
from kaggle_agents.core.state import AblationComponent, DevelopmentResult


class _Retry(RetryMixin):
    pass


@pytest.fixture(autouse=True)
def _default_oof_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_AGENTS_REQUIRE_OOF", raising=False)


def test_model_artifacts_require_train_ids_when_canonical_exists(
    tmp_path: Path,
) -> None:
    component = AblationComponent("candidate", "model", "train")

    expected = _expected_model_artifacts(component, tmp_path)
    assert expected == [
        "models/oof_candidate.npy",
        "models/test_candidate.npy",
    ]

    (tmp_path / "canonical").mkdir()
    (tmp_path / "canonical" / "metadata.json").write_text("{}", encoding="utf-8")
    expected = _expected_model_artifacts(component, tmp_path)
    assert "models/train_ids_candidate.npy" in expected
    assert "models/test_ids_candidate.npy" not in expected


def test_mlebench_model_artifacts_match_strict_validation(tmp_path: Path) -> None:
    # Strict post-acceptance validation requires train_ids AND test_ids in
    # mlebench mode regardless of canonical presence; the executor must
    # enforce the same set or components die after training with no retry.
    component = AblationComponent("candidate", "model", "train")

    expected = _expected_model_artifacts(component, tmp_path, "mlebench")
    assert expected == [
        "models/oof_candidate.npy",
        "models/test_candidate.npy",
        "models/train_ids_candidate.npy",
        "models/test_ids_candidate.npy",
    ]


def test_non_model_components_have_no_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessing = AblationComponent("cache_images", "preprocessing", "resize")
    assert _expected_model_artifacts(preprocessing, tmp_path) is None

    monkeypatch.setenv("KAGGLE_AGENTS_REQUIRE_OOF", "0")
    model = AblationComponent("candidate", "model", "train")
    assert _expected_model_artifacts(model, tmp_path) is None


def _cached_state(tmp_path: Path, code: str) -> dict:
    return {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "development_results": [
            DevelopmentResult(code=code, success=True, execution_time=1.0)
        ],
    }


def test_cached_model_reuse_requires_oof_artifact_on_disk(tmp_path: Path) -> None:
    # The aerial-cactus run re-planned "image_augmentation_preprocessing" as a
    # MODEL component; the name-keyed cache reused the old preprocessing run
    # (no OOF evidence) and the candidate died at promotion.
    component = AblationComponent("candidate", "model", "train")
    state = _cached_state(tmp_path, "print('candidate')")

    assert _Retry()._should_skip_component(component, state) is None

    (tmp_path / "models").mkdir()
    np.save(tmp_path / "models" / "oof_candidate.npy", np.array([0.1, 0.9]))
    reused = _Retry()._should_skip_component(component, state)
    assert reused is not None
    assert reused.success is True


def test_cached_non_model_reuse_is_unchanged(tmp_path: Path) -> None:
    component = AblationComponent("cache_images", "preprocessing", "resize")
    state = _cached_state(tmp_path, "print('cache_images')")

    reused = _Retry()._should_skip_component(component, state)
    assert reused is not None


def test_ensemble_component_needs_combinable_predictions(tmp_path: Path) -> None:
    assert _has_combinable_model_predictions({}, tmp_path) is False
    assert (
        _has_combinable_model_predictions({"oof_availability": {"m": False}}, tmp_path)
        is False
    )
    assert (
        _has_combinable_model_predictions({"oof_availability": {"m": True}}, tmp_path)
        is True
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    np.save(models_dir / "rejected_oof_bad.npy", np.array([0.5]))
    assert _has_combinable_model_predictions({}, tmp_path) is False

    np.save(models_dir / "oof_good.npy", np.array([0.5]))
    assert _has_combinable_model_predictions({}, tmp_path) is True


def test_missing_artifact_error_gets_reuse_hint() -> None:
    hinted = _maybe_add_artifact_hint(
        "Missing expected artifacts: models/test_candidate.npy"
    )
    assert "Do NOT retrain from scratch" in hinted
    assert "allow_pickle=False" in hinted

    untouched = _maybe_add_artifact_hint("ValueError: shapes do not match")
    assert untouched == "ValueError: shapes do not match"


def _write_csv(path: Path, ids: list, target: list | None = None) -> None:
    import pandas as pd

    data = {"id": ids}
    if target is not None:
        data["has_cactus"] = target
    pd.DataFrame(data).to_csv(path, index=False)


def test_canonical_string_train_ids_survive_unpickled_resave(tmp_path: Path) -> None:
    # String IDs from pandas arrive as object dtype; stored raw they poison
    # the whole artifact chain (np.save(..., allow_pickle=False) crashes and
    # pickled saves are refused by the trusted scorer).
    from kaggle_agents.utils.data_contract import prepare_canonical_data

    _write_csv(
        tmp_path / "train.csv",
        [f"img_{i}.jpg" for i in range(10)],
        [i % 2 for i in range(10)],
    )
    _write_csv(tmp_path / "test.csv", ["t_0.jpg", "t_1.jpg"])

    prepare_canonical_data(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        target_col="has_cactus",
        output_dir=tmp_path,
        id_col="id",
        n_folds=2,
    )

    saved = np.load(tmp_path / "canonical" / "train_ids.npy", allow_pickle=False)
    assert saved.dtype.kind == "U"
    # The exact operation generated code performs with the loaded IDs.
    np.save(tmp_path / "resaved.npy", saved, allow_pickle=False)


def test_canonical_integer_train_ids_keep_numeric_dtype(tmp_path: Path) -> None:
    from kaggle_agents.utils.data_contract import prepare_canonical_data

    _write_csv(tmp_path / "train.csv", list(range(10)), [i % 2 for i in range(10)])
    _write_csv(tmp_path / "test.csv", [100, 101])

    prepare_canonical_data(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        target_col="has_cactus",
        output_dir=tmp_path,
        id_col="id",
        n_folds=2,
    )

    saved = np.load(tmp_path / "canonical" / "train_ids.npy", allow_pickle=False)
    assert saved.dtype.kind in "iu"


def test_preamble_normalizes_legacy_object_train_ids() -> None:
    import inspect

    from kaggle_agents.agents.developer import code_generator

    src = inspect.getsource(code_generator)
    assert "CANONICAL_TRAIN_IDS.dtype == object" in src
