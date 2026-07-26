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

    untouched = _maybe_add_artifact_hint("ValueError: shapes do not match")
    assert untouched == "ValueError: shapes do not match"
