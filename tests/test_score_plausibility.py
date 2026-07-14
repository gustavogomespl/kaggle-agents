"""Guards against implausible CV scores poisoning hill-climbing.

Regression tests for a run where a component printed a broken
"Final Validation Performance: 0.000000" (RMSE): the hill-climb accepted it as
the new best, and every later real component was rejected against the
impossible baseline while the rejected components' artifacts stayed on disk.
"""

from types import SimpleNamespace

import numpy as np

from kaggle_agents.agents.developer.validation import (
    ValidationMixin,
    quarantine_component_artifacts,
)
from kaggle_agents.agents.ensemble.prediction_pairs import find_prediction_pairs
from kaggle_agents.core.state import AblationComponent, CompetitionInfo


class _Validator(ValidationMixin):
    pass


COMPONENT = AblationComponent("model_x", "model", "train")


def _exec_result(stdout: str) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout)


class TestScorePlausibility:
    def test_zero_or_negative_lower_better_is_implausible(self):
        assert ValidationMixin._is_score_implausible(0.0, "rmse") is True
        assert ValidationMixin._is_score_implausible(-0.1, "mae") is True

    def test_perfect_higher_better_is_plausible(self):
        # Saturated comps legitimately reach AUC ~1.0 (higher is better).
        assert ValidationMixin._is_score_implausible(1.0, "auc") is False
        assert ValidationMixin._is_score_implausible(0.0, "accuracy") is False
        assert ValidationMixin._is_score_implausible(None, "rmse") is False

    def test_zero_rmse_component_rejected(self):
        keep, score = _Validator()._validate_component_improvement(
            COMPONENT,
            _exec_result("Final Validation Performance: 0.000000\n"),
            {"competition_info": CompetitionInfo("demo", "", "rmse", "regression")},
        )
        assert keep is False
        assert score is None

    def test_real_rmse_accepted_when_baseline_not_poisoned(self):
        # After an implausible rejection the baseline stays unset, so the next
        # real component must be accepted instead of losing to a fake 0.0.
        keep, score = _Validator()._validate_component_improvement(
            COMPONENT,
            _exec_result("Final Validation Performance: 0.1198\n"),
            {"competition_info": CompetitionInfo("demo", "", "rmse", "regression")},
        )
        assert keep is True
        assert score == 0.1198

    def test_perfect_auc_component_accepted(self):
        keep, score = _Validator()._validate_component_improvement(
            COMPONENT,
            _exec_result("Final Validation Performance: 1.0\n"),
            {
                "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
                "baseline_cv_score": 0.98,
            },
        )
        assert keep is True
        assert score == 1.0


class TestQuarantineArtifacts:
    def test_quarantine_hides_pairs_from_ensemble_glob(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        for prefix in ("oof_", "test_", "test_ids_", "train_ids_"):
            np.save(models / f"{prefix}bad.npy", np.zeros(3))
        np.save(models / "oof_good.npy", np.zeros(3))
        np.save(models / "test_good.npy", np.zeros(3))

        renamed = quarantine_component_artifacts(models, "bad")

        assert sorted(renamed) == [
            "oof_bad.npy",
            "test_bad.npy",
            "test_ids_bad.npy",
            "train_ids_bad.npy",
        ]
        assert set(find_prediction_pairs(models)) == {"good"}
        assert (models / "rejected_oof_bad.npy").exists()

    def test_quarantine_overwrites_previous_quarantine(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "rejected_oof_bad.npy", np.zeros(1))
        np.save(models / "oof_bad.npy", np.ones(2))

        renamed = quarantine_component_artifacts(models, "bad")

        assert renamed == ["oof_bad.npy"]
        assert len(np.load(models / "rejected_oof_bad.npy")) == 2

    def test_quarantine_noop_without_artifacts(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        assert quarantine_component_artifacts(models, "ghost") == []
