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
    def test_zero_lower_better_needs_a_trusted_recompute(self):
        # A stdout-declared 0.0 loss is almost always a broken validation calc
        # (the incident that poisoned hill-climbing); only a score recomputed
        # from canonical OOF artifacts may legitimately be exactly 0.0.
        assert ValidationMixin._is_score_implausible(0.0, "rmse") is True
        assert ValidationMixin._is_score_implausible(0.0, "rmse", trusted=True) is False
        assert ValidationMixin._is_score_implausible(-0.1, "mae") is True
        assert (
            ValidationMixin._is_score_implausible(-0.1, "mae", trusted=True) is True
        )
        assert ValidationMixin._is_score_implausible(float("inf"), "rmse") is True
        assert ValidationMixin._is_score_implausible(float("nan"), "auc") is True

    def test_perfect_higher_better_is_plausible(self):
        # Saturated comps legitimately reach AUC ~1.0 (higher is better).
        assert ValidationMixin._is_score_implausible(1.0, "auc") is False
        assert ValidationMixin._is_score_implausible(0.0, "accuracy") is False
        assert ValidationMixin._is_score_implausible(None, "rmse") is False

    def test_zero_rmse_from_stdout_is_rejected(self):
        # Default (kaggle) mode promotes stdout-declared scores, so the exact
        # failure from the incident log must reject before touching baselines.
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
        quarantined = list((models / ".rejected" / "bad").glob("*/oof_bad.npy"))
        assert len(quarantined) == 1

    def test_quarantine_preserves_previous_rejections(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "oof_bad.npy", np.ones(2))

        first = quarantine_component_artifacts(models, "bad")
        np.save(models / "oof_bad.npy", np.ones(3))
        second = quarantine_component_artifacts(models, "bad")

        assert first == ["oof_bad.npy"]
        assert second == ["oof_bad.npy"]
        quarantined = list((models / ".rejected" / "bad").glob("*/oof_bad.npy"))
        assert sorted(len(np.load(path)) for path in quarantined) == [2, 3]

    def test_quarantine_noop_without_artifacts(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        assert quarantine_component_artifacts(models, "ghost") == []
