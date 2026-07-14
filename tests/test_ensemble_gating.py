"""Ensemble agent gating: accepted-only pairs and no unscored overwrite.

Regression tests for a run where the final ensemble averaged the test
predictions of hill-climb REJECTED components (their .npy artifacts were still
on disk) and unconditionally overwrote submission.csv with an average that had
no validation score, replacing the scored best single model.
"""

import numpy as np
import pandas as pd

from kaggle_agents.agents.ensemble.agent import EnsembleAgent
from kaggle_agents.core.state import CompetitionInfo


def _make_pair(models_dir, name, n=2):
    np.save(models_dir / f"oof_{name}.npy", np.linspace(0.0, 1.0, n))
    np.save(models_dir / f"test_{name}.npy", np.linspace(0.0, 1.0, n))


def _base_state(tmp_path, **extra):
    (tmp_path / "models").mkdir(exist_ok=True)
    sample = tmp_path / "sample_submission.csv"
    if not sample.exists():
        pd.DataFrame({"id": [1, 2], "value": [0.0, 0.0]}).to_csv(sample, index=False)
    state = {
        "working_directory": str(tmp_path),
        "sample_submission_path": str(sample),
        "competition_info": CompetitionInfo("demo", "", "rmse", "regression"),
        "current_iteration": 0,
    }
    state.update(extra)
    return state


class TestAcceptedPairFilter:
    def test_non_accepted_pairs_dropped(self, tmp_path, monkeypatch):
        state = _base_state(tmp_path, oof_available_kept=True)
        models = tmp_path / "models"
        _make_pair(models, "kept")
        _make_pair(models, "rolled_back")

        seen = {}
        monkeypatch.setattr(EnsembleAgent, "_try_oof_stacking", lambda self, **kw: None)

        def _capture(self, pairs, *args, **kwargs):
            seen["pairs"] = set(pairs)
            return True

        monkeypatch.setattr(EnsembleAgent, "_ensemble_from_predictions", _capture)

        result = EnsembleAgent()(state)

        assert result.get("ensemble_created") is True
        assert seen["pairs"] == {"kept"}

    def test_all_pairs_kept_without_state_keys(self, tmp_path, monkeypatch):
        # Resumed/legacy states have no oof_available_* keys; the glob results
        # must be kept (disk quarantine already hides rejected files).
        state = _base_state(tmp_path)
        models = tmp_path / "models"
        _make_pair(models, "a")
        _make_pair(models, "b")

        seen = {}
        monkeypatch.setattr(EnsembleAgent, "_try_oof_stacking", lambda self, **kw: None)

        def _capture(self, pairs, *args, **kwargs):
            seen["pairs"] = set(pairs)
            return True

        monkeypatch.setattr(EnsembleAgent, "_ensemble_from_predictions", _capture)

        EnsembleAgent()(state)

        assert seen["pairs"] == {"a", "b"}

    def test_filter_to_zero_pairs_restores_best(self, tmp_path):
        state = _base_state(tmp_path, oof_available_other=True)
        _make_pair(tmp_path / "models", "rolled_back")
        best_content = "id,value\n1,0.9\n2,0.8\n"
        (tmp_path / "submission_best.csv").write_text(best_content, encoding="utf-8")

        result = EnsembleAgent()(state)

        assert result["ensemble_skipped"] is True
        assert result["skip_reason"] == "no_prediction_pairs"
        restored = pd.read_csv(tmp_path / "submission.csv")
        assert restored["value"].tolist() == [0.9, 0.8]


class TestUnscoredFallbackGate:
    def test_unscored_average_blocked_by_scored_best(self, tmp_path, monkeypatch):
        state = _base_state(tmp_path, best_single_model_score=0.095)
        models = tmp_path / "models"
        _make_pair(models, "a")
        _make_pair(models, "b")
        (tmp_path / "submission_best.csv").write_text(
            "id,value\n1,0.9\n2,0.8\n", encoding="utf-8"
        )
        (tmp_path / "submission.csv").write_text(
            "id,value\n1,0.1\n2,0.1\n", encoding="utf-8"
        )

        monkeypatch.setattr(EnsembleAgent, "_try_oof_stacking", lambda self, **kw: None)

        def _must_not_run(self, *args, **kwargs):
            raise AssertionError("unscored average must not overwrite a scored best")

        monkeypatch.setattr(EnsembleAgent, "_ensemble_from_predictions", _must_not_run)

        result = EnsembleAgent()(state)

        assert result["ensemble_skipped"] is True
        assert result["skip_reason"] == "unscored_fallback_kept_scored_best"
        restored = pd.read_csv(tmp_path / "submission.csv")
        assert restored["value"].tolist() == [0.9, 0.8]

    def test_unscored_average_runs_without_scored_best(self, tmp_path, monkeypatch):
        # Cold start: no scored best exists, so the simple average is still
        # better than no submission at all.
        state = _base_state(tmp_path)
        models = tmp_path / "models"
        _make_pair(models, "a")
        _make_pair(models, "b")
        (tmp_path / "submission_best.csv").write_text(
            "id,value\n1,0.9\n2,0.8\n", encoding="utf-8"
        )

        called = {}
        monkeypatch.setattr(EnsembleAgent, "_try_oof_stacking", lambda self, **kw: None)

        def _capture(self, *args, **kwargs):
            called["ran"] = True
            return True

        monkeypatch.setattr(EnsembleAgent, "_ensemble_from_predictions", _capture)

        result = EnsembleAgent()(state)

        assert called.get("ran") is True
        assert result.get("ensemble_created") is True
