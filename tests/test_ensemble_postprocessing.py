"""Tests for metric-aware postprocessing and the OOF-stacking route guards."""

import json

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.agents.ensemble.agent import EnsembleAgent
from kaggle_agents.agents.ensemble.fallback import recover_from_checkpoints
from kaggle_agents.agents.ensemble.postprocessing import (
    apply_rounding,
    labels_from_oof_tuning,
    metric_label_kind,
    tune_binary_threshold,
    tune_qwk_rounding,
)
from kaggle_agents.agents.ensemble.stacking import load_cv_folds
from kaggle_agents.agents.ensemble.submission import format_ensemble_predictions
from kaggle_agents.prompts.templates.developer.component_guidance import (
    COMPONENT_GUIDANCE,
)


class TestMetricLabelKind:
    def test_qwk_metrics(self):
        assert metric_label_kind("quadratic_weighted_kappa") == "qwk"
        assert metric_label_kind("qwk") == "qwk"
        assert metric_label_kind("kappa") == "qwk"

    def test_threshold_metrics(self):
        assert metric_label_kind("accuracy") == "threshold"
        assert metric_label_kind("f1") == "threshold"
        assert metric_label_kind("mcc") == "threshold"

    def test_probability_metrics_need_nothing(self):
        assert metric_label_kind("auc") is None
        assert metric_label_kind("log_loss") is None
        assert metric_label_kind("rmse") is None
        assert metric_label_kind(None) is None


class TestTuneBinaryThreshold:
    def test_tuned_never_worse_than_baseline(self):
        rng = np.random.default_rng(42)
        y = (rng.random(1000) < 0.1).astype(int)  # 10% positives
        # Scores correlated with y but shifted: optimal threshold well below 0.5
        oof = np.clip(0.15 * y + rng.normal(0.12, 0.05, 1000), 0, 1)

        threshold, tuned, baseline = tune_binary_threshold(oof, y, "f1")
        assert 0.0 < threshold < 1.0
        assert tuned >= baseline

    def test_shifted_scores_move_threshold(self):
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.5).astype(int)
        # Well-separated but compressed into [0.0, 0.4]: best accuracy cut ~0.2
        oof = np.where(y == 1, 0.30, 0.10) + rng.normal(0, 0.03, 2000)

        threshold, tuned, baseline = tune_binary_threshold(oof, y, "accuracy")
        assert threshold < 0.45
        assert tuned > baseline  # 0.5 classifies everything as 0 -> ~50% acc

    def test_balanced_calibrated_keeps_half(self):
        rng = np.random.default_rng(1)
        y = (rng.random(4000) < 0.5).astype(int)
        oof = np.clip(np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.1, 4000), 0, 1)

        threshold, tuned, baseline = tune_binary_threshold(oof, y, "accuracy")
        assert tuned >= baseline
        assert abs(threshold - 0.5) < 0.25

    def test_explicit_macro_average_is_used(self):
        y = np.array([0, 0, 0, 1])
        oof = np.full(4, 0.1)

        _threshold, _tuned, baseline = tune_binary_threshold(oof, y, "f1_macro")

        assert baseline == pytest.approx(3 / 7)

    def test_label_scale_threshold_is_not_limited_to_probability_range(self):
        y = np.array([1, 2, 1, 2])
        oof = np.array([1.1, 1.9, 1.2, 1.8])

        threshold, tuned, _baseline = tune_binary_threshold(
            oof, y, "quadratic_weighted_kappa"
        )

        assert 1.2 < threshold < 1.8
        assert tuned == pytest.approx(1.0)


class TestQwkRounding:
    def test_tuned_never_worse_than_plain_rounding(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 5, 1500)
        # Biased continuous predictions (systematic +0.4 shift)
        oof = y + 0.4 + rng.normal(0, 0.35, 1500)

        boundaries, tuned, baseline, classes = tune_qwk_rounding(oof, y)
        assert list(classes) == [0, 1, 2, 3, 4]
        assert len(boundaries) == 4
        assert tuned >= baseline

    def test_apply_rounding_maps_to_original_labels(self):
        classes = np.array([1, 2, 3, 5])  # non-contiguous labels
        preds = np.array([0.2, 1.7, 2.9, 10.0])
        labels = apply_rounding(preds, [1.5, 2.5, 4.0], classes)
        assert labels.tolist() == [1, 2, 3, 5]

    def test_binary_degenerates_to_threshold(self):
        rng = np.random.default_rng(3)
        y = (rng.random(500) < 0.5).astype(int)
        oof = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.1, 500), 0, 1)

        boundaries, tuned, _baseline, _classes = tune_qwk_rounding(oof, y)
        assert len(boundaries) == 1
        assert tuned >= 0.0


class TestLabelsFromOofTuning:
    def test_qwk_path_returns_original_class_labels(self):
        rng = np.random.default_rng(11)
        y = rng.integers(1, 6, 800)  # labels 1..5 (not 0-indexed)
        oof = y + rng.normal(0, 0.3, 800)
        test = y[:100] + rng.normal(0, 0.3, 100)

        labels, info = labels_from_oof_tuning(test, oof, y, "quadratic_weighted_kappa")
        assert info["rule"] == "qwk_rounding"
        assert set(np.unique(labels)).issubset({1, 2, 3, 4, 5})
        assert info["oof_score_tuned"] >= info["oof_score_baseline"]

    def test_threshold_path(self):
        rng = np.random.default_rng(13)
        y = (rng.random(600) < 0.3).astype(int)
        oof = np.clip(0.2 * y + rng.normal(0.15, 0.05, 600), 0, 1)
        test = np.clip(rng.normal(0.2, 0.08, 50), 0, 1)

        labels, info = labels_from_oof_tuning(test, oof, y, "accuracy")
        assert info["rule"] == "binary_threshold"
        assert set(np.unique(labels)).issubset({0, 1})

    def test_threshold_path_preserves_string_labels(self):
        y = np.array(["no", "yes", "no", "yes"])
        oof = np.array([0.1, 0.9, 0.2, 0.8])

        labels, info = labels_from_oof_tuning(
            np.array([0.8, 0.2]), oof, y, "accuracy"
        )

        assert info["rule"] == "binary_threshold"
        assert labels.tolist() == ["yes", "no"]

    def test_threshold_path_preserves_nonzero_numeric_labels(self):
        y = np.array([1, 2, 1, 2])
        oof = np.array([0.1, 0.9, 0.2, 0.8])

        labels, _info = labels_from_oof_tuning(
            np.array([0.8, 0.2]), oof, y, "accuracy"
        )

        assert labels.tolist() == [2, 1]


class TestFormatEnsemblePredictionsTuning:
    def _sample_sub(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"id": range(n), "target": np.zeros(n, dtype=int)})

    def test_without_oof_keeps_legacy_half_threshold(self):
        preds = np.array([0.4, 0.6, 0.49, 0.51])
        out = format_ensemble_predictions(
            preds, self._sample_sub(4), "classification", "accuracy"
        )
        assert out.tolist() == [0, 1, 0, 1]

    def test_with_oof_uses_tuned_threshold(self):
        rng = np.random.default_rng(5)
        y = (rng.random(2000) < 0.5).astype(int)
        # Scores compressed low: tuned cut ~0.2, so 0.3 must become positive
        oof = np.where(y == 1, 0.30, 0.10) + rng.normal(0, 0.03, 2000)
        preds = np.array([0.05, 0.30, 0.35])

        out = format_ensemble_predictions(
            preds,
            self._sample_sub(3),
            "classification",
            "accuracy",
            oof_preds=oof,
            y_true=y,
        )
        assert out.tolist() == [0, 1, 1]  # 0.5 threshold would give [0, 0, 0]

    def test_probability_metric_unchanged(self):
        preds = np.array([0.4, 0.6])
        sample = pd.DataFrame({"id": [0, 1], "target": [0.5, 0.5]})
        out = format_ensemble_predictions(preds, sample, "classification", "auc")
        assert np.allclose(out, preds)

    def test_multiclass_argmax_unchanged(self):
        preds = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        out = format_ensemble_predictions(
            preds, self._sample_sub(2), "classification", "accuracy"
        )
        assert out.tolist() == [1, 0]

    def test_oof_length_mismatch_falls_back(self):
        preds = np.array([0.4, 0.6])
        out = format_ensemble_predictions(
            preds,
            self._sample_sub(2),
            "classification",
            "accuracy",
            oof_preds=np.array([0.1, 0.2, 0.3]),
            y_true=np.array([0, 1]),  # mismatched lengths -> legacy path
        )
        assert out.tolist() == [0, 1]

    def test_string_labels_are_preserved_in_submission(self):
        y = np.array(["negative", "positive", "negative", "positive"])
        oof = np.array([0.1, 0.9, 0.2, 0.8])
        sample = pd.DataFrame({"id": [1, 2], "target": ["negative", "negative"]})

        out = format_ensemble_predictions(
            np.array([0.8, 0.2]),
            sample,
            "classification",
            "accuracy",
            oof_preds=oof,
            y_true=y,
        )

        assert out.tolist() == ["positive", "negative"]


class TestTryOofStackingGuards:
    """The stacking route must degrade gracefully to the mean path (return None)."""

    @pytest.fixture
    def agent(self):
        return EnsembleAgent()

    def test_requires_two_pairs(self, agent, temp_data_dir):
        outcome = agent._try_oof_stacking(
            state={},
            prediction_pairs={"only_one": (temp_data_dir / "a.npy", temp_data_dir / "b.npy")},
            models_dir=temp_data_dir,
            sample_path=temp_data_dir / "sample_submission.csv",
            output_path=temp_data_dir / "submission.csv",
            problem_type="binary_classification",
            metric_name="auc",
            current_iteration=0,
        )
        assert outcome is None

    def test_requires_sample_submission(self, agent, temp_data_dir):
        pairs = {
            "m1": (temp_data_dir / "oof_m1.npy", temp_data_dir / "test_m1.npy"),
            "m2": (temp_data_dir / "oof_m2.npy", temp_data_dir / "test_m2.npy"),
        }
        outcome = agent._try_oof_stacking(
            state={},
            prediction_pairs=pairs,
            models_dir=temp_data_dir,
            sample_path=temp_data_dir / "missing.csv",
            output_path=temp_data_dir / "submission.csv",
            problem_type="binary_classification",
            metric_name="auc",
            current_iteration=0,
        )
        assert outcome is None

    def test_requires_canonical_y(self, agent, temp_data_dir):
        (temp_data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n")
        pairs = {
            "m1": (temp_data_dir / "oof_m1.npy", temp_data_dir / "test_m1.npy"),
            "m2": (temp_data_dir / "oof_m2.npy", temp_data_dir / "test_m2.npy"),
        }
        outcome = agent._try_oof_stacking(
            state={"canonical_contract": None},
            prediction_pairs=pairs,
            models_dir=temp_data_dir,
            sample_path=temp_data_dir / "sample_submission.csv",
            output_path=temp_data_dir / "submission.csv",
            problem_type="binary_classification",
            metric_name="auc",
            current_iteration=0,
        )
        assert outcome is None

    def test_load_canonical_training_data(self, agent, temp_data_dir):
        y = np.array([0, 1, 0, 1])
        train_ids = np.array(["a", "b", "c", "d"])
        y_path = temp_data_dir / "y.npy"
        ids_path = temp_data_dir / "train_ids.npy"
        np.save(y_path, y)
        np.save(ids_path, train_ids)

        state = {
            "canonical_contract": {
                "y_path": str(y_path),
                "train_ids_path": str(ids_path),
                "folds_path": str(temp_data_dir / "folds.npy"),
            }
        }
        loaded_y, loaded_ids, folds_path = agent._load_canonical_training_data(state)
        assert loaded_y is not None
        assert loaded_y.tolist() == [0, 1, 0, 1]
        assert loaded_ids.tolist() == ["a", "b", "c", "d"]
        assert folds_path is not None

    def test_comparable_cv_score_ignores_leaderboard_best(self, agent):
        score, source = agent._get_comparable_cv_score(
            {
                "best_score": 0.99,
                "best_single_model_score": 0.81,
                "baseline_cv_score": 0.79,
            }
        )

        assert score == pytest.approx(0.81)
        assert source == "best_single_model_score"

    def test_comparable_cv_score_falls_back_to_baseline(self, agent):
        score, source = agent._get_comparable_cv_score(
            {"best_score": 0.99, "baseline_cv_score": 0.79}
        )

        assert score == pytest.approx(0.79)
        assert source == "baseline_cv_score"


class TestCanonicalFolds:
    def test_loads_npy_contract(self, temp_data_dir):
        folds_path = temp_data_dir / "folds.npy"
        expected = np.array([0, 1, 0, 1])
        np.save(folds_path, expected)

        loaded = load_cv_folds("model", temp_data_dir, folds_path, len(expected))

        assert np.array_equal(loaded, expected)

    def test_rejects_npy_length_mismatch(self, temp_data_dir):
        folds_path = temp_data_dir / "folds.npy"
        np.save(folds_path, np.array([0, 1]))

        assert load_cv_folds("model", temp_data_dir, folds_path, 4) is None


class TestCheckpointRecovery:
    def test_component_prompt_requires_recoverable_partial_test_predictions(self):
        guidance = COMPONENT_GUIDANCE["model"]

        assert "test_pred_sum += fold_test_preds" in guidance
        assert "_test_partial.npy" in guidance
        assert "completed_folds" in guidance

    def test_uses_partial_test_predictions(self, temp_data_dir):
        models_dir = temp_data_dir / "models"
        checkpoints = models_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        component = "lgbm"
        oof = np.array([0.1, 0.8, 0.2, 0.0])
        test = np.array([0.25, 0.75])

        np.save(checkpoints / f"{component}_oof_partial.npy", oof)
        np.save(checkpoints / f"{component}_test_partial.npy", test)
        (checkpoints / f"{component}_checkpoint_state.json").write_text(
            json.dumps(
                {
                    "component_name": component,
                    "completed_folds": [0, 1],
                    "min_folds": 2,
                }
            ),
            encoding="utf-8",
        )

        recovered = recover_from_checkpoints(models_dir)

        assert component in recovered
        assert np.array_equal(recovered[component][0], oof)
        assert np.array_equal(recovered[component][1], test)

    def test_skips_checkpoint_without_partial_test_predictions(self, temp_data_dir):
        models_dir = temp_data_dir / "models"
        checkpoints = models_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        component = "lgbm"
        np.save(checkpoints / f"{component}_oof_partial.npy", np.array([0.1, 0.8]))
        (checkpoints / f"{component}_checkpoint_state.json").write_text(
            json.dumps(
                {
                    "component_name": component,
                    "completed_folds": [0, 1],
                    "min_folds": 2,
                }
            ),
            encoding="utf-8",
        )

        assert recover_from_checkpoints(models_dir) == {}
