"""Metric-aware postprocessing must reach the single-model path.

Stacking needs two models, so a run that accepted exactly one shipped whatever
decision rule the generated script happened to pick -- a fixed 0.5 threshold or
a plain argmax -- even on competitions scored on hard labels. The tuning
helpers were implemented and tested, but nothing reachable called them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.agents.ensemble.agent import EnsembleAgent
from kaggle_agents.agents.submission_agent import SubmissionAgent
from kaggle_agents.utils.submission_artifacts import sha256_file


COMPONENT = "model_a"
N_TRAIN = 400
N_TEST = 60


@pytest.fixture
def scenario(tmp_path: Path) -> dict:
    """A single accepted model whose optimal threshold is far from 0.5."""
    rng = np.random.default_rng(0)

    # Labels are positive well below 0.5, so the fixed rule under-predicts.
    y = (rng.random(N_TRAIN) < 0.45).astype(int)
    oof = np.where(
        y == 1,
        rng.uniform(0.20, 0.55, N_TRAIN),
        rng.uniform(0.02, 0.30, N_TRAIN),
    )
    test_preds = rng.uniform(0.05, 0.55, N_TEST)

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    y_path = canonical / "y.npy"
    np.save(y_path, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    np.save(models_dir / f"oof_{COMPONENT}.npy", oof)
    np.save(models_dir / f"test_{COMPONENT}.npy", test_preds)
    test_ids = np.array([f"t{i}" for i in range(N_TEST)])
    np.save(models_dir / f"test_ids_{COMPONENT}.npy", test_ids)

    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": test_ids, "target": np.zeros(N_TEST, dtype=int)}).to_csv(
        sample_path, index=False
    )

    return {
        "workspace": tmp_path,
        "models_dir": models_dir,
        "sample_path": sample_path,
        "output_path": tmp_path / "submission.csv",
        "pairs": {
            COMPONENT: (
                models_dir / f"oof_{COMPONENT}.npy",
                models_dir / f"test_{COMPONENT}.npy",
            )
        },
        "y": y,
        "oof": oof,
        "test_preds": test_preds,
        "state": {
            "working_directory": str(tmp_path),
            "canonical_contract": {"y_path": str(y_path)},
        },
    }


def _run(scenario: dict, metric_name: str, **state_overrides):
    state = {**scenario["state"], **state_overrides}
    return EnsembleAgent()._try_single_model_postprocessing(
        state=state,
        prediction_pairs=scenario["pairs"],
        models_dir=scenario["models_dir"],
        sample_path=scenario["sample_path"],
        output_path=scenario["output_path"],
        problem_type="classification",
        metric_name=metric_name,
        current_iteration=0,
    )


class TestSingleModelPostprocessing:
    def test_label_metric_gets_an_oof_tuned_threshold(self, scenario):
        outcome = _run(scenario, "accuracy")

        assert outcome is not None
        written = pd.read_csv(scenario["output_path"])
        tuned_labels = written["target"].to_numpy()
        fixed_labels = (scenario["test_preds"] >= 0.5).astype(int)

        # The fixed rule is what the generated script would have produced.
        assert not np.array_equal(tuned_labels, fixed_labels)
        assert set(np.unique(tuned_labels)) <= {0, 1}

    def test_probability_metric_is_left_alone(self, scenario):
        """Thresholding an AUC submission would destroy information."""
        assert _run(scenario, "auc") is None
        assert not scenario["output_path"].exists()

    def test_rmse_is_left_alone(self, scenario):
        assert _run(scenario, "rmse") is None

    def test_two_models_are_left_to_the_stacking_path(self, scenario):
        scenario["pairs"]["model_b"] = scenario["pairs"][COMPONENT]

        assert _run(scenario, "accuracy") is None

    def test_missing_canonical_labels_disable_tuning(self, scenario):
        outcome = _run(scenario, "accuracy", canonical_contract={})

        assert outcome is None

    def test_temporal_oof_mask_disables_tuning(self, scenario):
        mask_path = scenario["workspace"] / "canonical" / "oof_eligible_mask.npy"
        np.save(mask_path, np.ones(N_TRAIN, dtype=bool))

        assert _run(scenario, "accuracy") is None

    def test_missing_test_ids_refuses_positional_alignment(self, scenario):
        (scenario["models_dir"] / f"test_ids_{COMPONENT}.npy").unlink()

        assert _run(scenario, "accuracy") is None
        assert not scenario["output_path"].exists()

    def test_labels_follow_submission_id_order(self, scenario):
        """Shuffled submission IDs must not silently produce positional labels."""
        shuffled = pd.read_csv(scenario["sample_path"]).iloc[::-1].reset_index(drop=True)
        shuffled.to_csv(scenario["sample_path"], index=False)

        outcome = _run(scenario, "accuracy")

        assert outcome is not None
        written = pd.read_csv(scenario["output_path"])
        assert written["id"].tolist() == shuffled["id"].tolist()

        by_id = dict(zip(written["id"].astype(str), written["target"], strict=True))
        model_ids = np.load(
            scenario["models_dir"] / f"test_ids_{COMPONENT}.npy", allow_pickle=True
        )
        # Every model row landed on its own ID, not on a positional neighbour.
        assert len(by_id) == len(model_ids)

    def test_a_worse_tuned_rule_never_replaces_the_existing_artifact(self, scenario):
        outcome = _run(scenario, "accuracy", best_single_model_score=10.0)

        assert outcome is None
        assert not scenario["output_path"].exists()


class TestPostprocessingProvenance:
    def test_outcome_binds_its_score_to_the_written_bytes(self, scenario):
        outcome = _run(scenario, "accuracy")

        assert outcome["ensemble_score_source"] == "host_oof_postprocessing"
        assert outcome["ensemble_submission_owner"] == "ensemble"
        assert outcome["ensemble_submission_sha256"] == sha256_file(
            scenario["output_path"]
        )
        assert np.isfinite(outcome["ensemble_oof_score"])

    def test_submission_agent_accepts_the_postprocessing_provenance(self, scenario):
        outcome = _run(scenario, "accuracy")
        state = {
            **scenario["state"],
            "run_id": "postproc-run",
            "ensemble_oof_score": outcome["ensemble_oof_score"],
            "ensemble_submission_sha256": outcome["ensemble_submission_sha256"],
            "ensemble_submission_owner": "ensemble",
            "ensemble_score_source": "host_oof_postprocessing",
        }

        score, source, owner = SubmissionAgent._resolve_mlebench_cv_provenance(
            state, scenario["workspace"], scenario["output_path"]
        )

        assert score == pytest.approx(outcome["ensemble_oof_score"])
        assert source == "host_oof_postprocessing"
        assert owner == "ensemble"

    def test_provenance_is_rejected_when_the_artifact_changes(self, scenario):
        outcome = _run(scenario, "accuracy")
        state = {
            **scenario["state"],
            "run_id": "postproc-run",
            "ensemble_oof_score": outcome["ensemble_oof_score"],
            "ensemble_submission_sha256": outcome["ensemble_submission_sha256"],
            "ensemble_submission_owner": "ensemble",
            "ensemble_score_source": "host_oof_postprocessing",
        }
        tampered = pd.read_csv(scenario["output_path"])
        tampered["target"] = 1 - tampered["target"]
        tampered.to_csv(scenario["output_path"], index=False)

        score, source, owner = SubmissionAgent._resolve_mlebench_cv_provenance(
            state, scenario["workspace"], scenario["output_path"]
        )

        assert (score, source, owner) == (None, None, None)


class TestMeanFallbackOof:
    def test_aligned_oof_is_available_for_the_mean_path(self, scenario):
        oof = EnsembleAgent()._load_aligned_oof(
            scenario["pairs"], scenario["y"], scenario["models_dir"]
        )

        assert oof is not None
        assert oof.shape == (N_TRAIN,)
        assert np.allclose(oof, scenario["oof"])

    def test_shape_disagreement_fails_closed(self, scenario):
        np.save(
            scenario["models_dir"] / f"oof_{COMPONENT}.npy",
            np.zeros(N_TRAIN - 1),
        )

        assert (
            EnsembleAgent()._load_aligned_oof(
                scenario["pairs"], scenario["y"], scenario["models_dir"]
            )
            is None
        )

    def test_mean_path_applies_the_tuned_rule_when_oof_is_supplied(self, scenario):
        agent = EnsembleAgent()
        oof = agent._load_aligned_oof(
            scenario["pairs"], scenario["y"], scenario["models_dir"]
        )

        assert agent._ensemble_from_predictions(
            scenario["pairs"],
            scenario["sample_path"],
            scenario["output_path"],
            scenario["models_dir"],
            None,
            problem_type="classification",
            metric_name="accuracy",
            oof_preds=oof,
            y_true=scenario["y"],
        )

        tuned = pd.read_csv(scenario["output_path"])["target"].to_numpy()
        assert not np.array_equal(tuned, (scenario["test_preds"] >= 0.5).astype(int))


class TestNonBinaryLabelScoring:
    """score_predictions re-thresholds hard labels at 0.5, so scoring the tuned
    rule through it collapsed a label set like {1,2} into one class. The score
    now comes from the tuner itself, evaluated on the original label values."""

    @pytest.fixture
    def shifted_labels(self, scenario, tmp_path):
        """Same scenario with labels {1,2} instead of {0,1}."""
        y = np.load(tmp_path / "canonical" / "y.npy") + 1
        np.save(tmp_path / "canonical" / "y.npy", y)
        scenario["y"] = y
        return scenario

    def test_shifted_binary_labels_are_scored_correctly(self, shifted_labels):
        outcome = _run(shifted_labels, "accuracy")

        assert outcome is not None
        # A collapsed-to-one-class score would sit near the base rate; the true
        # tuned accuracy is well above it.
        assert outcome["ensemble_oof_score"] > 0.6
        written = pd.read_csv(shifted_labels["output_path"])
        assert set(np.unique(written["target"])) <= {1, 2}

    def test_score_matches_the_metric_on_original_labels(self, shifted_labels):
        from sklearn.metrics import accuracy_score

        from kaggle_agents.agents.ensemble.postprocessing import tune_binary_threshold

        outcome = _run(shifted_labels, "accuracy")
        threshold, _, _ = tune_binary_threshold(
            shifted_labels["oof"], shifted_labels["y"], "accuracy"
        )
        classes = np.unique(shifted_labels["y"])
        oof_labels = classes[(shifted_labels["oof"] >= threshold).astype(int)]

        assert outcome["ensemble_oof_score"] == pytest.approx(
            accuracy_score(shifted_labels["y"], oof_labels), abs=1e-6
        )

    def test_minimization_metrics_are_refused(self, scenario):
        """The score convention here only holds for maximize label metrics."""
        assert _run(scenario, "log_loss") is None

    def test_tuned_score_shares_the_trusted_score_convention(self, scenario):
        """The guard compares the tuned rule against trusted_component_scores,
        so both must express the same quantity. For a maximize label metric the
        trusted convention (negate a lower-is-better loss) reduces to the raw
        metric value."""
        from sklearn.metrics import accuracy_score

        from kaggle_agents.agents.ensemble.scoring import score_predictions
        from kaggle_agents.core.config import is_metric_minimization

        loss = score_predictions(
            scenario["oof"], scenario["y"], "classification", "accuracy"
        )
        trusted = loss if is_metric_minimization("accuracy") else -loss
        fixed_rule = accuracy_score(
            scenario["y"], (scenario["oof"] >= 0.5).astype(int)
        )

        assert trusted == pytest.approx(fixed_rule)

        outcome = _run(scenario, "accuracy")
        # Same quantity, better rule.
        assert outcome["ensemble_oof_score"] >= trusted
