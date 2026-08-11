"""Ensemble agent gating: accepted-only pairs and no unscored overwrite.

Regression tests for a run where the final ensemble averaged the test
predictions of hill-climb REJECTED components (their .npy artifacts were still
on disk) and unconditionally overwrote submission.csv with an average that had
no validation score, replacing the scored best single model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from kaggle_agents.agents.ensemble.agent import EnsembleAgent
from kaggle_agents.agents.ensemble.scoring import filter_by_score_threshold
from kaggle_agents.agents.ensemble.submission import safe_restore_submission
from kaggle_agents.core.state import (
    CompetitionInfo,
    DevelopmentResult,
    KaggleState,
    create_initial_state,
)
from kaggle_agents.utils.oof_validation import validate_oof_stack
from kaggle_agents.utils.submission_artifacts import (
    snapshot_accepted_submission,
    snapshot_best_candidate_submission,
)
from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node
from kaggle_agents.workflow.routing import route_after_robustness_gate


def _make_pair(models_dir, name, n=2):
    np.save(models_dir / f"oof_{name}.npy", np.linspace(0.0, 1.0, n))
    np.save(models_dir / f"test_{name}.npy", np.linspace(0.0, 1.0, n))


def _make_text_pair(models_dir, name):
    np.save(
        models_dir / f"oof_{name}.npy",
        np.array(["one", "two"], dtype=str),
        allow_pickle=False,
    )
    np.save(
        models_dir / f"test_{name}.npy",
        np.array(["three", "four"], dtype=str),
        allow_pickle=False,
    )


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
    state.setdefault(
        "robustness_approved_components",
        {
            str(name): True
            for name, available in state.get("oof_availability", {}).items()
            if available is True
        },
    )
    return state


def _accepted_text_snapshot(tmp_path, state):
    submission = tmp_path / "submission.csv"
    expected = b"id,after\n1,three\n2,four\n"
    submission.write_bytes(expected)
    snapshot, digest = snapshot_accepted_submission(
        tmp_path,
        submission,
        run_id=state["run_id"],
        iteration=0,
    )
    state.update(
        {
            "accepted_submission_path": str(snapshot),
            "accepted_submission_snapshot_path": str(snapshot),
            "accepted_submission_sha256": digest,
        }
    )
    submission.write_bytes(b"id,after\n1,tampered\n2,tampered\n")
    return submission, expected


def _record_forbidden_text_array_loads(monkeypatch):
    original_load = np.load
    forbidden_loads = []

    def recording_load(path, *args, **kwargs):
        name = Path(path).name
        if name == "y.npy" or name.startswith("oof_"):
            forbidden_loads.append(name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    return forbidden_loads


class TestAcceptedPairFilter:
    def test_non_accepted_pairs_dropped(self, tmp_path, monkeypatch):
        state = _base_state(
            tmp_path,
            oof_availability={"kept": True, "rolled_back": False},
        )
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

    def test_pairs_without_explicit_acceptance_fail_closed(self, tmp_path, monkeypatch):
        state = _base_state(tmp_path)
        models = tmp_path / "models"
        _make_pair(models, "a")
        _make_pair(models, "b")

        monkeypatch.setattr(EnsembleAgent, "_try_oof_stacking", lambda self, **kw: None)

        def _must_not_run(self, *args, **kwargs):
            raise AssertionError("unaccepted filesystem artifacts must not be ensembled")

        monkeypatch.setattr(EnsembleAgent, "_ensemble_from_predictions", _must_not_run)

        result = EnsembleAgent()(state)

        assert result["ensemble_skipped"] is True
        assert result["skip_reason"] == "no_prediction_pairs"

    def test_filter_to_zero_pairs_restores_best(self, tmp_path):
        state = _base_state(tmp_path, oof_availability={"other": True})
        _make_pair(tmp_path / "models", "rolled_back")
        best_content = "id,value\n1,0.9\n2,0.8\n"
        (tmp_path / "submission_best.csv").write_text(best_content, encoding="utf-8")

        result = EnsembleAgent()(state)

        assert result["ensemble_skipped"] is True
        assert result["skip_reason"] == "no_prediction_pairs"
        restored = pd.read_csv(tmp_path / "submission.csv")
        assert restored["value"].tolist() == [0.9, 0.8]

    def test_mlebench_restores_only_hash_verified_immutable_snapshot(
        self,
        tmp_path,
    ):
        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="ensemble-run",
            oof_availability={"rolled_back": False},
            robustness_approved_components={"model_a": True},
            best_candidate_submission_component_name="model_a",
        )
        submission = tmp_path / "submission.csv"
        expected = b"id,value\n1,0.9\n2,0.8\n"
        submission.write_bytes(expected)
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id="ensemble-run",
            iteration=0,
        )
        state.update(
            {
                "best_candidate_submission_snapshot_path": str(snapshot),
                "best_candidate_submission_sha256": digest,
            }
        )

        # Both mutable paths disagree with the immutable state artifact.
        submission.write_text("id,value\n1,0.1\n2,0.2\n", encoding="utf-8")
        (tmp_path / "submission_best.csv").write_text(
            "id,value\n1,0.3\n2,0.4\n",
            encoding="utf-8",
        )

        result = EnsembleAgent()(state)

        assert result["skip_reason"] == "no_prediction_pairs"
        assert submission.read_bytes() == expected

    def test_mlebench_does_not_restore_best_snapshot_after_owner_rejection(
        self, tmp_path
    ):
        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="ensemble-run",
            robustness_approved_components={"rejected": False},
            best_candidate_submission_component_name="rejected",
        )
        submission = tmp_path / "submission.csv"
        submission.write_bytes(b"id,value\n1,0.7\n2,0.6\n")
        accepted_snapshot, accepted_digest = snapshot_accepted_submission(
            tmp_path,
            submission,
            run_id="ensemble-run",
            iteration=0,
        )
        submission.write_bytes(b"id,value\n1,0.99\n2,0.98\n")
        rejected_snapshot, rejected_digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id="ensemble-run",
            iteration=1,
        )
        state.update(
            {
                "accepted_submission_path": str(accepted_snapshot),
                "accepted_submission_snapshot_path": str(accepted_snapshot),
                "accepted_submission_sha256": accepted_digest,
                "best_candidate_submission_snapshot_path": str(
                    rejected_snapshot
                ),
                "best_candidate_submission_sha256": rejected_digest,
            }
        )
        submission.write_bytes(b"id,value\n1,0.1\n2,0.2\n")

        result = EnsembleAgent()(state)

        assert result["skip_reason"] == "no_prediction_pairs"
        assert submission.read_bytes() == b"id,value\n1,0.7\n2,0.6\n"

    def test_mlebench_rejects_shape_valid_tampered_snapshot(self, tmp_path):
        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="ensemble-run",
            robustness_approved_components={"model_a": True},
            best_candidate_submission_component_name="model_a",
        )
        submission = tmp_path / "submission.csv"
        submission.write_text("id,value\n1,0.9\n2,0.8\n", encoding="utf-8")
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id="ensemble-run",
            iteration=0,
        )
        state.update(
            {
                "best_candidate_submission_snapshot_path": str(snapshot),
                "best_candidate_submission_sha256": digest,
            }
        )

        # The replacement is schema-valid, so only immutable hash checking can
        # distinguish it from the selected candidate.
        snapshot.chmod(0o644)
        snapshot.write_text("id,value\n1,0.2\n2,0.1\n", encoding="utf-8")
        submission.write_text("id,value\n1,0.4\n2,0.3\n", encoding="utf-8")
        (tmp_path / "submission_best.csv").write_text(
            "id,value\n1,0.7\n2,0.6\n",
            encoding="utf-8",
        )

        result = EnsembleAgent()(state)

        assert result["skip_reason"] == "verified_snapshot_unavailable"
        assert result["workflow_valid"] is False
        assert not submission.exists()

    def test_mlebench_missing_snapshot_does_not_use_mutable_best(self, tmp_path):
        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="ensemble-run",
            robustness_approved_components={"model_a": True},
            best_candidate_submission_component_name="model_a",
        )
        submission = tmp_path / "submission.csv"
        submission.write_text("id,value\n1,0.1\n2,0.2\n", encoding="utf-8")
        (tmp_path / "submission_best.csv").write_text(
            "id,value\n1,0.9\n2,0.8\n",
            encoding="utf-8",
        )

        result = EnsembleAgent()(state)

        assert result["skip_reason"] == "verified_snapshot_unavailable"
        assert result["workflow_valid"] is False
        assert not submission.exists()

    def test_legacy_ensemble_delegates_mlebench_snapshot_policy(self, tmp_path):
        from kaggle_agents.agents.ensemble_agent import (
            EnsembleAgent as LegacyEnsembleAgent,
        )

        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="ensemble-run",
        )
        (tmp_path / "submission.csv").write_text(
            "id,value\n1,0.1\n2,0.2\n",
            encoding="utf-8",
        )
        (tmp_path / "submission_best.csv").write_text(
            "id,value\n1,0.9\n2,0.8\n",
            encoding="utf-8",
        )

        result = LegacyEnsembleAgent()(state)

        assert result["skip_reason"] == "verified_snapshot_unavailable"
        assert result["workflow_valid"] is False
        assert not (tmp_path / "submission.csv").exists()

    def test_invalid_regular_restore_never_falls_back_to_raw_copy(self, tmp_path):
        source = tmp_path / "submission_best.csv"
        destination = tmp_path / "submission.csv"
        sample = tmp_path / "sample_submission.csv"
        source.write_text("id,value\n1,0.9\n", encoding="utf-8")
        destination.write_text("id,value\n1,0.4\n2,0.3\n", encoding="utf-8")
        pd.DataFrame({"id": [1, 2], "value": [0.0, 0.0]}).to_csv(
            sample,
            index=False,
        )
        original = destination.read_bytes()

        restored = safe_restore_submission(source, destination, sample)

        assert restored is False
        assert destination.read_bytes() == original

    def test_rejected_a_stays_excluded_after_b_passes_gate(
        self, tmp_path, monkeypatch
    ):
        models = tmp_path / "models"
        models.mkdir()
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": [1, 2], "value": [0.0, 0.0]}).to_csv(
            sample,
            index=False,
        )
        result_a = DevelopmentResult(
            code='COMPONENT_NAME = "a"\n',
            success=True,
        )
        _make_pair(models, "a")
        (tmp_path / "submission.csv").write_text(
            "id,value\n1,0.1\n2,0.2\n",
            encoding="utf-8",
        )

        state = create_initial_state("demo", str(tmp_path))
        state.update(
            {
                "sample_submission_path": str(sample),
                "competition_info": CompetitionInfo(
                    "demo",
                    "",
                    "rmse",
                    "regression",
                ),
                "development_results": [result_a],
                "component_results": {"a": result_a},
                "oof_availability": {"a": True},
                "trusted_component_scores": {"a": 0.4},
                "robustness_approved_components": {"a": False},
                "robustness_passed": False,
                "robustness_failure_details": {
                    "failed_modules": ["debugging"],
                    "failed_components": ["a"],
                    "issues": ["component-specific failure"],
                    "suggestions": ["regenerate a"],
                },
            }
        )

        rejected = robustness_gate_node(state)
        assert rejected["robustness_gate_action"] == "recover"
        assert rejected["oof_availability"]["a"] is False
        assert "a" not in rejected["trusted_component_scores"]
        assert rejected["robustness_approved_components"]["a"] is False
        assert rejected["failed_component_names"] == ["a"]
        assert not (models / "oof_a.npy").exists()
        assert list(
            (tmp_path / ".rejected_candidates").glob(
                "robustness_*/models/a/oof_a.npy"
            )
        )

        # Simulate the bounded recovery: B passes. Recreate stale A artifacts
        # to prove that a later filesystem scan cannot resurrect the rejection.
        _make_pair(models, "a")
        _make_pair(models, "b")
        result_b = DevelopmentResult(
            code='COMPONENT_NAME = "b"\n',
            success=True,
        )
        state.update(rejected)
        state.update(
            {
                "development_results": [result_a, result_b],
                "component_results": {"b": result_b},
                "oof_availability": {"a": False, "b": True},
                "robustness_approved_components": {"a": False, "b": True},
                "robustness_passed": True,
                "robustness_failure_details": {},
            }
        )

        seen = {}
        monkeypatch.setattr(
            EnsembleAgent,
            "_try_oof_stacking",
            lambda self, **kwargs: None,
        )

        def _capture(self, pairs, *args, **kwargs):
            seen["pairs"] = set(pairs)
            return True

        monkeypatch.setattr(
            EnsembleAgent,
            "_ensemble_from_predictions",
            _capture,
        )

        workflow = StateGraph(KaggleState)
        workflow.add_node("gate", robustness_gate_node)
        workflow.add_node("ensemble", EnsembleAgent())
        workflow.set_entry_point("gate")
        workflow.add_conditional_edges(
            "gate",
            route_after_robustness_gate,
            {
                "pass": "ensemble",
                "recover": END,
                "fail": END,
            },
        )
        workflow.add_edge("ensemble", END)
        workflow.compile().invoke(state)

        assert seen["pairs"] == {"b"}


class TestSeq2SeqEnsembleGuard:
    @pytest.mark.parametrize("pair_count", [1, 2])
    def test_restores_verified_snapshot_without_loading_y_or_oof(
        self,
        tmp_path,
        monkeypatch,
        pair_count,
    ):
        names = [f"text_{index}" for index in range(pair_count)]
        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="seq2seq-run",
            competition_info=CompetitionInfo(
                "text normalization",
                "",
                "accuracy",
                "seq2seq",
            ),
            oof_availability=dict.fromkeys(names, True),
            robustness_approved_components=dict.fromkeys(names, True),
            trusted_component_scores=dict.fromkeys(names, 0.9),
        )
        pd.DataFrame(
            {"id": [1, 2], "after": ["placeholder", "placeholder"]}
        ).to_csv(
            state["sample_submission_path"],
            index=False,
        )
        for name in names:
            _make_text_pair(tmp_path / "models", name)
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        np.save(
            canonical / "y.npy",
            np.array(["one", "two"], dtype=object),
            allow_pickle=True,
        )
        state["canonical_contract"] = {
            "y_path": str(canonical / "y.npy"),
        }
        submission, expected = _accepted_text_snapshot(tmp_path, state)
        forbidden_loads = _record_forbidden_text_array_loads(monkeypatch)

        result = EnsembleAgent()(state)

        assert result["ensemble_skipped"] is True
        assert result["skip_reason"] == "seq2seq_kept_verified_snapshot"
        assert submission.read_bytes() == expected
        assert forbidden_loads == []

    def test_missing_verified_snapshot_fails_closed_before_array_load(
        self,
        tmp_path,
        monkeypatch,
    ):
        state = _base_state(
            tmp_path,
            run_mode="mlebench",
            run_id="seq2seq-run",
            competition_info=CompetitionInfo(
                "text normalization",
                "",
                "accuracy",
                "seq_to_seq",
            ),
            oof_availability={"text": True},
            robustness_approved_components={"text": True},
            trusted_component_scores={"text": 0.9},
        )
        _make_text_pair(tmp_path / "models", "text")
        output = tmp_path / "submission.csv"
        output.write_text("id,value\n1,mutable\n2,mutable\n", encoding="utf-8")
        forbidden_loads = _record_forbidden_text_array_loads(monkeypatch)

        result = EnsembleAgent()(state)

        assert result["ensemble_skipped"] is True
        assert result["skip_reason"] == "verified_snapshot_unavailable"
        assert result["workflow_valid"] is False
        assert not output.exists()
        assert forbidden_loads == []


class TestUnscoredFallbackGate:
    def test_unscored_average_blocked_by_scored_best(self, tmp_path, monkeypatch):
        state = _base_state(
            tmp_path,
            best_single_model_score=0.095,
            oof_availability={"a": True, "b": True},
        )
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
        state = _base_state(
            tmp_path,
            oof_availability={"a": True, "b": True},
        )
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


class TestMleOOFIdentityGate:
    def test_missing_identity_artifacts_are_blocking_when_required(
        self, tmp_path
    ):
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "oof_a.npy", np.array([0.2, 0.8]))
        np.save(models / "test_a.npy", np.array([0.4]))
        pairs = {
            "a": (models / "oof_a.npy", models / "test_a.npy")
        }

        relaxed, _ = validate_oof_stack(
            pairs,
            models,
            train_ids=np.array(["row-a", "row-b"]),
            expected_class_order=["no", "yes"],
            problem_type="classification",
        )
        required, results = validate_oof_stack(
            pairs,
            models,
            train_ids=np.array(["row-a", "row-b"]),
            expected_class_order=["no", "yes"],
            problem_type="classification",
            require_identity_artifacts=True,
        )

        assert set(relaxed) == {"a"}
        assert required == {}
        assert results[0].is_valid is False
        assert any("train_ids" in error for error in results[0].errors)
        assert any("class_order" in error for error in results[0].errors)

    def test_global_class_order_cannot_authorize_mle_component(
        self, tmp_path
    ):
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "oof_a.npy", np.array([[0.8, 0.2], [0.2, 0.8]]))
        np.save(models / "test_a.npy", np.array([[0.6, 0.4]]))
        np.save(models / "train_ids_a.npy", np.array(["row-a", "row-b"], dtype=str))
        np.save(models / "class_order.npy", np.array(["no", "yes"], dtype=str))
        pairs = {"a": (models / "oof_a.npy", models / "test_a.npy")}

        required, results = validate_oof_stack(
            pairs,
            models,
            train_ids=np.array(["row-a", "row-b"], dtype=str),
            expected_class_order=["no", "yes"],
            problem_type="classification",
            require_identity_artifacts=True,
        )

        assert required == {}
        assert any("class_order" in error for error in results[0].errors)


class TestPredictionIdContract:
    def _save_prediction_pair(self, models_dir, name, ids, predictions):
        predictions = np.asarray(predictions, dtype=float)
        np.save(models_dir / f"oof_{name}.npy", predictions)
        np.save(models_dir / f"test_{name}.npy", predictions)
        np.save(models_dir / f"test_ids_{name}.npy", np.asarray(ids))

    def test_equal_shape_models_are_always_aligned_by_id(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        sample_path = tmp_path / "sample_submission.csv"
        output_path = tmp_path / "submission.csv"
        pd.DataFrame(
            {"id": ["a", "b", "c"], "value": [0.0, 0.0, 0.0]}
        ).to_csv(sample_path, index=False)

        self._save_prediction_pair(
            models_dir, "ordered", ["a", "b", "c"], [0.1, 0.2, 0.8]
        )
        self._save_prediction_pair(
            models_dir, "reversed", ["c", "b", "a"], [0.6, 0.4, 0.2]
        )
        pairs = {
            name: (
                models_dir / f"oof_{name}.npy",
                models_dir / f"test_{name}.npy",
            )
            for name in ("ordered", "reversed")
        }

        created = EnsembleAgent()._ensemble_from_predictions(
            pairs,
            sample_path,
            output_path,
            models_dir=models_dir,
            problem_type="regression",
            metric_name="rmse",
        )

        assert created is True
        result = pd.read_csv(output_path)
        assert result["id"].tolist() == ["a", "b", "c"]
        assert result["value"].to_numpy() == pytest.approx([0.15, 0.3, 0.7])

    def test_missing_ids_fail_closed(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        sample_path = tmp_path / "sample_submission.csv"
        output_path = tmp_path / "submission.csv"
        pd.DataFrame({"id": [1, 2], "value": [0.0, 0.0]}).to_csv(
            sample_path, index=False
        )
        _make_pair(models_dir, "model")

        created = EnsembleAgent()._ensemble_from_predictions(
            {
                "model": (
                    models_dir / "oof_model.npy",
                    models_dir / "test_model.npy",
                )
            },
            sample_path,
            output_path,
            models_dir=models_dir,
            problem_type="regression",
            metric_name="rmse",
        )

        assert created is False
        assert not output_path.exists()

    def test_long_format_predictions_are_expanded_from_verified_id_grid(
        self, tmp_path
    ):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        sample_path = tmp_path / "sample_submission.csv"
        output_path = tmp_path / "submission.csv"
        pd.DataFrame(
            {"id": [40, 41, 100, 101], "probability": [0.0] * 4}
        ).to_csv(sample_path, index=False)

        predictions = np.array([[0.2, 0.8], [0.7, 0.3]])
        np.save(models_dir / "oof_model.npy", predictions)
        np.save(models_dir / "test_model.npy", predictions)
        np.save(models_dir / "test_ids_model.npy", np.array([2, 5]))

        created = EnsembleAgent()._ensemble_from_predictions(
            {
                "model": (
                    models_dir / "oof_model.npy",
                    models_dir / "test_model.npy",
                )
            },
            sample_path,
            output_path,
            models_dir=models_dir,
            expected_n_test=2,
            problem_type="classification",
            metric_name="log_loss",
        )

        assert created is True
        result = pd.read_csv(output_path)
        assert result["id"].tolist() == [40, 41, 100, 101]
        assert result["probability"].to_numpy() == pytest.approx(
            [0.2, 0.8, 0.7, 0.3]
        )

    def test_constant_predictions_fail_closed(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        sample_path = tmp_path / "sample_submission.csv"
        output_path = tmp_path / "submission.csv"
        pd.DataFrame(
            {"id": ["a", "b", "c"], "value": [0.0, 0.0, 0.0]}
        ).to_csv(sample_path, index=False)
        self._save_prediction_pair(
            models_dir, "constant", ["a", "b", "c"], [0.5, 0.5, 0.5]
        )

        created = EnsembleAgent()._ensemble_from_predictions(
            {
                "constant": (
                    models_dir / "oof_constant.npy",
                    models_dir / "test_constant.npy",
                )
            },
            sample_path,
            output_path,
            models_dir=models_dir,
            problem_type="regression",
            metric_name="rmse",
        )

        assert created is False
        assert not output_path.exists()


def test_score_filter_rejects_models_whose_oof_score_is_not_finite(tmp_path):
    pairs = {}
    for name in ("a", "b"):
        oof = tmp_path / f"oof_{name}.npy"
        test = tmp_path / f"test_{name}.npy"
        np.save(oof, np.array([0.1, 0.9]))
        np.save(test, np.array([0.2, 0.8]))
        pairs[name] = (oof, test)

    filtered, scores = filter_by_score_threshold(
        pairs,
        np.array([0, 1]),
        "unsupported_metric",
    )

    assert filtered == {}
    assert all(np.isinf(score) for score in scores.values())
