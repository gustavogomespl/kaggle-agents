"""Integration-level unit tests for workflow gates and causal ablations."""

from types import SimpleNamespace

import numpy as np

from kaggle_agents.agents.developer.agent import DeveloperAgent
from kaggle_agents.agents.meta_evaluator.agent import MetaEvaluatorAgent
from kaggle_agents.agents.planner.agent import PlannerAgent, determine_planning_mode
from kaggle_agents.agents.planner.plan_refinement import refine_ablation_plan
from kaggle_agents.agents.robustness_agent import RobustnessAgent
from kaggle_agents.core.config import AgentConfig, reset_config, set_config
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    DevelopmentResult,
    ValidationResult,
)
from kaggle_agents.utils.submission_artifacts import (
    sha256_file,
    snapshot_accepted_submission,
    snapshot_best_candidate_submission,
    verified_accepted_submission,
)
from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node
from kaggle_agents.workflow.routing import (
    route_after_meta_evaluator,
    route_after_robustness_gate,
)


def _toggles(**values):
    defaults = {
        "disable_search": False,
        "disable_robustness": False,
        "disable_meta_evaluator": False,
        "disable_ensemble": False,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _failure_state(tmp_path, **overrides):
    state = {
        "working_directory": str(tmp_path),
        "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
        "robustness_passed": False,
        "robustness_abstained": False,
        "robustness_recovery_count": 0,
        "max_robustness_recoveries": 1,
        "robustness_failure_details": {
            "failed_modules": ["leakage"],
            "issues": ["target leakage"],
            "suggestions": ["fit transforms inside each fold"],
        },
        "refinement_guidance": {},
        "submissions": [],
        "current_iteration": 0,
    }
    state.update(overrides)
    return state


class TestRobustnessGate:
    def test_pass_reaches_ensemble(self, tmp_path):
        state = _failure_state(tmp_path, robustness_passed=True)
        updates = robustness_gate_node(state)
        assert updates["robustness_gate_action"] == "pass"
        assert updates["workflow_valid"] is True
        assert route_after_robustness_gate(updates) == "pass"

    def test_mle_ablation_cannot_approve_unscored_oof(self, tmp_path):
        models = tmp_path / "models"
        canonical = tmp_path / "canonical"
        models.mkdir()
        canonical.mkdir()
        np.save(canonical / "train_ids.npy", np.array(["a", "b"]))
        np.save(models / "train_ids_model_a.npy", np.array(["a", "b"]))
        np.save(models / "oof_model_a.npy", np.array([0.2, 0.8]))
        np.save(models / "test_model_a.npy", np.array([0.4]))
        state = _failure_state(
            tmp_path,
            run_mode="mlebench",
            robustness_passed=True,
            robustness_abstained=True,
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": True},
            trusted_component_scores={},
            canonical_contract={
                "train_ids_path": str(canonical / "train_ids.npy")
            },
        )

        updates = robustness_gate_node(state)

        assert updates["robustness_gate_action"] == "recover"
        assert updates["oof_availability"]["model_a"] is False
        assert "trusted_oof_evidence" in updates[
            "refinement_guidance"
        ]["planner_guidance"]

    def test_mle_gate_rejects_stale_oof_row_identity(self, tmp_path):
        models = tmp_path / "models"
        canonical = tmp_path / "canonical"
        models.mkdir()
        canonical.mkdir()
        np.save(canonical / "train_ids.npy", np.array(["a", "b"]))
        np.save(models / "train_ids_model_a.npy", np.array(["b", "a"]))
        np.save(models / "oof_model_a.npy", np.array([0.2, 0.8]))
        np.save(models / "test_model_a.npy", np.array([0.4]))
        state = _failure_state(
            tmp_path,
            run_mode="mlebench",
            robustness_passed=True,
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": True},
            trusted_component_scores={"model_a": 0.8},
            canonical_contract={
                "train_ids_path": str(canonical / "train_ids.npy")
            },
        )

        updates = robustness_gate_node(state)

        assert updates["robustness_gate_action"] == "recover"
        assert updates["oof_availability"]["model_a"] is False
        assert "canonical OOF row order" in updates[
            "refinement_guidance"
        ]["planner_guidance"]

    def test_mle_gate_rejects_global_multiclass_order_fallback(self, tmp_path):
        models = tmp_path / "models"
        canonical = tmp_path / "canonical"
        models.mkdir()
        canonical.mkdir()
        np.save(canonical / "train_ids.npy", np.array(["a", "b"]))
        np.save(models / "train_ids_model_a.npy", np.array(["a", "b"], dtype=str))
        np.save(
            models / "oof_model_a.npy",
            np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
        )
        np.save(models / "test_model_a.npy", np.array([[0.2, 0.3, 0.5]]))
        np.save(models / "class_order.npy", np.array(["a", "b", "c"], dtype=str))
        state = _failure_state(
            tmp_path,
            run_mode="mlebench",
            robustness_passed=True,
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": True},
            trusted_component_scores={"model_a": 0.8},
            submission_contract={"class_order": ["a", "b", "c"]},
            canonical_contract={
                "train_ids_path": str(canonical / "train_ids.npy")
            },
        )

        updates = robustness_gate_node(state)

        assert updates["robustness_gate_action"] == "recover"
        assert "component-specific multiclass class order" in updates[
            "refinement_guidance"
        ]["planner_guidance"]

    def test_quality_objection_keeps_the_snapshot_it_disputes(self, tmp_path):
        """Suspecting a model is bad is not evidence that its artifact is fake.

        The snapshot is hash-verified and was chosen without any grading score.
        Revoking it converts "we doubt this model" into "this run produced
        nothing", which tells the benchmark strictly less: the grader would
        have accepted the file, and a weak score reports the same doubt without
        throwing the evidence away.
        """
        models = tmp_path / "models"
        models.mkdir()
        np.save(models / "oof_model_a.npy", np.array([0.2, 0.8]))
        np.save(models / "test_model_a.npy", np.array([0.4]))
        submission = tmp_path / "submission.csv"
        submission.write_text(
            "id,target\n1,0.8\n",
            encoding="utf-8",
        )
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id="gate-run",
            iteration=0,
        )
        state = _failure_state(
            tmp_path,
            run_id="gate-run",
            best_single_model_name="model_a",
            best_single_model_score=0.8,
            baseline_cv_score=0.8,
            best_candidate_submission_component_name="model_a",
            best_candidate_submission_snapshot_path=str(snapshot),
            best_candidate_submission_sha256=digest,
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": False},
            trusted_component_scores={"model_a": 0.8},
            robustness_failure_details={
                "failed_modules": ["leakage"],
                "failed_components": ["model_a"],
                "issues": ["target leakage"],
                "suggestions": ["regenerate model_a"],
            },
        )

        updates = robustness_gate_node(state)

        assert updates["robustness_gate_action"] == "recover"
        # Demoted out of the ensemble, but still gradeable.
        assert updates["oof_availability"]["model_a"] is False
        assert updates["robustness_approved_components"]["model_a"] is False
        for revoked_key in (
            "best_candidate_submission_snapshot_path",
            "best_candidate_submission_sha256",
            "best_candidate_submission_component_name",
            "best_single_model_name",
            "best_single_model_score",
        ):
            assert revoked_key not in updates

    def test_integrity_failure_revokes_the_snapshot_it_disputes(self, tmp_path):
        """Unverifiable evidence is the one case where destruction is right."""
        models = tmp_path / "models"
        canonical = tmp_path / "canonical"
        models.mkdir()
        canonical.mkdir()
        np.save(canonical / "train_ids.npy", np.array(["a", "b"]))
        np.save(models / "train_ids_model_a.npy", np.array(["a", "b"]))
        np.save(models / "oof_model_a.npy", np.array([0.2, 0.8]))
        # test_model_a.npy is absent: the pair cannot be verified.
        submission = tmp_path / "submission.csv"
        submission.write_text("id,target\n1,0.8\n", encoding="utf-8")
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id="gate-run",
            iteration=0,
        )
        state = _failure_state(
            tmp_path,
            run_mode="mlebench",
            run_id="gate-run",
            robustness_passed=True,
            best_single_model_name="model_a",
            best_single_model_score=0.8,
            best_candidate_submission_component_name="model_a",
            best_candidate_submission_snapshot_path=str(snapshot),
            best_candidate_submission_sha256=digest,
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": True},
            trusted_component_scores={"model_a": 0.8},
            canonical_contract={
                "train_ids_path": str(canonical / "train_ids.npy")
            },
        )

        updates = robustness_gate_node(state)

        assert updates["best_candidate_submission_snapshot_path"] is None
        assert updates["best_candidate_submission_sha256"] is None
        assert updates["best_candidate_submission_component_name"] is None
        assert updates["best_single_model_name"] is None

    def test_first_failure_requests_one_targeted_recovery(self, tmp_path):
        updates = robustness_gate_node(_failure_state(tmp_path))
        assert updates["robustness_gate_action"] == "recover"
        assert updates["robustness_recovery_count"] == 1
        assert updates["force_refinement"] is True
        assert updates["current_component_index"] == 0
        assert "target leakage" in updates["refinement_guidance"]["developer_guidance"]

    def test_second_failure_stops_invalid_without_submission(self, tmp_path):
        state = _failure_state(tmp_path, robustness_recovery_count=1)
        updates = robustness_gate_node(state)
        assert updates["robustness_gate_action"] == "fail"
        assert updates["workflow_valid"] is False
        assert updates["should_continue"] is False
        assert updates["termination_reason"] == "robustness_failed_no_valid_submission"

    def test_second_failure_restores_previously_valid_submission(self, tmp_path):
        run_id = "test-run"
        previous = tmp_path / "submission.csv"
        expected = b"id,target\r\n1,0.8\r\n"
        previous.write_bytes(expected)
        snapshot, digest = snapshot_accepted_submission(
            tmp_path,
            previous,
            run_id=run_id,
            iteration=0,
        )
        previous.write_bytes(b"id,target\n1,0.1\n")
        state = _failure_state(
            tmp_path,
            robustness_recovery_count=1,
            run_id=run_id,
            accepted_submission_path=str(snapshot),
            accepted_submission_snapshot_path=str(snapshot),
            accepted_submission_sha256=digest,
        )
        updates = robustness_gate_node(state)
        assert updates["workflow_valid"] is True
        assert updates["termination_reason"] == (
            "robustness_quality_objection_graded_verified_snapshot"
        )
        assert previous.read_bytes() == expected

    def test_quality_objection_grades_the_snapshot_its_owner_produced(
        self, tmp_path
    ):
        """Even when the disputed component is the one that owns the snapshot.

        This is the case the split exists for: the accepted artifact is
        hash-verified, the objection is a suspicion about the code that made
        it, and blocking here is how a run that produced a scoring submission
        came to report that it produced nothing.
        """
        run_id = "test-run"
        submission = tmp_path / "submission.csv"
        submission.write_text("id,target\n1,0.8\n", encoding="utf-8")
        snapshot, digest = snapshot_accepted_submission(
            tmp_path,
            submission,
            run_id=run_id,
            iteration=0,
        )
        state = _failure_state(
            tmp_path,
            robustness_recovery_count=1,
            run_id=run_id,
            accepted_submission_path=str(snapshot),
            accepted_submission_snapshot_path=str(snapshot),
            accepted_submission_sha256=digest,
            accepted_submission_cv_score=0.9,
            accepted_submission_score_owner="model_a",
            accepted_submission_score_source="trusted_component_scores",
            robustness_failure_details={
                "failed_modules": ["leakage"],
                "failed_components": ["model_a"],
                "issues": ["target leakage"],
                "suggestions": ["regenerate model_a"],
            },
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": False},
        )

        updates = robustness_gate_node(state)

        assert updates["workflow_valid"] is True
        assert updates["termination_reason"] == (
            "robustness_quality_objection_graded_verified_snapshot"
        )
        assert submission.read_text(encoding="utf-8") == "id,target\n1,0.8\n"
        assert updates["oof_availability"]["model_a"] is False
        for preserved_key in (
            "accepted_submission_path",
            "accepted_submission_snapshot_path",
            "accepted_submission_sha256",
            "accepted_submission_score_owner",
        ):
            assert preserved_key not in updates
        assert (
            verified_accepted_submission({**state, **updates}, tmp_path)
            is not None
        )

    def test_integrity_failure_discards_the_snapshot_its_owner_produced(
        self, tmp_path
    ):
        """The same setup, with evidence that cannot be verified, still blocks."""
        run_id = "test-run"
        models = tmp_path / "models"
        canonical = tmp_path / "canonical"
        models.mkdir()
        canonical.mkdir()
        np.save(canonical / "train_ids.npy", np.array(["a", "b"]))
        np.save(models / "train_ids_model_a.npy", np.array(["b", "a"]))
        np.save(models / "oof_model_a.npy", np.array([0.2, 0.8]))
        np.save(models / "test_model_a.npy", np.array([0.4]))
        submission = tmp_path / "submission.csv"
        submission.write_text("id,target\n1,0.8\n", encoding="utf-8")
        snapshot, digest = snapshot_accepted_submission(
            tmp_path,
            submission,
            run_id=run_id,
            iteration=0,
        )
        state = _failure_state(
            tmp_path,
            run_mode="mlebench",
            robustness_passed=True,
            robustness_recovery_count=1,
            run_id=run_id,
            accepted_submission_path=str(snapshot),
            accepted_submission_snapshot_path=str(snapshot),
            accepted_submission_sha256=digest,
            accepted_submission_score_owner="model_a",
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": True},
            trusted_component_scores={"model_a": 0.8},
            canonical_contract={
                "train_ids_path": str(canonical / "train_ids.npy")
            },
        )

        updates = robustness_gate_node(state)

        assert updates["workflow_valid"] is False
        assert updates["termination_reason"] == (
            "robustness_failed_no_valid_submission"
        )
        assert not submission.exists()
        for revoked_key in (
            "accepted_submission_path",
            "accepted_submission_snapshot_path",
            "accepted_submission_sha256",
            "accepted_submission_score_owner",
        ):
            assert updates[revoked_key] is None

    def test_second_failure_restores_verified_best_candidate_owned_by_another_component(
        self, tmp_path
    ):
        submission = tmp_path / "submission.csv"
        expected = b"id,target\r\n1,0.9\r\n"
        submission.write_bytes(expected)
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id="gate-run",
            iteration=0,
        )
        submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
        state = _failure_state(
            tmp_path,
            run_id="gate-run",
            robustness_recovery_count=1,
            best_candidate_submission_component_name="model_b",
            best_candidate_submission_snapshot_path=str(snapshot),
            best_candidate_submission_sha256=digest,
            robustness_failure_details={
                "failed_modules": ["leakage"],
                "failed_components": ["model_a"],
                "issues": ["target leakage"],
                "suggestions": ["regenerate model_a"],
            },
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": False},
        )

        updates = robustness_gate_node(state)

        assert updates["workflow_valid"] is True
        assert updates["termination_reason"] == (
            "robustness_quality_objection_graded_verified_snapshot"
        )
        assert submission.read_bytes() == expected
        # The runner grades ONLY the accepted registry. When the run never
        # reached the SubmissionAgent (this scenario: the gate stops the run
        # before ensemble/submission), workflow_valid=True with an unbound
        # snapshot makes the runner answer "No hash-verified accepted
        # submission generated in this run" and the run grades as nothing.
        graded = verified_accepted_submission({**state, **updates}, tmp_path)
        assert graded is not None
        assert sha256_file(graded) == digest

    def test_integrity_failure_still_grades_anothers_verified_snapshot(
        self, tmp_path
    ):
        """Integrity discards the disputed candidate, not the whole run.

        model_a's evidence cannot be verified (its rows do not line up with
        the canonical order), but the surviving snapshot belongs to model_b,
        whose integrity is not in question. The run must end gradeable, with
        the snapshot bound to the registry the runner reads.
        """
        run_id = "gate-run"
        models = tmp_path / "models"
        canonical = tmp_path / "canonical"
        models.mkdir()
        canonical.mkdir()
        np.save(canonical / "train_ids.npy", np.array(["a", "b"]))
        np.save(models / "train_ids_model_a.npy", np.array(["b", "a"]))
        np.save(models / "oof_model_a.npy", np.array([0.2, 0.8]))
        np.save(models / "test_model_a.npy", np.array([0.4]))
        submission = tmp_path / "submission.csv"
        expected = b"id,target\r\n1,0.9\r\n"
        submission.write_bytes(expected)
        snapshot, digest = snapshot_best_candidate_submission(
            tmp_path,
            submission,
            run_id=run_id,
            iteration=0,
        )
        state = _failure_state(
            tmp_path,
            run_mode="mlebench",
            robustness_passed=True,
            robustness_recovery_count=1,
            run_id=run_id,
            best_candidate_submission_component_name="model_b",
            best_candidate_submission_snapshot_path=str(snapshot),
            best_candidate_submission_sha256=digest,
            oof_availability={"model_a": True},
            robustness_approved_components={"model_a": True},
            trusted_component_scores={"model_a": 0.8},
            canonical_contract={
                "train_ids_path": str(canonical / "train_ids.npy")
            },
        )

        updates = robustness_gate_node(state)

        assert updates["workflow_valid"] is True
        assert updates["termination_reason"] == (
            "robustness_failed_preserved_best_submission"
        )
        assert submission.read_bytes() == expected
        graded = verified_accepted_submission({**state, **updates}, tmp_path)
        assert graded is not None
        assert sha256_file(graded) == digest

    def test_second_failure_rejects_mutable_hill_climb_best_from_disk(self, tmp_path):
        best = tmp_path / "submission_best.csv"
        best.write_text("id,target\n1,0.9\n", encoding="utf-8")
        state = _failure_state(tmp_path, robustness_recovery_count=1)
        updates = robustness_gate_node(state)
        assert updates["robustness_gate_action"] == "fail"
        assert updates["workflow_valid"] is False
        assert updates["termination_reason"] == "robustness_failed_no_valid_submission"
        assert not (tmp_path / "submission.csv").exists()

    def test_second_failure_quarantines_existing_mutable_submission_csv(self, tmp_path):
        existing = tmp_path / "submission.csv"
        existing.write_text("id,target\n1,0.7\n", encoding="utf-8")
        updates = robustness_gate_node(_failure_state(tmp_path, robustness_recovery_count=1))
        assert updates["workflow_valid"] is False
        assert not existing.exists()
        quarantined = list(
            (tmp_path / ".rejected_candidates").glob(
                "robustness_*/submission.csv"
            )
        )
        assert len(quarantined) == 1
        assert quarantined[0].read_text(encoding="utf-8") == "id,target\n1,0.7\n"

    def test_second_failure_rejects_tampered_accepted_snapshot(self, tmp_path):
        run_id = "test-run"
        submission = tmp_path / "submission.csv"
        submission.write_text("id,target\n1,0.8\n", encoding="utf-8")
        snapshot, digest = snapshot_accepted_submission(
            tmp_path,
            submission,
            run_id=run_id,
            iteration=0,
        )
        snapshot.chmod(0o644)
        snapshot.write_text("id,target\n1,1.0\n", encoding="utf-8")

        updates = robustness_gate_node(
            _failure_state(
                tmp_path,
                robustness_recovery_count=1,
                run_id=run_id,
                accepted_submission_path=str(snapshot),
                accepted_submission_snapshot_path=str(snapshot),
                accepted_submission_sha256=digest,
            )
        )

        assert updates["workflow_valid"] is False
        assert updates["termination_reason"] == "robustness_failed_no_valid_submission"

    def test_recovery_guidance_names_flagged_component(self, tmp_path):
        # Naming the validated component invalidates the developer skip-cache
        # and steers the correction plan at it — otherwise recovery revalidates
        # the same stale code and can never converge.
        flagged_code = 'COMPONENT_NAME = "ensemble_weighted_averaging"\nprint("x")\n'
        state = _failure_state(
            tmp_path,
            development_results=[DevelopmentResult(code=flagged_code, success=True)],
        )
        updates = robustness_gate_node(state)
        assert updates["robustness_gate_action"] == "recover"
        guidance = updates["refinement_guidance"]
        assert "ensemble_weighted_averaging" in guidance["planner_guidance"]
        assert "ensemble_weighted_averaging" in guidance["developer_guidance"]

    def test_agent_validates_every_eligible_prediction_pair(
        self, tmp_path, monkeypatch
    ):
        agent = object.__new__(RobustnessAgent)
        agent.config = SimpleNamespace(
            ablation_toggles=_toggles(),
            validation=SimpleNamespace(min_validation_score=0.7),
        )
        monkeypatch.setattr(
            "kaggle_agents.core.config.get_llm",
            lambda: SimpleNamespace(),
        )

        def _result(module, passed=True):
            return ValidationResult(
                module=module,
                passed=passed,
                score=1.0 if passed else 0.0,
                issues=[] if passed else ["component-specific failure"],
            )

        monkeypatch.setattr(
            agent,
            "_validate_debugging",
            lambda result, _working_dir: _result(
                "debugging",
                passed='COMPONENT_NAME = "a"' not in result.code,
            ),
        )
        monkeypatch.setattr(
            agent,
            "_validate_leakage",
            lambda _result_value, _working_dir, _state: _result("leakage"),
        )
        monkeypatch.setattr(
            agent,
            "_validate_data_usage",
            lambda _result_value, _working_dir, _state: _result("data_usage"),
        )
        monkeypatch.setattr(
            agent,
            "_validate_hyperparameters",
            lambda _result_value, _working_dir, _state: _result("hyperparameters"),
        )
        monkeypatch.setattr(
            agent,
            "_validate_format",
            lambda _result_value, _working_dir, _state: _result("format"),
        )
        monkeypatch.setattr(
            agent,
            "_validate_data_shapes",
            lambda _working_dir, _state: _result("data_shapes"),
        )
        monkeypatch.setattr(
            agent,
            "_check_model_performance_gap",
            lambda _state: _result("performance_gap"),
        )

        result_a = DevelopmentResult(
            code='COMPONENT_NAME = "a"\n',
            success=True,
        )
        result_b = DevelopmentResult(
            code='COMPONENT_NAME = "b"\n',
            success=True,
        )
        updates = agent(
            {
                "working_directory": str(tmp_path),
                "current_iteration": 0,
                "development_results": [result_a, result_b],
                "component_results": {"a": result_a, "b": result_b},
                "oof_availability": {"a": True, "b": True},
                "robustness_approved_components": {},
            }
        )

        assert updates["robustness_passed"] is False
        assert updates["robustness_approved_components"] == {
            "a": False,
            "b": True,
        }
        assert updates["robustness_failure_details"]["failed_components"] == ["a"]


class TestCausalAblations:
    def test_robustness_ablation_abstains_without_perfect_score(self, tmp_path):
        agent = object.__new__(RobustnessAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_robustness=True))
        updates = agent(
            {
                "working_directory": str(tmp_path),
                "current_iteration": 0,
                "development_results": [],
            }
        )
        assert updates["robustness_abstained"] is True
        assert updates["robustness_passed"] is True
        assert updates["overall_validation_score"] is None

    def test_meta_ablation_clears_stale_recovery_signals(self):
        agent = object.__new__(MetaEvaluatorAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_meta_evaluator=True))
        updates = agent(
            {
                "current_iteration": 3,
                "stagnation_detection": {"trigger_sota_search": True},
                "failure_analysis": {"error_patterns": ["memory_error"]},
            }
        )
        assert updates["stagnation_detection"] == {}
        assert updates["failure_analysis"] == {}
        assert updates["refinement_guidance"] == {}
        assert updates["trigger_debug_loop"] is False

    def test_meta_ablation_routes_directly_to_iteration_control(self):
        config = AgentConfig()
        config.ablation_toggles.disable_meta_evaluator = True
        set_config(config)
        try:
            route = route_after_meta_evaluator(
                {
                    "stagnation_detection": {"trigger_sota_search": True},
                    "failure_analysis": {"error_patterns": ["memory_error"]},
                }
            )
            assert route == "skip_recovery"
        finally:
            reset_config()

    def test_planner_removes_ensemble_components(self):
        agent = object.__new__(PlannerAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_ensemble=True))
        plan = [
            AblationComponent("model", "model", "train"),
            AblationComponent("stack", "ensemble", "blend"),
        ]
        finalized, hashes = agent._finalize_plan(plan, {"previous_plan_hashes": []})
        assert [component.name for component in finalized] == ["model"]
        assert len(hashes) == 1

    def test_developer_skips_stale_ensemble_component(self, tmp_path):
        agent = object.__new__(DeveloperAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_ensemble=True))
        updates = agent(
            {
                "working_directory": str(tmp_path),
                "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
                "ablation_plan": [AblationComponent("stack", "ensemble", "blend")],
                "current_component_index": 0,
                "current_iteration": 0,
            }
        )
        assert updates["current_component_index"] == 1
        assert updates["telemetry_events"][0]["event"] == "developer_ensemble_component_skipped"


class TestPlannerRefinementMode:
    def test_first_completed_cycle_uses_targeted_refinement(self):
        is_refinement, use_eureka = determine_planning_mode(
            {
                "current_iteration": 1,
                "crossover_guidance": {"preserve_components": ["model"]},
                "evolutionary_generation": 1,
            }
        )
        assert is_refinement is True
        assert use_eureka is False

    def test_guardrail_can_force_same_iteration_refinement(self):
        is_refinement, use_eureka = determine_planning_mode(
            {"current_iteration": 0, "force_refinement": True}
        )
        assert is_refinement is True
        assert use_eureka is False

    def test_eureka_remains_explicitly_available(self):
        is_refinement, use_eureka = determine_planning_mode(
            {
                "current_iteration": 2,
                "force_refinement": True,
                "force_eureka_planning": True,
            }
        )
        assert is_refinement is True
        assert use_eureka is True

    def test_duplicate_tabular_plan_uses_live_rotation(self):
        repeated = AblationComponent("repeat_model", "model", "train")
        repeated_hash = hash((("repeat_model", "model"),))
        state = {
            "ablation_plan": [repeated],
            "development_results": [DevelopmentResult(code="x", success=True)],
            "best_score": 0.0,
            "current_performance_score": 0.0,
            "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
            "domain_detected": "tabular_classification",
            "run_mode": "mlebench",
            "fast_mode": True,
            "previous_plan_hashes": [repeated_hash],
            "refinement_guidance": {},
            "failure_analysis": {},
            "failed_component_names": [],
        }

        plan = refine_ablation_plan(
            state=state,
            sota_analysis={},
            llm=None,
            use_dspy=True,
            refine_ablation_plan_prompt="{gap_analysis}{previous_plan}{test_results}{current_score}{memory_summary}",
            analyze_gaps_fn=lambda **_kwargs: {},
            create_refined_fallback_plan_fn=lambda *_args: [
                {
                    "name": "repeat_model",
                    "component_type": "model",
                    "code_outline": "train",
                    "estimated_impact": 0.1,
                }
            ],
            create_diversified_fallback_plan_fn=lambda *_args: [],
            get_memory_summary_for_planning_fn=lambda _state: "",
        )

        names = {component.name for component in plan}
        assert "catboost_fast_cv" in names
        assert "lightgbm_tuned_cv" in names


class TestRejectedComponentResolution:
    def test_explicit_rejections_suppress_legacy_fallbacks(self):
        # With an explicit rejection in hand, the flagged-component fallback
        # must not fire: it quarantined an innocent, fully evidenced
        # component named in the last development result.
        from kaggle_agents.workflow.nodes.robustness_gate import (
            _rejected_component_names,
        )

        state = {
            "robustness_failure_details": {},
            "robustness_approved_components": {},
            "oof_availability": {"model_a": True, "model_b": True},
            "development_results": [
                SimpleNamespace(code='COMPONENT_NAME = "model_a"')
            ],
        }

        rejected = _rejected_component_names(
            state, explicit_rejections={"model_b"}
        )

        assert rejected == ["model_b"]

    def test_legacy_fallback_still_resolves_flagged_component(self):
        from kaggle_agents.workflow.nodes.robustness_gate import (
            _rejected_component_names,
        )

        state = {
            "robustness_failure_details": {},
            "robustness_approved_components": {},
            "oof_availability": {},
            "development_results": [
                SimpleNamespace(code='COMPONENT_NAME = "model_a"')
            ],
        }

        assert _rejected_component_names(state) == ["model_a"]
