"""Regression tests for leakage-free MLE-bench execution."""

from __future__ import annotations

import builtins
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from kaggle_agents.agents.meta_evaluator.rewards import RewardsMixin
from kaggle_agents.agents.submission_agent import SubmissionAgent
from kaggle_agents.core.state import CompetitionInfo
from kaggle_agents.mlebench.runner import MLEBenchRunner
from kaggle_agents.utils.submission_artifacts import (
    restore_best_candidate_submission,
    snapshot_accepted_submission,
    snapshot_best_candidate_submission,
)


def _competition(metric: str = "auc") -> CompetitionInfo:
    return CompetitionInfo(
        name="protocol-test",
        description="",
        evaluation_metric=metric,
        problem_type="classification",
    )


def test_metric_preflight_keeps_explicit_caller_value() -> None:
    resolution = MLEBenchRunner._resolve_evaluation_metric(
        "opaque-task",
        "roc_auc",
    )

    assert resolution.canonical_name == "auc"
    assert resolution.raw_name == "roc_auc"
    assert resolution.source == "explicit_argument"


def test_metric_preflight_resolves_public_mlebench_registry(
    monkeypatch,
) -> None:
    registry = SimpleNamespace(
        get_competition=lambda competition_id: SimpleNamespace(
            id=competition_id,
            grader=SimpleNamespace(name="auc-roc"),
        )
    )
    monkeypatch.setattr(
        "kaggle_agents.mlebench.runner.import_module",
        lambda _module_name: SimpleNamespace(registry=registry),
    )

    resolution = MLEBenchRunner._resolve_evaluation_metric(
        "opaque-task",
        "unknown",
    )

    assert resolution.canonical_name == "auc"
    assert resolution.raw_name == "auc-roc"
    assert resolution.source == "mlebench_public_registry"


@pytest.mark.parametrize(
    ("public_name", "canonical_name"),
    [
        ("auc-roc", "auc"),
        ("column-wise ROC AUC", "auc"),
        ("mean-column-wise-roc-auc", "auc"),
        ("log-loss", "log_loss"),
        ("multi-class-log-loss", "log_loss"),
        ("root_mean_squared_error", "rmse"),
        ("root-mean-squared-error", "rmse"),
        ("mean-column-wise-rmsle", "rmsle"),
        ("accuracy", "accuracy"),
        ("multi-class-classification-accuracy", "accuracy"),
        ("quadratic-weighted-kappa", "quadratic_weighted_kappa"),
    ],
)
def test_metric_preflight_normalizes_mlebench_lite_aliases(
    public_name: str,
    canonical_name: str,
) -> None:
    resolution = MLEBenchRunner._resolve_evaluation_metric(
        "opaque-task",
        public_name,
    )

    assert resolution.canonical_name == canonical_name
    assert resolution.raw_name == public_name
    assert resolution.source == "explicit_argument"


def test_metric_preflight_rejects_special_metric_without_faithful_host_scorer() -> None:
    with pytest.raises(
        RuntimeError,
        match="not supported by the host-side canonical OOF scorer",
    ):
        MLEBenchRunner._resolve_evaluation_metric(
            "opaque-task",
            "mean-average-precision-at-10",
        )


def test_metric_preflight_aborts_before_data_or_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = MLEBenchRunner(
        mle_cache_path=tmp_path / "cache",
        workspace_base=tmp_path / "workspaces",
    )

    def missing_registry(_module_name):
        raise ModuleNotFoundError("mlebench")

    def unexpected_data_access(_competition_id):
        raise AssertionError("data preparation must not start before metric preflight")

    monkeypatch.setattr(
        "kaggle_agents.mlebench.runner.import_module",
        missing_registry,
    )
    monkeypatch.setattr(
        runner.data_adapter,
        "is_competition_prepared",
        unexpected_data_access,
    )

    result = runner.run("opaque-task", evaluation_metric="unknown")

    assert result.success is False
    assert "metric preflight failed before workflow execution" in (result.error or "")
    assert "Pass evaluation_metric explicitly" in (result.error or "")


def test_submission_agent_initializes_without_kaggle_credentials(monkeypatch) -> None:
    """Kaggle import/authentication stays lazy and package exits are contained."""
    agent = SubmissionAgent()
    assert agent.kaggle_api is None

    real_import = builtins.__import__

    def exiting_import(name, *args, **kwargs):
        if name == "kaggle.api.kaggle_api_extended":
            raise SystemExit(1)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", exiting_import)
    assert agent._ensure_kaggle_api() is False
    assert agent.kaggle_api is None


def test_mlebench_submission_only_validates_artifact(tmp_path: Path, monkeypatch) -> None:
    """MLE-bench workflow neither grades nor uploads before the runner finishes."""
    sample = pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]})
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)
    sample.assign(target=[0.2, 0.8]).to_csv(submission_path, index=False)
    candidate_snapshot, candidate_digest = snapshot_best_candidate_submission(
        tmp_path,
        submission_path,
        run_id="protocol-run",
        iteration=0,
    )

    agent = SubmissionAgent()

    def unexpected_upload(*args, **kwargs):
        raise AssertionError("MLE-bench workflow must not upload")

    monkeypatch.setattr(agent, "_upload_to_kaggle", unexpected_upload)
    updates = agent(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "protocol-run",
            # Generic scores deliberately disagree with the artifact-bound score.
            "current_performance_score": 0.99,
            "baseline_cv_score": 0.98,
            "best_single_model_score": 0.97,
            "best_candidate_submission_snapshot_path": str(candidate_snapshot),
            "best_candidate_submission_sha256": candidate_digest,
            "best_candidate_submission_component_name": "model_a",
            "robustness_approved_components": {"model_a": True},
            "trusted_component_scores": {"model_a": 0.73},
        }
    )

    result = updates["submissions"][0]
    assert result.valid is True
    assert result.public_score is None
    assert result.private_score is None
    assert result.cv_score == pytest.approx(0.73)
    snapshot_path = Path(result.file_path)
    assert snapshot_path == Path(updates["accepted_submission_snapshot_path"])
    assert snapshot_path.read_bytes() == submission_path.read_bytes()
    assert updates["accepted_submission_path"] == str(snapshot_path)
    assert updates["accepted_submission_sha256"] == hashlib.sha256(
        submission_path.read_bytes()
    ).hexdigest()
    assert updates["accepted_submission_cv_score"] == pytest.approx(0.73)
    assert updates["accepted_submission_score_owner"] == "model_a"
    assert updates["accepted_submission_score_source"] == "trusted_component_scores"
    assert "mlebench_grade" not in updates
    assert "best_score" not in updates


def test_mlebench_submission_does_not_inherit_generic_scores(
    tmp_path: Path,
) -> None:
    """Shape validity alone cannot attach an unrelated/default progress score."""
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(
        sample_path,
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(
        submission_path,
        index=False,
    )

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "unproven-score-run",
            "current_performance_score": 0.0,
            "best_single_model_score": 0.91,
            "baseline_cv_score": 0.89,
        }
    )

    result = updates["submissions"][0]
    assert result.valid is True
    assert result.cv_score is None
    assert updates["accepted_submission_cv_score"] is None
    assert updates["accepted_submission_score_owner"] is None
    assert updates["accepted_submission_score_source"] is None


def test_mlebench_submission_rejects_unapproved_component_score(
    tmp_path: Path,
) -> None:
    """A matching snapshot is insufficient when robustness rejected its owner."""
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(
        sample_path,
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(
        submission_path,
        index=False,
    )
    candidate_snapshot, candidate_digest = snapshot_best_candidate_submission(
        tmp_path,
        submission_path,
        run_id="rejected-score-run",
        iteration=0,
    )

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "rejected-score-run",
            "best_candidate_submission_snapshot_path": str(candidate_snapshot),
            "best_candidate_submission_sha256": candidate_digest,
            "best_candidate_submission_component_name": "rejected",
            "robustness_approved_components": {"rejected": False},
            "trusted_component_scores": {"rejected": 0.99},
        }
    )

    assert updates["submissions"][0].valid is True
    assert updates["submissions"][0].cv_score is None
    assert updates["accepted_submission_cv_score"] is None


def test_mlebench_submission_uses_hash_bound_host_ensemble_score(
    tmp_path: Path,
) -> None:
    """A host OOF ensemble score is reportable only for its exact CSV bytes."""
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(
        sample_path,
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(
        submission_path,
        index=False,
    )
    digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "ensemble-score-run",
            "ensemble_oof_score": 0.81,
            "ensemble_submission_sha256": digest,
            "ensemble_submission_owner": "ensemble",
            "ensemble_score_source": "host_oof_ensemble",
        }
    )

    assert updates["submissions"][0].cv_score == pytest.approx(0.81)
    assert updates["accepted_submission_cv_score"] == pytest.approx(0.81)
    assert updates["accepted_submission_score_owner"] == "ensemble"
    assert updates["accepted_submission_score_source"] == "host_oof_ensemble"


def test_mlebench_submission_rejects_stale_ensemble_score_after_csv_change(
    tmp_path: Path,
) -> None:
    """A later artifact cannot inherit an earlier ensemble's OOF score."""
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(
        sample_path,
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(
        submission_path,
        index=False,
    )
    stale_digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    pd.DataFrame({"id": [1, 2], "target": [0.3, 0.7]}).to_csv(
        submission_path,
        index=False,
    )

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "stale-ensemble-score-run",
            "ensemble_oof_score": 0.81,
            "ensemble_submission_sha256": stale_digest,
            "ensemble_submission_owner": "ensemble",
            "ensemble_score_source": "host_oof_ensemble",
        }
    )

    assert updates["submissions"][0].valid is True
    assert updates["submissions"][0].cv_score is None
    assert updates["accepted_submission_cv_score"] is None


def test_mlebench_submission_preserves_score_of_restored_accepted_snapshot(
    tmp_path: Path,
) -> None:
    """A verified prior snapshot retains its original host score provenance."""
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(
        sample_path,
        index=False,
    )
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(
        submission_path,
        index=False,
    )
    accepted_snapshot, accepted_digest = snapshot_accepted_submission(
        tmp_path,
        submission_path,
        run_id="restored-score-run",
        iteration=0,
    )

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "restored-score-run",
            "accepted_submission_path": str(accepted_snapshot),
            "accepted_submission_snapshot_path": str(accepted_snapshot),
            "accepted_submission_sha256": accepted_digest,
            "accepted_submission_cv_score": 0.81,
            "accepted_submission_score_owner": "ensemble",
            "accepted_submission_score_source": "host_oof_ensemble",
        }
    )

    assert updates["submissions"][0].cv_score == pytest.approx(0.81)
    assert updates["accepted_submission_cv_score"] == pytest.approx(0.81)
    assert updates["accepted_submission_score_owner"] == "ensemble"
    assert updates["accepted_submission_score_source"] == "host_oof_ensemble"


def test_mlebench_submission_rejects_upstream_invalid_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A shape-valid mutable CSV cannot bypass a failed snapshot gate."""
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1], "target": [0.5]}).to_csv(sample_path, index=False)
    pd.DataFrame({"id": [1], "target": [0.8]}).to_csv(
        submission_path,
        index=False,
    )
    agent = SubmissionAgent()

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("invalid upstream workflow must stop before CSV validation")

    monkeypatch.setattr(agent, "_validate_submission", unexpected_validation)
    updates = agent(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "protocol-run",
            "workflow_valid": False,
            "submission_validation_error": "verified snapshot unavailable",
            "ablation_plan": [],
        }
    )

    assert updates["submissions"][0].valid is False
    assert updates["workflow_valid"] is False
    assert "accepted_submission_snapshot_path" not in updates


def test_sample_submission_is_never_treated_as_generated_output(tmp_path: Path) -> None:
    """A template alone is not a valid final MLE-bench artifact."""
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(
        sample_path, index=False
    )

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "protocol-run",
        }
    )

    assert updates["submission_validation_error"] == "No submission file found"
    assert "submissions" not in updates


def test_sample_submission_symlink_cannot_be_promoted(tmp_path: Path) -> None:
    """Renaming the public template through a symlink is not artifact production."""
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,target\n1,0.5\n", encoding="utf-8")
    (tmp_path / "submission.csv").symlink_to(sample_path)

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "protocol-run",
        }
    )

    result = updates["submissions"][0]
    assert result.valid is False
    assert "sample submission template" in result.error
    assert "accepted_submission_snapshot_path" not in updates


def test_copied_sample_submission_cannot_be_promoted(tmp_path: Path) -> None:
    """A byte-for-byte copy of the public template is not generated output."""
    template = b"id,target\n1,0.5\n"
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_bytes(template)
    (tmp_path / "submission.csv").write_bytes(template)

    updates = SubmissionAgent()(
        {
            "working_directory": str(tmp_path),
            "competition_info": _competition(),
            "sample_submission_path": str(sample_path),
            "run_mode": "mlebench",
            "run_id": "protocol-run",
        }
    )

    assert updates["submissions"][0].valid is False
    assert "sample submission template" in updates["submission_validation_error"]
    assert "accepted_submission_sha256" not in updates


def test_runner_grades_only_explicit_hash_verified_snapshot(tmp_path: Path) -> None:
    runner = MLEBenchRunner(
        mle_cache_path=tmp_path / "cache",
        workspace_base=tmp_path / "workspaces",
    )
    workspace = tmp_path / "run"
    workspace.mkdir()
    (workspace / "sample_submission.csv").write_text(
        "id,target\n1,0.5\n", encoding="utf-8"
    )
    mutable = workspace / "submission.csv"
    mutable.write_text("id,target\n1,0.8\n", encoding="utf-8")

    # Mutable files and the public template are never implicit fallbacks.
    assert runner._find_submission(workspace) is None
    assert runner._find_submission(workspace, {}) is None

    snapshot, digest = snapshot_accepted_submission(
        workspace,
        mutable,
        run_id="protocol-run",
        iteration=1,
    )
    state = {
        "run_id": "protocol-run",
        "accepted_submission_path": str(snapshot),
        "accepted_submission_snapshot_path": str(snapshot),
        "accepted_submission_sha256": digest,
    }
    assert runner._find_submission(workspace, state) == snapshot.resolve()

    snapshot.chmod(0o644)
    snapshot.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert runner._find_submission(workspace, state) is None


def test_runner_creates_distinct_workspace_per_run_seed_and_ablation(tmp_path: Path) -> None:
    runner = MLEBenchRunner(
        mle_cache_path=tmp_path / "cache",
        workspace_base=tmp_path / "workspaces",
    )

    run_id_1, workspace_1 = runner._create_run_workspace("demo-comp", 42)
    run_id_2, workspace_2 = runner._create_run_workspace("demo-comp", 42)

    assert run_id_1 != run_id_2
    assert workspace_1 != workspace_2
    assert workspace_1.is_dir()
    assert workspace_2.is_dir()
    assert "seed-42" in workspace_1.parts
    assert runner._ablation_label() in workspace_1.parts


@pytest.mark.parametrize(
    "override_name",
    [
        "KAGGLE_AGENTS_FORCE_DATA_TYPE",
        "KAGGLE_AGENTS_DATA_TYPE",
        "KAGGLE_AGENTS_FORCE_DOMAIN",
    ],
)
def test_mlebench_runner_rejects_manual_domain_hint_channel(
    tmp_path: Path,
    monkeypatch,
    override_name: str,
) -> None:
    monkeypatch.setenv(override_name, "image_classification")
    runner = MLEBenchRunner(
        mle_cache_path=tmp_path / "cache",
        workspace_base=tmp_path / "workspaces",
    )

    result = runner.run("opaque-task")

    assert result.success is False
    assert "forbids manual domain overrides" in (result.error or "")
    assert override_name in (result.error or "")


@pytest.mark.parametrize(
    "hint_name",
    [
        "KAGGLE_AGENTS_TARGET_SCORE",
        "TARGET_SCORE",
    ],
)
def test_mlebench_runner_rejects_manual_target_score_hint(
    tmp_path: Path,
    monkeypatch,
    hint_name: str,
) -> None:
    monkeypatch.setenv(hint_name, "0.91")
    runner = MLEBenchRunner(
        mle_cache_path=tmp_path / "cache",
        workspace_base=tmp_path / "workspaces",
    )

    result = runner.run("opaque-task")

    assert result.success is False
    assert "forbids manual target-score hints" in (result.error or "")
    assert hint_name in (result.error or "")


def test_domain_node_rejects_override_in_mlebench_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from kaggle_agents.workflow.nodes.domain_detection import (
        domain_detection_node,
    )

    monkeypatch.setenv("KAGGLE_AGENTS_DATA_TYPE", "audio")

    with pytest.raises(RuntimeError, match="forbidden in MLE-bench"):
        domain_detection_node(
            {
                "run_mode": "mlebench",
                "competition_info": _competition(),
                "working_directory": str(tmp_path),
            }
        )


def test_best_candidate_snapshot_is_separate_and_restores_exact_bytes(tmp_path: Path) -> None:
    run_id = "protocol-run"
    submission = tmp_path / "submission.csv"
    expected = b"id,target\r\n1,0.8123456789012345\r\n"
    submission.write_bytes(expected)
    snapshot, digest = snapshot_best_candidate_submission(
        tmp_path,
        submission,
        run_id=run_id,
        iteration=1,
    )
    assert ".best_candidate_submissions" in snapshot.parts
    assert ".accepted_submissions" not in snapshot.parts

    submission.write_bytes(b"id,target\n1,0.1\n")
    restored = restore_best_candidate_submission(
        {
            "run_id": run_id,
            "best_candidate_submission_snapshot_path": str(snapshot),
            "best_candidate_submission_sha256": digest,
        },
        tmp_path,
    )

    assert restored == submission
    assert submission.read_bytes() == expected


@pytest.mark.parametrize("status", ["error", "failed", "cancelled", "canceled"])
def test_score_polling_stops_on_terminal_failure(status: str, monkeypatch) -> None:
    """Terminal Kaggle states stop polling after the first observed failure."""
    agent = SubmissionAgent()
    calls = 0

    def fetch_score(_competition_name):
        nonlocal calls
        calls += 1
        return None, None, status

    monkeypatch.setattr(agent, "_fetch_score", fetch_score)
    monkeypatch.setattr("kaggle_agents.agents.submission_agent.time.sleep", lambda _: None)

    score, percentile, error = agent._poll_for_score(
        "protocol-test", poll_timeout=600, poll_interval=20
    )

    assert calls == 1
    assert score is None
    assert percentile is None
    assert status in error


def test_routing_has_no_private_grade_dependency() -> None:
    """Routing source cannot branch on runner-only test-set feedback."""
    package_root = Path(__file__).parents[1] / "kaggle_agents"
    routing_source = (package_root / "workflow" / "routing.py").read_text(
        encoding="utf-8"
    )
    assert "mlebench_grade" not in routing_source
    assert "CV/OOF-guided refinement" in routing_source


def test_rewards_ignore_legacy_private_grade() -> None:
    """Reward shaping is invariant to injected test-set score/medal data."""
    evaluator = RewardsMixin()
    failure_analysis = {"success_components": [], "failed_components": []}
    base_state = {
        "run_mode": "mlebench",
        "competition_info": _competition(),
        "development_results": [],
        "current_performance_score": 0.65,
        "baseline_cv_score": 0.60,
        "target_score": 0.80,
        "overall_validation_score": 0.90,
    }
    injected_state = {
        **base_state,
        "mlebench_grade": {
            "valid_submission": True,
            "score": 0.99,
            "gold_medal": True,
        },
    }

    clean_rewards = evaluator._calculate_reward_signals(base_state, failure_analysis)
    injected_rewards = evaluator._calculate_reward_signals(injected_state, failure_analysis)

    assert injected_rewards == clean_rewards
    assert "r_medal" not in injected_rewards


def test_only_runner_contains_mlebench_grading_command() -> None:
    """The private grader command is isolated to final runner evaluation."""
    package_root = Path(__file__).parents[1] / "kaggle_agents"
    offenders = []
    for path in package_root.rglob("*.py"):
        if "grade-sample" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(package_root).as_posix())

    assert offenders == ["mlebench/runner.py"]


def test_in_process_runner_does_not_claim_os_private_label_isolation() -> None:
    """Publication telemetry must describe the actual, weaker execution boundary."""
    runner_source = (
        Path(__file__).parents[1] / "kaggle_agents" / "mlebench" / "runner.py"
    ).read_text(encoding="utf-8")

    assert '"os_private_label_isolation"' in runner_source
    assert '"not_enforced_by_in_process_runner"' in runner_source
    assert "grade externally" in runner_source


def test_workflow_components_do_not_read_private_grade_state() -> None:
    """Developer, submission, routing, and meta feedback are grade-independent."""
    package_root = Path(__file__).parents[1] / "kaggle_agents"
    workflow_components = [
        "agents/developer/agent.py",
        "agents/submission_agent.py",
        "agents/meta_evaluator/guidance.py",
        "agents/meta_evaluator/rewards.py",
        "workflow/routing.py",
    ]
    for relative_path in workflow_components:
        source = (package_root / relative_path).read_text(encoding="utf-8")
        assert "mlebench_grade" not in source, relative_path
