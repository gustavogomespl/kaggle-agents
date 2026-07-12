"""Regression tests for leakage-free MLE-bench execution."""

from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.agents.meta_evaluator.rewards import RewardsMixin
from kaggle_agents.agents.submission_agent import SubmissionAgent
from kaggle_agents.core.state import CompetitionInfo


def _competition(metric: str = "auc") -> CompetitionInfo:
    return CompetitionInfo(
        name="protocol-test",
        description="",
        evaluation_metric=metric,
        problem_type="classification",
    )


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
            "current_performance_score": 0.73,
        }
    )

    result = updates["submissions"][0]
    assert result.valid is True
    assert result.public_score is None
    assert result.private_score is None
    assert result.cv_score == pytest.approx(0.73)
    assert result.file_path == str(submission_path)
    assert "mlebench_grade" not in updates
    assert "best_score" not in updates


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
        }
    )

    assert updates["submission_validation_error"] == "No submission file found"
    assert "submissions" not in updates


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


def test_workflow_components_do_not_read_private_grade_state() -> None:
    """Developer, submission, routing, and meta feedback are grade-independent."""
    package_root = Path(__file__).parents[1] / "kaggle_agents"
    workflow_components = [
        "agents/developer/agent.py",
        "agents/submission_agent.py",
        "agents/meta_evaluator/guidance.py",
        "agents/meta_evaluator/rewards.py",
        "workflow/routing.py",
        "workflow.py",
    ]
    for relative_path in workflow_components:
        source = (package_root / relative_path).read_text(encoding="utf-8")
        assert "mlebench_grade" not in source, relative_path
