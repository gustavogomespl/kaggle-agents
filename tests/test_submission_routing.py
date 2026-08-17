"""Submission routing must fail closed after bounded regeneration."""

from pathlib import Path

from kaggle_agents.agents.submission_agent import SubmissionAgent
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    SubmissionResult,
)
from kaggle_agents.workflow.routing import route_after_submission


def test_missing_submission_fails_after_retry_budget() -> None:
    state = {"submissions": [], "retry_submission_count": 4}

    assert route_after_submission(state) == "fail"


def test_missing_submission_retries_last_predictive_component() -> None:
    state = {
        "submissions": [],
        "retry_submission_count": 0,
        "ablation_plan": [
            AblationComponent("features", "feature_engineering", "build"),
            AblationComponent("model", "model", "train"),
        ],
        "current_component_index": 2,
        "skip_remaining_components": True,
    }

    updates = SubmissionAgent._retry_updates(state, "missing")

    assert updates["retry_submission_count"] == 1
    assert updates["current_component_index"] == 1
    assert updates["skip_remaining_components"] is False
    assert route_after_submission({**state, **updates}) == "retry_developer"


def test_invalid_submission_fails_after_retry_budget() -> None:
    state = {
        "submissions": [{"valid": False, "error": "non-finite predictions"}],
        "retry_submission_count": 4,
    }

    assert route_after_submission(state) == "fail"


def test_retry_exhaustion_is_persisted_by_submission_node() -> None:
    updates = SubmissionAgent._retry_updates(
        {"retry_submission_count": 3, "ablation_plan": []},
        "invalid",
    )

    assert updates["retry_submission_count"] == 4
    assert updates["workflow_valid"] is False
    assert updates["termination_reason"] == "submission_invalid_after_retries"


def test_current_missing_artifact_is_not_masked_by_prior_valid_submission() -> None:
    state = {
        "submissions": [{"valid": True, "error": None}],
        "submission_validation_error": "No submission file found",
        "retry_submission_count": 1,
    }

    assert route_after_submission(state) == "retry_developer"


def test_retry_exhaustion_with_verified_accepted_artifact_keeps_run_gradable() -> None:
    # A later iteration failing 4x must not throw away an earlier accepted,
    # hash-verified artifact: the loop stops but the runner can still grade it.
    updates = SubmissionAgent._retry_updates(
        {
            "retry_submission_count": 3,
            "ablation_plan": [],
            "accepted_submission_snapshot_path": "/ws/.submission_store/x.csv",
            "accepted_submission_sha256": "0" * 64,
        },
        "invalid",
    )

    assert updates["retry_submission_count"] == 4
    assert updates["should_continue"] is False
    assert updates["termination_reason"] == "submission_invalid_after_retries"
    assert "workflow_valid" not in updates


def test_failed_kaggle_upload_advances_the_retry_counter(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "submission.csv").write_text("id,value\n1,0.5\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,value\n1,0.0\n", encoding="utf-8")

    agent = SubmissionAgent()
    monkeypatch.setattr(
        SubmissionAgent, "_validate_submission", lambda self, *a, **k: (True, "ok")
    )
    monkeypatch.setattr(
        SubmissionAgent,
        "_upload_to_kaggle",
        lambda self, **kwargs: SubmissionResult(
            submission_id=None,
            public_score=None,
            private_score=None,
            percentile=None,
            cv_score=None,
            file_path=str(tmp_path / "submission.csv"),
            valid=False,
            error="Daily submission limit reached",
        ),
    )

    state = {
        "working_directory": str(tmp_path),
        "competition_info": CompetitionInfo("demo", "", "rmse", "regression"),
        "retry_submission_count": 1,
        "submissions": [],
        "ablation_plan": [],
    }
    updates = agent(state)

    assert updates["retry_submission_count"] == 2
    assert updates["submission_validation_error"] == "Daily submission limit reached"
    assert route_after_submission({**state, **updates}) == "retry_developer"
