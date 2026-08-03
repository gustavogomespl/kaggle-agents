"""Regression tests for fail-closed Developer artifact retries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kaggle_agents.agents.developer.agent import DeveloperAgent
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    DevelopmentResult,
)


class _TimeoutConfig:
    def get_timeout(self, _component_type: str, _component_name: str) -> int:
        return 60


def _agent(implementation) -> DeveloperAgent:
    agent = object.__new__(DeveloperAgent)
    agent.config = SimpleNamespace(
        ablation=SimpleNamespace(testing_timeout=60),
        component_timeout=_TimeoutConfig(),
        ablation_toggles=None,
    )
    agent.executor = SimpleNamespace(timeout=60, run_mode="")
    agent._implement_component = implementation
    agent._last_reasoning_trace = None
    agent._last_target_source = None
    agent._last_target_source_metadata = None
    agent._last_self_evaluation = None
    agent._preference_collector = SimpleNamespace(
        get_pairs_for_state=lambda: []
    )
    return agent


def _state(
    tmp_path: Path,
    component: AblationComponent,
    *,
    current_index: int = 0,
) -> dict:
    return {
        "ablation_plan": [component],
        "current_component_index": current_index,
        "code_retry_count": 0,
        "working_directory": str(tmp_path),
        "competition_info": CompetitionInfo(
            "demo",
            "",
            "auc",
            "binary_classification",
        ),
        "run_mode": "mlebench",
        "oof_availability": {component.name: True},
        "robustness_approved_components": {component.name: True},
        "component_results": {
            component.name: DevelopmentResult(code="approved", success=True)
        },
        "trusted_component_scores": {component.name: 0.81},
    }


def test_failed_rerun_revokes_stale_approved_artifacts_before_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KAGGLE_AGENTS_MAX_COMPONENT_RETRIES", "3")
    models = tmp_path / "models"
    models.mkdir()
    component = AblationComponent("approved_model", "model", "fit")
    state = _state(tmp_path, component)

    def fail_after_writing(_component, _state):
        (models / "oof_approved_model.npy").write_bytes(b"failed-oof")
        (models / "test_approved_model.npy").write_bytes(b"failed-test")
        (tmp_path / "submission.csv").write_text(
            "id,target\n1,0.1\n",
            encoding="utf-8",
        )
        (tmp_path / "submission_approved_model.csv").write_text(
            "id,target\n1,0.1\n",
            encoding="utf-8",
        )
        return (
            DevelopmentResult(
                code="broken",
                success=False,
                stderr="runtime failure",
                errors=["runtime failure"],
            ),
            [],
        )

    agent = _agent(fail_after_writing)

    first = agent(state)

    assert first["current_component_index"] == 0
    assert first["code_retry_count"] == 1
    assert first["oof_availability"]["approved_model"] is False
    assert first["robustness_approved_components"]["approved_model"] is False
    assert "approved_model" not in first["component_results"]
    assert "approved_model" not in first["trusted_component_scores"]
    assert not (models / "oof_approved_model.npy").exists()
    assert not (models / "test_approved_model.npy").exists()
    assert not (tmp_path / "submission.csv").exists()
    assert not (tmp_path / "submission_approved_model.csv").exists()
    assert list(
        (models / ".rejected" / "approved_model").rglob(
            "oof_approved_model.npy"
        )
    )
    assert len(list((tmp_path / ".rejected_submissions").glob("*.csv"))) == 2

    state.update(first)
    second = agent(state)
    assert second["current_component_index"] == 0
    assert second["code_retry_count"] == 2

    state.update(second)
    third = agent(state)
    assert third["current_component_index"] == 1
    assert third["code_retry_count"] == 0
    assert third["failed_component_names"] == ["approved_model"]


@pytest.mark.parametrize(
    "robustness_approved",
    [True, False],
    ids=["after-robustness", "before-robustness"],
)
def test_component_cannot_overwrite_another_trusted_oof_pair(
    tmp_path: Path,
    monkeypatch,
    robustness_approved: bool,
) -> None:
    monkeypatch.setenv("KAGGLE_AGENTS_MAX_COMPONENT_RETRIES", "3")
    models = tmp_path / "models"
    models.mkdir()
    approved_oof = b"approved-a-oof"
    approved_test = b"approved-a-test"
    (models / "oof_model_a.npy").write_bytes(approved_oof)
    (models / "test_model_a.npy").write_bytes(approved_test)

    component_b = AblationComponent("model_b", "model", "fit")
    state = _state(tmp_path, component_b)
    state["oof_availability"] = {"model_a": True}
    state["robustness_approved_components"] = {
        "model_a": robustness_approved
    }
    approved_result = DevelopmentResult(code="approved-a", success=True)
    state["component_results"] = {"model_a": approved_result}
    state["trusted_component_scores"] = {"model_a": 0.84}

    def overwrite_a(_component, _state):
        (models / "oof_model_a.npy").write_bytes(b"tampered-by-b")
        (models / "test_model_a.npy").write_bytes(b"tampered-by-b")
        (models / "oof_model_b.npy").write_bytes(b"candidate-b-oof")
        (models / "test_model_b.npy").write_bytes(b"candidate-b-test")
        (tmp_path / "submission.csv").write_text(
            "id,target\n1,0.2\n",
            encoding="utf-8",
        )
        return DevelopmentResult(code="b", success=True), []

    updates = _agent(overwrite_a)(state)

    assert (models / "oof_model_a.npy").read_bytes() == approved_oof
    assert (models / "test_model_a.npy").read_bytes() == approved_test
    assert updates["oof_availability"]["model_a"] is True
    assert (
        updates["robustness_approved_components"]["model_a"]
        is robustness_approved
    )
    assert updates["component_results"]["model_a"] is approved_result
    assert updates["trusted_component_scores"]["model_a"] == 0.84
    assert updates["oof_availability"]["model_b"] is False
    assert updates["robustness_approved_components"]["model_b"] is False
    assert updates["current_component_index"] == 0
    assert updates["code_retry_count"] == 1
    assert "Cross-component artifact mutation blocked" in updates["rollback_reason"]
    assert not (models / "oof_model_b.npy").exists()
    assert not (models / "test_model_b.npy").exists()
    assert list(
        (models / ".rejected_cross_component" / "model_b").rglob(
            "oof_model_a.npy"
        )
    )


def test_low_score_retry_uses_same_cleanup_and_does_not_advance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KAGGLE_AGENTS_MAX_COMPONENT_RETRIES", "3")
    monkeypatch.setenv("KAGGLE_AGENTS_MIN_COMPONENT_SCORE", "0.9")
    monkeypatch.setenv("KAGGLE_AGENTS_STRICT_MODE", "0")
    models = tmp_path / "models"
    models.mkdir()
    component = AblationComponent("threshold_model", "model", "fit")
    state = _state(tmp_path, component)
    state["run_mode"] = "kaggle"

    def low_score_candidate(_component, _state):
        np.save(models / "oof_threshold_model.npy", np.array([0.2, 0.8]))
        np.save(models / "test_threshold_model.npy", np.array([0.4]))
        (tmp_path / "submission.csv").write_text(
            "id,target\n1,0.4\n",
            encoding="utf-8",
        )
        return DevelopmentResult(code="candidate", success=True), []

    agent = _agent(low_score_candidate)
    agent._validate_component_improvement = (
        lambda _component, _exec_result, _state: (True, 0.4)
    )

    updates = agent(state)

    assert updates["current_component_index"] == 0
    assert updates["code_retry_count"] == 1
    assert updates["oof_availability"]["threshold_model"] is False
    assert updates["robustness_approved_components"]["threshold_model"] is False
    assert "threshold_model" not in updates["component_results"]
    assert "threshold_model" not in updates["trusted_component_scores"]
    assert not (models / "oof_threshold_model.npy").exists()
    assert not (models / "test_threshold_model.npy").exists()
    assert not (tmp_path / "submission.csv").exists()
    assert "below required threshold" in updates["rollback_reason"]


def test_non_retryable_harness_failure_is_not_cleaned_up_as_a_model_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A preamble failure is not evidence about the component.

    ``_reject_model_candidate`` exists to invalidate a candidate that produced
    something untrustworthy. A program that never reached its own body produced
    nothing, so revoking the component's earlier evidence, charging it a retry
    and eventually listing it as a failed component would all be wrong.
    """
    monkeypatch.setenv("KAGGLE_AGENTS_MAX_COMPONENT_RETRIES", "3")
    models = tmp_path / "models"
    models.mkdir()
    (models / "oof_approved_model.npy").write_bytes(b"accepted-oof")
    component = AblationComponent("approved_model", "model", "fit")
    state = _state(tmp_path, component)
    state["failed_contract_fingerprints"] = {}

    def preamble_failure(_component, _state):
        return (
            DevelopmentResult(
                code="header-only",
                success=False,
                stderr="RuntimeError inside the injected canonical loader",
                errors=["RuntimeError inside the injected canonical loader"],
                failure_origin="harness",
                retryable=False,
                header_sha256="a" * 64,
                contract_fingerprint="b" * 64,
            ),
            [],
        )

    updates = _agent(preamble_failure)(state)

    # Advanced once, never retried, never blamed on the component.
    assert updates["current_component_index"] == 1
    assert updates["code_retry_count"] == 0
    assert "approved_model" not in (updates.get("failed_component_names") or [])
    assert "rollback_reason" not in updates
    # Prior accepted evidence is untouched.
    assert "oof_availability" not in updates
    assert "component_results" not in updates
    assert "trusted_component_scores" not in updates
    assert (models / "oof_approved_model.npy").read_bytes() == b"accepted-oof"
    # Terminal, and the contract is suppressed for the rest of the run.
    assert updates["workflow_valid"] is False
    assert updates["terminal_failure_origin"] == "harness"
    assert updates["terminal_failure_detail"]["contract_fingerprint"] == "b" * 64
    assert "b" * 64 in updates["failed_contract_fingerprints"]
    assert updates.get("skip_remaining_components") is not True
