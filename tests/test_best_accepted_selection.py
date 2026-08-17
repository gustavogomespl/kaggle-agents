"""The graded artifact must be the best of the run, not the last of the run.

Every iteration used to snapshot unconditionally, so a refinement that made
things worse still became the artifact handed to the grader. These tests also
pin the budget gate that replaced "refinement is off in mlebench".
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.agents.developer.refinement import RefinementMixin
from kaggle_agents.agents.submission_agent import SubmissionAgent
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.utils.submission_artifacts import (
    snapshot_accepted_submission,
    snapshot_best_candidate_submission,
)


RUN_ID = "selection-run"


def _competition(metric: str = "auc") -> CompetitionInfo:
    return CompetitionInfo(
        name="selection-test",
        description="",
        evaluation_metric=metric,
        problem_type="classification",
    )


def _write_submission(path: Path, values: list[float]) -> None:
    pd.DataFrame({"id": list(range(1, len(values) + 1)), "target": values}).to_csv(
        path, index=False
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    _write_submission(tmp_path / "sample_submission.csv", [0.5, 0.5])
    return tmp_path


def _accepted_state(workspace: Path, accepted_values: list[float], cv: float) -> dict:
    """State carrying an already-accepted artifact scored at ``cv``."""
    accepted_source = workspace / "accepted_source.csv"
    _write_submission(accepted_source, accepted_values)
    snapshot, digest = snapshot_accepted_submission(
        workspace, accepted_source, run_id=RUN_ID, iteration=0
    )
    return {
        "working_directory": str(workspace),
        "competition_info": _competition(),
        "sample_submission_path": str(workspace / "sample_submission.csv"),
        "run_mode": "mlebench",
        "run_id": RUN_ID,
        "accepted_submission_path": str(snapshot),
        "accepted_submission_snapshot_path": str(snapshot),
        "accepted_submission_sha256": digest,
        "accepted_submission_cv_score": cv,
        "accepted_submission_score_owner": "model_a",
        "accepted_submission_score_source": "trusted_component_scores",
        "robustness_approved_components": {"model_a": True},
    }


def _candidate(workspace: Path, values: list[float], cv: float, state: dict) -> dict:
    """Add a fresh candidate artifact with hash-bound CV evidence."""
    submission = workspace / "submission.csv"
    _write_submission(submission, values)
    snapshot, digest = snapshot_best_candidate_submission(
        workspace, submission, run_id=RUN_ID, iteration=1
    )
    state.update(
        {
            "best_candidate_submission_snapshot_path": str(snapshot),
            "best_candidate_submission_sha256": digest,
            "best_candidate_submission_component_name": "model_a",
            "trusted_component_scores": {"model_a": cv},
        }
    )
    return state


class TestBestOfRunSelection:
    def test_worse_later_iteration_does_not_replace_the_accepted_artifact(
        self, workspace: Path
    ):
        accepted_bytes = None
        state = _accepted_state(workspace, [0.2, 0.8], cv=0.84)
        accepted_bytes = Path(state["accepted_submission_snapshot_path"]).read_bytes()
        state = _candidate(workspace, [0.9, 0.1], cv=0.79, state=state)

        updates = SubmissionAgent()(state)

        result = updates["submissions"][0]
        assert result.valid is True
        assert result.cv_score == pytest.approx(0.84)
        # submission.csv was restored to the better artifact, byte for byte.
        assert (workspace / "submission.csv").read_bytes() == accepted_bytes
        # The accepted provenance must not be re-pointed at the worse artifact.
        assert "accepted_submission_sha256" not in updates

    def test_better_later_iteration_is_accepted(self, workspace: Path):
        state = _accepted_state(workspace, [0.2, 0.8], cv=0.79)
        state = _candidate(workspace, [0.9, 0.1], cv=0.84, state=state)

        updates = SubmissionAgent()(state)

        assert updates["accepted_submission_cv_score"] == pytest.approx(0.84)
        assert updates["submissions"][0].cv_score == pytest.approx(0.84)

    def test_ties_do_not_churn_the_accepted_artifact(self, workspace: Path):
        state = _accepted_state(workspace, [0.2, 0.8], cv=0.84)
        state = _candidate(workspace, [0.9, 0.1], cv=0.84, state=state)

        updates = SubmissionAgent()(state)

        assert updates["submissions"][0].cv_score == pytest.approx(0.84)
        assert "accepted_submission_sha256" not in updates

    def test_minimization_metric_respects_direction(self, workspace: Path):
        state = _accepted_state(workspace, [0.2, 0.8], cv=0.30)
        state["competition_info"] = _competition("rmse")
        state = _candidate(workspace, [0.9, 0.1], cv=0.45, state=state)

        updates = SubmissionAgent()(state)

        # Higher RMSE is worse, so the accepted artifact is kept.
        assert updates["submissions"][0].cv_score == pytest.approx(0.30)
        assert "accepted_submission_sha256" not in updates

    def test_minimization_metric_accepts_a_lower_score(self, workspace: Path):
        state = _accepted_state(workspace, [0.2, 0.8], cv=0.45)
        state["competition_info"] = _competition("rmse")
        state = _candidate(workspace, [0.9, 0.1], cv=0.30, state=state)

        updates = SubmissionAgent()(state)

        assert updates["accepted_submission_cv_score"] == pytest.approx(0.30)

    def test_unscored_artifact_cannot_displace_a_scored_one(self, workspace: Path):
        state = _accepted_state(workspace, [0.2, 0.8], cv=0.84)
        # A candidate with no hash-bound provenance at all.
        _write_submission(workspace / "submission.csv", [0.9, 0.1])

        updates = SubmissionAgent()(state)

        assert updates["submissions"][0].cv_score == pytest.approx(0.84)
        assert "accepted_submission_sha256" not in updates

    def test_first_accepted_artifact_is_unaffected(self, workspace: Path):
        """With nothing accepted yet there is nothing to compare against."""
        state = {
            "working_directory": str(workspace),
            "competition_info": _competition(),
            "sample_submission_path": str(workspace / "sample_submission.csv"),
            "run_mode": "mlebench",
            "run_id": RUN_ID,
            "robustness_approved_components": {"model_a": True},
        }
        state = _candidate(workspace, [0.2, 0.8], cv=0.73, state=state)

        updates = SubmissionAgent()(state)

        assert updates["accepted_submission_cv_score"] == pytest.approx(0.73)

    def test_unscored_lane_still_accepts(self, workspace: Path):
        """Domains without canonical labels never produce a CV score; those runs
        must keep working rather than freezing on their first artifact."""
        state = {
            "working_directory": str(workspace),
            "competition_info": _competition(),
            "sample_submission_path": str(workspace / "sample_submission.csv"),
            "run_mode": "mlebench",
            "run_id": RUN_ID,
        }
        _write_submission(workspace / "submission.csv", [0.2, 0.8])

        updates = SubmissionAgent()(state)

        assert updates["submissions"][0].valid is True
        assert updates["accepted_submission_cv_score"] is None


class _Ablation:
    enable_refinement = True


class _Config:
    ablation = _Ablation()


class _Refiner(RefinementMixin):
    """Minimal host for the mixin's gating logic."""

    config = _Config()


class TestRefinementBudgetGate:
    def _component(self) -> AblationComponent:
        return AblationComponent(name="model_a", component_type="model", code="")

    def test_refinement_is_no_longer_disabled_in_mlebench(self):
        """Regression guard: this loop returned 0 in the benchmarked mode, so
        the only mechanism that improves a working model never ran."""
        assert _Refiner()._get_refinement_iterations({"run_mode": "mlebench"}) > 0

    def test_iteration_count_is_configurable_per_mode(self, monkeypatch):
        monkeypatch.setenv("MLEBENCH_REFINEMENT_ITERS", "0")

        assert _Refiner()._get_refinement_iterations({"run_mode": "mlebench"}) == 0
        assert _Refiner()._get_refinement_iterations({"run_mode": "kaggle"}) == 2

    def test_refines_when_the_clock_allows_another_pass(self):
        state = {"run_mode": "mlebench", "run_deadline_ts": time.time() + 20_000}

        assert (
            _Refiner()._should_run_refinement(
                self._component(),
                state,
                new_cv_score=0.8,
                execution_time_s=100.0,
                component_timeout_s=2700,
            )
            is True
        )

    def test_does_not_refine_when_the_pass_would_not_fit(self):
        state = {"run_mode": "mlebench", "run_deadline_ts": time.time() + 300}

        assert (
            _Refiner()._should_run_refinement(
                self._component(),
                state,
                new_cv_score=0.8,
                execution_time_s=1200.0,
                component_timeout_s=2700,
            )
            is False
        )

    def test_component_that_used_most_of_its_budget_is_not_rerun(self):
        state = {"run_mode": "mlebench", "run_deadline_ts": time.time() + 20_000}

        assert (
            _Refiner()._should_run_refinement(
                self._component(),
                state,
                new_cv_score=0.8,
                execution_time_s=2000.0,
                component_timeout_s=2700,
            )
            is False
        )

    def test_non_model_components_are_never_refined(self):
        component = AblationComponent(
            name="prep", component_type="preprocessing", code=""
        )
        state = {"run_mode": "mlebench", "run_deadline_ts": time.time() + 20_000}

        assert (
            _Refiner()._should_run_refinement(
                component,
                state,
                new_cv_score=0.8,
                execution_time_s=10.0,
                component_timeout_s=2700,
            )
            is False
        )
