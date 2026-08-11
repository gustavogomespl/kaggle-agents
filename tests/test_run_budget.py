"""Run-level budget, signal diagnosis, and crash-recovery guarantees.

These cover the three ways a run used to lose its whole cost or its whole
result: a CPU ceiling that fired long before the wall-clock timeout, an
unbounded run, and a crash that discarded an artifact already accepted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import signal
import subprocess
import time
from pathlib import Path

import pytest

from kaggle_agents.mlebench.runner import MLEBenchRunner, classify_failure_origin
from kaggle_agents.tools.code_executor.process import (
    MIN_CPU_TIME_S,
    cpu_time_budget_for,
    describe_signal_exit,
)
from kaggle_agents.utils.run_budget import (
    budget_exhausted,
    clamp_timeout_to_budget,
    format_remaining,
    remaining_budget_s,
    run_deadline,
)
from kaggle_agents.utils.submission_artifacts import (
    snapshot_accepted_submission,
    verified_accepted_submission,
)
from kaggle_agents.workflow.routing import (
    route_after_iteration_control,
    route_after_meta_evaluator,
)


class TestCpuTimeBudget:
    """RLIMIT_CPU is summed across threads, so a fixed cap silently becomes a
    much tighter wall-clock deadline as soon as generated code uses more than
    one core."""

    def test_budget_scales_with_wall_timeout_and_cores(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 8)

        budget = cpu_time_budget_for(3000)

        # 8 saturated cores for 3000s of wall clock is 24000 CPU-seconds; the
        # ceiling must sit above that, not at the old fixed 7200.
        assert budget >= 3000 * 8

    def test_budget_never_tightens_below_the_historical_floor(self, monkeypatch):
        monkeypatch.setattr("os.cpu_count", lambda: 1)

        assert cpu_time_budget_for(60) == MIN_CPU_TIME_S

    @pytest.mark.parametrize("bad", [None, 0, -1])
    def test_missing_wall_timeout_falls_back_to_a_default(self, bad):
        assert cpu_time_budget_for(bad) >= MIN_CPU_TIME_S

    def test_old_fixed_cap_would_have_killed_a_multicore_component(self, monkeypatch):
        """Regression guard for the behaviour this replaced."""
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        wall_timeout = 2700

        # The previous constant expressed as an effective wall-clock deadline.
        assert wall_timeout > 7200 / 8
        assert cpu_time_budget_for(wall_timeout) / 8 >= wall_timeout


class TestSignalDiagnosis:
    """Death by signal must not read as a defect in the generated program: the
    repair loop would spend its attempts rewriting correct code."""

    def test_normal_exit_has_no_signal_cause(self):
        assert describe_signal_exit(0) is None
        assert describe_signal_exit(1) is None
        assert describe_signal_exit(None) is None

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="POSIX signal numbers"
    )
    def test_cpu_limit_kill_is_named_and_absolved(self):
        message = describe_signal_exit(-int(signal.SIGXCPU))

        assert "SIGXCPU" in message
        assert "RLIMIT_CPU" in message
        assert "not a defect in the code" in message

    def test_unknown_signal_still_reports_a_number(self):
        assert describe_signal_exit(-999) == "Killed by signal 999"


class TestRunBudgetHelpers:
    def test_unbudgeted_run_is_never_constrained(self):
        state = {"run_deadline_ts": None}

        assert run_deadline(state) is None
        assert remaining_budget_s(state) is None
        assert budget_exhausted(state, reserve_s=600) is False
        assert clamp_timeout_to_budget(state, 2700) == 2700
        assert format_remaining(state) == "unbudgeted"

    def test_malformed_deadline_is_treated_as_unbudgeted(self):
        for value in ("not-a-number", -5, 0):
            assert run_deadline({"run_deadline_ts": value}) is None

    def test_exhaustion_respects_the_finalization_reserve(self):
        now = time.time()
        state = {"run_deadline_ts": now + 300}

        assert budget_exhausted(state, reserve_s=600, now=now) is True
        assert budget_exhausted(state, reserve_s=60, now=now) is False

    def test_timeout_is_clamped_to_what_is_left(self):
        now = time.time()
        state = {"run_deadline_ts": now + 1800}

        clamped = clamp_timeout_to_budget(state, 2700, reserve_s=600, now=now)

        assert clamped == 1200

    def test_timeout_is_untouched_when_budget_is_ample(self):
        now = time.time()
        state = {"run_deadline_ts": now + 20_000}

        assert clamp_timeout_to_budget(state, 2700, reserve_s=600, now=now) == 2700

    def test_clamp_never_returns_a_non_positive_timeout(self):
        now = time.time()
        state = {"run_deadline_ts": now + 1}

        assert clamp_timeout_to_budget(state, 2700, reserve_s=600, now=now) > 0


class TestBudgetStopsTheLoop:
    """Iteration counts alone never bounded a run; the deadline is a second,
    independent stop condition -- and it is a clock, not a score."""

    def _state(self, remaining_s: float | None) -> dict:
        state = {
            "run_mode": "mlebench",
            "current_iteration": 1,
            "max_iterations": 3,
            "needs_refinement": True,
        }
        if remaining_s is not None:
            state["run_deadline_ts"] = time.time() + remaining_s
        return state

    def test_refines_while_budget_remains(self):
        assert route_after_iteration_control(self._state(20_000)) == "refine"

    def test_unbudgeted_run_keeps_previous_behaviour(self):
        assert route_after_iteration_control(self._state(None)) == "refine"

    def test_ends_when_budget_is_exhausted_before_max_iterations(self):
        state = self._state(60)

        assert route_after_iteration_control(state) == "end"
        # The iteration budget was not the reason.
        assert state["current_iteration"] < state["max_iterations"]

    def test_recovery_routes_are_skipped_when_out_of_budget(self):
        state = self._state(60)
        state["stagnation_detection"] = {"trigger_sota_search": True}

        assert route_after_meta_evaluator(state) == "skip_recovery"

    def test_recovery_routes_still_fire_with_budget_left(self):
        state = self._state(20_000)
        state["stagnation_detection"] = {"trigger_sota_search": True}

        assert route_after_meta_evaluator(state) == "sota_search"


class TestPlantedSnapshotsAreNotGraded:
    """The snapshot store lives inside the workspace the generated code writes
    to, and the filesystem guard only protects canonical/. Recovery therefore
    must never trust a file just because its name matches its own hash."""

    def test_planted_snapshot_without_state_evidence_is_ignored(self, tmp_path):
        """A candidate can create a correctly-named, correctly-hashed file."""
        workspace = tmp_path
        store = workspace / ".accepted_submissions" / "run1"
        store.mkdir(parents=True)
        body = b"id,target\n1,1.0\n"
        digest = hashlib.sha256(body).hexdigest()
        planted = store / f"iteration-9999-{digest}.csv"
        planted.write_bytes(body)

        # No state ever bound this digest, so nothing resolves it.
        assert verified_accepted_submission({"run_id": "run1"}, workspace) is None

    def test_only_the_state_bound_digest_resolves(self, tmp_path):
        workspace = tmp_path
        source = workspace / "submission.csv"
        source.write_text("id,target\n1,0.9\n", encoding="utf-8")
        snapshot, digest = snapshot_accepted_submission(
            workspace, source, run_id="run1", iteration=1
        )

        # A second, planted artifact sits in the same store.
        planted_body = b"id,target\n1,0.0\n"
        planted_digest = hashlib.sha256(planted_body).hexdigest()
        (snapshot.parent / f"iteration-9999-{planted_digest}.csv").write_bytes(
            planted_body
        )

        state = {
            "run_id": "run1",
            "accepted_submission_path": str(snapshot),
            "accepted_submission_snapshot_path": str(snapshot),
            "accepted_submission_sha256": digest,
        }
        resolved = verified_accepted_submission(state, workspace)

        assert resolved == snapshot
        assert resolved.read_bytes() != planted_body


def _load_sweep_module():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "mlebench_eval.py"
    spec = importlib.util.spec_from_file_location("mlebench_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSweepResume:
    """A sweep is the entire GPU budget: a crash must not restart it. Resume
    must also stay outcome-independent, or it becomes a search for good runs."""

    def test_completed_attempt_is_final_regardless_of_outcome(self):
        sweep = _load_sweep_module()

        for outcome in (
            {"valid_submission": True, "any_medal": True},
            {"valid_submission": False, "any_medal": False},
            {"valid_submission": True, "gold_medal": False, "score": None},
            {"failure_origin": "agent"},
        ):
            row = {"terminal_status": "completed", **outcome}
            assert sweep.is_final_result(row) is True

    def test_harness_exception_is_retryable(self):
        sweep = _load_sweep_module()

        assert sweep.is_final_result({"terminal_status": "harness_exception"}) is False
        assert sweep.is_final_result({}) is False

    @pytest.mark.parametrize("origin", ["infrastructure", "harness"])
    def test_infrastructure_failures_are_invalid_attempts(self, origin):
        """A 401 or a missing grader is not a failed run by the agent."""
        sweep = _load_sweep_module()
        row = {"terminal_status": "completed", "failure_origin": origin}

        assert sweep.is_final_result(row) is False

    def test_run_key_separates_seeds_arms_and_protocols(self):
        sweep = _load_sweep_module()
        base = {
            "competition_id": "c",
            "seed": 42,
            "arm": "full",
            "config_fingerprint": "abc123",
        }

        assert sweep._run_key(base) != sweep._run_key({**base, "seed": 43})
        assert sweep._run_key(base) != sweep._run_key({**base, "arm": "without-search"})
        assert sweep._run_key(base) != sweep._run_key(
            {**base, "config_fingerprint": "def456"}
        )
        assert sweep._run_key(base) == sweep._run_key(dict(base))

    def test_fingerprint_changes_with_anything_that_defines_the_protocol(self):
        sweep = _load_sweep_module()
        base = dict(model="gemini-3-flash", budget=25200, commit="abc", arm="full")
        reference = sweep.config_fingerprint(**base)

        assert sweep.config_fingerprint(**{**base, "model": "gpt-4o-mini"}) != reference
        assert sweep.config_fingerprint(**{**base, "budget": 18000}) != reference
        assert sweep.config_fingerprint(**{**base, "commit": "def"}) != reference
        assert sweep.config_fingerprint(**base) == reference


class TestFailureOrigin:
    """The protocol counts an agent failure once and never reruns it, while an
    infrastructure or harness failure is an invalid attempt. Conflating them
    lowers the reported rate and makes the attempt unrecoverable."""

    @pytest.mark.parametrize(
        "message",
        [
            "openai.AuthenticationError: 401 Unauthorized",
            "403 Forbidden",
            "RateLimitError: 429 Too Many Requests",
            "503 Service Unavailable",
            "insufficient_quota",
        ],
    )
    def test_provider_failures_are_infrastructure(self, message):
        assert classify_failure_origin(message) == "infrastructure"

    @pytest.mark.parametrize(
        "message",
        [
            "mlebench command not found. Install with: pip install -e ...",
            "Private directory is empty",
            "OSError: [Errno 28] No space left on device",
        ],
    )
    def test_environment_failures_are_harness(self, message):
        assert classify_failure_origin(message) == "harness"

    def test_unrecognised_failures_stay_with_the_agent(self):
        """Conservative default: a rerun must never be a second chance."""
        assert classify_failure_origin("KeyError: 'target'") == "agent"
        assert classify_failure_origin("ValueError: shape mismatch") == "agent"

    def test_no_error_is_no_failure(self):
        assert classify_failure_origin(None) is None
        assert classify_failure_origin("") is None


class TestBackbonePreflight:
    """One missing variable silently swaps the backbone under test -- the exact
    confound a matched comparison exists to control -- and a whole sweep can
    finish before anyone notices."""

    def _clear(self, monkeypatch):
        for name in (
            "LLM_MODEL",
            "LLM_PROVIDER",
            "KAGGLE_AGENTS_EXPECTED_MODEL",
            "KAGGLE_AGENTS_ALLOW_DEFAULT_BACKBONE",
            "PLANNER_MODEL",
            "PLANNER_PROVIDER",
            "DEVELOPER_MODEL",
            "DEVELOPER_PROVIDER",
            "EVALUATOR_MODEL",
            "EVALUATOR_PROVIDER",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_undeclared_backbone_aborts(self, monkeypatch):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        self._clear(monkeypatch)

        with pytest.raises(RuntimeError, match="LLM_MODEL and LLM_PROVIDER"):
            MLEBenchRunner._preflight_backbone()

    def test_consistent_declaration_passes_and_is_recorded(self, monkeypatch):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "google/gemini-3-flash-preview")

        resolved = MLEBenchRunner._preflight_backbone()

        assert resolved["base"] == "openai/google/gemini-3-flash-preview"
        assert set(resolved) == {"base", "planner", "developer", "evaluator"}

    def test_role_override_with_a_different_model_aborts(self, monkeypatch):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "google/gemini-3-flash-preview")
        monkeypatch.setenv("PLANNER_MODEL", "gpt-4o-mini")

        with pytest.raises(RuntimeError, match="per-role overrides disagree"):
            MLEBenchRunner._preflight_backbone()

    def test_role_override_repeating_the_same_model_is_fine(self, monkeypatch):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setenv("DEVELOPER_MODEL", "m")
        monkeypatch.setenv("DEVELOPER_PROVIDER", "openai")

        assert MLEBenchRunner._preflight_backbone()["developer"] == "openai/m"

    def test_expected_model_mismatch_aborts(self, monkeypatch):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("KAGGLE_AGENTS_EXPECTED_MODEL", "google/gemini-3-flash-preview")

        with pytest.raises(RuntimeError, match="expected"):
            MLEBenchRunner._preflight_backbone()

    def test_escape_hatch_disables_the_check(self, monkeypatch):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        self._clear(monkeypatch)
        monkeypatch.setenv("KAGGLE_AGENTS_ALLOW_DEFAULT_BACKBONE", "true")

        assert MLEBenchRunner._preflight_backbone() == {}


class TestGradingTimeoutIsAHarnessCondition:
    """A 60s cap zeroed pixel-scale submissions deterministically, and the
    timeout message matched no failure signature, so the run was billed to
    the agent and became ineligible for rerun — with a perfect hash-verified
    artifact sitting on disk. Grading runs once at the very end of a
    multi-hour run; the cap must be generous and a timeout must surface as a
    harness condition."""

    def _runner(self, tmp_path: Path):
        return MLEBenchRunner(
            mle_cache_path=tmp_path / "cache", workspace_base=tmp_path / "ws"
        )

    def _timing_out_run(self, monkeypatch, captured: dict):
        def fake_run(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            raise subprocess.TimeoutExpired(
                cmd="mlebench", timeout=kwargs.get("timeout")
            )

        monkeypatch.setattr(
            "kaggle_agents.mlebench.runner.subprocess.run", fake_run
        )

    def test_timeout_reports_grading_unavailable(self, tmp_path, monkeypatch):
        self._timing_out_run(monkeypatch, {})
        grading = self._runner(tmp_path)._grade_submission(
            "comp", tmp_path / "s.csv"
        )
        assert grading["valid_submission"] is False
        assert grading.get("grading_unavailable") is True

    def test_timeout_budget_is_generous_and_configurable(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}
        self._timing_out_run(monkeypatch, captured)
        runner = self._runner(tmp_path)

        monkeypatch.delenv("KAGGLE_AGENTS_GRADING_TIMEOUT_S", raising=False)
        runner._grade_submission("comp", tmp_path / "s.csv")
        assert captured["timeout"] >= 600

        monkeypatch.setenv("KAGGLE_AGENTS_GRADING_TIMEOUT_S", "1234")
        runner._grade_submission("comp", tmp_path / "s.csv")
        assert captured["timeout"] == 1234


class TestGradingSurvivesACrash:
    """Telemetry and the single grading pass used to live inside the same try
    as ``workflow.invoke``: any exception discarded both, so a run holding a
    verified accepted artifact was recorded as having produced nothing."""

    def _runner(self, tmp_path: Path):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        return MLEBenchRunner(
            mle_cache_path=tmp_path / "cache", workspace_base=tmp_path / "ws"
        )

    def _result(self, error: str | None = None):
        from kaggle_agents.mlebench.runner import MLEBenchResult

        return MLEBenchResult(competition_id="c", success=False, error=error)

    def _accepted(self, workspace: Path, run_id: str) -> dict:
        """Accept an artifact and return the state the graph would have emitted."""
        source = workspace / "submission.csv"
        source.write_text("id,target\n1,0.9\n", encoding="utf-8")
        snapshot, digest = snapshot_accepted_submission(
            workspace, source, run_id=run_id, iteration=1
        )
        return {
            "run_id": run_id,
            "accepted_submission_path": str(snapshot),
            "accepted_submission_snapshot_path": str(snapshot),
            "accepted_submission_sha256": digest,
        }

    def test_crashed_run_is_still_graded_on_its_accepted_artifact(
        self, tmp_path, monkeypatch
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        observed = self._accepted(workspace, "crash-run")

        runner = self._runner(tmp_path)
        graded: list[Path] = []

        def fake_grade(competition_id, path):
            graded.append(path)
            return {"valid_submission": True, "score": 0.91, "bronze_medal": True}

        monkeypatch.setattr(runner, "_grade_submission", fake_grade)
        result = self._result(error="boom")

        # final_state is None: exactly what an exception from the graph leaves.
        # The last streamed state still binds the accepted digest.
        runner._finalize_run(
            result, workspace, "crash-run", None, {}, last_observed_state=observed
        )

        assert len(graded) == 1
        assert result.valid_submission is True
        assert result.bronze_medal is True
        assert result.score == pytest.approx(0.91)
        # The attempt stays on the ledger as a failure, not a clean completion.
        assert result.success is False
        assert result.error == "boom"

    def test_telemetry_is_written_even_without_a_final_state(
        self, tmp_path, monkeypatch
    ):
        import json

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = self._runner(tmp_path)
        monkeypatch.setattr(runner, "_grade_submission", lambda *a: {})

        runner._finalize_run(
            self._result(error="boom"), workspace, "run", None, {}, last_observed_state={}
        )

        telemetry = json.loads((workspace / "telemetry.json").read_text())
        assert telemetry["terminal_status"] == "workflow_exception"
        assert telemetry["error"] == "boom"

    def test_keyboard_interrupt_is_annotated_finalized_once_and_reraised(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = self._runner(tmp_path)
        finalized = []

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt("stop")

        def record_finalization(result, *_args, **_kwargs):
            finalized.append(result)

        monkeypatch.setattr(runner, "_resolve_evaluation_metric", interrupt)
        monkeypatch.setattr(runner, "_finalize_run", record_finalization)

        with pytest.raises(KeyboardInterrupt, match="stop") as excinfo:
            runner.run("competition")

        assert len(finalized) == 1
        result = finalized[0]
        assert result.failure_origin == "harness"
        assert result.terminal_failure_detail == {
            "reason": "keyboard_interrupt",
            "exception_type": "KeyboardInterrupt",
        }
        assert "KeyboardInterrupt" in result.error
        assert "KeyboardInterrupt" in result.traceback
        assert excinfo.value.mlebench_result is result

    def test_finalization_error_does_not_replace_keyboard_interrupt(
        self,
        tmp_path,
        monkeypatch,
    ):
        runner = self._runner(tmp_path)
        finalized = []

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt("stop")

        def broken_finalization(result, *_args, **_kwargs):
            finalized.append(result)
            raise RuntimeError("telemetry disk failed")

        monkeypatch.setattr(runner, "_resolve_evaluation_metric", interrupt)
        monkeypatch.setattr(runner, "_finalize_run", broken_finalization)

        with pytest.raises(KeyboardInterrupt, match="stop") as excinfo:
            runner.run("competition")

        assert len(finalized) == 1
        assert excinfo.value.mlebench_result is finalized[0]
        assert finalized[0].terminal_failure_detail["finalization_error"] == (
            "RuntimeError: telemetry disk failed"
        )

    def test_terminal_agent_state_takes_precedence_over_late_interrupt(
        self,
        tmp_path,
        monkeypatch,
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = self._runner(tmp_path)
        monkeypatch.setattr(runner, "_grade_submission", lambda *_args: {})
        result = self._result(error="KeyboardInterrupt: stop")
        result.failure_origin = "harness"
        result.terminal_failure_detail = {
            "reason": "keyboard_interrupt",
            "exception_type": "KeyboardInterrupt",
        }
        final_state = {
            "workflow_valid": False,
            "terminal_failure_origin": "agent",
            "terminal_failure_detail": {"reason": "model_failure"},
        }

        runner._finalize_run(
            result,
            workspace,
            "run",
            final_state,
            {},
        )

        assert result.failure_origin == "agent"
        assert result.terminal_failure_detail == {"reason": "model_failure"}
        telemetry = json.loads((workspace / "telemetry.json").read_text())
        assert telemetry["terminal_status"] == "completed"
        assert telemetry["failure_origin"] == "agent"

    def test_interrupt_after_final_state_never_becomes_clean_success(
        self,
        tmp_path,
        monkeypatch,
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        observed = self._accepted(workspace, "late-interrupt")
        observed["workflow_valid"] = True
        runner = self._runner(tmp_path)
        monkeypatch.setattr(
            runner,
            "_grade_submission",
            lambda *_args: {"valid_submission": True, "score": 0.8},
        )
        result = self._result(error="KeyboardInterrupt: stop")
        result.failure_origin = "harness"
        result.terminal_failure_detail = {
            "reason": "keyboard_interrupt",
            "exception_type": "KeyboardInterrupt",
        }

        runner._finalize_run(
            result,
            workspace,
            "late-interrupt",
            observed,
            {},
        )

        assert result.valid_submission is True
        assert result.success is False
        assert result.failure_origin == "harness"
        telemetry = json.loads((workspace / "telemetry.json").read_text())
        assert telemetry["terminal_status"] == "harness_exception"

    def test_nothing_is_graded_when_no_artifact_was_accepted(
        self, tmp_path, monkeypatch
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = self._runner(tmp_path)

        def unexpected(*args):
            raise AssertionError("must not grade without an accepted artifact")

        monkeypatch.setattr(runner, "_grade_submission", unexpected)
        result = self._result(error="boom")

        runner._finalize_run(result, workspace, "run", None, {}, last_observed_state={})

        assert result.valid_submission is False

    def test_a_planted_snapshot_is_not_graded_after_a_crash(
        self, tmp_path, monkeypatch
    ):
        """Generated code can write into the store; only state evidence counts."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        store = workspace / ".accepted_submissions" / "crash-run"
        store.mkdir(parents=True)
        body = b"id,target\n1,1.0\n"
        digest = hashlib.sha256(body).hexdigest()
        (store / f"iteration-9999-{digest}.csv").write_bytes(body)

        runner = self._runner(tmp_path)

        def unexpected(*args):
            raise AssertionError("a planted artifact must never be graded")

        monkeypatch.setattr(runner, "_grade_submission", unexpected)
        result = self._result(error="boom")

        runner._finalize_run(
            result, workspace, "crash-run", None, {}, last_observed_state={}
        )

        assert result.valid_submission is False

    def test_a_missing_workspace_is_a_no_op(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)
        monkeypatch.setattr(
            runner, "_grade_submission", lambda *a: pytest.fail("no workspace")
        )

        runner._finalize_run(self._result(), None, None, None, {})

    def test_a_harness_truncated_run_still_grades_what_it_accepted(
        self, tmp_path, monkeypatch
    ):
        """Iteration 2 dies in the injected preamble; iteration 1's artifact stands.

        The run is invalid and eligible for rerun, but it demonstrably produced
        a hash-verified submission. Refusing to grade it would report a valid
        result as nothing at all, and the sweep would lose both the score and
        the reason.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        observed = self._accepted(workspace, "harness-run")
        observed.update(
            {
                "workflow_valid": False,
                "terminal_failure_origin": "harness",
                "terminal_failure_detail": {
                    "reason": "injected_header_failure",
                    "component": "model_b",
                    "contract_fingerprint": "c" * 64,
                },
                # Both legacy keys exist and are None: the run must still
                # report a non-empty error.
                "submission_validation_error": None,
                "termination_reason": None,
            }
        )

        runner = self._runner(tmp_path)
        graded: list[Path] = []

        def fake_grade(competition_id, path):
            graded.append(path)
            return {"valid_submission": True, "score": 0.66, "above_median": True}

        monkeypatch.setattr(runner, "_grade_submission", fake_grade)
        result = self._result()

        runner._finalize_run(result, workspace, "harness-run", observed, {})

        assert len(graded) == 1
        assert result.valid_submission is True
        assert result.score == pytest.approx(0.66)
        # Invalid attempt, eligible for rerun - not a clean completion.
        assert result.success is False
        assert result.failure_origin == "harness"
        assert classify_failure_origin(result.error) is not None
        assert result.terminal_failure_detail["contract_fingerprint"] == "c" * 64
        assert result.error
