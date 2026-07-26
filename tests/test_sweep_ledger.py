"""The sweep ledger is the paper's raw data, produced once on a fixed budget.

These drive `run_evaluation` end to end with a stubbed runner, because the
failures they guard against -- a denominator that counts competitions nobody
asked for, an infrastructure failure recorded as an agent failure, a resume that
reuses another protocol's results -- only appear in the assembled ledger.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import kaggle_agents.mlebench as mlebench_pkg
from kaggle_agents.mlebench.runner import MLEBenchResult


@pytest.fixture
def sweep():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "mlebench_eval.py"
    spec = importlib.util.spec_from_file_location("mlebench_eval_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def declared_backbone(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("RUN_SEED", "42")


def _result(competition: str, **overrides) -> MLEBenchResult:
    base = {
        "competition_id": competition,
        "success": True,
        "valid_submission": True,
        "score": 0.9,
        "bronze_medal": True,
        "above_median": True,
        "agent_execution_time": 3600.0,
        "iterations": 3,
    }
    base.update(overrides)
    return MLEBenchResult(**base)


def _stub_runner(monkeypatch, results: dict[str, MLEBenchResult], calls: list):
    def fake_solve(competition_id, **kwargs):
        calls.append(competition_id)
        outcome = results[competition_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(mlebench_pkg, "solve_mlebench", fake_solve)


class TestDenominator:
    def test_summary_counts_only_the_competitions_requested(
        self, sweep, tmp_path, monkeypatch
    ):
        """Reusing an output directory must not inflate the numerator."""
        calls: list[str] = []
        _stub_runner(
            monkeypatch,
            {c: _result(c) for c in ("comp-a", "comp-b", "comp-c")},
            calls,
        )

        sweep.run_evaluation(["comp-a", "comp-b", "comp-c"], output_dir=str(tmp_path))
        _, summary = sweep.run_evaluation(["comp-a"], output_dir=str(tmp_path))

        assert summary["total_competitions"] == 1
        assert summary["completed"] == 1
        assert summary["missing"] == 0
        assert summary["any_medal_percentage"] <= 1.0

    def test_rates_never_exceed_one(self, sweep, tmp_path, monkeypatch):
        calls: list[str] = []
        _stub_runner(monkeypatch, {c: _result(c) for c in ("a", "b")}, calls)

        _, summary = sweep.run_evaluation(["a", "b"], output_dir=str(tmp_path))

        assert 0.0 <= summary["valid_submission_percentage"] <= 1.0
        assert 0.0 <= summary["any_medal_percentage"] <= 1.0
        assert summary["missing"] >= 0


class TestResume:
    def test_completed_competitions_are_not_rerun(self, sweep, tmp_path, monkeypatch):
        calls: list[str] = []
        _stub_runner(monkeypatch, {c: _result(c) for c in ("a", "b")}, calls)

        sweep.run_evaluation(["a", "b"], output_dir=str(tmp_path))
        sweep.run_evaluation(["a", "b"], output_dir=str(tmp_path))

        assert calls == ["a", "b"]

    def test_a_different_protocol_does_not_reuse_results(
        self, sweep, tmp_path, monkeypatch
    ):
        """A changed backbone or budget must invalidate the resume."""
        calls: list[str] = []
        _stub_runner(monkeypatch, {"a": _result("a")}, calls)

        sweep.run_evaluation(["a"], output_dir=str(tmp_path))
        monkeypatch.setenv("LLM_MODEL", "a-different-model")
        sweep.run_evaluation(["a"], output_dir=str(tmp_path))

        assert calls == ["a", "a"]

    def test_infrastructure_failures_are_retried(self, sweep, tmp_path, monkeypatch):
        calls: list[str] = []
        outcomes = {
            "a": _result(
                "a",
                success=False,
                valid_submission=False,
                bronze_medal=False,
                error="401 Unauthorized",
                failure_origin="infrastructure",
            )
        }
        _stub_runner(monkeypatch, outcomes, calls)

        sweep.run_evaluation(["a"], output_dir=str(tmp_path))
        outcomes["a"] = _result("a")
        sweep.run_evaluation(["a"], output_dir=str(tmp_path))

        assert calls == ["a", "a"]

    def test_agent_failures_are_never_retried(self, sweep, tmp_path, monkeypatch):
        """A rerun must not be a second chance at a better outcome."""
        calls: list[str] = []
        _stub_runner(
            monkeypatch,
            {
                "a": _result(
                    "a",
                    success=False,
                    valid_submission=False,
                    bronze_medal=False,
                    error="ValueError: shape mismatch",
                    failure_origin="agent",
                )
            },
            calls,
        )

        sweep.run_evaluation(["a"], output_dir=str(tmp_path))
        sweep.run_evaluation(["a"], output_dir=str(tmp_path))

        assert calls == ["a"]


class TestLedger:
    def test_every_attempt_stays_recorded(self, sweep, tmp_path, monkeypatch):
        """The protocol requires invalid attempts to remain visible."""
        calls: list[str] = []
        outcomes = {
            "a": _result(
                "a",
                success=False,
                valid_submission=False,
                bronze_medal=False,
                error="503 Service Unavailable",
                failure_origin="infrastructure",
            )
        }
        _stub_runner(monkeypatch, outcomes, calls)

        sweep.run_evaluation(["a"], output_dir=str(tmp_path))
        outcomes["a"] = _result("a")
        sweep.run_evaluation(["a"], output_dir=str(tmp_path))

        rows = json.loads((tmp_path / "results.json").read_text())
        assert len(rows) == 2
        assert [r["failure_origin"] for r in rows] == ["infrastructure", None]
        # The retried competition is counted once, not twice.
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["completed"] == 1
        assert summary["invalid_attempts"] == 1

    def test_rows_carry_seed_arm_and_protocol(self, sweep, tmp_path, monkeypatch):
        calls: list[str] = []
        _stub_runner(monkeypatch, {"a": _result("a")}, calls)

        sweep.run_evaluation(["a"], output_dir=str(tmp_path))

        row = json.loads((tmp_path / "results.json").read_text())[0]
        assert row["seed"] == 42
        assert row["arm"] == "full"
        assert row["config_fingerprint"]
        assert row["attempted_at"]

    def test_a_raising_runner_is_a_retryable_harness_attempt(
        self, sweep, tmp_path, monkeypatch
    ):
        calls: list[str] = []
        _stub_runner(monkeypatch, {"a": RuntimeError("exploded")}, calls)

        sweep.run_evaluation(["a"], output_dir=str(tmp_path))

        row = json.loads((tmp_path / "results.json").read_text())[0]
        assert row["terminal_status"] == "harness_exception"
        assert sweep.is_final_result(row) is False

    def test_ledger_write_is_atomic(self, sweep, tmp_path):
        target = tmp_path / "results.json"
        target.write_text('["previous"]', encoding="utf-8")

        sweep._write_json_atomic(target, [{"new": True}])

        assert json.loads(target.read_text()) == [{"new": True}]
        assert not list(tmp_path.glob("*.tmp"))
