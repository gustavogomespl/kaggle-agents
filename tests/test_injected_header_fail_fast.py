"""Immutable-preamble failures are harness failures, not model quality.

Everything above ``# === END PATH CONSTANTS ===`` is generator-owned: path
constants, canonical loaders and the injected helpers. When that region raises,
no candidate the LLM could write would repair it, so both retry levels (fixer /
debugger inside the component, and the outer component retry) are wasted budget
and the attempt is not evidence about the model.

The classification is deliberately conservative: harness only when the executor
proves the candidate body was never reached *in the exact script it launched*.
Anything ambiguous stays an ordinary retryable failure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kaggle_agents.agents.developer.execution_failures import (
    INJECTED_HEADER_END_MARKER,
    INJECTED_INPUT_MANIFEST_PREFIX,
    GeneratedContractStructureError,
    HeaderInputManifest,
    ProtectedInputMutationError,
    RepeatedInjectedContractError,
    annotate_generated_execution,
    execute_generated_candidate,
    execution_failure_to_development_result,
    generated_contract_fingerprint,
    generated_header,
    generated_header_sha256,
    parse_exact_header_manifest,
    render_header_manifest_line,
    require_one_exact_generated_header_and_manifest,
    sanitize_candidate_body,
)
from kaggle_agents.agents.developer.retry import preserve_injected_header
from kaggle_agents.agents.developer.target_source import (
    CanonicalTargetContractError,
    ProtectedInput,
)
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.tools.code_executor.dataclasses import ExecutionResult
from kaggle_agents.utils.submission_artifacts import verified_accepted_submission


SCRIPT = "/ws/_exec_0123456789abcdef0123456789abcdef.py"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _manifest(
    fingerprint: str = "targetfp",
    protected: tuple[ProtectedInput, ...] = (),
) -> HeaderInputManifest:
    return HeaderInputManifest(
        target_source_fingerprint=fingerprint,
        protected_inputs=protected,
    )


def _header(
    component: str = "model_a",
    fingerprint: str = "targetfp",
    protected: tuple[ProtectedInput, ...] = (),
    canonical_body: str = "CANONICAL_Y = _load()",
) -> str:
    """A header shaped like the real one: constants, loaders, helpers, marker."""
    return (
        "# === PATH CONSTANTS (AUTO-INJECTED - DO NOT MODIFY) ===\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        f'COMPONENT_NAME = "{component}"\n'
        "\n"
        "# === CANONICAL DATA CONTRACT ===\n"
        f"{canonical_body}\n"
        "\n"
        "def write_submission(preds):\n"
        "    raise RuntimeError('helper')\n"
        "\n"
        f"{render_header_manifest_line(_manifest(fingerprint, protected))}\n"
        f"{INJECTED_HEADER_END_MARKER}\n"
    )


def _marker_line(code: str) -> int:
    for index, line in enumerate(code.splitlines(), start=1):
        if line.rstrip("\r\n") == INJECTED_HEADER_END_MARKER:
            return index
    raise AssertionError("no marker")


def _frame(line: int, script: str = SCRIPT, func: str = "<module>") -> str:
    return f'  File "{script}", line {line}, in {func}\n    boom\n'


def _failed(
    code: str,
    *,
    stderr: str,
    candidate_body_reached: bool | None,
    script: str | None = SCRIPT,
    errors: list[str] | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        success=False,
        stdout="",
        stderr=stderr,
        execution_time=1.0,
        exit_code=1,
        artifacts_created=[],
        errors=errors if errors is not None else ["RuntimeError: boom"],
        executed_script_path=script,
        candidate_body_reached=candidate_body_reached,
    )


BODY = "print('body')\nwrite_submission([1])\n"


# ---------------------------------------------------------------------------
# 1-2. Preamble failures are harness failures
# ---------------------------------------------------------------------------


def test_failure_deep_in_the_injected_canonical_section_is_harness() -> None:
    code = _header() + BODY
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr="Traceback (most recent call last):\n" + _frame(marker - 3),
        candidate_body_reached=False,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin == "harness"
    assert annotated.retryable is False
    assert annotated.header_sha256 == generated_header_sha256(code)
    assert annotated.contract_fingerprint == generated_contract_fingerprint(code)


def test_failure_near_the_start_of_the_preamble_is_harness() -> None:
    code = _header() + BODY
    result = _failed(
        code,
        stderr="Traceback (most recent call last):\n" + _frame(2),
        candidate_body_reached=False,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin == "harness"
    assert annotated.retryable is False


# ---------------------------------------------------------------------------
# 3-5. Anything that reached the body stays retryable
# ---------------------------------------------------------------------------


def test_direct_candidate_body_failure_after_the_marker_is_retryable() -> None:
    code = _header() + BODY
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr="Traceback (most recent call last):\n" + _frame(marker + 2),
        candidate_body_reached=True,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin is None
    assert annotated.retryable is True


def test_injected_helper_raising_below_a_body_frame_is_retryable() -> None:
    """``write_submission()`` is defined in the header but called by the body."""
    code = _header() + BODY
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr=(
            "Traceback (most recent call last):\n"
            + _frame(marker + 2)
            + _frame(marker - 3, func="write_submission")
        ),
        candidate_body_reached=True,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin is None
    assert annotated.retryable is True


def test_helper_failure_without_a_body_frame_is_retryable_when_token_observed() -> None:
    """atexit/callback/child paths can lose the body frame; the token cannot."""
    code = _header() + BODY
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr=(
            "Traceback (most recent call last):\n"
            + _frame(marker - 3, func="write_submission")
        ),
        candidate_body_reached=True,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin is None
    assert annotated.retryable is True


# ---------------------------------------------------------------------------
# 6. Only the exact launched script is evidence
# ---------------------------------------------------------------------------


def test_a_stale_or_spoofed_exec_path_is_never_harness() -> None:
    code = _header() + BODY
    marker = _marker_line(code)
    stale = "/ws/_exec_ffffffffffffffffffffffffffffffff.py"
    result = _failed(
        code,
        stderr="Traceback (most recent call last):\n" + _frame(marker - 3, script=stale),
        candidate_body_reached=False,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin is None
    assert annotated.retryable is True


def test_no_traceback_frame_at_all_is_never_harness() -> None:
    code = _header() + BODY
    result = _failed(code, stderr="killed", candidate_body_reached=False)

    assert annotate_generated_execution(code, result).retryable is True


# ---------------------------------------------------------------------------
# 7-8. Fingerprint normalization
# ---------------------------------------------------------------------------


def test_headers_differing_only_in_component_name_share_a_fingerprint() -> None:
    first = _header(component="model_a") + BODY
    second = _header(component="model_b") + BODY

    assert generated_header_sha256(first) != generated_header_sha256(second)
    assert generated_contract_fingerprint(first) == generated_contract_fingerprint(second)


def test_different_protected_bytes_change_the_normalized_fingerprint() -> None:
    same_paths_a = (ProtectedInput("canonical/y.npy", 10, "a" * 64),)
    same_paths_b = (ProtectedInput("canonical/y.npy", 10, "b" * 64),)
    first = _header(fingerprint="fp-a", protected=same_paths_a) + BODY
    second = _header(fingerprint="fp-b", protected=same_paths_b) + BODY

    assert generated_contract_fingerprint(first) != generated_contract_fingerprint(second)


# ---------------------------------------------------------------------------
# 9. Structure validation before launch
# ---------------------------------------------------------------------------


class _RecordingExecutor:
    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or ExecutionResult(
            success=True,
            stdout="",
            stderr="",
            execution_time=0.1,
            exit_code=0,
            artifacts_created=[],
            errors=[],
            executed_script_path=SCRIPT,
            candidate_body_reached=True,
        )
        self.timeout = 60
        self.run_mode = ""
        self.mlebench_cache_path = ""

    def execute(self, code: str, **kwargs: Any) -> ExecutionResult:
        self.calls.append({"code": code, **kwargs})
        return self.result

    def validate_syntax(self, code: str) -> tuple[bool, str | None]:
        return True, None


@pytest.mark.parametrize(
    "code",
    [
        "print('no header')\n",
        _header() + BODY + INJECTED_HEADER_END_MARKER + "\n",
        _header().replace(INJECTED_INPUT_MANIFEST_PREFIX, "# other: ", 1) + BODY,
        _header()
        + render_header_manifest_line(_manifest("second"))
        + "\n"
        + INJECTED_HEADER_END_MARKER
        + "\n"
        + BODY,
    ],
)
def test_wrapper_rejects_zero_or_multiple_markers_or_manifests(code: str) -> None:
    executor = _RecordingExecutor()

    with pytest.raises(GeneratedContractStructureError):
        execute_generated_candidate(executor, code, working_dir="/ws")

    assert executor.calls == []


def test_structure_validation_never_substring_truncates() -> None:
    """A marker-looking prefix inside a longer line is not a marker."""
    code = (
        _header().replace(
            INJECTED_HEADER_END_MARKER,
            INJECTED_HEADER_END_MARKER,
        )
        + f"{INJECTED_HEADER_END_MARKER} EXTRA\n"
        + BODY
    )

    manifest = require_one_exact_generated_header_and_manifest(code)

    assert manifest.target_source_fingerprint == "targetfp"
    assert generated_header(code).endswith(INJECTED_HEADER_END_MARKER + "\n")


@pytest.mark.parametrize(
    "line",
    [
        INJECTED_INPUT_MANIFEST_PREFIX + "!!!not-base64!!!",
        INJECTED_INPUT_MANIFEST_PREFIX
        + __import__("base64").urlsafe_b64encode(b"{not json").decode(),
    ],
)
def test_malformed_manifest_payloads_are_structure_errors(line: str) -> None:
    header = _header().replace(
        render_header_manifest_line(_manifest()),
        line,
    )

    with pytest.raises(GeneratedContractStructureError):
        parse_exact_header_manifest(header)


@pytest.mark.parametrize(
    "relative",
    ["/abs/canonical/y.npy", "../escape.npy", "canonical/../../escape.npy"],
)
def test_absolute_or_escaping_protected_paths_are_rejected(relative: str) -> None:
    header = _header(protected=(ProtectedInput(relative, 4, "c" * 64),))

    with pytest.raises(GeneratedContractStructureError):
        parse_exact_header_manifest(header)


def test_duplicate_protected_paths_are_rejected() -> None:
    duplicate = (
        ProtectedInput("canonical/y.npy", 4, "c" * 64),
        ProtectedInput("canonical/y.npy", 4, "d" * 64),
    )
    header = _header(protected=duplicate)

    with pytest.raises(GeneratedContractStructureError):
        parse_exact_header_manifest(header)


def test_non_canonical_manifest_encoding_is_rejected() -> None:
    import base64

    payload = json.dumps(
        {"protected_inputs": [], "target_source_fingerprint": "fp", "extra": 1}
    ).encode()
    header = _header().replace(
        render_header_manifest_line(_manifest()),
        INJECTED_INPUT_MANIFEST_PREFIX
        + base64.urlsafe_b64encode(payload).decode().rstrip("="),
    )

    with pytest.raises(GeneratedContractStructureError):
        parse_exact_header_manifest(header)


# ---------------------------------------------------------------------------
# 10-11. Generic inputs and non-preamble failures
# ---------------------------------------------------------------------------


def test_generic_execute_without_a_generated_header_still_runs(tmp_path: Path) -> None:
    from kaggle_agents.tools.code_executor.executor import CodeExecutor

    executor = CodeExecutor(timeout=60)
    result = executor.execute(
        "import numpy as np\nprint('plain script')\n",
        str(tmp_path),
        component_type="preprocessing",
    )

    assert result.success is True
    assert "plain script" in result.stdout
    assert result.candidate_body_reached is None
    assert result.executed_script_path is not None


@pytest.mark.parametrize(
    ("errors", "body_reached", "script"),
    [
        (["Timeout: execution exceeded 60s"], False, SCRIPT),
        (["Process terminated by signal SIGKILL (9)"], False, SCRIPT),
        (["Missing expected artifacts: models/oof_a.npy"], True, SCRIPT),
        (["Pre-execution validation failed: forbidden call"], None, None),
    ],
)
def test_timeouts_signals_missing_artifacts_and_prevalidation_are_not_preamble(
    errors: list[str],
    body_reached: bool | None,
    script: str | None,
) -> None:
    code = _header() + BODY
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr="Traceback (most recent call last):\n" + _frame(marker - 3),
        candidate_body_reached=body_reached,
        script=script,
        errors=errors,
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin != "harness"
    assert annotated.retryable is True


def test_syntax_error_after_the_marker_is_not_a_preamble_failure() -> None:
    code = _header() + BODY
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr=(
            f'  File "{SCRIPT}", line {marker + 2}\n'
            "    def (\n"
            "        ^\n"
            "SyntaxError: invalid syntax\n"
        ),
        candidate_body_reached=False,
        errors=["SyntaxError: invalid syntax"],
    )

    annotated = annotate_generated_execution(code, result)

    assert annotated.failure_origin is None
    assert annotated.retryable is True


# ---------------------------------------------------------------------------
# 12. Body/header collisions are sanitized, never terminal
# ---------------------------------------------------------------------------


def test_body_echoing_the_marker_or_manifest_is_sanitized_at_assembly() -> None:
    body = (
        "print('start')\n"
        f"{INJECTED_HEADER_END_MARKER}\n"
        f"{INJECTED_INPUT_MANIFEST_PREFIX}Zm9v\n"
        "print('end')\n"
    )

    sanitized, removed = sanitize_candidate_body(body)

    assert INJECTED_HEADER_END_MARKER not in sanitized
    assert INJECTED_INPUT_MANIFEST_PREFIX not in sanitized
    assert len(removed) == 2
    assert "print('start')" in sanitized and "print('end')" in sanitized

    assembled = _header() + sanitized
    manifest = require_one_exact_generated_header_and_manifest(assembled)
    assert manifest.target_source_fingerprint == "targetfp"


def test_sanitized_body_collision_is_never_harness() -> None:
    body, _ = sanitize_candidate_body(
        f"{INJECTED_HEADER_END_MARKER}\nraise RuntimeError('body')\n"
    )
    code = _header() + body
    marker = _marker_line(code)
    result = _failed(
        code,
        stderr="Traceback (most recent call last):\n" + _frame(marker + 1),
        candidate_body_reached=True,
    )

    assert annotate_generated_execution(code, result).failure_origin is None


# ---------------------------------------------------------------------------
# Executor instrumentation
# ---------------------------------------------------------------------------


class TestExecutorInstrumentation:
    def _executor(self, tmp_path: Path):
        from kaggle_agents.tools.code_executor.executor import CodeExecutor

        return CodeExecutor(timeout=60)

    def test_body_reached_token_is_detected_and_stripped_from_stdout(
        self, tmp_path: Path
    ) -> None:
        code = _header(canonical_body="CANONICAL_Y = [1]") + "print('candidate body')\n"
        result = self._executor(tmp_path).execute(
            code, str(tmp_path), component_type="preprocessing"
        )

        assert result.success is True
        assert result.candidate_body_reached is True
        assert "candidate body" in result.stdout
        assert "KAGGLE_AGENTS" not in result.stdout
        assert result.executed_script_path is not None
        assert not list(tmp_path.glob("_exec_*.py"))

    def test_preamble_failure_reports_body_not_reached(self, tmp_path: Path) -> None:
        code = (
            _header(canonical_body="raise RuntimeError('canonical load failed')")
            + "print('candidate body')\n"
        )
        result = self._executor(tmp_path).execute(
            code, str(tmp_path), component_type="preprocessing"
        )

        assert result.success is False
        assert result.candidate_body_reached is False
        annotated = annotate_generated_execution(code, result)
        assert annotated.failure_origin == "harness"
        assert annotated.retryable is False

    def test_temporary_script_names_are_collision_safe(self, tmp_path: Path) -> None:
        executor = self._executor(tmp_path)
        first = executor.execute(
            "import numpy as np\nprint(1)\n",
            str(tmp_path),
            component_type="preprocessing",
        )
        second = executor.execute(
            "import numpy as np\nprint(2)\n",
            str(tmp_path),
            component_type="preprocessing",
        )

        assert first.executed_script_path != second.executed_script_path


# ---------------------------------------------------------------------------
# Protected inputs stay immutable for the execution being classified
# ---------------------------------------------------------------------------


class TestProtectedInputs:
    def _protected(self, tmp_path: Path, relative: str, payload: bytes):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return ProtectedInput(
            relative,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )

    def _executor(self):
        from kaggle_agents.tools.code_executor.executor import CodeExecutor

        return CodeExecutor(timeout=60)

    def test_prelaunch_mismatch_is_agent_origin_and_not_retryable(
        self, tmp_path: Path
    ) -> None:
        entry = self._protected(tmp_path, "labels.csv", b"id,y\n1,0\n")
        (tmp_path / "labels.csv").write_bytes(b"id,y\n1,1\n")
        code = _header(protected=(entry,), canonical_body="X = 1") + "print('body')\n"

        result = self._executor().execute(
            code, str(tmp_path), component_type="preprocessing"
        )

        assert result.success is False
        assert result.failure_origin == "agent"
        assert result.retryable is False
        assert result.candidate_body_reached is not True

    def test_unchanged_protected_file_is_hashed_at_most_once(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from kaggle_agents.tools.code_executor import canonical_integrity

        entry = self._protected(tmp_path, "labels.csv", b"id,y\n1,0\n")
        code = _header(protected=(entry,), canonical_body="X = 1") + "print('body')\n"

        calls: list[str] = []
        real = canonical_integrity._sha256_file

        def counting(path: Path) -> str:
            if Path(path).name == "labels.csv":
                calls.append(str(path))
            return real(path)

        canonical_integrity.reset_protected_input_digest_cache()
        monkeypatch.setattr(canonical_integrity, "_sha256_file", counting)

        executor = self._executor()
        executor.execute(code, str(tmp_path), component_type="preprocessing")
        executor.execute(code, str(tmp_path), component_type="preprocessing")

        assert len(calls) <= 1

    def test_mutation_by_the_candidate_is_restored_and_reported(
        self, tmp_path: Path
    ) -> None:
        entry = self._protected(tmp_path, "labels.csv", b"id,y\n1,0\n")
        code = (
            _header(protected=(entry,), canonical_body="X = 1")
            + "Path('labels.csv').write_bytes(b'tampered')\n"
        )

        result = self._executor().execute(
            code, str(tmp_path), component_type="preprocessing"
        )

        assert result.success is False
        assert result.failure_origin == "agent"
        assert (tmp_path / "labels.csv").read_bytes() == b"id,y\n1,0\n"


# ---------------------------------------------------------------------------
# Developer transitions: every typed failure through DeveloperAgent.__call__
# ---------------------------------------------------------------------------


class _TimeoutConfig:
    def get_timeout(self, _component_type: str, _component_name: str) -> int:
        return 60


def _agent(executor: Any) -> Any:
    from kaggle_agents.agents.developer.agent import DeveloperAgent

    agent = object.__new__(DeveloperAgent)
    agent.config = SimpleNamespace(
        ablation=SimpleNamespace(
            testing_timeout=60,
            enable_code_preview=False,
            save_generated_code=False,
            enable_refinement=False,
        ),
        component_timeout=_TimeoutConfig(),
        ablation_toggles=None,
    )
    agent.executor = executor
    agent._last_reasoning_trace = None
    agent._last_target_source = None
    agent._last_target_source_metadata = None
    agent._last_self_evaluation = None
    agent._preference_collector = SimpleNamespace(get_pairs_for_state=lambda: [])
    return agent


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "train.csv").write_text("id,y\n1,0\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("id\n2\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,y\n2,0\n", encoding="utf-8")
    return tmp_path


def _state(tmp_path: Path, component: AblationComponent) -> dict[str, Any]:
    return {
        "ablation_plan": [component],
        "current_component_index": 0,
        "code_retry_count": 0,
        "working_directory": str(tmp_path),
        "competition_info": CompetitionInfo("demo", "", "auc", "binary_classification"),
        "run_mode": "mlebench",
        "data_files": {},
        "development_results": [],
        "component_results": {},
        "oof_availability": {},
        "robustness_approved_components": {},
        "trusted_component_scores": {},
        "failed_contract_fingerprints": {},
    }


class _PreambleFailingExecutor(_RecordingExecutor):
    """Fails inside the injected preamble, exactly as the real executor reports."""

    def execute(self, code: str, **kwargs: Any) -> ExecutionResult:
        self.calls.append({"code": code, **kwargs})
        marker = _marker_line(code)
        return _failed(
            code,
            stderr="Traceback (most recent call last):\n" + _frame(marker - 2),
            candidate_body_reached=False,
        )


def _instrument(agent: Any, calls: dict[str, int]) -> None:
    def _fix(code: str, *_args: Any, **_kwargs: Any) -> str:
        calls["fix"] += 1
        # Mirrors the real fixer: an LLM rewrite always comes back through
        # preserve_injected_header().
        return preserve_injected_header(code, "print('fixed body')")

    def _debug(code: str, exec_result: Any, *_args: Any, **_kwargs: Any):
        calls["debug"] += 1
        return code, exec_result, False

    agent._fix_code_error = _fix
    agent._debug_code = _debug
    agent._get_meta_feedback = lambda *a, **k: "feedback"


def _harness_event(updates: dict[str, Any], name: str) -> dict[str, Any]:
    """The one telemetry event a transition must emit, by exact name."""
    events = [
        event
        for event in (updates.get("telemetry_events") or [])
        if event.get("event") == name
    ]
    assert len(events) == 1, (
        f"expected exactly one {name!r} event, got "
        f"{[event.get('event') for event in (updates.get('telemetry_events') or [])]}"
    )
    assert events[0].get("category") == "harness"
    return events[0].get("detail") or {}


def _generate_from_contract(calls: dict[str, int]):
    def _generate(component, competition_info, working_dir, domain, state=None, **kwargs):
        calls["llm"] = calls.get("llm", 0) + 1
        prepared = kwargs.get("prepared_contract")
        assert prepared is not None, "the preparer must run before generation"
        return prepared.path_header + "\nprint('candidate body')\n"

    return _generate


class TestDeveloperTransitions:
    def test_injected_preamble_failure_is_terminal_harness_without_retries(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        calls = {"fix": 0, "debug": 0, "llm": 0}
        executor = _PreambleFailingExecutor()
        agent = _agent(executor)
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        updates = agent(state)

        assert len(executor.calls) == 1
        assert calls["fix"] == 0
        assert calls["debug"] == 0
        assert updates["code_retry_count"] == 0
        assert updates["terminal_failure_origin"] == "harness"
        assert updates["workflow_valid"] is False
        assert updates["terminal_failure_detail"]["contract_fingerprint"]
        assert component.name not in (updates.get("failed_component_names") or [])
        assert updates["current_component_index"] == 1
        assert updates.get("skip_remaining_components") is not True
        fingerprints = updates["failed_contract_fingerprints"]
        assert updates["terminal_failure_detail"]["contract_fingerprint"] in fingerprints
        detail = _harness_event(updates, "injected_header_failure")
        assert detail["component_name"] == component.name
        assert detail["origin"] == "harness"
        assert detail["reason"] == "injected_header_failure"
        assert (
            detail["contract_fingerprint"]
            == updates["terminal_failure_detail"]["contract_fingerprint"]
        )

    def test_prior_accepted_evidence_survives_a_preamble_failure(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        snapshot = working / ".accepted_submissions" / "run-1"
        snapshot.mkdir(parents=True)
        payload = b"id,y\n2,0.5\n"
        accepted = snapshot / f"{hashlib.sha256(payload).hexdigest()}.csv"
        accepted.write_bytes(payload)
        state.update(
            {
                "run_id": "run-1",
                "accepted_submission_path": str(accepted),
                "accepted_submission_snapshot_path": str(accepted),
                "accepted_submission_sha256": hashlib.sha256(payload).hexdigest(),
                "component_results": {"earlier": SimpleNamespace(success=True)},
                "trusted_component_scores": {"earlier": 0.7},
                "oof_availability": {"earlier": True},
                "robustness_approved_components": {"earlier": True},
            }
        )
        calls = {"fix": 0, "debug": 0, "llm": 0}
        agent = _agent(_PreambleFailingExecutor())
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        original_digest = state["accepted_submission_sha256"]

        updates = agent(state)

        assert accepted.read_bytes() == payload
        # Preservation is by omission: the transition must not rewrite the
        # accepted registry or the component evidence at all. Asserting only
        # the merged value would pass even if the transition wrote those keys
        # back with different contents.
        for key in (
            "accepted_submission_path",
            "accepted_submission_snapshot_path",
            "accepted_submission_sha256",
            "component_results",
            "trusted_component_scores",
            "oof_availability",
            "robustness_approved_components",
        ):
            assert key not in updates, f"{key} must be preserved by omission"
        merged = {**state, **updates}
        assert merged["accepted_submission_sha256"] == original_digest
        assert verified_accepted_submission(merged, working) is not None
        assert "earlier" in merged["component_results"]
        assert merged["trusted_component_scores"]["earlier"] == 0.7

    def test_repeated_contract_skips_llm_and_executor_for_the_next_component(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        first = AblationComponent("prep_a", "preprocessing", "clean")
        second = AblationComponent("prep_b", "preprocessing", "clean")
        state = _state(working, first)
        state["ablation_plan"] = [first, second]
        calls = {"fix": 0, "debug": 0, "llm": 0}
        executor = _PreambleFailingExecutor()
        agent = _agent(executor)
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        first_updates = agent(state)
        state.update(first_updates)
        state["current_component_index"] = 1

        second_updates = agent(state)

        assert len(executor.calls) == 1
        assert calls["llm"] == 1
        assert second_updates["current_component_index"] == 2
        assert second_updates["terminal_failure_origin"] == "harness"
        detail = _harness_event(second_updates, "duplicate_injected_contract_skipped")
        assert detail["component_name"] == second.name
        assert (
            detail["contract_fingerprint"]
            in first_updates["failed_contract_fingerprints"]
        )

    def test_a_body_exception_keeps_the_existing_bounded_retries(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        calls = {"fix": 0, "debug": 0, "llm": 0}

        class _BodyFailure(_RecordingExecutor):
            def execute(self, code: str, **kwargs: Any) -> ExecutionResult:
                self.calls.append({"code": code})
                marker = _marker_line(code)
                return _failed(
                    code,
                    stderr="Traceback (most recent call last):\n" + _frame(marker + 2),
                    candidate_body_reached=True,
                )

        executor = _BodyFailure()
        agent = _agent(executor)
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        updates = agent(state)

        assert len(executor.calls) >= 2
        assert calls["fix"] >= 1
        assert calls["debug"] == 1
        assert updates.get("terminal_failure_origin") is None
        assert updates["code_retry_count"] == 1

    def test_canonical_contract_error_is_terminal_harness_with_zero_calls(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        state["canonical_contract"] = {"train_ids_path": str(working / "missing.npy")}
        calls = {"fix": 0, "debug": 0, "llm": 0}
        executor = _RecordingExecutor()
        agent = _agent(executor)
        _instrument(agent, calls)

        def _raise(*_args: Any, **_kwargs: Any):
            raise CanonicalTargetContractError(
                "canonical contract is corrupt",
                [{"code": "missing_file", "path": "canonical/y.npy"}],
            )

        agent._prepare_generated_contract = _raise
        agent._generate_code = _generate_from_contract(calls)

        updates = agent(state)

        assert executor.calls == []
        assert calls["llm"] == 0
        assert calls["fix"] == 0 and calls["debug"] == 0
        assert updates["terminal_failure_origin"] == "harness"
        assert updates["workflow_valid"] is False
        assert updates["skip_remaining_components"] is True
        assert updates["terminal_failure_detail"]["violations"]
        assert component.name not in (updates.get("failed_component_names") or [])
        detail = _harness_event(updates, "generated_contract_unavailable")
        assert detail["reason"] == "canonical_target_contract_error"
        assert detail["component_name"] == component.name
        assert detail["violations"]

    def test_generated_contract_structure_error_is_terminal_harness(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        calls = {"fix": 0, "debug": 0, "llm": 0}
        executor = _RecordingExecutor()
        agent = _agent(executor)
        _instrument(agent, calls)

        def _raise(*_args: Any, **_kwargs: Any):
            raise GeneratedContractStructureError("two markers emitted")

        agent._prepare_generated_contract = _raise
        agent._generate_code = _generate_from_contract(calls)

        updates = agent(state)

        assert executor.calls == []
        assert calls["llm"] == 0
        assert updates["terminal_failure_origin"] == "harness"
        assert updates["skip_remaining_components"] is True
        detail = _harness_event(updates, "generated_contract_unavailable")
        assert detail["reason"] == "generated_contract_structure_error"

    def test_protected_input_mutation_is_terminal_agent_origin(
        self, tmp_path: Path
    ) -> None:
        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        calls = {"fix": 0, "debug": 0, "llm": 0}

        class _MutationExecutor(_RecordingExecutor):
            def execute(self, code: str, **kwargs: Any) -> ExecutionResult:
                self.calls.append({"code": code})
                return replace(
                    _failed(
                        code,
                        stderr="protected input changed",
                        candidate_body_reached=None,
                        errors=[str(ProtectedInputMutationError("labels.csv changed"))],
                    ),
                    failure_origin="agent",
                    retryable=False,
                )

        executor = _MutationExecutor()
        agent = _agent(executor)
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        updates = agent(state)

        assert len(executor.calls) == 1
        assert calls["fix"] == 0 and calls["debug"] == 0
        assert updates["terminal_failure_origin"] == "agent"
        assert updates["workflow_valid"] is False
        assert component.name not in (updates.get("failed_component_names") or [])
        detail = _harness_event(updates, "injected_header_failure")
        assert detail["origin"] == "agent"
        assert detail["reason"] == "protected_input_contract_failure"


# ---------------------------------------------------------------------------
# The no-LLM contract preparer
# ---------------------------------------------------------------------------


class TestPreparedContract:
    def test_a_repeated_fingerprint_raises_before_any_prompt_work(
        self, tmp_path: Path
    ) -> None:
        from kaggle_agents.agents.developer.agent import DeveloperAgent

        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        agent = _agent(_RecordingExecutor())
        agent.llm = SimpleNamespace(
            invoke=lambda *a, **k: pytest.fail("no LLM call is allowed here")
        )

        prepared = DeveloperAgent._prepare_generated_contract(
            agent,
            component,
            state["competition_info"],
            working,
            "tabular",
            state,
        )

        assert prepared.contract_fingerprint == generated_contract_fingerprint(
            prepared.path_header
        )
        assert prepared.header_sha256 == generated_header_sha256(prepared.path_header)

        state["failed_contract_fingerprints"] = {
            prepared.contract_fingerprint: {"component": "prep_a"}
        }

        with pytest.raises(RepeatedInjectedContractError):
            DeveloperAgent._prepare_generated_contract(
                agent,
                component,
                state["competition_info"],
                working,
                "tabular",
                state,
            )

    def test_generated_header_carries_exactly_one_marker_and_manifest(
        self, tmp_path: Path
    ) -> None:
        from kaggle_agents.agents.developer.agent import DeveloperAgent

        working = _workspace(tmp_path)
        component = AblationComponent("prep_a", "preprocessing", "clean")
        state = _state(working, component)
        agent = _agent(_RecordingExecutor())

        prepared = DeveloperAgent._prepare_generated_contract(
            agent,
            component,
            state["competition_info"],
            working,
            "tabular",
            state,
        )

        lines = prepared.path_header.splitlines()
        assert lines.count(INJECTED_HEADER_END_MARKER) == 1
        assert lines[-1] == INJECTED_HEADER_END_MARKER
        assert sum(1 for line in lines if line.startswith(INJECTED_INPUT_MANIFEST_PREFIX)) == 1
        manifest = parse_exact_header_manifest(prepared.path_header)
        assert manifest.target_source_fingerprint == (
            prepared.target_source.target_source_fingerprint
        )


# ---------------------------------------------------------------------------
# The standalone refinement loop keeps the trusted header
# ---------------------------------------------------------------------------


def test_standalone_refinement_loop_preserves_the_injected_header(
    tmp_path: Path,
) -> None:
    from kaggle_agents.agents.developer.refinement import RefinementMixin
    from kaggle_agents.core.state import DevelopmentResult

    code = _header(canonical_body="X = 1") + "print('body')\n"
    executed: list[str] = []

    class _Refiner(RefinementMixin):
        def __init__(self) -> None:
            self.config = SimpleNamespace(ablation=SimpleNamespace(enable_refinement=True))
            self.llm = SimpleNamespace(
                invoke=lambda *a, **k: SimpleNamespace(
                    content="```python\nprint('rewritten without header')\n```"
                )
            )
            self.executor = SimpleNamespace(
                execute=lambda code, *a, **k: executed.append(code)
                or ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    execution_time=0.0,
                    exit_code=1,
                    artifacts_created=[],
                    errors=["boom"],
                    executed_script_path=SCRIPT,
                    candidate_body_reached=True,
                )
            )

        def _extract_code_from_response(self, response: str) -> str:
            return response.split("```python")[1].split("```")[0]

        def _extract_cv_score(self, _stdout: str):
            return None

        def _get_refinement_iterations(self, _state):
            return 1

    refiner = _Refiner()
    result = DevelopmentResult(code=code, success=True)
    refiner._run_refinement_loop(
        result,
        AblationComponent("m", "model", "fit"),
        {"competition_info": CompetitionInfo("d", "", "auc", "binary_classification")},
        0.5,
        tmp_path,
        {},
    )

    assert executed, "the refinement loop must still execute a candidate"
    for candidate in executed:
        assert candidate.splitlines().count(INJECTED_HEADER_END_MARKER) == 1
        require_one_exact_generated_header_and_manifest(candidate)


# ---------------------------------------------------------------------------
# Robustness stops; the runner reports and still grades what was accepted
# ---------------------------------------------------------------------------


def test_robustness_agent_node_short_circuits_without_constructing_the_agent(
    monkeypatch,
) -> None:
    import kaggle_agents.agents.robustness_agent as robustness_module

    constructed: list[int] = []

    class _Tripwire:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append(1)
            raise AssertionError("RobustnessAgent must not be constructed")

    monkeypatch.setattr(robustness_module, "RobustnessAgent", _Tripwire)

    updates = robustness_module.robustness_agent_node(
        {
            "terminal_failure_origin": "harness",
            "terminal_failure_detail": {"reason": "injected_header_failure"},
            "current_iteration": 1,
        }
    )

    assert constructed == []
    assert updates["robustness_passed"] is False
    assert updates.get("robustness_abstained") is not True
    detail = _harness_event(updates, "robustness_skipped_terminal_failure")
    assert detail["origin"] == "harness"
    assert detail["reason"] == "injected_header_failure"


def test_robustness_gate_returns_invalid_without_spending_recovery_budget() -> None:
    from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node

    updates = robustness_gate_node(
        {
            "working_directory": "/tmp",
            "terminal_failure_origin": "harness",
            "terminal_failure_detail": {"contract_fingerprint": "abc"},
            "robustness_recovery_count": 0,
            "max_robustness_recoveries": 1,
            "accepted_submission_sha256": "f" * 64,
        }
    )

    assert updates["workflow_valid"] is False
    assert updates["robustness_gate_action"] == "fail"
    assert updates["robustness_recovery_count"] == 0
    assert "accepted_submission_sha256" not in updates
    detail = _harness_event(updates, "robustness_gate_terminal_failure")
    assert detail["origin"] == "harness"
    assert detail["contract_fingerprint"] == "abc"


class TestRunnerPropagation:
    def _runner(self, tmp_path: Path):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        return MLEBenchRunner(
            mle_cache_path=tmp_path / "cache", workspace_base=tmp_path / "ws"
        )

    def _accepted_state(self, workspace: Path, run_id: str) -> dict[str, Any]:
        from kaggle_agents.utils.submission_artifacts import snapshot_accepted_submission

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

    def test_terminal_harness_still_grades_the_accepted_snapshot(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from kaggle_agents.mlebench.runner import MLEBenchResult

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        state = self._accepted_state(workspace, "harness-run")
        state.update(
            {
                "workflow_valid": False,
                "terminal_failure_origin": "harness",
                "terminal_failure_detail": {
                    "reason": "injected_header_failure",
                    "contract_fingerprint": "a" * 64,
                    "component": "model_a",
                },
                "submission_validation_error": None,
                "termination_reason": None,
            }
        )
        runner = self._runner(tmp_path)
        graded: list[Path] = []

        def fake_grade(_competition_id: str, path: Path) -> dict[str, Any]:
            graded.append(path)
            return {"valid_submission": True, "score": 0.77, "bronze_medal": True}

        monkeypatch.setattr(runner, "_grade_submission", fake_grade)
        result = MLEBenchResult(competition_id="c", success=False)

        runner._finalize_run(result, workspace, "harness-run", state, {})

        assert len(graded) == 1
        assert result.valid_submission is True
        assert result.score == pytest.approx(0.77)
        assert result.failure_origin == "harness"
        assert result.terminal_failure_detail["contract_fingerprint"] == "a" * 64
        assert result.error

    def test_terminal_harness_without_an_artifact_grades_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from kaggle_agents.mlebench.runner import MLEBenchResult

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = self._runner(tmp_path)
        monkeypatch.setattr(
            runner,
            "_grade_submission",
            lambda *a, **k: pytest.fail("nothing may be graded"),
        )
        result = MLEBenchResult(competition_id="c", success=False)

        runner._finalize_run(
            result,
            workspace,
            "harness-run",
            {
                "workflow_valid": False,
                "terminal_failure_origin": "harness",
                "terminal_failure_detail": {"reason": "injected_header_failure"},
            },
            {},
        )

        assert result.valid_submission is False
        assert result.failure_origin == "harness"
        assert result.error

    def test_unknown_terminal_origins_are_not_copied(self, tmp_path: Path) -> None:
        from kaggle_agents.mlebench.runner import MLEBenchResult

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        result = MLEBenchResult(competition_id="c", success=False)

        self._runner(tmp_path)._finalize_run(
            result,
            workspace,
            "run",
            {"workflow_valid": False, "terminal_failure_origin": "cosmic_rays"},
            {},
        )

        assert result.failure_origin is None


def test_execution_failure_to_development_result_copies_classification() -> None:
    code = _header() + BODY
    marker = _marker_line(code)
    annotated = annotate_generated_execution(
        code,
        _failed(
            code,
            stderr="Traceback (most recent call last):\n" + _frame(marker - 2),
            candidate_body_reached=False,
        ),
    )

    development = execution_failure_to_development_result(code, annotated, "full")

    assert development.success is False
    assert development.failure_origin == "harness"
    assert development.retryable is False
    assert development.header_sha256 == annotated.header_sha256
    assert development.contract_fingerprint == annotated.contract_fingerprint
    assert development.run_fidelity == "full"


# ---------------------------------------------------------------------------
# The executed preamble is byte-identical to the fingerprinted one
# ---------------------------------------------------------------------------


class TestAssembledProgramKeepsTheTrustedHeader:
    """Post-processing rewrites whole-file patterns.

    ``_strip_nrows_param`` and ``_rewrite_base_dir_references`` are regex passes
    over the complete program, so they can rewrite generator-owned lines. If the
    header that runs is not the header that was fingerprinted, the recorded
    ``header_sha256``/``contract_fingerprint`` describe bytes nothing executed -
    and the whole classification rests on those bytes.
    """

    def _agent_with_body(self, body: str) -> Any:
        from kaggle_agents.agents.developer.agent import DeveloperAgent

        agent = _agent(_RecordingExecutor())
        agent.use_dspy = False
        agent.llm = SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(
                content=f"```python\n{body}```"
            )
        )
        agent._extract_code_from_response = (
            lambda response: DeveloperAgent._extract_code_from_response(agent, response)
        )
        return agent

    def _generate(self, agent: Any, tmp_path: Path):
        from kaggle_agents.agents.developer.agent import DeveloperAgent

        working = _workspace(tmp_path)
        # A model component: its header carries the injected write_submission
        # helper, whose template read uses ``nrows=0``. That is the exact
        # generator-owned text the whole-file nrows pass would otherwise strip,
        # turning a header-row read into a full read of the template.
        component = AblationComponent("model_a", "model", "fit")
        state = _state(working, component)
        state["submission_contract"] = {"id_col": "id", "target_cols": ["y"]}
        prepared = DeveloperAgent._prepare_generated_contract(
            agent,
            component,
            state["competition_info"],
            working,
            "tabular",
            state,
        )
        generated = DeveloperAgent._generate_code(
            agent,
            component,
            state["competition_info"],
            working,
            "tabular",
            state,
            prepared_contract=prepared,
        )
        return prepared, generated

    def test_post_processing_cannot_change_the_fingerprinted_header(
        self, tmp_path: Path
    ) -> None:
        body = (
            "import pandas as pd\n"
            # Triggers _strip_nrows_param, which rewrites the WHOLE program.
            "train = pd.read_csv(TRAIN_PATH, nrows=1000)\n"
            # Triggers _rewrite_base_dir_references.
            "extra = BASE_DIR / 'features.csv'\n"
            # Triggers _strip_path_redefinitions.
            "MODELS_DIR = Path('/tmp/mine')\n"
            "print('Final Validation Performance: 1.0')\n"
        )
        prepared, generated = self._generate(self._agent_with_body(body), tmp_path)

        assert generated_header(generated) == prepared.path_header
        assert generated_header_sha256(generated) == prepared.header_sha256
        assert generated_contract_fingerprint(generated) == prepared.contract_fingerprint
        require_one_exact_generated_header_and_manifest(generated)

    def test_a_body_that_echoes_the_header_is_sanitized_end_to_end(
        self, tmp_path: Path
    ) -> None:
        """RED item 12, through the real generator: agent-side, never terminal."""
        body = (
            "import pandas as pd\n"
            f"{INJECTED_HEADER_END_MARKER}\n"
            f"{INJECTED_INPUT_MANIFEST_PREFIX}Zm9ybmVk\n"
            "print('Final Validation Performance: 1.0')\n"
        )
        prepared, generated = self._generate(self._agent_with_body(body), tmp_path)

        lines = generated.splitlines()
        assert lines.count(INJECTED_HEADER_END_MARKER) == 1
        assert (
            sum(1 for line in lines if line.startswith(INJECTED_INPUT_MANIFEST_PREFIX))
            == 1
        )
        assert generated_header(generated) == prepared.path_header
        # The collision was stripped from the body, not fatal.
        body_after_marker = generated.split(INJECTED_HEADER_END_MARKER, 1)[1]
        assert INJECTED_INPUT_MANIFEST_PREFIX not in body_after_marker
        assert "print('Final Validation Performance: 1.0')" in body_after_marker


# ---------------------------------------------------------------------------
# The gate must not lose a verifiable snapshot on its way out
# ---------------------------------------------------------------------------


def _best_candidate_state(working: Path, run_id: str = "gate-run") -> dict[str, Any]:
    """A run that produced a verifiable candidate but never reached submission."""
    from kaggle_agents.utils.submission_artifacts import (
        snapshot_best_candidate_submission,
    )

    source = working / "submission.csv"
    source.write_text("id,target\n1,0.75\n", encoding="utf-8")
    snapshot, digest = snapshot_best_candidate_submission(
        working, source, run_id=run_id, iteration=1
    )
    return {
        "working_directory": str(working),
        "run_id": run_id,
        "current_iteration": 1,
        "best_candidate_submission_snapshot_path": str(snapshot),
        "best_candidate_submission_sha256": digest,
        "best_candidate_submission_component_name": "model_a",
        "trusted_component_scores": {"model_a": 0.81},
    }


class TestGateBindsWhatItCanStillGrade:
    def test_terminal_stop_binds_a_verifiable_best_candidate_for_grading(
        self, tmp_path: Path
    ) -> None:
        from kaggle_agents.utils.submission_artifacts import (
            verified_accepted_submission,
        )
        from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node

        working = tmp_path
        state = _best_candidate_state(working)
        state.update(
            {
                "terminal_failure_origin": "harness",
                "terminal_failure_detail": {
                    "reason": "injected_header_failure",
                    "component": "model_b",
                    "contract_fingerprint": "d" * 64,
                },
            }
        )

        updates = robustness_gate_node(state)

        assert updates["workflow_valid"] is False
        assert updates["robustness_recovery_count"] == 0
        merged = {**state, **updates}
        bound = verified_accepted_submission(merged, working)
        assert bound is not None, "the runner grades only the accepted registry"
        assert bound.read_bytes() == b"id,target\n1,0.75\n"
        # Trusted provenance travels only when it demonstrably describes these
        # bytes.
        assert merged["accepted_submission_cv_score"] == 0.81
        assert merged["accepted_submission_score_owner"] == "model_a"

    def test_an_unverifiable_snapshot_is_never_bound(self, tmp_path: Path) -> None:
        from kaggle_agents.utils.submission_artifacts import (
            verified_accepted_submission,
        )
        from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node

        working = tmp_path
        state = _best_candidate_state(working)
        state["best_candidate_submission_sha256"] = "0" * 64  # digest mismatch
        state.update(
            {
                "terminal_failure_origin": "harness",
                "terminal_failure_detail": {"reason": "injected_header_failure"},
            }
        )

        updates = robustness_gate_node(state)

        assert updates["workflow_valid"] is False
        assert verified_accepted_submission({**state, **updates}, working) is None
        assert "accepted_submission_sha256" not in updates


# ---------------------------------------------------------------------------
# Developer -> gate -> runner, end to end
# ---------------------------------------------------------------------------


class TestGradingSurvivesTheWholeChain:
    def _runner(self, tmp_path: Path):
        from kaggle_agents.mlebench.runner import MLEBenchRunner

        return MLEBenchRunner(
            mle_cache_path=tmp_path / "cache", workspace_base=tmp_path / "ws"
        )

    def _grade(self, runner, monkeypatch, graded: list[Path]):
        def fake_grade(_competition_id: str, path: Path) -> dict[str, Any]:
            graded.append(path)
            return {"valid_submission": True, "score": 0.73, "above_median": True}

        monkeypatch.setattr(runner, "_grade_submission", fake_grade)

    def test_accepted_snapshot_reaches_grading_through_developer_and_gate(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from kaggle_agents.mlebench.runner import MLEBenchResult
        from kaggle_agents.utils.submission_artifacts import (
            snapshot_accepted_submission,
        )
        from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node

        working = _workspace(tmp_path)
        payload = b"id,target\n2,0.5\n"
        (working / "submission.csv").write_bytes(payload)
        snapshot, digest = snapshot_accepted_submission(
            working, working / "submission.csv", run_id="chain-run", iteration=1
        )
        component = AblationComponent("prep_b", "preprocessing", "clean")
        state = _state(working, component)
        state.update(
            {
                "run_id": "chain-run",
                "current_iteration": 2,
                "accepted_submission_path": str(snapshot),
                "accepted_submission_snapshot_path": str(snapshot),
                "accepted_submission_sha256": digest,
            }
        )
        calls = {"fix": 0, "debug": 0, "llm": 0}
        agent = _agent(_PreambleFailingExecutor())
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        # 1. Developer records the terminal harness failure.
        state.update(agent(state))
        assert state["terminal_failure_origin"] == "harness"

        # 2. The gate stops without spending recovery budget.
        state.update(robustness_gate_node(state))
        assert state["workflow_valid"] is False

        # 3. The single grading pass still grades what the run accepted.
        runner = self._runner(tmp_path)
        graded: list[Path] = []
        self._grade(runner, monkeypatch, graded)
        result = MLEBenchResult(competition_id="c", success=False)
        runner._finalize_run(result, working, "chain-run", state, {})

        assert graded == [snapshot]
        assert snapshot.read_bytes() == payload
        assert result.valid_submission is True
        assert result.failure_origin == "harness"
        assert result.success is False
        assert result.terminal_failure_detail["reason"] == "injected_header_failure"

    def test_best_candidate_bound_by_the_gate_reaches_grading(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from kaggle_agents.mlebench.runner import MLEBenchResult
        from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node

        working = _workspace(tmp_path)
        component = AblationComponent("prep_b", "preprocessing", "clean")
        state = _state(working, component)
        state.update(_best_candidate_state(working, run_id="chain-best"))
        calls = {"fix": 0, "debug": 0, "llm": 0}
        agent = _agent(_PreambleFailingExecutor())
        _instrument(agent, calls)
        agent._generate_code = _generate_from_contract(calls)

        state.update(agent(state))
        state.update(robustness_gate_node(state))

        runner = self._runner(tmp_path)
        graded: list[Path] = []
        self._grade(runner, monkeypatch, graded)
        result = MLEBenchResult(competition_id="c", success=False)
        runner._finalize_run(result, working, "chain-best", state, {})

        assert len(graded) == 1
        assert graded[0].read_bytes() == b"id,target\n1,0.75\n"
        assert result.valid_submission is True
        assert result.failure_origin == "harness"
        assert result.score == pytest.approx(0.73)
