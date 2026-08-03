"""Classify failures of the injected, generator-owned preamble.

Everything a generated program contains above ``# === END PATH CONSTANTS ===``
is written by this repository, not by the model: path constants, canonical
loaders and the injected ``write_submission`` / ``save_component_artifacts`` /
``validate_probabilities`` helpers. The marker name is historical; its actual
placement is the end of the complete generator-owned preamble, immediately
before the candidate body.

When that region raises, no candidate rewrite can repair it. Spending the
fixer, the debugger and the outer component retry on it burns the run's budget
and, worse, records the attempt as evidence about model quality. So a proven
preamble failure is terminal, non-retryable and attributed to the harness.

The proof is deliberately narrow. A failure is only a preamble failure when the
executor reports that the candidate body was never reached *in the exact script
it launched*, and every traceback frame naming that script is at or before the
marker line. A body-reached token, a frame after the marker, a stale or spoofed
``_exec_*.py`` path, a timeout, a signal, a missing artifact or a pre-execution
rejection all keep the failure ordinary and retryable. False negatives cost one
normal retry; a false positive would blame the harness for a candidate defect
and silently stop the run.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from ...core.state.results import DevelopmentResult
from ...tools.code_executor.canonical_integrity import ProtectedInputMutationError
from ...tools.code_executor.dataclasses import ExecutionResult
from .target_source import DeveloperTargetSource, ProtectedInput


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...tools.code_executor.executor import CodeExecutor


INJECTED_HEADER_END_MARKER = "# === END PATH CONSTANTS ==="
INJECTED_INPUT_MANIFEST_PREFIX = "# KAGGLE_AGENTS_INPUT_MANIFEST_V1: "

_TRACEBACK_FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)', re.MULTILINE)

# Failure shapes that are never evidence about the preamble, whatever the
# traceback looks like. A killed process leaves no reliable frames, and a
# missing artifact is reported after the program already ran.
_NON_PREAMBLE_ERROR_SIGNATURES = (
    "timeout:",
    "execution timeout",
    "terminated by signal",
    "killed by signal",
    "missing expected artifacts",
    "pre-execution validation failed",
    "canonical contract integrity violation",
)


@dataclass(frozen=True)
class HeaderInputManifest:
    """The generator's declaration of what the preamble reads eagerly."""

    target_source_fingerprint: str
    protected_inputs: tuple[ProtectedInput, ...]


@dataclass(frozen=True)
class PreparedGeneratedContract:
    """One no-LLM preparation of the immutable contract for one component."""

    target_source: DeveloperTargetSource
    path_header: str
    header_sha256: str
    contract_fingerprint: str
    # Prompt-side material built during the same no-LLM pass. Keeping it here
    # is what makes "resolve and render exactly once per component" true: the
    # generator consumes this object instead of rebuilding paths and headers.
    prompt_inputs: dict[str, Any] = field(default_factory=dict)


class RepeatedInjectedContractError(RuntimeError):
    def __init__(self, contract_fingerprint: str) -> None:
        self.contract_fingerprint = contract_fingerprint
        super().__init__(
            "Injected generated contract already failed in this run: "
            f"{contract_fingerprint}"
        )


class GeneratedContractStructureError(RuntimeError):
    """Generator-owned marker/manifest malformation detected before launch.

    Body-introduced collisions are sanitized at assembly time and stay
    agent-side; this error is reserved for the generator's own output.
    """


# ``ProtectedInputMutationError`` is imported above rather than defined here:
# the executor's targeted integrity branch must be able to name the type
# without importing this package. It is re-exported (see ``__all__``) because
# it is part of the Developer's failure taxonomy.


# ---------------------------------------------------------------------------
# Header extraction and identity
# ---------------------------------------------------------------------------


def generated_header(code: str) -> str | None:
    """The complete generator-owned prefix, or None without exactly one marker."""
    lines = code.splitlines(keepends=True)
    marker_lines = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n") == INJECTED_HEADER_END_MARKER
    ]
    if len(marker_lines) != 1:
        return None
    return "".join(lines[: marker_lines[0] + 1])


def generated_header_sha256(code: str) -> str | None:
    """Byte-exact identity of this component's header."""
    header = generated_header(code)
    return None if header is None else hashlib.sha256(header.encode()).hexdigest()


def generated_contract_fingerprint(code: str) -> str | None:
    """Identity of the contract itself, independent of which component uses it.

    Two components in the same run share every staged path and every canonical
    byte; only ``COMPONENT_NAME`` differs. Normalizing that name is what lets a
    proven harness failure suppress the *next* component before it pays for an
    LLM call and an execution.
    """
    header = generated_header(code)
    if header is None:
        return None
    normalized = re.sub(
        r"^COMPONENT_NAME\s*=.*$",
        'COMPONENT_NAME = "<component>"',
        header,
        flags=re.MULTILINE,
    )
    target_source_fingerprint = parse_exact_header_manifest(
        header
    ).target_source_fingerprint
    payload = normalized.encode() + b"\0" + target_source_fingerprint.encode()
    return hashlib.sha256(payload).hexdigest()


def injected_marker_line_number(code: str) -> int | None:
    """1-based line number of the single marker line, or None."""
    marker_lines = [
        index
        for index, line in enumerate(code.splitlines(), start=1)
        if line.rstrip("\r\n") == INJECTED_HEADER_END_MARKER
    ]
    if len(marker_lines) != 1:
        return None
    return marker_lines[0]


# ---------------------------------------------------------------------------
# The input manifest carried inside the header
# ---------------------------------------------------------------------------


def _canonical_manifest_payload(manifest: HeaderInputManifest) -> bytes:
    return json.dumps(
        {
            "protected_inputs": [
                [item.relative_path, int(item.size), item.sha256]
                for item in manifest.protected_inputs
            ],
            "target_source_fingerprint": manifest.target_source_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def render_header_manifest_line(manifest: HeaderInputManifest) -> str:
    """One exact generator-owned manifest line for the header."""
    protected = tuple(
        sorted(manifest.protected_inputs, key=lambda item: item.relative_path)
    )
    encoded = base64.urlsafe_b64encode(
        _canonical_manifest_payload(replace(manifest, protected_inputs=protected))
    ).decode("ascii")
    return INJECTED_INPUT_MANIFEST_PREFIX + encoded


def _manifest_lines(header: str) -> list[str]:
    return [
        line.rstrip("\r\n")
        for line in header.splitlines()
        if line.startswith(INJECTED_INPUT_MANIFEST_PREFIX)
    ]


def _validated_relative_path(raw: object) -> str:
    path = str(raw)
    if not path or path != path.strip():
        raise GeneratedContractStructureError(
            f"Protected input path is not a clean relative path: {path!r}"
        )
    pure = PurePosixPath(path)
    if pure.is_absolute() or Path(path).is_absolute():
        raise GeneratedContractStructureError(
            f"Protected input path must be workspace-relative: {path!r}"
        )
    if ".." in pure.parts or "\\" in path:
        raise GeneratedContractStructureError(
            f"Protected input path escapes the workspace: {path!r}"
        )
    return path


def parse_exact_header_manifest(header: str) -> HeaderInputManifest:
    """Decode the single manifest line, rejecting anything ambiguous."""
    lines = _manifest_lines(header)
    if len(lines) != 1:
        raise GeneratedContractStructureError(
            "The generated header must carry exactly one input manifest line; "
            f"found {len(lines)}"
        )
    encoded = lines[0][len(INJECTED_INPUT_MANIFEST_PREFIX) :].strip()
    try:
        raw = base64.b64decode(encoded.encode("ascii"), altchars=b"-_", validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise GeneratedContractStructureError(
            f"Input manifest is not valid base64url: {exc}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeneratedContractStructureError(
            f"Input manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "protected_inputs",
        "target_source_fingerprint",
    }:
        raise GeneratedContractStructureError(
            "Input manifest has an unexpected shape; expected exactly "
            "protected_inputs and target_source_fingerprint"
        )

    fingerprint = payload["target_source_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise GeneratedContractStructureError(
            "Input manifest declares no target source fingerprint"
        )

    entries = payload["protected_inputs"]
    if not isinstance(entries, list):
        raise GeneratedContractStructureError(
            "Input manifest protected_inputs must be a list"
        )
    protected: list[ProtectedInput] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 3:
            raise GeneratedContractStructureError(
                f"Malformed protected input entry: {entry!r}"
            )
        relative = _validated_relative_path(entry[0])
        if relative in seen:
            raise GeneratedContractStructureError(
                f"Duplicate protected input path in manifest: {relative}"
            )
        seen.add(relative)
        size, digest = entry[1], entry[2]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise GeneratedContractStructureError(
                f"Protected input size is not a non-negative integer: {size!r}"
            )
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GeneratedContractStructureError(
                f"Protected input digest is not a sha256 hex digest: {digest!r}"
            )
        protected.append(ProtectedInput(relative, size, digest))

    manifest = HeaderInputManifest(
        target_source_fingerprint=fingerprint,
        protected_inputs=tuple(protected),
    )
    # Canonical encoding: re-rendering must reproduce the same line. This
    # rejects reordered keys, padded/unpadded variants and reordered paths, so
    # the fingerprint of a contract is a function of the contract alone.
    if render_header_manifest_line(manifest) != lines[0]:
        raise GeneratedContractStructureError(
            "Input manifest is not in canonical encoding"
        )
    return manifest


def require_one_exact_generated_header_and_manifest(code: str) -> HeaderInputManifest:
    """Validate marker/manifest structure before anything is launched."""
    header = generated_header(code)
    if header is None:
        count = sum(
            1
            for line in code.splitlines()
            if line.rstrip("\r\n") == INJECTED_HEADER_END_MARKER
        )
        raise GeneratedContractStructureError(
            "The generated program must contain exactly one "
            f"{INJECTED_HEADER_END_MARKER!r} line; found {count}"
        )
    if _manifest_lines(code) != _manifest_lines(header):
        raise GeneratedContractStructureError(
            "An input manifest line appears after the injected header marker"
        )
    return parse_exact_header_manifest(header)


def sanitize_candidate_body(body: str) -> tuple[str, list[str]]:
    """Strip body lines that collide with generator-owned header structure.

    A candidate that echoes the marker or a manifest-prefixed comment is doing
    ordinary agent-side work (copying the header into its own output). Removing
    those lines at assembly time keeps ``GeneratedContractStructureError``
    reserved for genuine generator malformation, and keeps the collision a
    normal bounded regeneration concern rather than a terminal harness stop.
    """
    kept: list[str] = []
    removed: list[str] = []
    for line in body.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if stripped == INJECTED_HEADER_END_MARKER or stripped.startswith(
            INJECTED_INPUT_MANIFEST_PREFIX
        ):
            removed.append(stripped)
            continue
        kept.append(line)
    return "".join(kept), removed


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _normalized(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(str(path)).resolve())
    except OSError:  # pragma: no cover - resolve() is total on supported OSes
        return str(path)


def _exact_script_frames(result: ExecutionResult) -> list[int]:
    """Traceback line numbers that name the exact script this run launched."""
    launched = _normalized(result.executed_script_path)
    if launched is None:
        return []
    text = "\n".join(
        [result.stderr or "", *(str(error) for error in (result.errors or []))]
    )
    lines: list[int] = []
    for match in _TRACEBACK_FRAME.finditer(text):
        if _normalized(match.group("path")) != launched:
            continue
        try:
            lines.append(int(match.group("line")))
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue
    return lines


def _is_non_preamble_failure(result: ExecutionResult) -> bool:
    text = " ".join(
        [result.stderr or "", *(str(error) for error in (result.errors or []))]
    ).lower()
    return any(signature in text for signature in _NON_PREAMBLE_ERROR_SIGNATURES)


def is_injected_preamble_failure(code: str, result: ExecutionResult) -> bool:
    """Whether this failure is proven to have happened above the marker."""
    if result.success:
        return False
    if result.candidate_body_reached is not False:
        # None means "not instrumented" (no generated header, or the process
        # never got far enough to be observed); True means the body ran.
        return False
    if _is_non_preamble_failure(result):
        return False
    marker_line = injected_marker_line_number(code)
    if marker_line is None:
        return False
    frames = _exact_script_frames(result)
    if not frames:
        return False
    return all(line <= marker_line for line in frames)


def annotate_generated_execution(
    code: str,
    result: ExecutionResult,
) -> ExecutionResult:
    """Attach contract identity and, when proven, the harness classification."""
    annotated = replace(
        result,
        header_sha256=result.header_sha256 or generated_header_sha256(code),
        contract_fingerprint=(
            result.contract_fingerprint or _safe_contract_fingerprint(code)
        ),
    )
    if result.failure_origin is not None:
        # The executor already classified this one (for example a protected
        # input mutation). Never overwrite a typed origin with a guess.
        return annotated
    if is_injected_preamble_failure(code, annotated):
        return replace(annotated, failure_origin="harness", retryable=False)
    return annotated


def _safe_contract_fingerprint(code: str) -> str | None:
    try:
        return generated_contract_fingerprint(code)
    except GeneratedContractStructureError:
        return None


def execution_failure_to_development_result(
    code: str,
    result: ExecutionResult,
    run_fidelity: Literal["full", "debug"],
) -> DevelopmentResult:
    return DevelopmentResult(
        code=code,
        success=False,
        stdout=result.stdout,
        stderr=result.stderr,
        execution_time=result.execution_time,
        artifacts_created=list(result.artifacts_created),
        errors=list(result.errors),
        run_fidelity=run_fidelity,
        failure_origin=result.failure_origin,
        retryable=result.retryable,
        header_sha256=result.header_sha256,
        contract_fingerprint=result.contract_fingerprint,
    )


def execute_generated_candidate(
    executor: CodeExecutor,
    code: str,
    **kwargs: Any,
) -> ExecutionResult:
    """The only way generated candidate code reaches the executor.

    Validating the structure here means a malformed generator-owned header is
    caught before a subprocess is launched, and every result that comes back
    carries the classification fields the retry levels read.
    """
    require_one_exact_generated_header_and_manifest(code)
    raw = executor.execute(code=code, **kwargs)
    return annotate_generated_execution(code, raw)


__all__ = [
    "INJECTED_HEADER_END_MARKER",
    "INJECTED_INPUT_MANIFEST_PREFIX",
    "GeneratedContractStructureError",
    "HeaderInputManifest",
    "PreparedGeneratedContract",
    "ProtectedInputMutationError",
    "RepeatedInjectedContractError",
    "annotate_generated_execution",
    "execute_generated_candidate",
    "execution_failure_to_development_result",
    "generated_contract_fingerprint",
    "generated_header",
    "generated_header_sha256",
    "injected_marker_line_number",
    "is_injected_preamble_failure",
    "parse_exact_header_manifest",
    "render_header_manifest_line",
    "require_one_exact_generated_header_and_manifest",
    "sanitize_candidate_body",
]
