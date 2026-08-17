"""Filesystem and Kaggle-network guardrails for generated MLE-bench code.

The benchmark runner stages public inputs into the run workspace. Generated
programs therefore never need to read the original MLE-bench cache, whose
adjacent ``prepared/private`` tree is reserved for the external grader.

This module adds defense-in-depth layers that:

1. reject generated source that explicitly asks for benchmark-cache/private
   paths or Kaggle retrieval; and
2. install a Python audit hook in the child interpreter that blocks runtime
   access to the original MLE-bench cache and the SearchAgent notebook cache
   (including through symlinks), Kaggle DNS/socket access, and Kaggle commands
   launched through subprocesses.

The audit hook is not a replacement for a container/mount namespace against
actively malicious native code. Paper evaluations should still keep grading
labels outside the agent container and apply an OS-level network policy. In
particular, a native binary or a direct connection to a previously resolved IP
can bypass Python hostname checks. The guard prevents accidental and ordinary
Python/subprocess access in the in-process Colab harness without blocking
package or model registries such as PyPI and Hugging Face.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path


_MLEBENCH_CACHE_ENV_NAMES = (
    "MLEBENCH_DATA_DIR",
    "MLEBENCH_CACHE_DIR",
)
_SEARCH_CACHE_ENV_NAME = "KAGGLE_AGENTS_SEARCH_CACHE_DIR"

_KAGGLE_MODULES = {"kaggle", "kagglehub"}
_KAGGLE_API_IDENTIFIERS = {"kaggleapi"}
_KAGGLE_HOST_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9-])(?:[a-z0-9-]+\.)*kaggle\.com"
    r"(?::\d+)?(?:[^a-z0-9-]|$)"
)
_KAGGLE_CLI_RE = re.compile(
    r"(?i)(?:^|[;&|]\s*|\s)(?:[\"']?)(?:[\w./-]+/)?kaggle"
    r"(?:[\"']?)(?=\s|$)"
)


def _literal_identifies_kaggle_retrieval(value: str) -> bool:
    """Return whether a source literal identifies Kaggle retrieval."""
    normalized = value.strip().lower()
    return bool(
        _KAGGLE_HOST_RE.search(normalized)
        or re.search(r"\b(?:from|import)\s+kaggle(?:hub)?\b", normalized)
        or re.search(
            r"(?:^|[;&|]\s*)kaggle\s+"
            r"(?:api|competitions|config|datasets|kernels|models)\b",
            normalized,
        )
    )


def _kaggle_ast_violation(node: ast.AST) -> str:
    """Describe explicit Kaggle API/CLI/network usage in one AST node."""
    if isinstance(node, ast.Import) and any(
        alias.name.split(".", maxsplit=1)[0].lower() in _KAGGLE_MODULES
        for alias in node.names
    ):
        return (
            "Kaggle API/CLI retrieval is forbidden in MLE-bench generated "
            "code; external retrieval is SearchAgent-only"
        )
    if isinstance(node, ast.ImportFrom):
        root_module = (node.module or "").split(".", maxsplit=1)[0].lower()
        if root_module in _KAGGLE_MODULES:
            return (
                "Kaggle API/CLI retrieval is forbidden in MLE-bench generated "
                "code; external retrieval is SearchAgent-only"
            )
    if isinstance(node, (ast.Name, ast.Attribute)):
        identifier = (
            node.id if isinstance(node, ast.Name) else node.attr
        ).lower()
        if identifier in _KAGGLE_API_IDENTIFIERS:
            return (
                "Kaggle API access is forbidden in MLE-bench generated code; "
                "external retrieval is SearchAgent-only"
            )
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _literal_identifies_kaggle_retrieval(node.value)
    ):
        return (
            "Kaggle URLs/API/CLI commands are forbidden in MLE-bench "
            "generated code; external retrieval is SearchAgent-only"
        )
    return ""


def is_mlebench_execution(source: Mapping[str, str] | None = None) -> bool:
    """Return whether generated code is executing under the MLE-bench protocol."""
    env = os.environ if source is None else source
    return str(env.get("KAGGLE_AGENTS_RUN_MODE", "")).strip().lower() == "mlebench"


def validate_mlebench_filesystem_access(code: str) -> tuple[bool, str]:
    """Reject explicit private-storage or Kaggle-retrieval attempts."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # The regular syntax validator reports the more useful diagnostic.
        return True, ""

    forbidden_literals = (
        "mlebench_data_dir",
        "mlebench_cache_dir",
        "kaggle_agents_search_cache_dir",
        "prepared/private",
        "prepared\\private",
        ".cache/mle-bench",
        ".cache\\mle-bench",
    )
    for node in ast.walk(tree):
        kaggle_violation = _kaggle_ast_violation(node)
        if kaggle_violation:
            return False, kaggle_violation
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        normalized = node.value.strip().lower()
        if any(marker in normalized for marker in forbidden_literals):
            return (
                False,
                "MLE-bench private/search-cache paths are host/grader-only; "
                "use the staged files in the run working directory",
            )

    root_enumeration_patterns = (
        r"os\.walk\s*\(\s*['\"]/(?:['\"])",
        r"path\s*\(\s*['\"]/(?:['\"])\s*\)\s*\.\s*rglob",
        r"glob\s*\(\s*['\"]/\*\*",
    )
    compact = re.sub(r"\s+", " ", code.lower())
    if any(re.search(pattern, compact) for pattern in root_enumeration_patterns):
        return False, "Filesystem-wide discovery is forbidden in MLE-bench generated code"

    return True, ""


def _mlebench_cache_roots(source: Mapping[str, str] | None = None) -> list[Path]:
    """Return original cache roots that generated code must never inspect."""
    env = os.environ if source is None else source
    candidates: list[Path] = []
    for name in _MLEBENCH_CACHE_ENV_NAMES:
        value = env.get(name)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            Path.home() / ".cache" / "mle-bench" / "data",
            Path("/root/.cache/mle-bench/data"),
            Path("/content/.cache/mle-bench/data"),
        ]
    )

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        key = os.path.normcase(str(normalized))
        if key not in seen:
            roots.append(normalized)
            seen.add(key)
    return roots


def _mlebench_search_cache_roots(
    source: Mapping[str, str] | None = None,
) -> list[Path]:
    """Return SearchAgent notebook-cache roots hidden from generated code."""
    env = os.environ if source is None else source
    candidates = [
        Path.cwd() / ".cache" / "notebooks",
        Path("/content/kaggle_competitions/.cache/notebooks"),
    ]
    configured = env.get(_SEARCH_CACHE_ENV_NAME)
    if configured:
        candidates.insert(0, Path(configured).expanduser())

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        key = os.path.normcase(str(normalized))
        if key not in seen:
            roots.append(normalized)
            seen.add(key)
    return roots


def install_mlebench_runtime_guard(
    working_dir: str | Path,
    source: Mapping[str, str] | None = None,
) -> Path:
    """Write a ``sitecustomize`` audit hook and return its import directory."""
    workspace = Path(working_dir).resolve()
    guard_dir = workspace / ".agent_runtime"
    guard_dir.mkdir(parents=True, exist_ok=True)
    cache_roots = [str(path) for path in _mlebench_cache_roots(source)]
    search_cache_roots = [
        str(path) for path in _mlebench_search_cache_roots(source)
    ]
    canonical_root = str(workspace / "canonical")

    hook_source = f'''"""Auto-generated MLE-bench runtime boundary."""
import os
import re
import socket
import sys

_BLOCKED_ROOTS = tuple(
    os.path.normcase(os.path.realpath(path))
    for path in {json.dumps(cache_roots)}
)
_SEARCH_CACHE_ROOTS = tuple(
    os.path.normcase(os.path.realpath(path))
    for path in {json.dumps(search_cache_roots)}
)
_CANONICAL_ROOT = os.path.normcase(
    os.path.realpath({json.dumps(canonical_root)})
)
_PRIVATE_MARKER = os.sep + "prepared" + os.sep + "private"
_KAGGLE_HOST_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9-])(?:[a-z0-9-]+\\.)*kaggle\\.com"
    r"(?::\\d+)?(?:[^a-z0-9-]|$)"
)
_KAGGLE_CLI_RE = re.compile(
    r"(?i)(?:^|[;&|]\\s*|\\s)(?:[\\\"']?)(?:[\\w./-]+/)?kaggle"
    r"(?:[\\\"']?)(?=\\s|$)"
)


def _coerce_paths(value):
    if isinstance(value, (str, bytes, os.PathLike)):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _coerce_paths(item)
    elif isinstance(value, dict):
        # ``subprocess.Popen`` and ``os.exec*`` audit events may carry an
        # explicit child environment. A command such as ``cat "$LABEL_PATH"``
        # hides the resolved path from argv, so inspect both environment names
        # and values before allowing the child process.
        for key, item in value.items():
            yield from _coerce_paths(key)
            yield from _coerce_paths(item)


def _blocked_kind(value):
    if isinstance(value, int):
        return ""
    try:
        path = os.fsdecode(value)
    except (TypeError, ValueError):
        return ""
    if not path:
        return ""
    resolved = os.path.normcase(os.path.realpath(path))
    private_marker = os.path.normcase(_PRIVATE_MARKER)
    if private_marker in resolved + os.sep:
        return "private"
    for root in _BLOCKED_ROOTS:
        if resolved == root or resolved.startswith(root + os.sep):
            return "private"
    for root in _SEARCH_CACHE_ROOTS:
        if resolved == root or resolved.startswith(root + os.sep):
            return "search_cache"
    if resolved.startswith(os.path.normcase(os.sep + "proc" + os.sep)):
        if resolved.endswith(os.sep + "environ") or resolved.endswith(os.sep + "cmdline"):
            return "process_metadata"
    return ""


def _deny(value):
    for path in _coerce_paths(value):
        blocked_kind = _blocked_kind(path)
        if blocked_kind == "search_cache":
            raise PermissionError(
                "MLE-bench SearchAgent notebook cache access blocked: "
                "retrieved source is host-only"
            )
        if blocked_kind:
            raise PermissionError(
                "MLE-bench private data access blocked: generated code may only "
                "use public files staged in its working directory"
            )


def _hostname(value):
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        candidate = os.fsdecode(value).strip().lower()
    except (TypeError, ValueError):
        return ""
    if "://" in candidate:
        candidate = candidate.split("://", 1)[1]
    candidate = candidate.split("/", 1)[0].rsplit("@", 1)[-1]
    if candidate.startswith("["):
        candidate = candidate.split("]", 1)[0].lstrip("[")
    elif candidate.count(":") == 1:
        candidate = candidate.rsplit(":", 1)[0]
    return candidate.rstrip(".")


def _is_kaggle_host(value):
    host = _hostname(value)
    return host == "kaggle.com" or host.endswith(".kaggle.com")


def _deny_kaggle_network(value):
    for item in _coerce_paths(value):
        if _is_kaggle_host(item):
            raise PermissionError(
                "MLE-bench Kaggle network access blocked: external "
                "competition retrieval is SearchAgent-only"
            )


def _command_identifies_kaggle(value):
    try:
        command = os.fsdecode(value).strip().lower()
    except (TypeError, ValueError):
        return False
    return bool(
        _KAGGLE_HOST_RE.search(command)
        or _KAGGLE_CLI_RE.search(command)
        or re.search(r"\\b(?:from|import)\\s+kaggle(?:hub)?\\b", command)
    )


def _deny_command(value):
    _deny(value)
    for item in _coerce_paths(value):
        try:
            command = os.path.normcase(os.fsdecode(item))
        except (TypeError, ValueError):
            continue
        private_marker = os.path.normcase(_PRIVATE_MARKER)
        if private_marker in command:
            raise PermissionError(
                "MLE-bench private data access blocked in child command"
            )
        if any(root and root in command for root in _BLOCKED_ROOTS):
            raise PermissionError(
                "MLE-bench cache access blocked in child command"
            )
        if any(root and root in command for root in _SEARCH_CACHE_ROOTS):
            raise PermissionError(
                "MLE-bench SearchAgent notebook cache access blocked in child "
                "command"
            )
        if _command_identifies_kaggle(item):
            raise PermissionError(
                "MLE-bench Kaggle retrieval blocked in child command: "
                "external competition retrieval is SearchAgent-only"
            )


def _is_canonical_path(value):
    if isinstance(value, int):
        return False
    try:
        resolved = os.path.normcase(os.path.realpath(os.fsdecode(value)))
    except (TypeError, ValueError):
        return False
    return (
        resolved == _CANONICAL_ROOT
        or resolved.startswith(_CANONICAL_ROOT + os.sep)
    )


def _open_requests_write(args):
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None
    if isinstance(mode, str) and any(marker in mode for marker in "wax+"):
        return True
    if isinstance(flags, int):
        write_flags = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
        )
        return bool(flags & write_flags)
    return False


def _deny_canonical_mutation(event, args):
    if not args:
        return
    if event == "open":
        mutates = _open_requests_write(args)
    else:
        mutates = event in {{
            "os.chmod",
            "os.chown",
            "os.link",
            "os.mkdir",
            "os.remove",
            "os.rename",
            "os.rmdir",
            "os.symlink",
            "os.truncate",
        }}
    if mutates and _is_canonical_path(args[0]):
        raise PermissionError(
            "MLE-bench canonical contract is host-owned and read-only"
        )
    if event in {{"os.link", "os.rename", "os.symlink"}} and len(args) > 1:
        if _is_canonical_path(args[1]):
            raise PermissionError(
                "MLE-bench canonical contract is host-owned and read-only"
            )


def _audit(event, args):
    if event in {{
        "open",
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.link",
        "os.listdir",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.scandir",
        "os.symlink",
        "os.truncate",
    }}:
        if args:
            _deny(args[0])
        if event == "os.rename" and len(args) > 1:
            _deny(args[1])
        _deny_canonical_mutation(event, args)
    elif event in {{
        "os.exec",
        "os.posix_spawn",
        "os.system",
        "subprocess.Popen",
    }}:
        _deny_command(args)
    elif event in {{
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
    }}:
        _deny_kaggle_network(args)


sys.addaudithook(_audit)

# Audit event coverage varies across Python/platform versions. Patch the common
# resolver/connection helpers too; socket.connect itself remains covered by the
# audit hook above.
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_GETHOSTBYNAME = socket.gethostbyname
_ORIGINAL_GETHOSTBYNAME_EX = socket.gethostbyname_ex
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


def _guarded_getaddrinfo(host, *args, **kwargs):
    _deny_kaggle_network(host)
    return _ORIGINAL_GETADDRINFO(host, *args, **kwargs)


def _guarded_gethostbyname(host):
    _deny_kaggle_network(host)
    return _ORIGINAL_GETHOSTBYNAME(host)


def _guarded_gethostbyname_ex(host):
    _deny_kaggle_network(host)
    return _ORIGINAL_GETHOSTBYNAME_EX(host)


def _guarded_create_connection(address, *args, **kwargs):
    _deny_kaggle_network(address)
    return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)


socket.getaddrinfo = _guarded_getaddrinfo
socket.gethostbyname = _guarded_gethostbyname
socket.gethostbyname_ex = _guarded_gethostbyname_ex
socket.create_connection = _guarded_create_connection
'''
    (guard_dir / "sitecustomize.py").write_text(hook_source, encoding="utf-8")
    return guard_dir


__all__ = [
    "install_mlebench_runtime_guard",
    "is_mlebench_execution",
    "validate_mlebench_filesystem_access",
]
