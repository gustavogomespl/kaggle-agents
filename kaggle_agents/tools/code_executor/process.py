"""
Process management utilities for code execution.

Contains functions for subprocess management and resource limits.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
from collections.abc import Mapping


# Feature flag for resource limits (can be disabled via environment variable)
ENABLE_RESOURCE_LIMITS = os.getenv("KAGGLE_AGENTS_ENABLE_LIMITS", "true").lower() == "true"


_SENSITIVE_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_CONFIG_FILE",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AZURE_CONFIG_DIR",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
    "DATABASE_URL",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "KAGGLE_AGENTS_SEARCH_CACHE_DIR",
    "KAGGLE_KEY",
    "KAGGLE_CONFIG_DIR",
    "KAGGLE_USERNAME",
    "LANGSMITH_API_KEY",
    "MLEBENCH_CACHE_DIR",
    "MLEBENCH_DATA_DIR",
    "OPENAI_API_KEY",
    "SSH_AUTH_SOCK",
    "WANDB_API_KEY",
}
_SENSITIVE_ENV_SUFFIXES = (
    "_ACCESS_TOKEN",
    "_API_KEY",
    "_CONNECTION_STRING",
    "_CREDENTIALS",
    "_DATABASE_URL",
    "_KEY",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)


def build_subprocess_env(
    source: Mapping[str, str] | None = None,
    home_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Build the environment exposed to generated code.

    Generated programs are untrusted: their prompts may include content retrieved
    from third-party notebooks. Runtime and ML configuration are preserved, while
    credentials are removed before spawning the subprocess.

    Set ``KAGGLE_AGENTS_ALLOW_GENERATED_CODE_SECRETS=true`` only for an explicitly
    trusted local workflow that genuinely needs credentials inside generated code.
    """
    env = dict(os.environ if source is None else source)
    if env.get("KAGGLE_AGENTS_RUN_MODE", "").strip().lower() == "mlebench":
        # The original benchmark cache is grader-only. Generated code uses the
        # public copy staged in its working directory.
        env.pop("MLEBENCH_DATA_DIR", None)
        env.pop("MLEBENCH_CACHE_DIR", None)
        env.pop("KAGGLE_AGENTS_SEARCH_CACHE_DIR", None)
    allow_secrets = env.get("KAGGLE_AGENTS_ALLOW_GENERATED_CODE_SECRETS", "false").lower()
    if allow_secrets in {"1", "true", "yes"}:
        return env

    sanitized = {
        key: value
        for key, value in env.items()
        if key.upper() not in _SENSITIVE_ENV_NAMES
        and not key.upper().endswith(_SENSITIVE_ENV_SUFFIXES)
    }
    if home_dir is not None:
        isolated_home = os.fspath(home_dir)
        real_home = env.get("HOME", "")
        sanitized["HOME"] = isolated_home
        sanitized["USERPROFILE"] = isolated_home
        sanitized["XDG_CONFIG_HOME"] = os.path.join(isolated_home, ".config")
        sanitized["XDG_CACHE_HOME"] = os.path.join(isolated_home, ".cache")
        # Pretrained-weight caches are not credentials: keep them shared so
        # backbones (EfficientNet/DeBERTa/...) are not re-downloaded per
        # workspace. HF_HOME itself stays isolated (it holds the hub token
        # file); only the hub cache subdirectory is shared.
        if real_home:
            sanitized.setdefault(
                "TORCH_HOME", os.path.join(real_home, ".cache", "torch")
            )
            sanitized.setdefault(
                "HF_HUB_CACHE", os.path.join(real_home, ".cache", "huggingface", "hub")
            )
    return sanitized


def _nvidia_gpu_present() -> bool:
    """Whether the host has an NVIDIA GPU driver available."""
    return os.path.exists("/proc/driver/nvidia") or shutil.which("nvidia-smi") is not None


def set_resource_limits(memory_mb: int = 8192, cpu_time_s: int = 3600) -> None:
    """
    Set resource limits for subprocess (Unix only).

    Falls back silently on Windows or if limits cannot be set.

    Args:
        memory_mb: Maximum memory in MB (default 8GB)
        cpu_time_s: Maximum CPU time in seconds (default 1 hour)
    """
    if not ENABLE_RESOURCE_LIMITS:
        return

    # RLIMIT only works on Unix
    if platform.system() == "Windows":
        return

    try:
        import resource

        # Memory limit (soft, hard). Skipped on GPU hosts: CUDA maps device and
        # pinned memory into the process address space, so an RLIMIT_AS cap
        # fires as fake CUDA OOM / pthread_create failures with free VRAM.
        if not _nvidia_gpu_present():
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_s, cpu_time_s))

        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    except (ImportError, OSError, ValueError) as e:
        # Fallback: just log warning, don't fail
        print(f"[WARN] Could not set resource limits: {e}")


def start_new_process_group() -> None:
    """Start process in new group for kill tree to work.

    Note: This may fail in containerized environments (Docker, Colab, etc.)
    where setpgrp() is not permitted. We catch and ignore such errors.
    """
    if platform.system() != "Windows":
        try:
            os.setpgrp()
        except (OSError, PermissionError):
            # Silently ignore - setpgrp may not be allowed in containers/Colab
            pass


def kill_process_tree(process: subprocess.Popen) -> None:
    """
    Kill process and all its children.

    Uses process group kill on Unix, falls back to simple kill on Windows.

    Args:
        process: The subprocess to kill
    """
    if platform.system() == "Windows":
        # Windows: just terminate the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return

    # Unix: kill entire process group
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)

        # Wait for graceful termination
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if not terminated
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)

    except (ProcessLookupError, OSError):
        # Process already terminated
        pass


def kill_process_group_by_id(process_group_id: int | None) -> None:
    """Kill descendants left after their process-group leader has exited."""
    if (
        process_group_id is None
        or process_group_id <= 0
        or platform.system() == "Windows"
    ):
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
