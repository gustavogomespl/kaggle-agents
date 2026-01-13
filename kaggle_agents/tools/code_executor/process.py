"""
Process management utilities for code execution.

Contains functions for subprocess management and resource limits.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess


# Feature flag for resource limits (can be disabled via environment variable)
ENABLE_RESOURCE_LIMITS = os.getenv("KAGGLE_AGENTS_ENABLE_LIMITS", "true").lower() == "true"


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

        # Memory limit (soft, hard)
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
