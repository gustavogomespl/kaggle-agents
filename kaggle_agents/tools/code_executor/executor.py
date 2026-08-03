"""
Code Executor - Main execution logic.

Contains the CodeExecutor class for executing Python code in a sandboxed environment.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue

from .canonical_integrity import (
    CanonicalIntegrityError,
    ProtectedInputMutationError,
    ProtectedInputSnapshot,
    describe_protected_input_changes,
    snapshot_canonical_contract,
    snapshot_protected_inputs,
    verify_and_restore_canonical_contract,
    verify_and_restore_protected_inputs,
)
from .dataclasses import ExecutionResult
from .error_parser import ErrorParserMixin
from .filesystem_guard import (
    install_mlebench_runtime_guard,
    is_mlebench_execution,
    validate_mlebench_filesystem_access,
)
from .process import (
    build_subprocess_env,
    describe_signal_exit,
    kill_process_group_by_id,
    kill_process_tree,
    set_resource_limits,
    start_new_process_group,
)
from .sanitizer import CodeSanitizerMixin
from .submission import SubmissionValidationMixin


@dataclass(frozen=True)
class _GeneratedContractInstrumentation:
    """Per-execution proof of where the candidate body began.

    ``source`` is the script actually launched: the received program with one
    flushed sentinel print inserted immediately after the injected marker. The
    token is unguessable per execution, so candidate code cannot fabricate the
    "the body ran" evidence that keeps a failure retryable, and the sentinel
    line is removed from the stdout the caller sees.
    """

    source: str
    token: str
    protected_inputs: tuple[tuple[str, int, str], ...]

    @property
    def sentinel_line(self) -> str:
        return f'print("{self.token}", flush=True)'


def _apply_protected_input_verdict(
    execution_result: ExecutionResult,
    snapshot: ProtectedInputSnapshot,
) -> None:
    """Restore mutated preamble inputs and reject the candidate that did it."""
    try:
        changes = verify_and_restore_protected_inputs(snapshot)
    except Exception as exc:
        changes = [f"verification_or_restore_failed={type(exc).__name__}:{exc}"]
    if not changes:
        return

    message = (
        "Protected preamble input integrity violation: the candidate changed a "
        "file the injected header had already fingerprinted "
        f"({'; '.join(changes)}). The original bytes were restored and the "
        "candidate is rejected."
    )
    print(f"   ⚠️  {message}")
    execution_result.success = False
    execution_result.errors = list(execution_result.errors or [])
    execution_result.errors.append(message)
    execution_result.stderr = f"{execution_result.stderr}\n{message}".strip()
    # Candidate-caused, so agent-origin; and no rewrite of this candidate can
    # restore a contract it has already invalidated.
    execution_result.failure_origin = "agent"
    execution_result.retryable = False


class CodeExecutor(CodeSanitizerMixin, SubmissionValidationMixin, ErrorParserMixin):
    """
    Execute Python code in a sandboxed environment.

    Features:
    - Subprocess isolation (separate Python process)
    - Timeout enforcement
    - Error parsing and categorization
    - Artifact detection and validation
    - Working directory management
    """

    def __init__(
        self,
        timeout: int = 300,
        run_mode: str = "",
        mlebench_cache_path: str | None = None,
    ):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds (default: 5 minutes)
            run_mode: Optional workflow mode. ``mlebench`` enables the
                benchmark filesystem boundary for every retry/debug execution.
            mlebench_cache_path: Optional host-only MLE-bench cache root to
                block even when it is outside the standard cache locations.
        """
        # Lazy import to avoid circular dependency
        from ...core.config import get_config

        self.config = get_config()
        self.timeout = timeout
        self.run_mode = run_mode.strip().lower()
        self.mlebench_cache_path = str(mlebench_cache_path or "").strip()

    def execute(
        self,
        code: str,
        working_dir: str,
        expected_artifacts: list = None,
        component_type: str | None = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute
            working_dir: Working directory for execution
            expected_artifacts: List of expected output files/directories
            component_type: Type of component ('preprocessing', 'feature_engineering', 'model', etc.)
                           Used to validate that preprocessing doesn't train models.

        Returns:
            ExecutionResult with execution details
        """
        received_code = code
        # AUTO-SANITIZE CODE (remove sys.exit, etc.)
        code, _fixes_applied = self.sanitize_code(code)

        # PRE-EXECUTION VALIDATION (MLE-STAR Pattern)
        is_valid, validation_msg = self.validate_code_before_execution(code, component_type)
        if not is_valid:
            print(f"   ⚠️  Code validation failed: {validation_msg}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Pre-execution validation failed: {validation_msg}",
                execution_time=0.0,
                exit_code=-1,
                artifacts_created=[],
                errors=[validation_msg],
            )

        execution_env = dict(os.environ)
        if self.run_mode:
            execution_env["KAGGLE_AGENTS_RUN_MODE"] = self.run_mode
        if self.mlebench_cache_path:
            # The guard generator consumes this path before the child
            # environment is sanitized. ``build_subprocess_env`` removes it,
            # so generated code cannot discover the host cache through env.
            execution_env["MLEBENCH_DATA_DIR"] = self.mlebench_cache_path
        mlebench_execution = is_mlebench_execution(execution_env)
        if mlebench_execution:
            is_valid, validation_msg = validate_mlebench_filesystem_access(code)
            if not is_valid:
                print(f"   ⚠️  MLE-bench filesystem validation failed: {validation_msg}")
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Pre-execution validation failed: {validation_msg}",
                    execution_time=0.0,
                    exit_code=-1,
                    artifacts_created=[],
                    errors=[validation_msg],
                )

        working_path = Path(working_dir) if isinstance(working_dir, str) else working_dir
        working_path.mkdir(parents=True, exist_ok=True)

        instrumentation = self._generated_contract_instrumentation(received_code, code)

        canonical_snapshot = None
        protected_snapshot: ProtectedInputSnapshot | None = None
        expected_artifact_transaction: tuple[
            Path, list[tuple[Path, Path]]
        ] | None = None
        process_group_id: int | None = None

        def finalize_execution_result(
            execution_result: ExecutionResult,
        ) -> ExecutionResult:
            """Restore host-owned inputs and commit only fresh candidate outputs."""
            nonlocal canonical_snapshot, protected_snapshot
            nonlocal expected_artifact_transaction
            if protected_snapshot is not None:
                snapshot, protected_snapshot = protected_snapshot, None
                _apply_protected_input_verdict(execution_result, snapshot)
            if canonical_snapshot is not None:
                try:
                    changes = verify_and_restore_canonical_contract(
                        canonical_snapshot
                    )
                except Exception as exc:
                    changes = [
                        "verification_or_restore_failed="
                        f"{type(exc).__name__}:{exc}"
                    ]
                finally:
                    canonical_snapshot = None

                if changes:
                    detail = "; ".join(changes)
                    message = (
                        "Canonical contract integrity violation: generated code "
                        "changed host-owned evaluation artifacts "
                        f"({detail}). The candidate is rejected and original "
                        "canonical bytes were restored."
                    )
                    print(f"   ⚠️  {message}")
                    execution_result.success = False
                    execution_result.errors = list(
                        execution_result.errors or []
                    )
                    execution_result.errors.append(message)
                    execution_result.stderr = (
                        f"{execution_result.stderr}\n{message}".strip()
                    )

            if expected_artifact_transaction is not None:
                backup_root, moved = expected_artifact_transaction
                if not execution_result.success:
                    for destination, backup in moved:
                        if destination.is_dir():
                            shutil.rmtree(destination)
                        elif destination.exists() or destination.is_symlink():
                            destination.unlink()
                        if backup.exists():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(backup), str(destination))
                shutil.rmtree(backup_root, ignore_errors=True)
                expected_artifact_transaction = None
            return execution_result

        runtime_guard_dir = None

        # Create temporary script file before the guarded try so final cleanup
        # is safe even when integrity-boundary setup fails. A UUID name keeps
        # concurrent executions in one workspace from sharing a path: the
        # classifier only trusts frames naming the exact script it launched.
        script_file = working_path / f"_exec_{uuid.uuid4().hex}.py"

        try:
            if mlebench_execution:
                try:
                    canonical_snapshot = snapshot_canonical_contract(
                        working_path
                    )
                except CanonicalIntegrityError as exc:
                    raise CanonicalIntegrityError(
                        "Canonical contract integrity setup failed: "
                        f"{exc}"
                    ) from exc
                runtime_guard_dir = install_mlebench_runtime_guard(
                    working_path,
                    source=execution_env,
                )

            if instrumentation is not None and instrumentation.protected_inputs:
                # Everything the immutable preamble reads must be immutable for
                # the execution being classified. A mismatch here means the
                # header was rendered against bytes that no longer exist, which
                # is an agent-origin integrity failure, not a harness failure.
                self._verify_protected_inputs_before_launch(
                    working_path, instrumentation
                )
                protected_snapshot = snapshot_protected_inputs(
                    working_path,
                    instrumentation.protected_inputs,
                    # canonical/ already has its own controller-memory snapshot
                    # in mlebench executions; do not hold those bytes twice.
                    skip_relatives=(
                        [
                            relative
                            for relative, _size, _digest in (
                                instrumentation.protected_inputs
                            )
                            if relative.startswith("canonical/")
                        ]
                        if canonical_snapshot is not None
                        else ()
                    ),
                )

            if expected_artifacts:
                # Evidence from a previous attempt must never satisfy this
                # attempt's existence check. Move it out of the child-visible
                # workspace, then restore it if the execution fails.
                backup_root = Path(
                    tempfile.mkdtemp(prefix="kaggle-agents-artifacts-")
                )
                moved: list[tuple[Path, Path]] = []
                expected_artifact_transaction = (backup_root, moved)
                root = working_path.resolve()
                for index, raw_path in enumerate(expected_artifacts):
                    relative = Path(str(raw_path))
                    if (
                        relative.is_absolute()
                        or relative == Path(".")
                        or ".." in relative.parts
                    ):
                        raise ValueError(
                            f"Expected artifact path must be workspace-relative: "
                            f"{raw_path!r}"
                        )
                    destination = (working_path / relative).resolve()
                    try:
                        destination.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(
                            "Expected artifact path escapes the workspace: "
                            f"{raw_path!r}"
                        ) from exc
                    if destination.exists() or destination.is_symlink():
                        backup = backup_root / str(index)
                        shutil.move(str(destination), str(backup))
                        moved.append((destination, backup))
            # Track artifacts before execution
            artifacts_before = self._get_artifacts(working_path)

            # Write code to file
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(code if instrumentation is None else instrumentation.source)

            # Keep common credential discovery paths away from the real user
            # home. This reduces accidental exposure but is not an OS sandbox.
            generated_home = working_path / ".agent_home"
            generated_home.mkdir(parents=True, exist_ok=True)

            # Execute in subprocess with REAL-TIME STREAMING
            start_time = time.time()

            # Prepare preexec_fn for Unix resource limits
            def preexec_setup():
                try:
                    start_new_process_group()
                    # CPU-seconds are derived from this call's wall budget: a
                    # fixed cap is summed across threads and would kill
                    # multi-core training long before the wall timeout.
                    set_resource_limits(memory_mb=16384, wall_timeout_s=self.timeout)
                except Exception:
                    # Silently ignore all preexec errors (Colab/container compatibility)
                    pass

            # Start process with line-buffered output for real-time streaming
            subprocess_env = build_subprocess_env(
                source=execution_env,
                home_dir=generated_home,
            )
            if runtime_guard_dir is not None:
                existing_pythonpath = subprocess_env.get("PYTHONPATH", "")
                pythonpath_parts = [str(runtime_guard_dir)]
                if existing_pythonpath:
                    pythonpath_parts.append(existing_pythonpath)
                subprocess_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

            process = subprocess.Popen(
                [sys.executable, "-u", str(script_file)],  # -u for unbuffered output
                cwd=str(working_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=subprocess_env,
                preexec_fn=preexec_setup if platform.system() != "Windows" else None,
                start_new_session=True if platform.system() != "Windows" else False,
            )
            if platform.system() != "Windows":
                # ``start_new_session=True`` makes the leader PID the process
                # group ID. Preserve it after the leader exits so background
                # descendants cannot mutate artifacts after host verification.
                process_group_id = process.pid

            # Queues for collecting output from threads
            stdout_queue: Queue = Queue()
            stderr_queue: Queue = Queue()
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            def read_stream(stream, queue, prefix=""):
                """Read from stream and put lines in queue."""
                try:
                    for line in iter(stream.readline, ""):
                        if line:
                            queue.put((prefix, line))
                except Exception:
                    pass
                finally:
                    stream.close()

            # Start reader threads
            stdout_thread = threading.Thread(
                target=read_stream, args=(process.stdout, stdout_queue, "")
            )
            stderr_thread = threading.Thread(
                target=read_stream, args=(process.stderr, stderr_queue, "⚠️ ")
            )
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Monitor progress with timeout and stream output in real-time
            progress_interval = 30  # Print progress every 30s
            last_output_time = start_time

            while True:
                # Check if process completed
                poll_result = process.poll()

                # Process any queued output (real-time streaming)
                while True:
                    try:
                        prefix, line = stdout_queue.get_nowait()
                        line_stripped = line.rstrip("\n\r")
                        stdout_lines.append(line)
                        if (
                            instrumentation is not None
                            and line_stripped == instrumentation.token
                        ):
                            # Internal phase sentinel: kept for classification,
                            # never echoed to the operator.
                            last_output_time = time.time()
                            continue
                        # Print structured logs with special formatting
                        if line_stripped.startswith("[LOG:"):
                            print(f"      📋 {line_stripped}")
                        elif "Fold" in line_stripped and "score" in line_stripped.lower():
                            print(f"      📊 {line_stripped}")
                        elif "Trial" in line_stripped or "trial" in line_stripped:
                            print(f"      🔬 {line_stripped}")
                        elif "✓" in line_stripped or "✅" in line_stripped or "⏱️" in line_stripped or "time" in line_stripped.lower():
                            print(f"      {line_stripped}")
                        elif "Final Validation Performance" in line_stripped:
                            print(f"      🎯 {line_stripped}")
                        # Regular output - only print important lines
                        elif any(
                            kw in line_stripped.lower()
                            for kw in [
                                "loading",
                                "training",
                                "fold",
                                "score",
                                "accuracy",
                                "auc",
                                "error",
                                "warning",
                                "saved",
                                "complete",
                            ]
                        ):
                            print(f"      {prefix}{line_stripped}")
                        last_output_time = time.time()
                    except Empty:
                        break

                while True:
                    try:
                        prefix, line = stderr_queue.get_nowait()
                        line_stripped = line.rstrip("\n\r")
                        stderr_lines.append(line)
                        # Only print non-Optuna stderr
                        if not re.match(r"\[I \d{4}-\d{2}-\d{2}", line_stripped):
                            print(f"      {prefix}{line_stripped}")
                    except Empty:
                        break

                if poll_result is not None:
                    kill_process_group_by_id(process_group_id)
                    process_group_id = None
                    # Process finished - drain remaining output
                    time.sleep(0.1)  # Brief pause to collect any remaining output
                    while not stdout_queue.empty():
                        try:
                            _, line = stdout_queue.get_nowait()
                            stdout_lines.append(line)
                        except Empty:
                            break
                    while not stderr_queue.empty():
                        try:
                            _, line = stderr_queue.get_nowait()
                            stderr_lines.append(line)
                        except Empty:
                            break
                    break

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    # CRITICAL: Kill entire process group, not just parent
                    kill_process_tree(process)
                    # Drain remaining output after kill
                    time.sleep(0.1)
                    while not stdout_queue.empty():
                        try:
                            _, line = stdout_queue.get_nowait()
                            stdout_lines.append(line)
                        except Empty:
                            break
                    while not stderr_queue.empty():
                        try:
                            _, line = stderr_queue.get_nowait()
                            stderr_lines.append(line)
                        except Empty:
                            break

                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)

                    stdout = "".join(stdout_lines)
                    stderr = "".join(stderr_lines)
                    stdout, body_reached = self._consume_body_phase_token(
                        stdout, stderr, instrumentation
                    )

                    # Track artifacts after execution (partial)
                    artifacts_after = self._get_artifacts(working_path)
                    artifacts_created = list(set(artifacts_after) - set(artifacts_before))
                    artifacts_created = [
                        a for a in artifacts_created if not a.endswith(script_file.name)
                    ]

                    return finalize_execution_result(
                        ExecutionResult(
                            success=False,
                            stdout=stdout,
                            stderr=(
                                stderr
                                + f"\nExecution timeout after {self.timeout}s"
                            ).strip(),
                            execution_time=self.timeout,
                            exit_code=-1,
                            artifacts_created=artifacts_created,
                            errors=[
                                f"Timeout: execution exceeded {self.timeout}s"
                            ],
                            executed_script_path=str(script_file),
                            candidate_body_reached=body_reached,
                        )
                    )

                # Print progress update if no output for a while
                time_since_output = time.time() - last_output_time
                if time_since_output >= progress_interval:
                    remaining = self.timeout - elapsed
                    print(
                        f"      ⏳ Execution in progress... ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)"
                    )
                    last_output_time = time.time()

                # Sleep briefly before next check
                time.sleep(0.1)

            # Wait for threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            # Combine collected output
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            stdout, body_reached = self._consume_body_phase_token(
                stdout, stderr, instrumentation
            )
            execution_time = time.time() - start_time

            # Create result object compatible with subprocess.run
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr

            result = Result(process.returncode, stdout, stderr)

            # Track artifacts after execution
            artifacts_after = self._get_artifacts(working_path)
            artifacts_created = list(set(artifacts_after) - set(artifacts_before))

            # Remove the script file from artifacts
            artifacts_created = [a for a in artifacts_created if not a.endswith(script_file.name)]

            # Parse errors
            errors = self._parse_errors(result.stderr, result.stdout)

            # Death by signal is not a defect in the generated program. Name it
            # explicitly so the repair loop does not rewrite correct code.
            signal_cause = describe_signal_exit(result.returncode)
            if signal_cause:
                print(f"   ⚠️  {signal_cause}")
                errors.append(signal_cause)

            # Check success
            success = result.returncode == 0 and not errors

            # Validate expected artifacts
            if expected_artifacts and success:
                missing = [a for a in expected_artifacts if not (working_path / a).exists()]
                if missing:
                    success = False
                    errors.append(f"Missing expected artifacts: {', '.join(missing)}")

            # EXTRACT PERFORMANCE METRIC (MLE-STAR Pattern)
            performance_score = None
            if success:
                performance_score = self.extract_performance_metric(result.stdout)
                if performance_score is not None:
                    print(f"   📊 Validation Performance: {performance_score:.6f}")
                else:
                    print("   ⚠️  Warning: Could not extract performance metric from output")

            return finalize_execution_result(
                ExecutionResult(
                    success=success,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_time=execution_time,
                    exit_code=result.returncode,
                    artifacts_created=artifacts_created,
                    errors=errors,
                    executed_script_path=str(script_file),
                    candidate_body_reached=body_reached,
                )
            )

        except ProtectedInputMutationError as exc:
            # Typed on purpose: the broad handler below would erase the origin
            # and the failure would re-enter the fix/debug loop, where no
            # candidate rewrite can repair an already-stale input contract.
            print(f"   ⚠️  {exc}")
            return finalize_execution_result(
                ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=str(exc),
                    execution_time=0.0,
                    exit_code=-1,
                    artifacts_created=[],
                    errors=[f"Protected preamble input changed: {exc}"],
                    failure_origin="agent",
                    retryable=False,
                )
            )

        except subprocess.TimeoutExpired:
            return finalize_execution_result(
                ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timeout after {self.timeout}s",
                    execution_time=self.timeout,
                    exit_code=-1,
                    artifacts_created=[],
                    errors=[f"Timeout: execution exceeded {self.timeout}s"],
                )
            )

        except Exception as e:
            return finalize_execution_result(
                ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=str(e),
                    execution_time=0.0,
                    exit_code=-1,
                    artifacts_created=[],
                    errors=[f"Execution error: {e!s}"],
                )
            )

        finally:
            kill_process_group_by_id(process_group_id)
            # Cleanup script file
            if script_file.exists():
                script_file.unlink()
            # Defensive cleanup for BaseException paths that bypass the normal
            # result finalizer. Ordinary returns set this to ``None``.
            if canonical_snapshot is not None:
                try:
                    verify_and_restore_canonical_contract(canonical_snapshot)
                finally:
                    canonical_snapshot = None
            if expected_artifact_transaction is not None:
                backup_root, moved = expected_artifact_transaction
                for destination, backup in moved:
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    elif destination.exists() or destination.is_symlink():
                        destination.unlink()
                    if backup.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(backup), str(destination))
                shutil.rmtree(backup_root, ignore_errors=True)
                expected_artifact_transaction = None

    @staticmethod
    def _generated_contract_instrumentation(
        received_code: str,
        sanitized_code: str,
    ) -> _GeneratedContractInstrumentation | None:
        """Insert the body-reached sentinel, or decline to instrument.

        Code without a generated header is a supported generic executor input:
        it runs uninstrumented and reports ``candidate_body_reached=None``. The
        same conservative answer is returned when sanitization changed the
        header, because then the launched script's marker line no longer
        matches the caller's copy and no classification would be trustworthy.
        """
        from ...agents.developer.execution_failures import (
            GeneratedContractStructureError,
            generated_header,
            parse_exact_header_manifest,
        )

        header = generated_header(sanitized_code)
        if header is None or generated_header(received_code) != header:
            return None
        try:
            manifest = parse_exact_header_manifest(header)
        except GeneratedContractStructureError:
            # Structure is validated by the caller before launch; an executor
            # that cannot read the manifest simply does not classify.
            return None

        token = f"__KAGGLE_AGENTS_BODY_REACHED_{uuid.uuid4().hex}__"
        instrumentation = _GeneratedContractInstrumentation(
            source="",
            token=token,
            protected_inputs=tuple(
                (item.relative_path, int(item.size), item.sha256)
                for item in manifest.protected_inputs
            ),
        )
        body = sanitized_code[len(header) :]
        if not header.endswith("\n"):
            header += "\n"
        source = f"{header}{instrumentation.sentinel_line}\n{body}"
        return replace(instrumentation, source=source)

    @staticmethod
    def _verify_protected_inputs_before_launch(
        working_path: Path,
        instrumentation: _GeneratedContractInstrumentation,
    ) -> None:
        """Fail closed when the header's declared inputs no longer match disk."""
        changes = describe_protected_input_changes(
            working_path, instrumentation.protected_inputs
        )
        if changes:
            raise ProtectedInputMutationError(
                "The injected header was rendered against inputs that have "
                "since changed: " + "; ".join(changes)
            )

    @staticmethod
    def _consume_body_phase_token(
        stdout: str,
        stderr: str,
        instrumentation: _GeneratedContractInstrumentation | None,
    ) -> tuple[str, bool | None]:
        """Detect the sentinel, then remove it from the caller-visible stdout."""
        if instrumentation is None:
            return stdout, None
        token = instrumentation.token
        observed = token in stdout or token in stderr
        if not observed:
            return stdout, False
        kept = [
            line
            for line in stdout.splitlines(keepends=True)
            if line.rstrip("\r\n") != token
        ]
        return "".join(kept), True

    def execute_with_retry(
        self,
        code: str,
        working_dir: str,
        max_retries: int = 3,
        expected_artifacts: list = None,
    ) -> tuple[ExecutionResult, int]:
        """
        Execute code with automatic retry on failure.

        Args:
            code: Python code to execute
            working_dir: Working directory
            max_retries: Maximum number of retry attempts
            expected_artifacts: Expected output files

        Returns:
            Tuple of (ExecutionResult, attempts_used)
        """
        for attempt in range(max_retries):
            print(f"   Attempt {attempt + 1}/{max_retries}...")

            result = self.execute(code, working_dir, expected_artifacts)

            if result.success:
                print("    Execution successful")
                return result, attempt + 1

            print(
                f"   L Execution failed: {result.errors[0] if result.errors else 'Unknown error'}"
            )

        return result, max_retries

    def validate_syntax(self, code: str) -> tuple:
        """
        Validate Python syntax without executing.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            return False, error_msg

    def _get_artifacts(self, directory: Path) -> list[str]:
        """
        Get list of files in directory (relative paths).

        Args:
            directory: Directory to scan

        Returns:
            List of relative file paths
        """
        artifacts = []

        if not directory.exists():
            return artifacts

        for item in directory.rglob("*"):
            if item.is_file():
                # Exclude temp files and Python cache
                if not any(x in str(item) for x in ["__pycache__", ".pyc", "_exec_"]):
                    rel_path = str(item.relative_to(directory))
                    artifacts.append(rel_path)

        return artifacts
