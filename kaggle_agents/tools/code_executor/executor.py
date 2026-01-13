"""
Code Executor - Main execution logic.

Contains the CodeExecutor class for executing Python code in a sandboxed environment.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from .dataclasses import ExecutionResult
from .error_parser import ErrorParserMixin
from .process import kill_process_tree, set_resource_limits, start_new_process_group
from .sanitizer import CodeSanitizerMixin
from .submission import SubmissionValidationMixin


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

    def __init__(self, timeout: int = 300):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds (default: 5 minutes)
        """
        # Lazy import to avoid circular dependency
        from ...core.config import get_config

        self.config = get_config()
        self.timeout = timeout

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

        working_path = Path(working_dir) if isinstance(working_dir, str) else working_dir
        working_path.mkdir(parents=True, exist_ok=True)

        # Track artifacts before execution
        artifacts_before = self._get_artifacts(working_path)

        # Create temporary script file
        script_file = working_path / f"_exec_{int(time.time())}.py"

        try:
            # Write code to file
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Execute in subprocess with REAL-TIME STREAMING
            start_time = time.time()

            # Prepare preexec_fn for Unix resource limits
            def preexec_setup():
                try:
                    start_new_process_group()
                    set_resource_limits(memory_mb=16384, cpu_time_s=7200)  # 16GB, 2 hours
                except Exception:
                    # Silently ignore all preexec errors (Colab/container compatibility)
                    pass

            # Start process with line-buffered output for real-time streaming
            process = subprocess.Popen(
                [sys.executable, "-u", str(script_file)],  # -u for unbuffered output
                cwd=str(working_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                preexec_fn=preexec_setup if platform.system() != "Windows" else None,
                start_new_session=True if platform.system() != "Windows" else False,
            )

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

                    # Track artifacts after execution (partial)
                    artifacts_after = self._get_artifacts(working_path)
                    artifacts_created = list(set(artifacts_after) - set(artifacts_before))
                    artifacts_created = [
                        a for a in artifacts_created if not a.endswith(script_file.name)
                    ]

                    return ExecutionResult(
                        success=False,
                        stdout=stdout,
                        stderr=(stderr + f"\nExecution timeout after {self.timeout}s").strip(),
                        execution_time=self.timeout,
                        exit_code=-1,
                        artifacts_created=artifacts_created,
                        errors=[f"Timeout: execution exceeded {self.timeout}s"],
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

            return ExecutionResult(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                exit_code=result.returncode,
                artifacts_created=artifacts_created,
                errors=errors,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timeout after {self.timeout}s",
                execution_time=self.timeout,
                exit_code=-1,
                artifacts_created=[],
                errors=[f"Timeout: execution exceeded {self.timeout}s"],
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                exit_code=-1,
                artifacts_created=[],
                errors=[f"Execution error: {e!s}"],
            )

        finally:
            # Cleanup script file
            if script_file.exists():
                script_file.unlink()

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
