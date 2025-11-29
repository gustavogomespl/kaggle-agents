"""
Code Executor Tool for safe code execution.

This module provides sandboxed Python code execution with:
- Subprocess isolation
- Timeout handling
- Error parsing and categorization
- Artifact validation
- Resource monitoring
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import get_config


@dataclass
class ExecutionResult:
    """Result from code execution."""

    success: bool
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int
    artifacts_created: list[str]
    errors: list[str]


class CodeExecutor:
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
        self.config = get_config()
        self.timeout = timeout

    def sanitize_code(self, code: str) -> tuple[str, list[str]]:
        """
        Automatically sanitize code by removing/replacing prohibited patterns.

        Args:
            code: Python code to sanitize

        Returns:
            Tuple of (sanitized_code, list_of_fixes_applied)
        """
        fixes_applied = []

        # Auto-fix sys.exit() calls
        if "sys.exit(1)" in code:
            code = code.replace("sys.exit(1)", 'raise ValueError("Missing required data or configuration")')
            fixes_applied.append("Replaced sys.exit(1) with ValueError")

        if "sys.exit(0)" in code:
            code = code.replace("sys.exit(0)", "pass  # Replaced sys.exit(0)")
            fixes_applied.append("Replaced sys.exit(0) with pass")

        # Generic sys.exit() with variable
        if "sys.exit(" in code:
            code = re.sub(r'sys\.exit\([^)]*\)', 'raise RuntimeError("Execution terminated")', code)
            fixes_applied.append("Replaced remaining sys.exit() calls with RuntimeError")

        # Auto-fix other termination calls
        if "exit()" in code and "sys.exit" not in code:
            code = code.replace("exit()", "pass  # Replaced exit()")
            fixes_applied.append("Replaced exit() with pass")

        if "quit()" in code:
            code = code.replace("quit()", "pass  # Replaced quit()")
            fixes_applied.append("Replaced quit() with pass")

        if "os._exit(" in code:
            code = re.sub(r'os\._exit\([^)]*\)', 'raise RuntimeError("Forced exit")', code)
            fixes_applied.append("Replaced os._exit() with RuntimeError")

        if fixes_applied:
            print(f"   🔧 Auto-sanitized code: {', '.join(fixes_applied)}")

        return code, fixes_applied

    def validate_code_before_execution(self, code: str) -> tuple[bool, str]:
        """
        Validates code meets requirements before execution (MLE-STAR pattern).

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, message)
        """
        # Check 1: Has required output format
        if "Final Validation Performance" not in code:
            return False, "Missing required output: 'Final Validation Performance: {score}'"

        # Check 2: No prohibited exit() calls (enhanced check)
        # Note: These should have been sanitized by sanitize_code() before validation
        prohibited_calls = ["sys.exit(", "quit()", "raise SystemExit", "os._exit("]
        for call in prohibited_calls:
            if call in code:
                return False, f"Code contains prohibited termination call: {call}"

        # Check 3: Early stopping rounds misuse (common API error)
        if "early_stopping_rounds=" in code and ".fit(" in code:
            # Check if it's being passed as a parameter to fit()
            if re.search(r'\.fit\([^)]*early_stopping_rounds=', code):
                return False, (
                    "API Error: early_stopping_rounds cannot be passed to fit(). "
                    "Use callbacks=[lgb.early_stopping(100)] or callbacks=[xgb.callback.EarlyStopping(100)] instead."
                )

        # Check 4: Has basic structure for ML code
        required_patterns = [
            ("import pandas" in code or "import numpy" in code, "Missing required imports (pandas or numpy)"),
        ]

        for has_pattern, error_msg in required_patterns:
            if not has_pattern:
                return False, error_msg

        # Check 5: Warning about categorical features (informational only)
        # This is a soft check - we warn but don't fail
        has_categorical_check = (
            "select_dtypes" in code or
            "LabelEncoder" in code or
            "OneHotEncoder" in code or
            "TargetEncoder" in code or
            "CatBoost" in code
        )

        if (not has_categorical_check and "LGBMClassifier" in code) or "XGBClassifier" in code:
            # This is just a warning, not a failure
            print("   ⚠️  Warning: LightGBM/XGBoost code without categorical encoding detected")

        return True, "Validation passed"

    def extract_performance_metric(self, stdout: str) -> float | None:
        """
        Extracts validation performance score from code output (MLE-STAR pattern).

        Args:
            stdout: Standard output from code execution

        Returns:
            Performance score if found, None otherwise
        """
        for line in stdout.splitlines():
            if "Final Validation Performance:" in line:
                try:
                    # Extract score after the colon
                    score_str = line.split(":")[-1].strip()
                    # Remove any non-numeric characters except decimal point and minus
                    score_str = re.sub(r'[^\d.\-]', '', score_str)
                    return float(score_str)
                except (ValueError, IndexError):
                    continue
        return None

    def execute(
        self,
        code: str,
        working_dir: Path | str,
        expected_artifacts: list[str] | None = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute
            working_dir: Working directory for execution
            expected_artifacts: List of expected output files/directories

        Returns:
            ExecutionResult with execution details
        """
        # AUTO-SANITIZE CODE (remove sys.exit, etc.)
        code, _fixes_applied = self.sanitize_code(code)

        # PRE-EXECUTION VALIDATION (MLE-STAR Pattern)
        is_valid, validation_msg = self.validate_code_before_execution(code)
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
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)

            # Execute in subprocess with progress monitoring
            start_time = time.time()

            # Start process
            process = subprocess.Popen(
                [sys.executable, str(script_file)],
                cwd=str(working_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Monitor progress with timeout
            progress_interval = 30  # Print progress every 30s
            last_progress_time = start_time

            while True:
                # Check if process completed
                poll_result = process.poll()
                if poll_result is not None:
                    # Process finished
                    break

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(process.args, self.timeout)

                # Print progress update
                if elapsed - (last_progress_time - start_time) >= progress_interval:
                    remaining = self.timeout - elapsed
                    print(f"      ⏳ Execution in progress... ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)")
                    last_progress_time = time.time()

                # Sleep briefly before next check
                time.sleep(1)

            # Get output
            stdout, stderr = process.communicate()
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
        working_dir: Path | str,
        max_retries: int = 3,
        expected_artifacts: list[str] | None = None,
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
                print("    Execution successful")
                return result, attempt + 1

            print(f"   L Execution failed: {result.errors[0] if result.errors else 'Unknown error'}")

        return result, max_retries

    def validate_syntax(self, code: str) -> tuple[bool, str | None]:
        """
        Validate Python syntax without executing.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
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

    def _filter_optuna_logs(self, output: str) -> str:
        """
        Filter out Optuna informational logs that are not errors.

        Optuna logs like "[I 2025-11-24 ...] Trial 0 finished with value: ..."
        are informational and should not be treated as errors.

        Args:
            output: stderr or stdout content

        Returns:
            Filtered output with Optuna info logs removed
        """
        # Pattern for Optuna info logs: [I YYYY-MM-DD HH:MM:SS,...]
        optuna_info_pattern = r'\[I \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\][^\n]*\n?'
        filtered = re.sub(optuna_info_pattern, '', output)

        # Also filter Optuna study creation messages
        study_pattern = r'A new study created in memory with name:[^\n]*\n?'
        filtered = re.sub(study_pattern, '', filtered)

        # Filter Optuna trial completion messages
        trial_pattern = r'Trial \d+ finished with value:[^\n]*\n?'
        filtered = re.sub(trial_pattern, '', filtered)

        # Filter Optuna sampler messages
        sampler_pattern = r'\[I[^\]]*\].*?(?:Sampler|TPE|CMA|Grid)[^\n]*\n?'
        filtered = re.sub(sampler_pattern, '', filtered)

        return filtered.strip()

    def _parse_errors(self, stderr: str, stdout: str) -> list[str]:
        """
        Parse and categorize errors from output.

        Args:
            stderr: Standard error output
            stdout: Standard output

        Returns:
            List of error messages
        """
        errors = []

        # Filter out Optuna informational logs before parsing
        stderr_filtered = self._filter_optuna_logs(stderr) if stderr else ""

        # Check stderr for Python exceptions
        if stderr_filtered:
            # Extract traceback info
            if "Traceback" in stderr_filtered:
                lines = stderr_filtered.split('\n')
                for _i, line in enumerate(lines):
                    if line.startswith("Traceback"):
                        # Get the actual error (usually last line)
                        error_line = lines[-2] if len(lines) > 1 else line
                        errors.append(error_line.strip())
                        break

            # Common error patterns
            error_patterns = [
                (r"ModuleNotFoundError: No module named '(\w+)'", "Missing module: {}"),
                (r"FileNotFoundError: .* '(.*)'", "File not found: {}"),
                (r"KeyError: '(.*)'", "Missing key: {}"),
                (r"ValueError: (.*)", "Value error: {}"),
                (r"TypeError: (.*)", "Type error: {}"),
                (r"MemoryError", "Out of memory"),
            ]

            for pattern, template in error_patterns:
                match = re.search(pattern, stderr_filtered)
                if match:
                    if "{}" in template:
                        errors.append(template.format(match.group(1)))
                    else:
                        errors.append(template)

            # If no specific error found, add generic stderr (but not if only Optuna logs)
            if not errors and stderr_filtered.strip():
                # Double-check it's not just leftover Optuna formatting
                if not re.match(r'^\s*\[.*?\]\s*$', stderr_filtered.strip()):
                    errors.append(f"Error: {stderr_filtered.strip()[:200]}")

        # Check stdout for warnings (but not Optuna trial info)
        stdout_filtered = self._filter_optuna_logs(stdout) if stdout else ""
        if "Warning:" in stdout_filtered or "WARNING:" in stdout_filtered:
            warnings = re.findall(r"(Warning:.*|WARNING:.*)", stdout_filtered)
            if warnings and len(errors) < 3:  # Limit warnings
                errors.extend(warnings[:3])

        return errors


class ArtifactValidator:
    """
    Validate generated artifacts (models, data files, submissions).

    This class provides domain-specific validation:
    - Model files (.pkl, .joblib)
    - Data files (train/test CSVs)
    - Submission files
    - Checkpoints
    """

    def __init__(self):
        """Initialize artifact validator."""
        pass

    def validate_model_artifacts(self, working_dir: Path | str) -> dict[str, Any]:
        """
        Validate that model artifacts were created correctly.

        Args:
            working_dir: Working directory

        Returns:
            Validation results dictionary
        """
        working_path = Path(working_dir) if isinstance(working_dir, str) else working_dir

        results = {
            "valid": False,
            "models_found": [],
            "issues": [],
        }

        # Check for model files
        model_extensions = [".pkl", ".joblib", ".h5", ".pth", ".pt"]
        models_dir = working_path / "models"

        model_files = []
        if models_dir.exists():
            for ext in model_extensions:
                model_files.extend(models_dir.glob(f"*{ext}"))

        if not model_files:
            results["issues"].append("No model files found in models/ directory")
        else:
            results["models_found"] = [f.name for f in model_files]
            results["valid"] = True

        return results

    def validate_submission_artifact(
        self,
        working_dir: Path | str,
        expected_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate submission file.

        Args:
            working_dir: Working directory
            expected_columns: Expected column names

        Returns:
            Validation results
        """
        import pandas as pd

        working_path = Path(working_dir) if isinstance(working_dir, str) else working_dir

        results = {
            "valid": False,
            "submission_path": None,
            "row_count": 0,
            "columns": [],
            "issues": [],
        }

        # Check for submission.csv
        submission_file = working_path / "submission.csv"

        if not submission_file.exists():
            results["issues"].append("submission.csv not found")
            return results

        try:
            # Read and validate
            df = pd.read_csv(submission_file)

            results["submission_path"] = str(submission_file)
            results["row_count"] = len(df)
            results["columns"] = df.columns.tolist()

            # Check for empty
            if len(df) == 0:
                results["issues"].append("Submission file is empty")
                return results

            # Check expected columns
            if expected_columns:
                missing = set(expected_columns) - set(df.columns)
                if missing:
                    results["issues"].append(f"Missing columns: {missing}")
                    return results

            # Check for NaN values
            if df.isnull().any().any():
                results["issues"].append("Submission contains NaN values")
                # Allow warning but not failure for NaNs

            results["valid"] = True

        except Exception as e:
            results["issues"].append(f"Error reading submission: {e!s}")

        return results


# ==================== Convenience Functions ====================

def execute_code(
    code: str,
    working_dir: Path | str,
    timeout: int = 300,
) -> ExecutionResult:
    """
    Convenience function to execute code.

    Args:
        code: Python code
        working_dir: Working directory
        timeout: Timeout in seconds

    Returns:
        ExecutionResult
    """
    executor = CodeExecutor(timeout=timeout)
    return executor.execute(code, working_dir)


def validate_code_syntax(code: str) -> tuple[bool, str | None]:
    """
    Convenience function to validate syntax.

    Args:
        code: Python code

    Returns:
        Tuple of (is_valid, error_message)
    """
    executor = CodeExecutor()
    return executor.validate_syntax(code)
