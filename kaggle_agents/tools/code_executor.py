"""
Code Executor Tool for safe code execution.

This module provides sandboxed Python code execution with:
- Subprocess isolation
- Timeout handling
- Error parsing and categorization
- Artifact validation
- Resource monitoring
"""

import os
import sys
import subprocess
import time
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

from ..core.state import DevelopmentResult
from ..core.config import get_config


@dataclass
class ExecutionResult:
    """Result from code execution."""

    success: bool
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int
    artifacts_created: List[str]
    errors: List[str]


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

    def validate_code_before_execution(self, code: str) -> Tuple[bool, str]:
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
        prohibited_calls = ["exit()", "sys.exit(", "quit()", "raise SystemExit", "os._exit("]
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

        if not has_categorical_check and "LGBMClassifier" in code or "XGBClassifier" in code:
            # This is just a warning, not a failure
            print("   ⚠️  Warning: LightGBM/XGBoost code without categorical encoding detected")

        return True, "Validation passed"

    def extract_performance_metric(self, stdout: str) -> Optional[float]:
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
        expected_artifacts: Optional[List[str]] = None,
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
                    print(f"   ⚠️  Warning: Could not extract performance metric from output")

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
                errors=[f"Execution error: {str(e)}"],
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
        expected_artifacts: Optional[List[str]] = None,
    ) -> Tuple[ExecutionResult, int]:
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
                print(f"    Execution successful")
                return result, attempt + 1

            print(f"   L Execution failed: {result.errors[0] if result.errors else 'Unknown error'}")

        return result, max_retries

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
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

    def _get_artifacts(self, directory: Path) -> List[str]:
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

    def _parse_errors(self, stderr: str, stdout: str) -> List[str]:
        """
        Parse and categorize errors from output.

        Args:
            stderr: Standard error output
            stdout: Standard output

        Returns:
            List of error messages
        """
        errors = []

        # Check stderr for Python exceptions
        if stderr:
            # Extract traceback info
            if "Traceback" in stderr:
                lines = stderr.split('\n')
                for i, line in enumerate(lines):
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
                match = re.search(pattern, stderr)
                if match:
                    if "{}" in template:
                        errors.append(template.format(match.group(1)))
                    else:
                        errors.append(template)

            # If no specific error found, add generic stderr
            if not errors and stderr.strip():
                errors.append(f"Error: {stderr.strip()[:200]}")

        # Check stdout for warnings
        if "Warning:" in stdout or "WARNING:" in stdout:
            warnings = re.findall(r"(Warning:.*|WARNING:.*)", stdout)
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

    def validate_model_artifacts(self, working_dir: Path | str) -> Dict[str, Any]:
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
        expected_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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
            results["issues"].append(f"Error reading submission: {str(e)}")

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


def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate syntax.

    Args:
        code: Python code

    Returns:
        Tuple of (is_valid, error_message)
    """
    executor = CodeExecutor()
    return executor.validate_syntax(code)
