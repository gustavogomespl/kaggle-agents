"""Code executor with sandboxing and error handling."""

import sys
import io
import traceback
import subprocess
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import logging

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Custom exception for code execution errors."""

    def __init__(self, message: str, error_type: str, traceback_str: str):
        """Initialize execution error.

        Args:
            message: Error message
            error_type: Type of error (e.g., "SyntaxError", "RuntimeError")
            traceback_str: Full traceback string
        """
        super().__init__(message)
        self.error_type = error_type
        self.traceback_str = traceback_str


class CodeExecutor:
    """Execute Python code with error capture and output validation."""

    def __init__(self, working_dir: Optional[Path] = None):
        """Initialize code executor.

        Args:
            working_dir: Working directory for code execution
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute_code(
        self, code: str, timeout: int = 300, capture_output: bool = True
    ) -> Tuple[bool, str, str]:
        """Execute Python code and capture results.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Write code to temporary file
        code_file = self.working_dir / "temp_execution.py"
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # Execute code in subprocess for isolation
            # Use just the filename since cwd is already set to working_dir
            result = subprocess.run(
                [sys.executable, code_file.name],
                cwd=str(self.working_dir),
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0
            stdout = result.stdout if capture_output else ""
            stderr = result.stderr if capture_output else ""

            if not success:
                logger.error(f"Code execution failed:\n{stderr}")
            else:
                logger.info("Code execution successful")

            return success, stdout, stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Code execution timed out after {timeout} seconds"
            logger.error(error_msg)
            return False, "", error_msg

        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, "", error_msg

        finally:
            # Clean up temporary file
            if code_file.exists():
                code_file.unlink()

    def execute_in_memory(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        locals_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, str, Dict[str, Any]]:
        """Execute code in memory without subprocess (faster but less isolated).

        Args:
            code: Python code to execute
            globals_dict: Global namespace (default: clean namespace)
            locals_dict: Local namespace (default: same as globals)

        Returns:
            Tuple of (success, stdout, stderr, updated_locals)
        """
        if globals_dict is None:
            globals_dict = {"__builtins__": __builtins__}

        if locals_dict is None:
            locals_dict = {}

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals_dict, locals_dict)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            logger.info("In-memory execution successful")
            return True, stdout, stderr, locals_dict

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            stderr = stderr_capture.getvalue() + "\n" + traceback_str

            logger.error(f"In-memory execution failed:\n{stderr}")
            return False, stdout_capture.getvalue(), stderr, locals_dict

    def execute_notebook_cell(
        self, code: str, notebook_globals: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        """Execute code as if it were a notebook cell (maintains state).

        Args:
            code: Python code to execute
            notebook_globals: Shared global namespace for notebook

        Returns:
            Tuple of (success, stdout, stderr)
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, notebook_globals)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            logger.info("Notebook cell execution successful")
            return True, stdout, stderr

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            stderr = stderr_capture.getvalue() + "\n" + traceback_str

            logger.error(f"Notebook cell execution failed:\n{stderr}")
            return False, stdout_capture.getvalue(), stderr

    def run_python_file(
        self, filepath: Path, args: Optional[list] = None, timeout: int = 300
    ) -> Tuple[bool, str, str]:
        """Run a Python file with optional arguments.

        Args:
            filepath: Path to Python file
            args: Command-line arguments
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        if not filepath.exists():
            return False, "", f"File not found: {filepath}"

        cmd = [sys.executable, str(filepath)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(filepath.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Execution timed out after {timeout} seconds"
            logger.error(error_msg)
            return False, "", error_msg

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg

    def validate_code_syntax(self, code: str) -> Tuple[bool, str]:
        """Validate Python code syntax without executing.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            return False, error_msg

    def parse_error_message(self, stderr: str) -> Dict[str, Any]:
        """Parse error message to extract useful information.

        Args:
            stderr: Standard error output

        Returns:
            Dictionary with parsed error information
        """
        error_info = {
            "error_type": "Unknown",
            "error_message": "",
            "line_number": None,
            "traceback": stderr,
        }

        lines = stderr.strip().split("\n")

        # Find the error type and message (usually the last line)
        if lines:
            last_line = lines[-1]
            if ": " in last_line:
                error_type, error_message = last_line.split(": ", 1)
                error_info["error_type"] = error_type
                error_info["error_message"] = error_message

        # Try to find line number
        for line in lines:
            if "line " in line.lower():
                try:
                    # Extract line number using regex or string parsing
                    import re

                    match = re.search(r"line (\d+)", line, re.IGNORECASE)
                    if match:
                        error_info["line_number"] = int(match.group(1))
                        break
                except:
                    pass

        return error_info

    def save_code(self, code: str, filename: str) -> Path:
        """Save code to file in working directory.

        Args:
            code: Python code to save
            filename: Name of file to save to

        Returns:
            Path to saved file
        """
        filepath = self.working_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Code saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Test the code executor
    executor = CodeExecutor()

    # Test 1: Simple code execution
    code1 = """
import pandas as pd
import numpy as np

print("Hello from code executor!")
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)
"""
    success, stdout, stderr = executor.execute_code(code1)
    print(f"Test 1 - Success: {success}")
    print(f"Output:\n{stdout}")

    # Test 2: Code with error
    code2 = """
print("Before error")
x = 1 / 0  # This will raise ZeroDivisionError
print("After error")
"""
    success, stdout, stderr = executor.execute_code(code2)
    print(f"\nTest 2 - Success: {success}")
    print(f"Error:\n{stderr}")
    error_info = executor.parse_error_message(stderr)
    print(f"Parsed error: {error_info}")

    # Test 3: Syntax validation
    code3 = "def foo(\nprint('invalid')"
    is_valid, error = executor.validate_code_syntax(code3)
    print(f"\nTest 3 - Valid: {is_valid}, Error: {error}")
