"""
Error Parser Utilities.

Advanced error parsing and classification for code execution failures.
Implements PREFACE pattern for error-guided prompt repair.
"""

import re
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ParsedError:
    """Structured error information."""
    error_type: str
    error_category: str
    line_number: Optional[int]
    function_name: Optional[str]
    error_message: str
    suggested_fix: str
    severity: str  # "critical", "high", "medium", "low"


class ErrorParser:
    """
    Parse and classify errors from code execution.

    Implements PREFACE pattern for error metadata extraction.
    """

    # Error category mappings
    ERROR_CATEGORIES = {
        "import": ["ImportError", "ModuleNotFoundError"],
        "file_io": ["FileNotFoundError", "PermissionError", "IOError"],
        "data": ["KeyError", "ValueError", "TypeError", "IndexError"],
        "memory": ["MemoryError", "RecursionError"],
        "runtime": ["RuntimeError", "AttributeError", "NameError"],
        "validation": ["AssertionError", "validation failed"],
        "timeout": ["TimeoutError", "timeout", "timed out"],
        "syntax": ["SyntaxError", "IndentationError"],
    }

    # Suggested fixes by error pattern
    ERROR_FIXES = {
        "ImportError": "Install missing package or check import statement",
        "ModuleNotFoundError": "Verify package is installed: pip install <package>",
        "FileNotFoundError": "Check file path exists and is correct",
        "KeyError": "Verify column/key name exists in data",
        "ValueError": "Check data types and value ranges",
        "TypeError": "Verify data types match expected types",
        "MemoryError": "Reduce dataset size or use chunking",
        "TimeoutError": "Optimize code or increase timeout limit",
        "validation failed": "Ensure code outputs 'Final Validation Performance: {score}'",
        "missing_output_format": "Add print(f'Final Validation Performance: {score}') at end",
    }

    def parse_error(self, error_msg: str, stderr: str = "") -> ParsedError:
        """
        Parse error message into structured format.

        Args:
            error_msg: Primary error message
            stderr: Full stderr output

        Returns:
            ParsedError object with structured information
        """
        if not error_msg:
            error_msg = stderr[:200] if stderr else "Unknown error"

        # Classify error type
        error_type = self._extract_error_type(error_msg)
        error_category = self._categorize_error(error_type)

        # Extract line number
        line_number = self._extract_line_number(stderr or error_msg)

        # Extract function name
        function_name = self._extract_function_name(stderr or error_msg)

        # Get suggested fix
        suggested_fix = self._get_suggested_fix(error_type, error_msg)

        # Determine severity
        severity = self._determine_severity(error_category, error_type)

        return ParsedError(
            error_type=error_type,
            error_category=error_category,
            line_number=line_number,
            function_name=function_name,
            error_message=error_msg,
            suggested_fix=suggested_fix,
            severity=severity,
        )

    def _extract_error_type(self, error_msg: str) -> str:
        """Extract error type from message."""
        # Common pattern: "ErrorType: message"
        match = re.search(r'(\w+Error|Timeout|validation failed):', error_msg, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback patterns
        if "import" in error_msg.lower():
            return "ImportError"
        elif "file not found" in error_msg.lower() or "no such file" in error_msg.lower():
            return "FileNotFoundError"
        elif "key" in error_msg.lower() and "not found" in error_msg.lower():
            return "KeyError"
        elif "validation" in error_msg.lower():
            return "validation_failed"
        elif "timeout" in error_msg.lower():
            return "TimeoutError"
        else:
            return "RuntimeError"

    def _categorize_error(self, error_type: str) -> str:
        """Categorize error into high-level category."""
        for category, error_types in self.ERROR_CATEGORIES.items():
            if any(error_type in et or et in error_type for et in error_types):
                return category
        return "runtime"

    def _extract_line_number(self, text: str) -> Optional[int]:
        """Extract line number from traceback."""
        # Pattern: "line 42"
        match = re.search(r'line (\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_function_name(self, text: str) -> Optional[str]:
        """Extract function name from traceback."""
        # Pattern: "in function_name"
        match = re.search(r'in (\w+)', text)
        if match:
            return match.group(1)
        return None

    def _get_suggested_fix(self, error_type: str, error_msg: str) -> str:
        """Get suggested fix for error."""
        # Check for exact match
        for pattern, fix in self.ERROR_FIXES.items():
            if pattern.lower() in error_type.lower():
                return fix

        # Check error message for specific hints
        if "final validation performance" in error_msg.lower():
            return self.ERROR_FIXES["missing_output_format"]

        # Generic fix
        return f"Review code logic and fix {error_type}"

    def _determine_severity(self, error_category: str, error_type: str) -> str:
        """Determine error severity."""
        critical_categories = ["syntax", "import"]
        high_categories = ["file_io", "validation", "timeout"]
        medium_categories = ["data", "memory"]

        if error_category in critical_categories:
            return "critical"
        elif error_category in high_categories:
            return "high"
        elif error_category in medium_categories:
            return "medium"
        else:
            return "low"

    def extract_execution_trace(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Extract execution trace for semantic alignment (CodeRL+ pattern).

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            Execution trace dictionary
        """
        trace = {
            "warnings": [],
            "print_statements": [],
            "performance_metric": None,
            "execution_steps": [],
        }

        # Extract warnings
        warning_pattern = r'(Warning:|WARNING:)(.+?)(?:\n|$)'
        warnings = re.findall(warning_pattern, stdout + stderr, re.IGNORECASE)
        trace["warnings"] = [w[1].strip() for w in warnings]

        # Extract print statements (execution steps)
        print_lines = [
            line.strip() for line in stdout.split('\n')
            if line.strip() and not line.startswith('Traceback')
        ]
        trace["print_statements"] = print_lines[:20]  # Keep first 20

        # Extract performance metric
        perf_pattern = r'Final Validation Performance:\s*([\d.]+)'
        match = re.search(perf_pattern, stdout, re.IGNORECASE)
        if match:
            trace["performance_metric"] = float(match.group(1))

        # Identify execution steps from prints
        step_keywords = [
            "loading", "preprocessing", "training", "validating",
            "predicting", "saving", "evaluating"
        ]
        for line in print_lines:
            line_lower = line.lower()
            for keyword in step_keywords:
                if keyword in line_lower:
                    trace["execution_steps"].append(keyword)
                    break

        return trace


def parse_error_simple(error_msg: str) -> str:
    """
    Simple error parsing for quick classification.

    Args:
        error_msg: Error message

    Returns:
        Error type string
    """
    parser = ErrorParser()
    parsed = parser.parse_error(error_msg)
    return parsed.error_type
