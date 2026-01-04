"""
Code Executor Tool for safe code execution.

This module provides sandboxed Python code execution with:
- Subprocess isolation
- Timeout handling
- Error parsing and categorization
- Artifact validation
- Resource monitoring
- Real-time stdout streaming for training logs
"""

import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any


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


@dataclass
class ExecutionProgress:
    """
    Track execution progress for checkpoint/resume support.

    This class captures incremental progress during long-running executions,
    enabling intelligent timeout decisions and partial result recovery.
    """

    # Fold tracking (for cross-validation)
    folds_completed: int = 0
    total_folds: int = 5
    fold_scores: list = None  # type: ignore  # list[float]

    # Score tracking
    current_cv_score: float = None  # type: ignore  # Optional[float]
    best_fold_score: float = None  # type: ignore  # Optional[float]

    # Artifact tracking
    models_saved: list = None  # type: ignore  # list[str]
    oof_predictions_saved: bool = False
    test_predictions_saved: bool = False

    # Time tracking
    elapsed_seconds: float = 0.0
    avg_fold_time: float = None  # type: ignore  # Optional[float]
    estimated_remaining: float = None  # type: ignore  # Optional[float]

    # Status
    current_phase: str = "initializing"  # initializing, training, validating, predicting
    last_output: str = ""

    def __post_init__(self):
        if self.fold_scores is None:
            self.fold_scores = []
        if self.models_saved is None:
            self.models_saved = []

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage based on folds completed."""
        if self.total_folds == 0:
            return 0.0
        return (self.folds_completed / self.total_folds) * 100

    @property
    def can_create_partial_submission(self) -> bool:
        """Check if we have enough data for a partial submission."""
        return self.folds_completed >= 1 and self.oof_predictions_saved

    def update_from_stdout(self, line: str) -> bool:
        """
        Update progress from a stdout line.

        Returns True if progress was updated.
        """
        updated = False
        line_lower = line.lower()

        # Detect fold completion patterns
        # Pattern: "Fold X best AUC: Y.YYYY" or "Fold X score: Y.YYYY"
        fold_patterns = [
            r"fold\s*(\d+)\s*(?:best\s+)?(?:auc|score|accuracy|f1|rmse|mae)[:=]\s*([\d.]+)",
            r"=+\s*fold\s*(\d+)\s*=+",
            r"fold\s*(\d+)\s*(?:of\s*\d+)?\s*(?:completed?|finished?|done)",
        ]

        for pattern in fold_patterns:
            match = re.search(pattern, line_lower)
            if match:
                try:
                    fold_num = int(match.group(1))
                    # Handle 0-indexed vs 1-indexed folds
                    completed = fold_num + 1 if fold_num < self.total_folds else fold_num
                    if completed > self.folds_completed:
                        self.folds_completed = min(completed, self.total_folds)
                        updated = True

                    # Try to extract score if present
                    if len(match.groups()) >= 2:
                        score = float(match.group(2))
                        self.fold_scores.append(score)
                        self.best_fold_score = max(self.fold_scores)
                except (ValueError, IndexError):
                    pass

        # Detect Final Validation Performance
        if "final validation performance" in line_lower:
            match = re.search(r"([\d.]+)", line)
            if match:
                self.current_cv_score = float(match.group(1))
                updated = True

        # Detect OOF/test prediction saves
        if "saved oof" in line_lower or "oof_pred" in line_lower:
            self.oof_predictions_saved = True
            updated = True

        if "saved test" in line_lower or "test_pred" in line_lower:
            self.test_predictions_saved = True
            updated = True

        # Detect model saves
        if (
            "saved model" in line_lower
            or "saving model" in line_lower
            or ".pkl" in line_lower
            or ".joblib" in line_lower
            or ".pth" in line_lower
        ):
            # Extract model name if possible
            match = re.search(r"(\w+\.(pkl|joblib|pth|pt|h5))", line_lower)
            if match and match.group(1) not in self.models_saved:
                self.models_saved.append(match.group(1))
                updated = True

        # Detect phase changes
        if "loading" in line_lower or "preparing" in line_lower:
            self.current_phase = "initializing"
        elif "training" in line_lower or "fitting" in line_lower:
            self.current_phase = "training"
        elif "validating" in line_lower or "evaluating" in line_lower:
            self.current_phase = "validating"
        elif "predicting" in line_lower or "inference" in line_lower:
            self.current_phase = "predicting"

        self.last_output = line.strip()[:200]
        return updated

    def estimate_remaining_time(self):
        """Estimate remaining time based on fold progress."""
        if self.folds_completed == 0 or self.elapsed_seconds == 0:
            return None

        self.avg_fold_time = self.elapsed_seconds / self.folds_completed
        remaining_folds = self.total_folds - self.folds_completed
        self.estimated_remaining = self.avg_fold_time * remaining_folds

        return self.estimated_remaining

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "folds_completed": self.folds_completed,
            "total_folds": self.total_folds,
            "fold_scores": self.fold_scores,
            "current_cv_score": self.current_cv_score,
            "best_fold_score": self.best_fold_score,
            "models_saved": self.models_saved,
            "oof_predictions_saved": self.oof_predictions_saved,
            "test_predictions_saved": self.test_predictions_saved,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_fold_time": self.avg_fold_time,
            "estimated_remaining": self.estimated_remaining,
            "current_phase": self.current_phase,
            "progress_percent": self.progress_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionProgress":
        """Create from dictionary."""
        return cls(
            folds_completed=data.get("folds_completed", 0),
            total_folds=data.get("total_folds", 5),
            fold_scores=data.get("fold_scores", []),
            current_cv_score=data.get("current_cv_score"),
            best_fold_score=data.get("best_fold_score"),
            models_saved=data.get("models_saved", []),
            oof_predictions_saved=data.get("oof_predictions_saved", False),
            test_predictions_saved=data.get("test_predictions_saved", False),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            avg_fold_time=data.get("avg_fold_time"),
            estimated_remaining=data.get("estimated_remaining"),
            current_phase=data.get("current_phase", "unknown"),
        )


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
        # Lazy import to avoid circular dependency
        from ..core.config import get_config

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
            code = code.replace(
                "sys.exit(1)", 'raise ValueError("Missing required data or configuration")'
            )
            fixes_applied.append("Replaced sys.exit(1) with ValueError")

        if "sys.exit(0)" in code:
            code = code.replace("sys.exit(0)", "pass  # Replaced sys.exit(0)")
            fixes_applied.append("Replaced sys.exit(0) with pass")

        # Generic sys.exit() with variable
        if "sys.exit(" in code:
            code = re.sub(r"sys\.exit\([^)]*\)", 'raise RuntimeError("Execution terminated")', code)
            fixes_applied.append("Replaced remaining sys.exit() calls with RuntimeError")

        # Auto-fix other termination calls
        if re.search(r"(?<!\\w)exit\\(\\)", code) and "sys.exit" not in code:
            code = re.sub(r"(?<!\\w)exit\\(\\)", "pass  # Replaced exit()", code)
            fixes_applied.append("Replaced exit() with pass")

        if re.search(r"(?<!\\w)quit\\(\\)", code):
            code = re.sub(r"(?<!\\w)quit\\(\\)", "pass  # Replaced quit()", code)
            fixes_applied.append("Replaced quit() with pass")

        if "os._exit(" in code:
            code = re.sub(r"os\._exit\([^)]*\)", 'raise RuntimeError("Forced exit")', code)
            fixes_applied.append("Replaced os._exit() with RuntimeError")

        if fixes_applied:
            print(f"   🔧 Auto-sanitized code: {', '.join(fixes_applied)}")

        return code, fixes_applied

    def detect_model_training(self, code: str) -> list[str]:
        """
        Detect model training patterns in code.

        Used to validate that preprocessing/feature_engineering components
        do not contain model training, which violates their contract.

        Args:
            code: Python code to analyze

        Returns:
            List of detected training patterns (empty if none found)
        """
        training_patterns = [
            # Sklearn-style fit patterns
            (r"\.fit\s*\(", ".fit() call"),
            (r"\.fit_transform\s*\(", ".fit_transform() call"),

            # Tree-based models
            (r"RandomForest(?:Classifier|Regressor)\s*\(", "RandomForest model"),
            (r"(?:LGBM|LightGBM)(?:Classifier|Regressor)\s*\(", "LightGBM model"),
            (r"(?:XGB|XGBoost)(?:Classifier|Regressor)\s*\(", "XGBoost model"),
            (r"CatBoost(?:Classifier|Regressor)\s*\(", "CatBoost model"),
            (r"GradientBoosting(?:Classifier|Regressor)\s*\(", "GradientBoosting model"),
            (r"ExtraTrees(?:Classifier|Regressor)\s*\(", "ExtraTrees model"),

            # Linear models
            (r"LogisticRegression\s*\(", "LogisticRegression model"),
            (r"(?:Linear|Ridge|Lasso|ElasticNet)(?:Regression|Classifier)\s*\(", "Linear model"),
            (r"(?:SVC|SVR)\s*\(", "SVM model"),

            # Neural network training
            (r"model\.train\s*\(", "PyTorch model.train()"),
            (r"\.fit\s*\([^)]*epochs", "Keras/TF model.fit() with epochs"),
            (r"optimizer\.step\s*\(", "PyTorch optimizer.step()"),
            (r"loss\.backward\s*\(", "PyTorch loss.backward()"),

            # Cross-validation (implies training)
            (r"cross_val_predict\s*\(", "cross_val_predict()"),
            (r"cross_val_score\s*\(", "cross_val_score()"),
        ]

        detected = []
        for pattern, description in training_patterns:
            if re.search(pattern, code):
                detected.append(description)

        return detected

    def validate_code_before_execution(
        self, code: str, component_type: str | None = None
    ) -> tuple[bool, str]:
        """
        Validates code meets requirements before execution (MLE-STAR pattern).

        Args:
            code: Python code to validate
            component_type: Type of component ('preprocessing', 'feature_engineering', 'model', etc.)
                           Used to validate that preprocessing doesn't train models.

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
            if re.search(r"\.fit\([^)]*early_stopping_rounds=", code):
                return False, (
                    "API Error: early_stopping_rounds cannot be passed to fit(). "
                    "Use callbacks=[lgb.early_stopping(100)] for LightGBM, "
                    "callbacks=[xgb.callback.EarlyStopping(100)] for XGBoost <2, "
                    "or pass early_stopping_rounds in the XGBoost 2.0+ constructor."
                )

        # Check 4: Has basic structure for ML code
        required_patterns = [
            (
                "import pandas" in code or "import numpy" in code,
                "Missing required imports (pandas or numpy)",
            ),
        ]

        for has_pattern, error_msg in required_patterns:
            if not has_pattern:
                return False, error_msg

        # Check 5: Block model training in preprocessing/feature_engineering components
        # These component types should ONLY transform data, not train ML models
        if component_type in ("preprocessing", "feature_engineering"):
            training_patterns = self.detect_model_training(code)

            if training_patterns:
                # Allow certain patterns for legitimate preprocessing uses
                # e.g., LabelEncoder.fit_transform for encoding, StandardScaler.fit for scaling
                allowed_patterns = {
                    ".fit() call",  # Could be scaler.fit() which is allowed
                    ".fit_transform() call",  # Could be encoder.fit_transform() which is allowed
                }

                # Check if patterns are for feature importance-based selection (allowed)
                is_feature_selection = (
                    "feature_importances_" in code
                    or "get_score(" in code  # XGBoost feature importance
                    or "feature_importance(" in code  # LightGBM
                    or "SelectFromModel" in code
                )

                # Filter out allowed patterns
                blocked_patterns = [
                    p for p in training_patterns
                    if p not in allowed_patterns or any(
                        model in p for model in [
                            "RandomForest", "LightGBM", "XGBoost", "CatBoost",
                            "LogisticRegression", "GradientBoosting", "SVM",
                            "Linear model", "ExtraTrees", "cross_val"
                        ]
                    )
                ]

                if blocked_patterns and not is_feature_selection:
                    return False, (
                        f"Model training detected in {component_type} component: "
                        f"{', '.join(blocked_patterns)}. "
                        f"Preprocessing/feature_engineering components MUST NOT train models. "
                        f"Move model training to a 'model' component instead."
                    )
                if blocked_patterns:
                    # It's feature selection - just warn but allow
                    print(
                        f"   ⚠️  {component_type} uses models for feature selection: "
                        f"{', '.join(blocked_patterns)}"
                    )

        # Check 6: Warning about categorical features (informational only)
        # This is a soft check - we warn but don't fail
        has_categorical_check = (
            "select_dtypes" in code
            or "LabelEncoder" in code
            or "OneHotEncoder" in code
            or "TargetEncoder" in code
            or "CatBoost" in code
        )

        if (not has_categorical_check and "LGBMClassifier" in code) or "XGBClassifier" in code:
            # This is just a warning, not a failure
            print("   ⚠️  Warning: LightGBM/XGBoost code without categorical encoding detected")

        return True, "Validation passed"

    def extract_performance_metric(self, stdout: str):
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
                    score_str = re.sub(r"[^\d.\-]", "", score_str)
                    return float(score_str)
                except (ValueError, IndexError):
                    continue
        return None

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

            # Start process with line-buffered output for real-time streaming
            process = subprocess.Popen(
                [sys.executable, "-u", str(script_file)],  # -u for unbuffered output
                cwd=str(working_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
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
            last_progress_time = start_time
            last_output_time = start_time

            while True:
                # Check if process completed
                poll_result = process.poll()

                # Process any queued output (real-time streaming)
                output_printed = False
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
                        output_printed = True
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
                        output_printed = True
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
                    process.kill()
                    process.wait()
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
                print("    Execution successful")
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
        optuna_info_pattern = r"\[I \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\][^\n]*\n?"
        filtered = re.sub(optuna_info_pattern, "", output)

        # Also filter Optuna study creation messages
        study_pattern = r"A new study created in memory with name:[^\n]*\n?"
        filtered = re.sub(study_pattern, "", filtered)

        # Filter Optuna trial completion messages
        trial_pattern = r"Trial \d+ finished with value:[^\n]*\n?"
        filtered = re.sub(trial_pattern, "", filtered)

        # Filter Optuna sampler messages
        sampler_pattern = r"\[I[^\]]*\].*?(?:Sampler|TPE|CMA|Grid)[^\n]*\n?"
        filtered = re.sub(sampler_pattern, "", filtered)

        return filtered.strip()

    def _filter_tqdm_logs(self, output: str) -> str:
        """
        Filter out tqdm progress bar output that is commonly written to stderr.

        Many ML scripts use tqdm, which writes progress updates to stderr by default.
        Those updates are not errors and should not cause the execution to be marked
        as failed.
        """
        if not output:
            return ""

        kept_lines: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Keep anything that looks like an actual exception/error.
            if any(tok in stripped for tok in ("Traceback", "Error", "Exception")):
                kept_lines.append(line)
                continue

            # Drop common tqdm progress patterns, e.g.:
            # "Fold0 Train Epoch1:  12%|█▏        | 33/275 [00:41<04:49,  1.20s/it, loss=1.62]"
            # "Validation:   0%|          | 0/138 [00:00<?, ?it/s]"
            has_bar = "%|" in stripped and "|" in stripped
            has_rate = ("it/s" in stripped) or ("s/it" in stripped)
            has_speed = bool(re.search(r"\b\d+(\.\d+)?\s*[kMGT]?B/s\b", stripped, re.IGNORECASE))
            has_counts = bool(re.search(r"\b\d+/\d+\b", stripped))
            if has_bar and (has_rate or has_speed or has_counts):
                continue
            if (has_rate or has_speed) and has_counts:
                continue

            kept_lines.append(line)

        return "\n".join(kept_lines).strip()

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

        # Filter out non-error stderr noise before parsing.
        # Optuna and tqdm commonly log to stderr even on successful runs.
        stderr_filtered = self._filter_optuna_logs(stderr) if stderr else ""
        stderr_filtered = self._filter_tqdm_logs(stderr_filtered)
        stderr_filtered = self._filter_framework_logs(stderr_filtered)
        if stderr_filtered:
            stderr_lines = []
            skip_next_context = False
            for line in stderr_filtered.splitlines():
                clean_line = line
                if clean_line.startswith("⚠️"):
                    clean_line = clean_line[len("⚠️") :]

                if skip_next_context:
                    if clean_line.strip() and clean_line[:1].isspace():
                        skip_next_context = False
                        continue
                    skip_next_context = False

                if "Warning" in clean_line or "WARNING" in clean_line:
                    skip_next_context = True
                    continue

                stderr_lines.append(line)
            stderr_filtered = "\n".join(stderr_lines).strip()

        # Check stderr for Python exceptions
        if stderr_filtered:
            # Extract traceback info
            if "Traceback" in stderr_filtered:
                lines = stderr_filtered.split("\n")
                error_line = ""
                for line in reversed(lines):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if re.match(r"^[A-Za-z_]+(Error|Exception):", stripped):
                        error_line = stripped
                        break
                if not error_line:
                    for line in reversed(lines):
                        stripped = line.strip()
                        if stripped:
                            error_line = stripped
                            break
                if error_line:
                    errors.append(error_line)

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
                if not re.match(r"^\s*\[.*?\]\s*$", stderr_filtered.strip()):
                    error_hint = re.search(
                        r"(Traceback|Error|Exception|Segmentation fault|SIGSEGV|Killed|Out of memory|OOM|CUDA error)",
                        stderr_filtered,
                        re.IGNORECASE,
                    )
                    if error_hint:
                        errors.append(f"Error: {stderr_filtered.strip()[:200]}")

        # Do not treat warnings as errors.

        return errors

    def _filter_framework_logs(self, output: str) -> str:
        """
        Filter out common non-fatal ML framework stderr noise.

        These messages are typically warnings or informational logs that
        shouldn't cause execution failure (e.g., cuFFT factory registration).
        """
        if not output:
            return ""

        drop_patterns = [
            r"Unable to register cuFFT factory",
            r"Unable to register cuDNN factory",
            r"Unable to register cuBLAS factory",
        ]

        kept_lines: list[str] = []
        for line in output.splitlines():
            if any(re.search(pattern, line) for pattern in drop_patterns):
                continue
            kept_lines.append(line)

        return "\n".join(kept_lines).strip()


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

    def validate_model_artifacts(self, working_dir: str) -> dict[str, Any]:
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
        working_dir: str,
        expected_columns: list = None,
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
    working_dir: str,
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


def validate_code_syntax(code: str) -> tuple:
    """
    Convenience function to validate syntax.

    Args:
        code: Python code

    Returns:
        Tuple of (is_valid, error_message)
    """
    executor = CodeExecutor()
    return executor.validate_syntax(code)
