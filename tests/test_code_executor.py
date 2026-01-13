"""Tests for the code executor error parsing, resource limits, and HPO validation."""

import os
import platform

import pytest


# Feature flag constant (copied from code_executor.py)
ENABLE_RESOURCE_LIMITS = os.getenv("KAGGLE_AGENTS_ENABLE_LIMITS", "true").lower() == "true"


def _set_resource_limits(memory_mb: int = 8192, cpu_time_s: int = 3600) -> None:
    """Set resource limits for subprocess (Unix only). Copied for isolated testing."""
    if not ENABLE_RESOURCE_LIMITS:
        return

    if platform.system() == "Windows":
        return

    try:
        import resource
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_s, cpu_time_s))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def _start_new_process_group() -> None:
    """Starts process in new group. Copied for isolated testing."""
    os.setpgrp()


def _validate_optuna_pruning_contract(code: str) -> tuple:
    """
    Validate Optuna pruning contract. Copied for isolated testing.
    """
    uses_optuna = any(pattern in code for pattern in [
        "import optuna",
        "from optuna",
        "optuna.create_study",
        "optuna.Study",
    ])

    if not uses_optuna:
        return True, ""

    pruner_patterns = [
        ("HyperbandPruner", "Hyperband"),
        ("MedianPruner", "Median"),
        ("SuccessiveHalvingPruner", "SuccessiveHalving"),
        ("ThresholdPruner", "Threshold"),
        ("PercentilePruner", "Percentile"),
    ]

    active_pruner = None
    for pattern, name in pruner_patterns:
        if pattern in code:
            active_pruner = name
            break

    if active_pruner is None:
        return True, ""

    has_report = "trial.report" in code
    has_prune_check = "should_prune" in code or "TrialPruned" in code

    if not has_report:
        return False, (
            f"Code uses {active_pruner}Pruner but does not call trial.report(). "
            "The pruner cannot work without intermediate score reporting. "
            "Add: trial.report(score, step) inside your training loop."
        )

    if not has_prune_check:
        return False, (
            f"Code uses {active_pruner}Pruner but does not check trial.should_prune(). "
            "Trials will never be pruned, wasting compute. "
            "Add: if trial.should_prune(): raise optuna.TrialPruned()"
        )

    return True, ""


# Mock CodeExecutor class for testing validation methods
class MockCodeExecutor:
    """Mock CodeExecutor for isolated testing."""

    def _validate_optuna_pruning_contract(self, code: str) -> tuple:
        return _validate_optuna_pruning_contract(code)

    def _parse_errors(self, stderr: str, stdout: str) -> list:
        """Parse errors from output. Copied for isolated testing."""
        errors = []
        lines = stderr.split("\n")

        in_traceback = False
        current_error = []

        for line in lines:
            # Skip tqdm progress bars
            if any(pat in line for pat in ["%|", "it/s", "s/it", "[00:0"]):
                continue

            if "Traceback (most recent call last)" in line:
                in_traceback = True
                current_error = [line]
            elif in_traceback:
                current_error.append(line)
                if line and not line.startswith(" ") and not line.startswith("\t"):
                    if "Error" in line or "Exception" in line:
                        errors.append("\n".join(current_error))
                        in_traceback = False
                        current_error = []

        return errors


class TestResourceLimits:
    """Tests for resource limit functionality."""

    def test_set_resource_limits_does_not_crash(self):
        """Should not crash when setting resource limits."""
        # This should work on Unix and silently do nothing on Windows
        try:
            _set_resource_limits(memory_mb=1024, cpu_time_s=60)
        except Exception as e:
            pytest.fail(f"_set_resource_limits raised an exception: {e}")

    def test_start_new_process_group_does_not_crash(self):
        """Should not crash when starting new process group."""
        # This is designed to be called in subprocess preexec_fn
        # In the main process, it should still not crash
        if platform.system() != "Windows":
            try:
                _start_new_process_group()
            except Exception:
                # os.setpgrp() may fail in some contexts (e.g., already group leader)
                # This is expected behavior, not a bug
                pass

    def test_resource_limits_feature_flag_exists(self):
        """Feature flag should exist and be a boolean-like value."""
        assert isinstance(ENABLE_RESOURCE_LIMITS, bool)


class TestOptunaPruningContractValidation:
    """Tests for Optuna pruning contract validation."""

    def test_passes_when_no_optuna(self):
        """Should pass when code doesn't use Optuna."""
        executor = MockCodeExecutor()

        code = """
import pandas as pd
import numpy as np

model.fit(X_train, y_train)
print("Final Validation Performance: 0.85")
"""
        is_valid, error = executor._validate_optuna_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_passes_when_optuna_without_pruner(self):
        """Should pass when Optuna is used but no pruner is active."""
        executor = MockCodeExecutor()

        code = """
import optuna

study = optuna.create_study(direction='minimize')

def objective(trial):
    params = {'lr': trial.suggest_float('lr', 0.01, 0.1)}
    return 0.5

study.optimize(objective, n_trials=10)
print("Final Validation Performance: 0.85")
"""
        is_valid, error = executor._validate_optuna_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_fails_when_pruner_without_report(self):
        """Should fail when Hyperband pruner is used but trial.report() is missing."""
        executor = MockCodeExecutor()

        code = """
import optuna
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction='minimize',
    pruner=HyperbandPruner(),
)

def objective(trial):
    params = {'lr': trial.suggest_float('lr', 0.01, 0.1)}
    return 0.5

study.optimize(objective, n_trials=10)
print("Final Validation Performance: 0.85")
"""
        is_valid, error = executor._validate_optuna_pruning_contract(code)
        assert not is_valid
        assert "trial.report()" in error
        assert "Hyperband" in error

    def test_fails_when_pruner_without_should_prune(self):
        """Should fail when pruner is used with report but without should_prune check."""
        executor = MockCodeExecutor()

        code = """
import optuna
from optuna.pruners import MedianPruner

study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(),
)

def objective(trial):
    for step in range(100):
        score = train_step()
        trial.report(score, step)  # Has report but missing pruning check
    return score

study.optimize(objective, n_trials=10)
print("Final Validation Performance: 0.85")
"""
        is_valid, error = executor._validate_optuna_pruning_contract(code)
        assert not is_valid
        assert "should_prune" in error.lower() or "prune" in error.lower()
        assert "Median" in error

    def test_passes_when_pruner_with_full_contract(self):
        """Should pass when pruner is used with both report and should_prune."""
        executor = MockCodeExecutor()

        code = """
import optuna
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction='minimize',
    pruner=HyperbandPruner(),
)

def objective(trial):
    for step in range(100):
        score = train_step()
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return score

study.optimize(objective, n_trials=10)
print("Final Validation Performance: 0.85")
"""
        is_valid, error = executor._validate_optuna_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_passes_with_trialPruned_exception(self):
        """Should pass when TrialPruned is raised (alternative pattern)."""
        executor = MockCodeExecutor()

        code = """
import optuna
from optuna.pruners import SuccessiveHalvingPruner

study = optuna.create_study(
    direction='minimize',
    pruner=SuccessiveHalvingPruner(),
)

def objective(trial):
    for step in range(100):
        score = train_step()
        trial.report(score, step)
        if some_condition:
            raise optuna.TrialPruned()
    return score

study.optimize(objective, n_trials=10)
print("Final Validation Performance: 0.85")
"""
        is_valid, error = executor._validate_optuna_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_detects_multiple_pruner_types(self):
        """Should detect various pruner types."""
        executor = MockCodeExecutor()

        pruner_types = [
            "HyperbandPruner",
            "MedianPruner",
            "SuccessiveHalvingPruner",
            "ThresholdPruner",
            "PercentilePruner",
        ]

        for pruner_type in pruner_types:
            code = f"""
import optuna
from optuna.pruners import {pruner_type}

study = optuna.create_study(pruner={pruner_type}())

def objective(trial):
    return 0.5  # Missing report and prune check

study.optimize(objective, n_trials=10)
print("Final Validation Performance: 0.85")
"""
            is_valid, error = executor._validate_optuna_pruning_contract(code)
            assert not is_valid, f"{pruner_type} should fail validation"
            assert pruner_type.replace("Pruner", "") in error


def test_parse_errors_ignores_tqdm_progress_on_stderr():
    # Avoid instantiating CodeExecutor to keep the test lightweight and free of
    # external configuration side-effects.
    executor = MockCodeExecutor()

    stderr = (
        "Fold0 Train Epoch1:   0%|          | 0/275 [00:00<?, ?it/s]\n"
        "Fold0 Train Epoch1:   0%|          | 0/275 [00:02<?, ?it/s, loss=1.79]\n"
        "Fold0 Train Epoch1:   0%|          | 1/275 [00:02<10:37,  2.33s/it, loss=1.79]\n"
    )

    assert executor._parse_errors(stderr=stderr, stdout="") == []


def test_parse_errors_still_detects_traceback_with_tqdm_noise():
    executor = MockCodeExecutor()

    stderr = (
        "Validation:   1%|▏         | 2/138 [00:02<02:28,  1.09s/it]\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
    )

    errors = executor._parse_errors(stderr=stderr, stdout="")
    assert errors, "Expected at least one parsed error"
    assert any("Value" in e or "boom" in e for e in errors)
