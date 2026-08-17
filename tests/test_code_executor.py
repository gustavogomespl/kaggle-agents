"""Tests for the code executor error parsing, resource limits, and HPO validation."""

import json
import os
import platform

import pytest

from kaggle_agents.tools.code_executor import CodeExecutor
from kaggle_agents.tools.code_executor.process import build_subprocess_env


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


class TestGpuAwareResourceLimits:
    """RLIMIT_AS must be skipped on GPU hosts: CUDA maps device/pinned memory
    into the process address space, so the cap fires as fake CUDA OOM and
    pthread_create failures with plenty of free VRAM."""

    def _record_calls(self, monkeypatch):
        import resource

        from kaggle_agents.tools.code_executor import process

        calls = []
        monkeypatch.setattr(process, "ENABLE_RESOURCE_LIMITS", True)
        monkeypatch.setattr(resource, "setrlimit", lambda limit, value: calls.append(limit))
        return process, resource, calls

    @pytest.mark.skipif(platform.system() == "Windows", reason="rlimits are Unix-only")
    def test_rlimit_as_skipped_on_gpu_host(self, monkeypatch):
        process, resource, calls = self._record_calls(monkeypatch)
        monkeypatch.setattr(process, "_nvidia_gpu_present", lambda: True)

        process.set_resource_limits()

        assert resource.RLIMIT_AS not in calls
        assert resource.RLIMIT_CPU in calls

    @pytest.mark.skipif(platform.system() == "Windows", reason="rlimits are Unix-only")
    def test_rlimit_as_applied_without_gpu(self, monkeypatch):
        process, resource, calls = self._record_calls(monkeypatch)
        monkeypatch.setattr(process, "_nvidia_gpu_present", lambda: False)

        process.set_resource_limits()

        assert resource.RLIMIT_AS in calls


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


class TestGeneratedCodeEnvironment:
    def test_secrets_are_removed_but_runtime_config_is_preserved(self):
        env = build_subprocess_env(
            {
                "OPENAI_API_KEY": "secret",
                "KAGGLE_USERNAME": "user",
                "CUSTOM_ACCESS_TOKEN": "token",
                "DATABASE_URL": "postgres://secret",
                "AZURE_OPENAI_KEY": "secret",
                "KAGGLE_CONFIG_DIR": "/real/home/.kaggle",
                "KAGGLE_AGENTS_CV_FOLDS": "5",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            home_dir="/tmp/generated-home",
        )

        assert "OPENAI_API_KEY" not in env
        assert "KAGGLE_USERNAME" not in env
        assert "CUSTOM_ACCESS_TOKEN" not in env
        assert "DATABASE_URL" not in env
        assert "AZURE_OPENAI_KEY" not in env
        assert "KAGGLE_CONFIG_DIR" not in env
        assert env["KAGGLE_AGENTS_CV_FOLDS"] == "5"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["HOME"] == "/tmp/generated-home"
        assert env["XDG_CONFIG_HOME"] == "/tmp/generated-home/.config"

    def test_explicit_trusted_override_preserves_environment(self):
        source = {
            "KAGGLE_AGENTS_ALLOW_GENERATED_CODE_SECRETS": "true",
            "OPENAI_API_KEY": "secret",
        }
        assert build_subprocess_env(source) == source

    def test_mlebench_cache_location_is_removed_even_with_trusted_override(self):
        env = build_subprocess_env(
            {
                "KAGGLE_AGENTS_RUN_MODE": "mlebench",
                "KAGGLE_AGENTS_ALLOW_GENERATED_CODE_SECRETS": "true",
                "MLEBENCH_DATA_DIR": "/grader/cache",
                "OPENAI_API_KEY": "explicitly-trusted",
            }
        )

        assert "MLEBENCH_DATA_DIR" not in env
        assert env["OPENAI_API_KEY"] == "explicitly-trusted"

    def test_executor_uses_isolated_home_and_scrubbed_environment(
        self, temp_data_dir, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
        code = """
import json
import os
import numpy as np
from pathlib import Path

Path("env.json").write_text(json.dumps({
    "home": os.environ.get("HOME"),
    "openai": os.environ.get("OPENAI_API_KEY"),
}))
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10).execute(
            code,
            str(temp_data_dir),
            expected_artifacts=["env.json"],
            component_type="model",
        )

        assert result.success, result.stderr
        payload = json.loads((temp_data_dir / "env.json").read_text())
        assert payload["home"] == str(temp_data_dir / ".agent_home")
        assert payload["openai"] is None

    def test_mlebench_rejects_explicit_private_cache_source(
        self, temp_data_dir, monkeypatch
    ):
        monkeypatch.setenv("KAGGLE_AGENTS_RUN_MODE", "mlebench")
        code = """
import numpy as np
from pathlib import Path

labels = Path("/root/.cache/mle-bench/data/task/prepared/private/labels.csv")
print(labels.read_text())
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10).execute(
            code,
            str(temp_data_dir),
            component_type="model",
        )

        assert result.success is False
        assert "grader-only" in result.stderr

    def test_mlebench_runtime_guard_blocks_dynamic_private_path(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "benchmark-cache"
        private_dir = cache_root / "task" / "prepared" / "private"
        private_dir.mkdir(parents=True)
        (private_dir / "labels.csv").write_text("id,label\n1,secret\n")
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        monkeypatch.setenv("KAGGLE_AGENTS_RUN_MODE", "mlebench")
        monkeypatch.setenv("MLEBENCH_DATA_DIR", str(cache_root))
        code = f"""
import numpy as np
from pathlib import Path

parts = {list(private_dir.parts)!r}
private_path = Path(*parts) / "labels.csv"
print(private_path.read_text())
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10).execute(
            code,
            str(work_dir),
            component_type="model",
        )

        assert result.success is False
        assert "private data access blocked" in result.stderr

    def test_mlebench_runtime_guard_allows_staged_public_files(
        self, tmp_path, monkeypatch
    ):
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        (work_dir / "train.csv").write_text("x,y\n1,0\n")
        monkeypatch.setenv("KAGGLE_AGENTS_RUN_MODE", "mlebench")
        code = """
import numpy as np
from pathlib import Path

assert "x,y" in Path("train.csv").read_text()
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10).execute(
            code,
            str(work_dir),
            component_type="model",
        )

        assert result.success, result.stderr

    def test_mlebench_runtime_guard_blocks_private_path_in_shell_command(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "benchmark-cache"
        private_dir = cache_root / "task" / "prepared" / "private"
        private_dir.mkdir(parents=True)
        (private_dir / "labels.csv").write_text("secret\n")
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        monkeypatch.setenv("KAGGLE_AGENTS_RUN_MODE", "mlebench")
        monkeypatch.setenv("MLEBENCH_DATA_DIR", str(cache_root))
        code = f"""
import numpy as np
import os
import numpy as np
from pathlib import Path

private_path = Path(*{list(private_dir.parts)!r}) / "labels.csv"
os.system("head -n 1 " + str(private_path))
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10).execute(
            code,
            str(work_dir),
            component_type="model",
        )

        assert result.success is False
        assert "private data access blocked" in result.stderr

    def test_mlebench_runtime_guard_checks_child_environment_paths(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "benchmark-cache"
        private_dir = cache_root / "task" / "prepared" / "private"
        private_dir.mkdir(parents=True)
        (private_dir / "labels.csv").write_text("PRIVATE_LABEL\n")
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        monkeypatch.setenv("KAGGLE_AGENTS_RUN_MODE", "mlebench")
        monkeypatch.setenv("MLEBENCH_DATA_DIR", str(cache_root))
        code = f"""
import numpy as np
import os
import subprocess
from pathlib import Path

private_path = Path(*{list(private_dir.parts)!r}) / "labels.csv"
child_env = dict(os.environ)
child_env["LABEL_PATH"] = str(private_path)
subprocess.run(
    ["sh", "-c", 'cat "$LABEL_PATH"'],
    env=child_env,
    check=True,
)
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10).execute(
            code,
            str(work_dir),
            component_type="model",
        )

        assert result.success is False
        assert "private data access blocked" in result.stderr
        assert "PRIVATE_LABEL" not in result.stdout

    def test_explicit_executor_mode_enables_guard_without_parent_env(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "benchmark-cache"
        private_dir = cache_root / "task" / "prepared" / "private"
        private_dir.mkdir(parents=True)
        (private_dir / "labels.csv").write_text("secret\n")
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        monkeypatch.delenv("KAGGLE_AGENTS_RUN_MODE", raising=False)
        monkeypatch.setenv("MLEBENCH_DATA_DIR", str(cache_root))
        code = f"""
import numpy as np
from pathlib import Path

private_path = Path(*{list(private_dir.parts)!r}) / "labels.csv"
print(private_path.read_text())
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
            code,
            str(work_dir),
            component_type="model",
        )

        assert result.success is False
        assert "private data access blocked" in result.stderr

    def test_explicit_custom_mlebench_cache_is_host_only_and_blocked(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "nonstandard-benchmark-cache"
        private_dir = cache_root / "task"
        private_dir.mkdir(parents=True)
        (private_dir / "secret-labels.csv").write_text("secret\n")
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        monkeypatch.delenv("KAGGLE_AGENTS_RUN_MODE", raising=False)
        monkeypatch.delenv("MLEBENCH_DATA_DIR", raising=False)
        code = f"""
import os
import numpy as np
from pathlib import Path

cache_env_name = "".join(["MLEBENCH", "_DATA_DIR"])
assert os.getenv(cache_env_name) is None
private_path = Path(*{list(private_dir.parts)!r}) / "secret-labels.csv"
print(private_path.read_text())
print("Final Validation Performance: 1.0")
"""

        result = CodeExecutor(
            timeout=10,
            run_mode="mlebench",
            mlebench_cache_path=str(cache_root),
        ).execute(code, str(work_dir), component_type="model")

        assert result.success is False
        assert "private data access blocked" in result.stderr
        assert "secret" not in result.stdout


class TestExpectedArtifactFreshness:
    def test_stale_expected_artifact_cannot_satisfy_current_execution(
        self, tmp_path
    ):
        artifact = tmp_path / "models" / "oof_candidate.npy"
        artifact.parent.mkdir()
        artifact.write_bytes(b"accepted-before")

        result = CodeExecutor(timeout=10).execute(
            'import numpy as np\nprint("Final Validation Performance: 1.0")',
            str(tmp_path),
            expected_artifacts=["models/oof_candidate.npy"],
            component_type="model",
        )

        assert result.success is False
        assert "Missing expected artifacts" in result.errors[-1]
        assert artifact.read_bytes() == b"accepted-before"

    def test_successful_execution_replaces_expected_artifact(self, tmp_path):
        artifact = tmp_path / "models" / "oof_candidate.npy"
        artifact.parent.mkdir()
        artifact.write_bytes(b"accepted-before")

        result = CodeExecutor(timeout=10).execute(
            """
import numpy as np
from pathlib import Path
Path("models/oof_candidate.npy").write_bytes(b"fresh")
print("Final Validation Performance: 1.0")
""",
            str(tmp_path),
            expected_artifacts=["models/oof_candidate.npy"],
            component_type="model",
        )

        assert result.success is True
        assert artifact.read_bytes() == b"fresh"

    def test_failed_execution_restores_previous_expected_artifact(self, tmp_path):
        artifact = tmp_path / "models" / "oof_candidate.npy"
        artifact.parent.mkdir()
        artifact.write_bytes(b"accepted-before")

        result = CodeExecutor(timeout=10).execute(
            """
import numpy as np
from pathlib import Path
Path("models/oof_candidate.npy").write_bytes(b"partial")
raise RuntimeError("training failed")
""",
            str(tmp_path),
            expected_artifacts=["models/oof_candidate.npy"],
            component_type="model",
        )

        assert result.success is False
        assert artifact.read_bytes() == b"accepted-before"


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


class TestStrictPerformanceMetricExtraction:
    """The submission gate only trusts the exact score marker line."""

    def test_exact_marker_extracted(self):
        executor = CodeExecutor(timeout=5)
        stdout = "training done\nFinal Validation Performance: 0.0152\n"
        assert executor.extract_performance_metric(stdout) == 0.0152

    def test_decorated_marker_rejected(self):
        # A generated ensemble once printed this mocked variant; the lenient
        # regex promoted it into submission_best.csv. The strict extractor
        # must return None so the gate ignores it.
        executor = CodeExecutor(timeout=5)
        stdout = "Final Validation Performance (rmse): 0.015200\n"
        assert executor.extract_performance_metric(stdout) is None

    def test_prefixed_marker_still_extracted(self):
        executor = CodeExecutor(timeout=5)
        stdout = "🎯 Final Validation Performance: 0.000000\n"
        assert executor.extract_performance_metric(stdout) == 0.0
