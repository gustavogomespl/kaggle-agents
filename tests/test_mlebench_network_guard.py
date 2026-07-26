"""Defense-in-depth tests for target-blind MLE-bench network execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_agents.tools.code_executor import CodeExecutor
from kaggle_agents.tools.code_executor.filesystem_guard import (
    validate_mlebench_filesystem_access,
)
from kaggle_agents.tools.code_executor.process import build_subprocess_env


@pytest.mark.parametrize(
    "source",
    [
        'import requests\nrequests.get("https://www.kaggle.com/c/example")',
        "from kaggle.api.kaggle_api_extended import KaggleApi",
        'import subprocess\nsubprocess.run("kaggle competitions download -c example", shell=True)',
    ],
)
def test_preflight_rejects_explicit_kaggle_retrieval(source: str) -> None:
    valid, message = validate_mlebench_filesystem_access(source)

    assert valid is False
    assert "SearchAgent-only" in message


@pytest.mark.parametrize(
    "source",
    [
        'import requests\nrequests.get("https://huggingface.co/model")',
        'import requests\nrequests.get("https://pypi.org/simple")',
        'print("Kaggle Agents fixed-budget evaluation")',
    ],
)
def test_preflight_does_not_block_registries_or_plain_prose(source: str) -> None:
    assert validate_mlebench_filesystem_access(source) == (True, "")


@pytest.mark.parametrize(
    "network_code",
    [
        """
import numpy as np
import socket
host = ".".join(("www", "kag" + "gle", "com"))
socket.getaddrinfo(host, 443)
""",
        """
import numpy as np
import requests
host = ".".join(("www", "kag" + "gle", "com"))
requests.get("https://" + host + "/", timeout=1)
""",
    ],
    ids=["socket", "requests"],
)
def test_runtime_guard_blocks_dynamic_kaggle_hostname(
    tmp_path: Path,
    network_code: str,
) -> None:
    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        network_code + '\nprint("Final Validation Performance: 1.0")\n',
        str(tmp_path),
        component_type="model",
    )

    assert result.success is False
    assert "Kaggle network access blocked" in result.stderr
    assert "SearchAgent-only" in result.stderr


def test_runtime_guard_blocks_kaggle_url_hidden_in_shell_command(
    tmp_path: Path,
) -> None:
    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        """
import numpy as np
import subprocess

host = ".".join(("www", "kag" + "gle", "com"))
command = "curl -fsS https://" + host + "/"
subprocess.run(["sh", "-c", command], check=True)
print("Final Validation Performance: 1.0")
""",
        str(tmp_path),
        component_type="model",
    )

    assert result.success is False
    assert "Kaggle retrieval blocked in child command" in result.stderr
    assert "SearchAgent-only" in result.stderr


def test_runtime_guard_allows_model_registry_command(tmp_path: Path) -> None:
    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        """
import numpy as np
import subprocess

subprocess.run(
    ["sh", "-c", "printf registry-ok https://huggingface.co/model"],
    check=True,
)
print("Final Validation Performance: 1.0")
""",
        str(tmp_path),
        component_type="model",
    )

    assert result.success is True
    assert "registry-ok" in result.stdout


def test_search_cache_env_is_hidden_even_with_trusted_secret_override() -> None:
    env = build_subprocess_env(
        {
            "KAGGLE_AGENTS_RUN_MODE": "mlebench",
            "KAGGLE_AGENTS_ALLOW_GENERATED_CODE_SECRETS": "true",
            "KAGGLE_AGENTS_SEARCH_CACHE_DIR": "/host/search-cache",
        }
    )

    assert "KAGGLE_AGENTS_SEARCH_CACHE_DIR" not in env


@pytest.mark.parametrize("through_symlink", [False, True])
def test_runtime_guard_blocks_search_notebook_cache_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    through_symlink: bool,
) -> None:
    search_cache = tmp_path / "host-search-cache"
    search_cache.mkdir()
    (search_cache / "retrieved.py").write_text(
        "SEARCH_CACHE_SECRET",
        encoding="utf-8",
    )
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    monkeypatch.setenv("KAGGLE_AGENTS_SEARCH_CACHE_DIR", str(search_cache))

    if through_symlink:
        (work_dir / "notebook-cache-link").symlink_to(
            search_cache,
            target_is_directory=True,
        )
        candidate_path = 'Path("notebook-cache-link") / "retrieved.py"'
    else:
        candidate_path = f"Path(*{list(search_cache.parts)!r}) / 'retrieved.py'"

    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        f"""
import numpy as np
from pathlib import Path

print(({candidate_path}).read_text())
print("Final Validation Performance: 1.0")
""",
        str(work_dir),
        component_type="model",
    )

    assert result.success is False
    assert "SearchAgent notebook cache access blocked" in result.stderr
    assert "SEARCH_CACHE_SECRET" not in result.stdout


def test_runtime_guard_blocks_default_cwd_search_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    search_cache = tmp_path / ".cache" / "notebooks"
    search_cache.mkdir(parents=True)
    (search_cache / "retrieved.py").write_text(
        "DEFAULT_CACHE_SECRET",
        encoding="utf-8",
    )
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()

    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        f"""
import numpy as np
from pathlib import Path

path = Path(*{list(search_cache.parts)!r}) / "retrieved.py"
print(path.read_text())
print("Final Validation Performance: 1.0")
""",
        str(work_dir),
        component_type="model",
    )

    assert result.success is False
    assert "SearchAgent notebook cache access blocked" in result.stderr
    assert "DEFAULT_CACHE_SECRET" not in result.stdout
