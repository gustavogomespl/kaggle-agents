"""Regression tests for the host-owned canonical evaluation contract."""

from __future__ import annotations

import json
import platform
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.tools.code_executor import CodeExecutor
from kaggle_agents.tools.code_executor import executor as executor_module
from kaggle_agents.tools.code_executor.canonical_integrity import (
    snapshot_canonical_contract,
    verify_and_restore_canonical_contract,
)
from kaggle_agents.utils.data_contract import prepare_canonical_data


def _write_canonical_contract(workspace: Path) -> dict[str, bytes]:
    canonical = workspace / "canonical"
    canonical.mkdir()
    np.save(canonical / "train_ids.npy", np.asarray(["a", "b"]))
    np.save(canonical / "y.npy", np.asarray([0.0, 1.0]))
    np.save(canonical / "folds.npy", np.asarray([0, 1]))
    (canonical / "feature_cols.json").write_text(
        json.dumps(["feature"]),
        encoding="utf-8",
    )
    (canonical / "metadata.json").write_text(
        json.dumps(
            {
                "n_folds": 2,
                "id_col": "id",
                "target_col": "target",
                "is_classification": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        path.name: path.read_bytes()
        for path in canonical.iterdir()
        if path.is_file()
    }


def test_failed_preparation_does_not_poison_the_execution_gate(tmp_path):
    """A canonical prep that dies mid-way must not leave a partial canonical/.

    The executor's integrity gate treats an existing canonical/ as a declared
    contract and fails closed when it is incomplete. A preparation crash that
    leaves the directory behind (observed when the public training data is a
    directory of JSON files, so pd.read_csv raises right after mkdir) therefore
    poisoned every later generated-code execution: the whole run produced zero
    components. Failure must leave the workspace exactly as canonical-less.
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    with pytest.raises((OSError, ValueError)):
        prepare_canonical_data(
            train_path=train_dir,
            test_path=train_dir,
            target_col="target",
            output_dir=tmp_path,
        )

    assert not (tmp_path / "canonical").exists()
    # The exact consequence that matters: the execution gate opens again.
    assert snapshot_canonical_contract(tmp_path) is None


def test_failed_re_preparation_keeps_a_complete_preexisting_contract(tmp_path):
    """Cleanup covers only the partial directory this preparation created.

    When a complete contract already exists, a failed re-preparation must not
    delete it: the artifacts on disk are still the ones every earlier
    component was validated against.
    """
    _write_canonical_contract(tmp_path)
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    with pytest.raises((OSError, ValueError)):
        prepare_canonical_data(
            train_path=train_dir,
            test_path=train_dir,
            target_col="target",
            output_dir=tmp_path,
        )

    assert (tmp_path / "canonical" / "metadata.json").is_file()


def test_snapshot_detects_mutation_and_restores_exact_contract(
    tmp_path: Path,
) -> None:
    expected = _write_canonical_contract(tmp_path)
    canonical = tmp_path / "canonical"
    original_mode = stat.S_IMODE((canonical / "metadata.json").stat().st_mode)
    snapshot = snapshot_canonical_contract(tmp_path)
    assert snapshot is not None

    np.save(canonical / "y.npy", np.asarray([1.0, 1.0]))
    (canonical / "metadata.json").chmod(0o600)
    (canonical / "candidate_owned.txt").write_text("not canonical")

    changes = verify_and_restore_canonical_contract(snapshot)

    assert any("modified=y.npy" in change for change in changes)
    assert any("added=candidate_owned.txt" in change for change in changes)
    assert "permissions_changed" in changes
    assert not (canonical / "candidate_owned.txt").exists()
    assert {
        path.name: path.read_bytes()
        for path in canonical.iterdir()
        if path.is_file()
    } == expected
    assert (
        stat.S_IMODE((canonical / "metadata.json").stat().st_mode)
        == original_mode
    )


def test_snapshot_accepts_complete_packed_image_contract_without_y_npy(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    np.save(canonical / "train_ids.npy", np.asarray(["a"], dtype=str))
    np.save(canonical / "test_ids.npy", np.asarray(["b"], dtype=str))
    np.save(canonical / "folds.npy", np.asarray([0], dtype=np.int64))
    np.save(
        canonical / "image_input_paths.npy",
        np.asarray(["train/a.png"], dtype=str),
    )
    np.save(
        canonical / "image_test_input_paths.npy",
        np.asarray(["test/b.png"], dtype=str),
    )
    np.savez(
        canonical / "image_targets.npz",
        values=np.asarray([0.5], dtype=np.float32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        shapes=np.asarray([[1, 1]], dtype=np.int32),
        image_ids=np.asarray(["a"], dtype=str),
    )
    (canonical / "feature_cols.json").write_text(
        json.dumps(["image_pixels"]),
        encoding="utf-8",
    )
    (canonical / "metadata.json").write_text(
        json.dumps(
            {
                "task_type": "image_to_image",
                "packed_image_contract": True,
            }
        ),
        encoding="utf-8",
    )

    snapshot = snapshot_canonical_contract(tmp_path)

    assert snapshot is not None
    assert "image_targets.npz" in snapshot.manifest
    assert "y.npy" not in snapshot.manifest


def test_mlebench_executor_hash_guard_rejects_and_restores_np_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The hash verifier remains authoritative if the audit hook is bypassed."""
    expected = _write_canonical_contract(tmp_path)

    # Exercise the host-side verifier directly, independently of the child
    # audit hook's defense-in-depth write denial.
    monkeypatch.setattr(
        executor_module,
        "install_mlebench_runtime_guard",
        lambda *_args, **_kwargs: None,
    )
    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        """
from pathlib import Path
import numpy as np

np.save(Path("canonical") / "y.npy", np.asarray([1.0, 1.0]))
print("Final Validation Performance: 1.0")
""",
        str(tmp_path),
        component_type="model",
    )

    assert result.success is False
    assert "Canonical contract integrity violation" in result.stderr
    assert (tmp_path / "canonical" / "y.npy").read_bytes() == expected["y.npy"]


def test_integrity_snapshot_has_no_same_uid_temporary_backup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A candidate cannot discover and corrupt a named restore source."""
    expected = _write_canonical_contract(tmp_path)
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_root))
    monkeypatch.setattr(
        executor_module,
        "install_mlebench_runtime_guard",
        lambda *_args, **_kwargs: None,
    )

    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        """
from pathlib import Path
import numpy as np

for candidate in Path("tmp").glob("kaggle-agents-canonical-integrity-*"):
    for path in candidate.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
            path.write_bytes(b"corrupted")
np.save(Path("canonical") / "y.npy", np.asarray([1.0, 1.0]))
""",
        str(tmp_path),
        component_type="model",
    )

    assert result.success is False
    assert list(temp_root.glob("kaggle-agents-canonical-integrity-*")) == []
    assert (tmp_path / "canonical" / "y.npy").read_bytes() == expected["y.npy"]


@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX process groups")
def test_executor_kills_background_descendants_before_integrity_verification(
    tmp_path: Path,
) -> None:
    expected = _write_canonical_contract(tmp_path)

    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        """
import subprocess
import sys
import numpy as np

child_code = '''
import time
from pathlib import Path
time.sleep(0.5)
Path("canonical/y.npy").write_bytes(b"late-corruption")
Path("background-survived.txt").write_text("survived")
'''
subprocess.Popen(
    [sys.executable, "-S", "-c", child_code],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print("Final Validation Performance: 1.0")
""",
        str(tmp_path),
        component_type="model",
    )

    time.sleep(0.8)
    assert result.success is True, result.stderr
    assert not (tmp_path / "background-survived.txt").exists()
    assert (tmp_path / "canonical" / "y.npy").read_bytes() == expected["y.npy"]


def test_mlebench_runtime_guard_denies_direct_canonical_write(
    tmp_path: Path,
) -> None:
    expected = _write_canonical_contract(tmp_path)

    result = CodeExecutor(timeout=10, run_mode="mlebench").execute(
        """
from pathlib import Path
import numpy as np

np.save(Path("canonical") / "y.npy", np.asarray([1.0, 1.0]))
print("Final Validation Performance: 1.0")
""",
        str(tmp_path),
        component_type="model",
    )

    assert result.success is False
    assert "canonical contract is host-owned and read-only" in result.stderr
    assert (tmp_path / "canonical" / "y.npy").read_bytes() == expected["y.npy"]


def test_regular_kaggle_executor_behavior_is_unchanged(tmp_path: Path) -> None:
    _write_canonical_contract(tmp_path)

    result = CodeExecutor(timeout=10).execute(
        """
from pathlib import Path
import numpy as np

np.save(Path("canonical") / "y.npy", np.asarray([1.0, 1.0]))
print("Final Validation Performance: 1.0")
""",
        str(tmp_path),
        component_type="model",
    )

    assert result.success is True
    assert np.load(tmp_path / "canonical" / "y.npy").tolist() == [1.0, 1.0]
