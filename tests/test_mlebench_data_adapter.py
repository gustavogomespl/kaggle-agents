from __future__ import annotations

from pathlib import Path

from kaggle_agents.mlebench.data_adapter import MLEBenchDataAdapter


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_workspace_links_train_test_dirs_when_named_train_images(tmp_path: Path) -> None:
    comp_id = "fake-image-competition"
    cache_root = tmp_path / "mle-cache"

    public_dir = cache_root / comp_id / "prepared" / "public"
    (public_dir / "train_images").mkdir(parents=True, exist_ok=True)
    (public_dir / "test_images").mkdir(parents=True, exist_ok=True)

    # Dummy media files (extension-based detection)
    (public_dir / "train_images" / "img_0.jpg").write_bytes(b"fake")
    (public_dir / "test_images" / "img_1.jpg").write_bytes(b"fake")

    _write_text(public_dir / "train.csv", "id,target\nimg_0.jpg,1\n")
    _write_text(public_dir / "sample_submission.csv", "id,target\nimg_1.jpg,0\n")

    workspace = tmp_path / "workspace" / comp_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(competition_id=comp_id, workspace_path=workspace)

    assert (workspace / "train").is_dir()
    assert (workspace / "test").is_dir()
    assert (workspace / "train.csv").is_file()
    assert (workspace / "sample_submission.csv").is_file()

    paths = adapter.get_state_paths(info)
    assert Path(paths["train_data_path"]).is_dir()
    assert Path(paths["test_data_path"]).is_dir()
    assert paths["data_files"]["data_type"] == "image"
