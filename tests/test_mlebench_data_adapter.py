from __future__ import annotations

from pathlib import Path

from kaggle_agents.mlebench.data_adapter import MLEBenchDataAdapter


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_workspace_stages_only_public_data_inside_run(tmp_path: Path) -> None:
    comp_id = "fake-image-competition"
    cache_root = tmp_path / "mle-cache"

    public_dir = cache_root / comp_id / "prepared" / "public"
    (public_dir / "train_images").mkdir(parents=True, exist_ok=True)
    (public_dir / "test_images").mkdir(parents=True, exist_ok=True)

    # Dummy media files (extension-based detection)
    (public_dir / "train_images" / "img_0.jpg").write_bytes(b"fake")
    (public_dir / "test_images" / "img_1.jpg").write_bytes(b"fake")
    private_labels = (
        cache_root / comp_id / "prepared" / "private" / "test.csv"
    )
    _write_text(private_labels, "id,target\nimg_1.jpg,1\n")

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
    assert info.ground_truth_path is None
    assert str(private_labels) not in repr(paths)
    assert Path(paths["train_data_path"]).is_dir()
    assert Path(paths["test_data_path"]).is_dir()
    assert paths["data_files"]["data_type"] == "image"
    for key in ("train_data_path", "test_data_path", "sample_submission_path"):
        resolved = Path(paths[key]).resolve()
        assert resolved.is_relative_to(workspace.resolve())
        assert not resolved.is_relative_to(public_dir.resolve())


def test_prepare_workspace_discovers_audio_in_arbitrarily_named_directories(
    tmp_path: Path,
) -> None:
    comp_id = "synthetic-audio-layout"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / comp_id / "prepared" / "public"
    audio_dir = public_dir / "bundle_x" / "recordings_y"
    audio_dir.mkdir(parents=True)
    for index in range(12):
        (audio_dir / f"clip-{index}.wav").write_bytes(b"audio")

    _write_text(
        public_dir / "bundle_x" / "targets_public.txt",
        "item_id;labels\n0;0\n1;1\n",
    )
    _write_text(
        public_dir / "sample_submission.csv",
        "item_id,prediction\nclip-10,0\nclip-11,0\n",
    )

    workspace = tmp_path / "workspace" / comp_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(comp_id, workspace)
    paths = adapter.get_state_paths(info)

    assert paths["data_files"]["data_type"] == "audio"
    assert Path(paths["data_files"]["audio_source"]).is_dir()
    assert Path(paths["train_data_path"]).is_dir()
    assert any(
        Path(label_path).name == "targets_public.txt"
        for label_path in paths["data_files"]["label_files"]
    )


def test_tabular_state_paths_prefer_csvs_over_supplementary_directories(
    tmp_path: Path,
) -> None:
    comp_id = "synthetic-tabular-with-row-assets"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / comp_id / "prepared" / "public"
    (public_dir / "train").mkdir(parents=True)
    (public_dir / "test").mkdir()
    _write_text(public_dir / "train" / "row-1.json", "{}")
    _write_text(public_dir / "test" / "row-2.json", "{}")
    _write_text(
        public_dir / "train.csv",
        "id,feature,target\n1,0.1,0.2\n2,0.3,0.4\n",
    )
    _write_text(
        public_dir / "test.csv",
        "id,feature\n3,0.5\n4,0.6\n",
    )
    _write_text(
        public_dir / "sample_submission.csv",
        "id,target\n3,0\n4,0\n",
    )

    workspace = tmp_path / "workspace" / comp_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(comp_id, workspace)
    paths = adapter.get_state_paths(info)

    assert paths["data_files"]["data_type"] == "tabular"
    assert Path(paths["train_data_path"]) == workspace / "train.csv"
    assert Path(paths["test_data_path"]) == workspace / "test.csv"
    assert Path(paths["data_files"]["train"]) == workspace / "train.csv"
    assert Path(paths["data_files"]["test"]) == workspace / "test.csv"
    assert (workspace / "train").is_dir()
    assert (workspace / "test").is_dir()


def test_sample_submission_directory_resolves_to_inner_csv(tmp_path: Path) -> None:
    # Kaggle packaging quirk: sample_submission.csv is a DIRECTORY containing
    # the real CSV. The staged alias name collides with the copied directory
    # and must not shadow the resolved file.
    comp_id = "fake-dir-sample-competition"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / comp_id / "prepared" / "public"

    (public_dir / "train_images").mkdir(parents=True, exist_ok=True)
    (public_dir / "train_images" / "img_0.jpg").write_bytes(b"fake")
    (public_dir / "test_images").mkdir(parents=True, exist_ok=True)
    (public_dir / "test_images" / "img_1.jpg").write_bytes(b"fake")
    _write_text(public_dir / "train.csv", "id,target\nimg_0.jpg,1\n")
    _write_text(
        public_dir / "sample_submission.csv" / "sample_submission.csv",
        "id,target\nimg_1.jpg,0\n",
    )

    workspace = tmp_path / "workspace" / comp_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(competition_id=comp_id, workspace_path=workspace)

    assert info.sample_submission_path is not None
    assert Path(info.sample_submission_path).is_file()

    paths = adapter.get_state_paths(info)
    assert Path(paths["sample_submission_path"]).is_file()
    assert Path(paths["sample_submission_path"]).resolve().is_relative_to(
        workspace.resolve()
    )


def test_adapter_preserves_ordered_multioutput_submission_columns(
    tmp_path: Path,
) -> None:
    comp_id = "synthetic-public-multioutput"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / comp_id / "prepared" / "public"
    _write_text(
        public_dir / "train.csv",
        "id,feature,label_b,label_a\n"
        "r0,0.1,1,0\n"
        "r1,0.2,0,1\n",
    )
    _write_text(
        public_dir / "test.csv",
        "id,feature\n"
        "t0,0.3\n"
        "t1,0.4\n",
    )
    # Template values are intentionally all zero and carry no target semantics.
    _write_text(
        public_dir / "sample_submission.csv",
        "id,label_a,label_b\n"
        "t0,0,0\n"
        "t1,0,0\n",
    )

    workspace = tmp_path / "workspace" / comp_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(comp_id, workspace)
    paths = adapter.get_state_paths(info)

    assert info.target_column == "label_a"
    assert info.target_columns == ["label_a", "label_b"]
    assert paths["target_col"] == "label_a"
    assert paths["target_cols"] == ["label_a", "label_b"]
    assert "target_type" not in paths
