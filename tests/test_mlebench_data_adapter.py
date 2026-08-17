from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kaggle_agents.mlebench.data_adapter import (
    MLEBenchDataAdapter,
    one_artifact_path,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_json_table_competition_is_materialized_as_csv(tmp_path: Path) -> None:
    """A competition whose tables ship as JSON gets workspace CSV copies.

    Shape: ``train/train.json`` + ``test/test.json`` and no CSV anywhere.
    Nothing downstream reads JSON — canonical prep and every generated
    component assume CSV tables — so such runs executed zero components:
    the prep failed (canonical-less lane), the injected header lost its
    canonical block while the prompts still mandated CANONICAL_* names, and
    the raw directory defeated every ``pd.read_csv(TRAIN_PATH)``. The
    workspace stage materializes the CSVs; the public tree is never touched.
    """
    comp_id = "fake-json-competition"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / comp_id / "prepared" / "public"
    train_records = [
        {
            "request_id": f"t_{i}",
            "request_text": f"body {i}",
            "subreddits": ["a", "b"],
            "target": i % 2,
        }
        for i in range(6)
    ]
    test_records = [
        {"request_id": f"u_{i}", "request_text": f"body {i}", "subreddits": ["a"]}
        for i in range(3)
    ]
    (public_dir / "train").mkdir(parents=True)
    (public_dir / "test").mkdir(parents=True)
    (public_dir / "train" / "train.json").write_text(
        json.dumps(train_records), encoding="utf-8"
    )
    (public_dir / "test" / "test.json").write_text(
        json.dumps(test_records), encoding="utf-8"
    )
    _write_text(
        public_dir / "sample_submission.csv",
        "request_id,target\nu_0,0\nu_1,0\nu_2,0\n",
    )

    workspace = tmp_path / "workspace" / comp_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(
        competition_id=comp_id, workspace_path=workspace
    )

    assert (workspace / "train.csv").is_file()
    assert (workspace / "test.csv").is_file()
    train_frame = pd.read_csv(workspace / "train.csv")
    assert len(train_frame) == 6
    assert "target" in train_frame.columns
    assert len(pd.read_csv(workspace / "test.csv")) == 3

    # The tabular lane consumes the materialized tables.
    paths = adapter.get_state_paths(info)
    assert Path(paths["train_data_path"]).name == "train.csv"
    assert Path(paths["test_data_path"]).name == "test.csv"

    # JSON tables are outside the bounded role resolver's delimited scope, so
    # it must leave both roles unresolved -- and must not clear the legacy
    # materialization fallback that is the only thing filling them here.
    assert one_artifact_path(info.public_artifacts, "train") is None
    assert one_artifact_path(info.public_artifacts, "test") is None
    assert one_artifact_path(info.public_artifacts, "submission") is not None

    # The public tree is read-only for this pipeline: no CSV appears there.
    assert sorted(p.name for p in (public_dir / "train").iterdir()) == [
        "train.json"
    ]


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

    # Every compatibility key above is derived from the typed records, which
    # travel into state alongside them.
    typed_roles = {
        record["role"]: Path(record["path"]).name
        for record in paths["data_files"]["public_artifacts"]
    }
    assert typed_roles["train"] == "train.csv"
    assert typed_roles["test"] == "test.csv"
    assert typed_roles["submission"] == "sample_submission.csv"


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
