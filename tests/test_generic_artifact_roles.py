"""
RED->GREEN tests for typed public artifacts, bounded ZIP provenance, and
exclusive public-role resolution.

Task 1 of docs/superpowers/plans/2026-08-02-generic-artifact-role-resolution.md:
`_extract_zips()` returns bounded, idempotent `ArchiveProvenance` records, and
`PublicArtifact.to_state()` serializes to a plain, JSON-safe dict.

Task 3: `resolve_public_artifacts()` assigns every public delimited table at
most one exclusive primary role (train/test/submission), everything else is an
inspector-verified auxiliary record, and the adapter stages those typed records
before deriving its legacy compatibility fields.

Task 6: one real miniature competition travels the whole path - prefixed
`*.csv.zip` archives, the adapter, the canonical preparation node, the no-LLM
contract preparation, the rendered preamble and a real subprocess execution -
and every boundary between those layers is asserted on what the production
code produced, never on a value the test wrote down.
"""

from __future__ import annotations

import json
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import kaggle_agents.agents.developer.code_generator as code_generator_module
from kaggle_agents.agents.developer.code_generator import CodeGeneratorMixin
from kaggle_agents.agents.developer.execution_failures import (
    INJECTED_HEADER_END_MARKER,
    execute_generated_candidate,
)
from kaggle_agents.agents.developer.target_source import (
    build_canonical_validation_marker,
    reset_target_source_caches,
)
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    create_initial_state,
)
from kaggle_agents.mlebench.data_adapter import MLEBenchDataAdapter, PublicArtifact
from kaggle_agents.mlebench.data_adapter.artifact_roles import (
    one_artifact_path,
    resolve_public_artifacts,
)
from kaggle_agents.tools.code_executor import CodeExecutor
from kaggle_agents.workflow.nodes.canonical_data import (
    canonical_data_preparation_node,
)


def test_zip_provenance_is_identical_after_archive_is_already_extracted(
    tmp_path: Path,
) -> None:
    """`_extract_zips` returns bounded provenance, unchanged across a rerun.

    The first call performs the real extraction; the second call finds the
    archive already extracted and skips re-extracting it, but must still open
    the central directory and report identical aggregate and per-member
    provenance both times.
    """
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    content = b"id,target\n1,0\n2,1\n"
    zip_path = public_dir / "train.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train.csv", content)

    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")

    first_run = adapter._extract_zips(public_dir)
    second_run = adapter._extract_zips(public_dir)

    assert first_run == second_run
    assert len(first_run) == 1

    provenance = first_run[0]
    assert provenance.archive_path == zip_path
    assert provenance.extraction_root == public_dir / "train"
    assert provenance.member_count == 1
    assert provenance.supported_tabular_member_count == 1
    assert provenance.supported_tabular_root_member_count == 1
    assert provenance.single_supported_tabular_root_member is True

    assert len(provenance.supported_members) == 1
    member = provenance.supported_members[0]
    assert member.member_name == "train.csv"
    assert member.extracted_path == public_dir / "train" / "train.csv"
    assert member.crc == zlib.crc32(content) & 0xFFFFFFFF
    assert member.file_size == len(content)
    assert member.at_archive_root is True

    # The archive really was extracted on the first call, and stays put.
    extracted_file = public_dir / "train" / "train.csv"
    assert extracted_file.is_file()
    assert extracted_file.read_bytes() == content


def test_public_artifact_state_is_json_safe(tmp_path: Path) -> None:
    """`PublicArtifact.to_state()` is a plain, JSON-round-trippable dict.

    Staging both the resolved table path and its source archive with
    `with_staged_paths()` must not leak `Path` objects (or any other
    non-JSON-safe value) into the serialized state.
    """
    workspace = tmp_path / "workspace"
    staged_table = workspace / "train.csv"
    staged_archive = workspace / "_archives" / "train.zip"

    artifact = PublicArtifact(
        path=tmp_path / "public" / "train" / "train.csv",
        role="train",
        layout="rectangular_table",
        source_archive=tmp_path / "public" / "train.zip",
        evidence=("exact_standard_name", "schema_corroborates_role"),
        fingerprint="deadbeef",
    )

    staged = artifact.with_staged_paths(staged_table, staged_archive)

    assert staged.path == staged_table
    assert staged.source_archive == staged_archive
    assert artifact.path != staged.path  # frozen: original is untouched

    state = staged.to_state()

    assert state == {
        "path": str(staged_table),
        "role": "train",
        "layout": "rectangular_table",
        "source_archive": str(staged_archive),
        "evidence": ["exact_standard_name", "schema_corroborates_role"],
        "fingerprint": "deadbeef",
    }

    encoded = json.dumps(state)
    assert json.loads(encoded) == state

    for value in state.values():
        assert isinstance(value, (str, list))
        if isinstance(value, list):
            assert all(isinstance(item, str) for item in value)


# --- Task 3 fixtures ----------------------------------------------------

# A token-level train table: the shape a real prefixed-archive competition
# publishes. Nothing here names a competition, only generic column concepts.
TOKEN_TRAIN_TABLE = (
    "sentence_id,token_id,class,before,after\n"
    "0,0,PLAIN,alpha,alpha\n"
    "0,1,PUNCT,.,.\n"
    "1,0,PLAIN,beta,beta\n"
    "1,1,DATE,2001,two thousand one\n"
)
TOKEN_TEST_TABLE = "sentence_id,token_id,before\n2,0,gamma\n2,1,.\n"
TOKEN_SUBMISSION_TABLE = "id,after\n2_0,gamma\n2_1,.\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _zip_one_member(archive: Path, member: str, content: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, content)


def _public_dir(tmp_path: Path, competition_id: str) -> Path:
    public_dir = tmp_path / "mle-cache" / competition_id / "prepared" / "public"
    public_dir.mkdir(parents=True)
    return public_dir


def _role_count(artifacts: Sequence[PublicArtifact], role: str) -> int:
    return sum(1 for artifact in artifacts if artifact.role == role)


def _artifact_named(
    artifacts: Sequence[PublicArtifact], name: str
) -> PublicArtifact:
    matches = [artifact for artifact in artifacts if artifact.path.name == name]
    assert len(matches) == 1, f"expected exactly one {name} record, got {matches}"
    return matches[0]


def _resolve_loose(public_dir: Path, data_type: str = "tabular"):
    return resolve_public_artifacts(public_dir, [], data_type)


# --- Step 3.1: the prefixed-archive end-to-end regression ----------------


def test_prefixed_zip_members_resolve_primary_roles_end_to_end(
    tmp_path: Path,
) -> None:
    """Prefixed `*.csv.zip` archives are the competition's primary tables.

    Their members never match `train.csv`/`test.csv`/`sample_submission*.csv`,
    so the legacy filename finders miss all three roles and the train table
    is misfiled as a label file. Role resolution has to read the archive and
    member stems instead, and the workspace must expose canonical aliases.
    """
    competition_id = "synthetic-prefixed-archives"
    public_dir = _public_dir(tmp_path, competition_id)
    cache_root = tmp_path / "mle-cache"
    _zip_one_member(
        public_dir / "en_train.csv.zip", "en_train.csv", TOKEN_TRAIN_TABLE
    )
    _zip_one_member(
        public_dir / "en_test_2.csv.zip", "en_test_2.csv", TOKEN_TEST_TABLE
    )
    _zip_one_member(
        public_dir / "en_sample_submission_2.csv.zip",
        "en_sample_submission_2.csv",
        TOKEN_SUBMISSION_TABLE,
    )
    # MLE-bench keeps the graded labels next to public/. Nothing may read it.
    private_test = cache_root / competition_id / "prepared" / "private" / "test.csv"
    _write(private_test, "id,after\n2_0,gamma\n2_1,.\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(competition_id, workspace)
    paths = adapter.get_state_paths(info)

    assert (workspace / "train.csv").is_file()
    assert (workspace / "test.csv").is_file()
    assert (workspace / "sample_submission.csv").is_file()
    assert (workspace / "train.csv").read_text(encoding="utf-8") == TOKEN_TRAIN_TABLE
    assert (workspace / "test.csv").read_text(encoding="utf-8") == TOKEN_TEST_TABLE
    assert (workspace / "sample_submission.csv").read_text(
        encoding="utf-8"
    ) == TOKEN_SUBMISSION_TABLE

    assert Path(paths["train_data_path"]) == workspace / "train.csv"
    assert Path(paths["test_data_path"]) == workspace / "test.csv"
    assert Path(paths["sample_submission_path"]) == workspace / "sample_submission.csv"

    # The train table is a primary role, never a label file.
    assert info.label_files == []
    assert paths["data_files"]["label_files"] == []
    for role in ("train", "test", "submission"):
        assert _role_count(info.public_artifacts, role) == 1

    train_artifact = next(a for a in info.public_artifacts if a.role == "train")
    assert train_artifact.role != "auxiliary"
    assert train_artifact.layout == "rectangular_table"
    assert train_artifact.source_archive is not None
    assert train_artifact.fingerprint

    # Everything serialized points inside the run workspace.
    workspace_root = workspace.resolve()
    for artifact in info.public_artifacts:
        assert artifact.path.resolve().is_relative_to(workspace_root)
        if artifact.source_archive is not None:
            assert artifact.source_archive.resolve().is_relative_to(workspace_root)

    serialized = json.dumps(paths) + json.dumps(
        [artifact.to_state() for artifact in info.public_artifacts]
    )
    assert str(private_test) not in serialized
    assert str(private_test.parent) not in serialized
    assert str(public_dir) not in serialized

    state_artifacts = paths["data_files"]["public_artifacts"]
    assert sorted(record["role"] for record in state_artifacts if record["role"] != "auxiliary") == [
        "submission",
        "test",
        "train",
    ]


def test_archives_copied_by_the_empty_public_fallback_are_typed(
    tmp_path: Path,
) -> None:
    """The fallback copies archives nothing has opened yet.

    Extraction runs before the fallback, so its provenance is empty for every
    archive the fallback introduces. Re-running the (idempotent) extraction
    afterwards is what lets those members reach role resolution at all.
    """
    competition_id = "synthetic-fallback-archives"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / competition_id / "prepared" / "public"
    public_dir.mkdir(parents=True)
    raw_dir = cache_root / competition_id / "raw"
    raw_dir.mkdir(parents=True)
    _zip_one_member(raw_dir / "en_train.csv.zip", "en_train.csv", TOKEN_TRAIN_TABLE)
    _write(raw_dir / "sample_submission.csv", TOKEN_SUBMISSION_TABLE)

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(competition_id, workspace)

    train = one_artifact_path(info.public_artifacts, "train")
    assert train == workspace / "train.csv"
    assert train.read_text(encoding="utf-8") == TOKEN_TRAIN_TABLE
    assert one_artifact_path(info.public_artifacts, "submission") is not None


def test_private_prepared_tables_never_enter_typed_records_or_state(
    tmp_path: Path,
) -> None:
    """`prepared/private/` is the grader's; no record or state key may cite it."""
    competition_id = "synthetic-private-neighbour"
    public_dir = _public_dir(tmp_path, competition_id)
    cache_root = tmp_path / "mle-cache"
    _write(public_dir / "train.csv", "id,feature,target\n1,0.1,0\n2,0.2,1\n")
    _write(public_dir / "test.csv", "id,feature\n3,0.3\n4,0.4\n")
    _write(public_dir / "sample_submission.csv", "id,target\n3,0\n4,0\n")
    private_test = cache_root / competition_id / "prepared" / "private" / "test.csv"
    _write(private_test, "id,target\n3,1\n4,0\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(competition_id, workspace)
    paths = adapter.get_state_paths(info)

    assert private_test.is_file()
    serialized = json.dumps(paths) + json.dumps(
        [artifact.to_state() for artifact in info.public_artifacts]
    )
    for forbidden in (
        str(private_test),
        str(private_test.parent),
        str(private_test.parent.resolve()),
        str(public_dir),
        str(public_dir.resolve()),
    ):
        assert forbidden not in serialized
    assert info.ground_truth_path is None


# --- Step 3.1: independent role-resolution regressions -------------------


def test_ambiguous_equally_ranked_train_candidates_fail_closed(
    tmp_path: Path,
) -> None:
    """Two identically ranked train candidates are an error, not a coin flip."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "train_part_a.csv", "id,feature,target\n1,0.1,0\n")
    _write(public_dir / "train_part_b.csv", "id,feature,target\n2,0.2,1\n")
    _write(public_dir / "sample_submission.csv", "id,target\n1,0\n")

    with pytest.raises(ValueError) as excinfo:
        _resolve_loose(public_dir)

    message = str(excinfo.value)
    assert "train" in message
    assert "train_part_a.csv" in message
    assert "train_part_b.csv" in message
    assert "role_token_in_member" in message
    # Diagnostics stay public-relative: no cache-side absolute path leaks.
    assert str(public_dir) not in message


def test_exclusive_assignment_never_gives_one_table_two_primary_roles(
    tmp_path: Path,
) -> None:
    """`train_test.csv` may win at most one role, and never displaces train.csv."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "train.csv", "id,feature,target\n1,0.1,0\n2,0.2,1\n")
    _write(public_dir / "train_test.csv", "id,feature,split\n1,0.1,train\n3,0.3,test\n")
    _write(public_dir / "sample_submission.csv", "id,target\n3,0\n")

    artifacts = _resolve_loose(public_dir)

    assert one_artifact_path(artifacts, "train").name == "train.csv"
    combined = _artifact_named(artifacts, "train_test.csv")
    assert combined.role in ("test", "auxiliary")
    assert [artifact.path for artifact in artifacts].count(combined.path) == 1
    for role in ("train", "test", "submission"):
        assert _role_count(artifacts, role) <= 1


def test_wide_multiclass_submission_columns_resolve_without_train_overlap(
    tmp_path: Path,
) -> None:
    """A submission template may declare targets that appear in no other table."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "train.csv", "id,text,author\na,hello there,x\nb,bye now,y\n")
    _write(public_dir / "test.csv", "id,text\nc,greetings\n")
    _write(
        public_dir / "sample_submission.csv",
        "id,alpha,beta,gamma\nc,0.33,0.33,0.34\n",
    )

    artifacts = _resolve_loose(public_dir, "text")

    assert one_artifact_path(artifacts, "train").name == "train.csv"
    assert one_artifact_path(artifacts, "test").name == "test.csv"
    assert one_artifact_path(artifacts, "submission").name == "sample_submission.csv"


def test_target_placeholder_test_table_still_resolves_the_test_role(
    tmp_path: Path,
) -> None:
    """A test table shipped with a placeholder target column is still the test table."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "train.csv", "id,feature,target\n1,0.1,1\n2,0.2,0\n")
    _write(public_dir / "test.csv", "id,feature,target\n3,0.3,0\n4,0.4,0\n")
    _write(public_dir / "sample_submission.csv", "id,target\n3,0\n4,0\n")

    artifacts = _resolve_loose(public_dir)

    assert one_artifact_path(artifacts, "train").name == "train.csv"
    assert one_artifact_path(artifacts, "test").name == "test.csv"
    assert one_artifact_path(artifacts, "submission").name == "sample_submission.csv"


def test_external_two_column_label_file_stays_auxiliary_sparse_labels(
    tmp_path: Path,
) -> None:
    """A nested label table is auxiliary evidence, never the train table."""
    competition_id = "synthetic-external-labels"
    public_dir = _public_dir(tmp_path, competition_id)
    images = public_dir / "images"
    images.mkdir()
    for index in range(4):
        (images / f"img_{index}.jpg").write_bytes(b"fake")
    _write(
        public_dir / "annotations" / "record_labels.csv",
        "record_id,label\nimg_0,cat\nimg_1,dog\n",
    )
    _write(public_dir / "sample_submission.csv", "record_id,label\nimg_2,cat\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    info = adapter.prepare_workspace(competition_id, workspace)

    labels = _artifact_named(info.public_artifacts, "record_labels.csv")
    assert labels.role == "auxiliary"
    assert labels.layout == "sparse_labels"
    assert one_artifact_path(info.public_artifacts, "train") is None
    assert [path.name for path in info.label_files] == ["record_labels.csv"]
    assert not (workspace / "train.csv").exists()


def test_idempotent_repeated_preparation_keeps_roles_and_fingerprints(
    tmp_path: Path,
) -> None:
    """Preparing an already-extracted tree twice yields identical typed records."""
    competition_id = "synthetic-idempotent-tree"
    public_dir = _public_dir(tmp_path, competition_id)
    _zip_one_member(
        public_dir / "en_train.csv.zip", "en_train.csv", TOKEN_TRAIN_TABLE
    )
    _write(public_dir / "sample_submission.csv", TOKEN_SUBMISSION_TABLE)

    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    first_workspace = tmp_path / "workspace" / "run-1"
    second_workspace = tmp_path / "workspace" / "run-2"
    first = adapter.prepare_workspace(competition_id, first_workspace)
    second = adapter.prepare_workspace(competition_id, second_workspace)

    def summary(info, workspace: Path):
        return sorted(
            (
                artifact.role,
                artifact.layout,
                str(artifact.path.relative_to(workspace)),
                artifact.fingerprint,
            )
            for artifact in info.public_artifacts
        )

    assert summary(first, first_workspace) == summary(second, second_workspace)
    assert _role_count(first.public_artifacts, "train") == 1


def test_exact_standard_names_retain_highest_precedence(tmp_path: Path) -> None:
    """An exact `train.csv` outranks any archive/member role token."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "train.csv", "id,feature,target\n1,0.1,0\n2,0.2,1\n")
    _zip_one_member(
        public_dir / "en_train.csv.zip", "en_train.csv", TOKEN_TRAIN_TABLE
    )
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    provenance = adapter._extract_zips(public_dir)

    artifacts = resolve_public_artifacts(public_dir, provenance, "tabular")

    assert one_artifact_path(artifacts, "train").name == "train.csv"
    archived = _artifact_named(artifacts, "en_train.csv")
    assert archived.role == "auxiliary"
    assert archived.source_archive == public_dir / "en_train.csv.zip"


def test_role_named_tsv_never_occupies_a_primary_role(tmp_path: Path) -> None:
    """Staging a TSV under a `.csv` alias would not change its delimiter."""
    competition_id = "synthetic-role-named-tsv"
    public_dir = _public_dir(tmp_path, competition_id)
    _write(public_dir / "train.tsv", "alpha\tbeta\n1\t2\n3\t4\n")
    _write(public_dir / "sample_submission.csv", "id,target\n1,0\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    info = adapter.prepare_workspace(competition_id, workspace)

    assert one_artifact_path(info.public_artifacts, "train") is None
    assert not (workspace / "train.csv").exists()
    assert info.train_csv_path is None
    delimited = _artifact_named(info.public_artifacts, "train.tsv")
    assert delimited.role == "auxiliary"
    assert delimited.layout == "unknown"
    assert info.label_files == []


# --- Audit ruling M1: camelCase stems ------------------------------------


def test_camel_case_sample_submission_resolves_the_submission_role(
    tmp_path: Path,
) -> None:
    """`sampleSubmission.csv` is the only delimited artifact in real comps."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    for name in ("train", "test"):
        media = public_dir / name
        media.mkdir()
        (media / "img_0.png").write_bytes(b"fake")
    _write(public_dir / "sampleSubmission.csv", "id,value\n1,0\n2,0\n")

    artifacts = _resolve_loose(public_dir, "image")

    assert one_artifact_path(artifacts, "submission").name == "sampleSubmission.csv"
    assert one_artifact_path(artifacts, "train") is None
    assert one_artifact_path(artifacts, "test") is None


def test_contest_stem_is_not_test_role_evidence(tmp_path: Path) -> None:
    """`contest` contains `test` lexically but is not a role token."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "contest_notes.csv", "id,note\n1,hello\n")
    _write(public_dir / "sample_submission.csv", "id,target\n1,0\n")

    artifacts = _resolve_loose(public_dir)

    assert one_artifact_path(artifacts, "test") is None
    assert _artifact_named(artifacts, "contest_notes.csv").role == "auxiliary"


# --- Audit ruling M2 / WATCH: weak train evidence ------------------------


def test_lone_labels_table_resolves_the_train_role(tmp_path: Path) -> None:
    """A bare `labels.csv` is the entire train table when nothing competes."""
    competition_id = "synthetic-lone-labels"
    public_dir = _public_dir(tmp_path, competition_id)
    images = public_dir / "images"
    images.mkdir()
    for index in range(4):
        (images / f"img_{index}.jpg").write_bytes(b"fake")
    _write(public_dir / "labels.csv", "id,breed\nimg_0,collie\nimg_1,poodle\n")
    _write(public_dir / "sample_submission.csv", "id,collie,poodle\nimg_2,0.5,0.5\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    info = adapter.prepare_workspace(competition_id, workspace)

    train = one_artifact_path(info.public_artifacts, "train")
    assert train == workspace / "train.csv"
    assert train.read_text(encoding="utf-8").startswith("id,breed")
    assert info.train_csv_path == workspace / "train.csv"
    assert info.label_files == []


def test_lone_train_labels_table_resolves_the_train_role(tmp_path: Path) -> None:
    """A sole `train_labels.csv` wins train on its explicit `train` token."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    media = public_dir / "train"
    media.mkdir()
    (media / "img_0.tif").write_bytes(b"fake")
    _write(public_dir / "train_labels.csv", "id,label\nimg_0,0\nimg_1,1\n")
    _write(public_dir / "sample_submission.csv", "id,label\nimg_2,0\n")

    artifacts = _resolve_loose(public_dir, "image")

    assert one_artifact_path(artifacts, "train").name == "train_labels.csv"
    assert one_artifact_path(artifacts, "submission").name == "sample_submission.csv"


def test_train_labels_never_displaces_a_resolved_train_table(
    tmp_path: Path,
) -> None:
    """The other direction of the same guard: `train.csv` keeps the role."""
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    _write(public_dir / "train.csv", "id,feature,target\n1,0.1,0\n2,0.2,1\n")
    _write(public_dir / "train_labels.csv", "id,label\n1,0\n2,1\n")
    _write(public_dir / "test.csv", "id,feature\n3,0.3\n")
    _write(public_dir / "sample_submission.csv", "id,target\n3,0\n")

    artifacts = _resolve_loose(public_dir)

    assert one_artifact_path(artifacts, "train").name == "train.csv"
    extra = _artifact_named(artifacts, "train_labels.csv")
    assert extra.role == "auxiliary"
    assert extra.layout == "sparse_labels"


# --- Step 3.4: staging, compatibility views and failed inspections -------


def test_typed_derivation_keeps_a_legacy_directory_sample_submission(
    tmp_path: Path,
) -> None:
    """A submission packed as a directory stays resolved by the legacy finder."""
    competition_id = "synthetic-nested-sample"
    public_dir = _public_dir(tmp_path, competition_id)
    media = public_dir / "train_images"
    media.mkdir()
    (media / "img_0.jpg").write_bytes(b"fake")
    _write(public_dir / "train.csv", "id,target\nimg_0,1\n")
    _write(
        public_dir / "sample_submission.csv" / "sample_submission.csv",
        "id,target\nimg_1,0\n",
    )

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    info = adapter.prepare_workspace(competition_id, workspace)

    assert info.sample_submission_path is not None
    assert Path(info.sample_submission_path).is_file()
    assert one_artifact_path(info.public_artifacts, "train") == workspace / "train.csv"


def test_recursive_audio_label_file_becomes_a_typed_sparse_label_record(
    tmp_path: Path,
) -> None:
    """A nested variable-width label file survives typed derivation."""
    competition_id = "synthetic-recursive-audio-labels"
    public_dir = _public_dir(tmp_path, competition_id)
    clips = public_dir / "bundle" / "clips"
    clips.mkdir(parents=True)
    for index in range(6):
        (clips / f"clip_{index}.wav").write_bytes(b"audio")
    _write(
        public_dir / "bundle" / "rec_labels.txt",
        "rec_id,[labels]\n0,3,7\n1\n2,4\n",
    )
    _write(public_dir / "sample_submission.csv", "rec_id,probability\n0,0\n1,0\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    info = adapter.prepare_workspace(competition_id, workspace)
    paths = adapter.get_state_paths(info)

    record = _artifact_named(info.public_artifacts, "rec_labels.txt")
    assert record.role == "auxiliary"
    assert record.layout == "sparse_labels"
    assert "variable_width_data_rows" in record.evidence
    assert [Path(path).name for path in paths["data_files"]["label_files"]] == [
        "rec_labels.txt"
    ]
    # Auxiliary artifacts keep their copied, regular staged relative path and
    # are never re-rooted as workspace-level alias symlinks.
    label_path = Path(paths["data_files"]["label_files"][0])
    assert label_path == workspace / "bundle" / "rec_labels.txt"
    assert label_path.is_file()
    assert not label_path.is_symlink()

    # The typed derivation is the sole producer of `label_files`. The legacy
    # lane staged a workspace-root alias per filename-matched label file and
    # then had its list discarded, leaving stray root symlinks pointing at
    # paths nothing owns any more.
    stray_root_alias = workspace / "rec_labels.txt"
    assert not stray_root_alias.exists()
    assert not stray_root_alias.is_symlink()


def test_label_named_candidate_failing_inspection_stages_as_unknown(
    tmp_path: Path,
) -> None:
    """Staging never aborts over a file nothing may ever consume."""
    competition_id = "synthetic-unreadable-labels"
    public_dir = _public_dir(tmp_path, competition_id)
    media = public_dir / "train"
    media.mkdir()
    (media / "img_0.png").write_bytes(b"fake")
    _write(public_dir / "train_annotations.txt", "alpha\nbeta\ngamma\n")
    _write(public_dir / "sample_submission.csv", "id,target\nimg_1,0\n")

    workspace = tmp_path / "workspace" / competition_id
    adapter = MLEBenchDataAdapter(mle_cache_path=tmp_path / "mle-cache")
    info = adapter.prepare_workspace(competition_id, workspace)

    record = _artifact_named(info.public_artifacts, "train_annotations.txt")
    assert record.role == "auxiliary"
    assert record.layout == "unknown"
    assert record.evidence
    assert info.label_files == []


# --- Task 6: the integrated prefixed-ZIP -> canonical -> candidate path ---

# A real, if tiny, token-level competition: ten train rows spread over five
# sentence groups, two test rows and the template that names them. Nothing
# here identifies a competition; the schemas are the generic token-level
# shape (a group key, a position, a train-only annotation, a source and a
# target). One target deliberately carries a comma inside a quoted field, so
# every layer that re-serializes it has to keep it one value.
GROUPED_TRAIN_TABLE = (
    "sentence_id,token_id,class,before,after\n"
    "0,0,PLAIN,The,The\n"
    '0,1,MONEY,$3.16,"three dollars, sixteen cents"\n'
    "1,0,CARDINAL,2,two\n"
    "1,1,PLAIN,cats,cats\n"
    "2,0,PLAIN,met,met\n"
    "2,1,PLAIN,at,at\n"
    "3,0,CARDINAL,5,five\n"
    "3,1,PLAIN,dogs,dogs\n"
    "4,0,PLAIN,on,on\n"
    "4,1,DATE,Monday,Monday\n"
)
# The public test table carries NO identifier column: `sentence_id` repeats
# and `token_id` is a position, so canonical prep has to name test rows by
# their position.
GROUPED_TEST_TABLE = "sentence_id,token_id,before\n5,0,2\n5,1,dogs\n"
# Template IDs are `<sentence_id>_<token_id>` in test-row order, and the
# placeholder is deliberately not one of the predictions so "the predictions
# were written" cannot pass by coincidence.
GROUPED_SAMPLE_TABLE = "id,after\n5_0,placeholder\n5_1,placeholder\n"

# The exact targets the train table declares, in file order. The test never
# writes this array into an artifact: it is the value canonical preparation
# must reproduce in `canonical/y.npy`.
GROUPED_TARGETS = [
    "The",
    "three dollars, sixteen cents",
    "two",
    "cats",
    "met",
    "at",
    "five",
    "dogs",
    "on",
    "Monday",
]

BODY_SENTINEL = "[TEST] CANDIDATE BODY REACHED"

# The candidate body, exactly as a model would return it. It reads only names
# the injected preamble defines, and proves at RUNTIME that the sparse
# preload name is absent - which is why the string appears here even though
# the generator-owned preamble must never contain it.
CANDIDATE_BODY = '''expected_targets = [
    "The",
    "three dollars, sixteen cents",
    "two",
    "cats",
    "met",
    "at",
    "five",
    "dogs",
    "on",
    "Monday",
]
assert CANONICAL_Y.tolist() == expected_targets
assert SUBMISSION_ID_COL == "id"
assert SUBMISSION_TARGET_COLS == ["after"]
assert "_PRELOADED_TARGETS_DF" not in globals()
write_submission(
    np.asarray(["two", "dogs"], dtype=str),
)
print("Final Validation Performance: 1.0")
print("[TEST] CANDIDATE BODY REACHED")
'''


class _HeaderGenerator(CodeGeneratorMixin):
    """Small real-generator harness that replaces only the LLM response."""

    use_dspy = False
    config = SimpleNamespace()
    llm = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=CANDIDATE_BODY)
    )

    @staticmethod
    def _get_dataset_info(*_args, **_kwargs) -> str:
        return ""

    @staticmethod
    def _get_domain_template(_domain: str, _component_type: str) -> str:
        return ""

    @staticmethod
    def _extract_code_from_response(response: str) -> str:
        return response.strip()


def test_prefixed_zip_reaches_candidate_body_through_canonical_contract(  # noqa: PLR0915 - one linear cross-layer run is the point; splitting it into helpers would hide which boundary broke
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One miniature competition crosses every layer, on production code only.

    The archives are prefixed, so no legacy filename finder resolves a role;
    the target contract, the folds, the test identity and the submission
    identity are all produced by the real nodes; and the program that finally
    runs is the one the generator rendered, executed in a real subprocess.
    Every assertion below reads a value some production layer produced.
    """
    reset_target_source_caches()
    competition_id = "generic-prefixed-seq2seq"
    cache_root = tmp_path / "mle-cache"
    public_dir = cache_root / competition_id / "prepared" / "public"
    public_dir.mkdir(parents=True)
    _zip_one_member(
        public_dir / "en_train.csv.zip", "en_train.csv", GROUPED_TRAIN_TABLE
    )
    _zip_one_member(
        public_dir / "en_test_2.csv.zip", "en_test_2.csv", GROUPED_TEST_TABLE
    )
    _zip_one_member(
        public_dir / "en_sample_submission_2.csv.zip",
        "en_sample_submission_2.csv",
        GROUPED_SAMPLE_TABLE,
    )
    # The graded labels live next to public/. Nothing this run produces may
    # name that directory.
    private_test = (
        cache_root / competition_id / "prepared" / "private" / "test.csv"
    )
    _write(private_test, "id,after\n5_0,two\n5_1,dogs\n")

    workspace = tmp_path / "workspace" / competition_id

    # 1-3: the real adapter, its state paths and a real initial state.
    adapter = MLEBenchDataAdapter(mle_cache_path=cache_root)
    info = adapter.prepare_workspace(competition_id, workspace_path=workspace)
    state_paths = adapter.get_state_paths(info)
    state = create_initial_state(competition_id, str(workspace))
    state.update(state_paths)

    # 4: only the workflow inputs the real node inspects.
    state["run_mode"] = "mlebench"
    state["mlebench_cache_path"] = str(cache_root)
    state["domain_detected"] = "seq_to_seq"
    state["target_col"] = "after"
    state["target_cols"] = ["after"]
    state["target_type"] = "single"
    state["timeout_per_component"] = 2800
    state["submission_contract"] = {
        "id_col": "id",
        "target_cols": ["after"],
        "expected_rows": 2,
    }

    # 5: the real canonical node writes the contract; nothing is hand-made.
    state.update(canonical_data_preparation_node(state))

    # --- the node-created contract -------------------------------------
    assert state["canonical_data_prepared"] is True
    assert state["target_col"] == "after"
    assert state["target_cols"] == ["after"]
    assert state["target_type"] == "single"

    contract = state["canonical_contract"]
    metadata = state["canonical_metadata"]
    assert contract["n_train"] == 10
    assert contract["n_test"] == 2
    assert contract["n_folds"] == 5
    for path_field in (
        "canonical_dir",
        "train_ids_path",
        "y_path",
        "folds_path",
        "feature_cols_path",
        "metadata_path",
        "test_ids_path",
    ):
        assert contract[path_field], f"contract declares no {path_field}"
        assert Path(contract[path_field]).exists()
    for hash_field in (
        "y_hash",
        "folds_hash",
        "train_ids_hash",
        "train_schema_hash",
    ):
        assert contract[hash_field]

    assert metadata["is_classification"] is False
    assert metadata["source_col"] == "before"
    assert metadata["group_col"] == "sentence_id"
    # Train-only annotation: absent from the public test table, so production
    # inference must decline to name it.
    assert metadata["class_col"] is None
    assert metadata["is_seq2seq"] is True

    # The comma-bearing target survives literally, as ONE value.
    canonical_y = np.load(contract["y_path"], allow_pickle=True)
    assert canonical_y.tolist() == GROUPED_TARGETS
    assert "three dollars, sixteen cents" in canonical_y.tolist()

    # Every sentence group is validated in exactly one fold, and the five
    # groups land in five distinct folds - no group is split across folds.
    folds = np.load(contract["folds_path"])
    staged_train = pd.read_csv(Path(state["data_files"]["train_csv"]))
    assert len(folds) == len(staged_train) == 10
    folds_by_group = {
        int(group): set(frame.tolist())
        for group, frame in pd.Series(folds).groupby(
            staged_train["sentence_id"].to_numpy()
        )
    }
    assert len(folds_by_group) == 5
    assert all(len(assigned) == 1 for assigned in folds_by_group.values())
    assert len({next(iter(a)) for a in folds_by_group.values()}) == 5

    # The marker is the node's own proof, and it describes this contract.
    marker = state["canonical_contract_validation"]
    assert marker
    assert marker == build_canonical_validation_marker(contract, metadata)

    # Declared test identity: the contract names it, the state echoes it, and
    # the array really has one name per public test row.
    assert contract["test_ids_path"] == state["canonical_test_ids_path"]
    canonical_test_ids = np.load(
        contract["test_ids_path"], allow_pickle=False
    )
    assert len(canonical_test_ids) == 2

    # 6: the real prompt builders; only the LLM response is replaced.
    monkeypatch.setattr(
        code_generator_module,
        "build_dynamic_instructions",
        lambda **_kwargs: "",
    )
    real_compose = code_generator_module.compose_generate_prompt
    composed_prompts: list[str] = []

    def _spy_compose(**kwargs) -> str:
        prompt = real_compose(**kwargs)
        composed_prompts.append(prompt)
        return prompt

    monkeypatch.setattr(
        code_generator_module, "compose_generate_prompt", _spy_compose
    )

    # 7: the no-LLM contract preparation, asserted before any generation.
    component = AblationComponent(
        "grouped_seq_model", "model", "train a sequence model"
    )
    competition_info = CompetitionInfo(
        competition_id, "", "accuracy", "sequence_generation"
    )
    generator = _HeaderGenerator()
    prepared = generator._prepare_generated_contract(
        component,
        competition_info,
        workspace,
        "seq_to_seq",
        state,
    )
    target_source = prepared.target_source
    assert target_source.mode == "canonical"
    assert target_source.canonical_authoritative is True
    assert target_source.label_files == ()
    assert target_source.canonical_target_path == Path(contract["y_path"])
    assert target_source.canonical_test_ids_path == Path(
        contract["test_ids_path"]
    )
    assert target_source.target_source_fingerprint
    assert prepared.header_sha256
    assert prepared.contract_fingerprint

    manifest_paths = {
        item.relative_path for item in target_source.protected_inputs
    }
    assert manifest_paths
    assert (
        str(Path(contract["test_ids_path"]).relative_to(workspace))
        in manifest_paths
    )

    generated = generator._generate_code(
        component,
        competition_info,
        workspace,
        "seq_to_seq",
        state,
        prepared_contract=prepared,
    )
    header = generated.split(INJECTED_HEADER_END_MARKER, 1)[0]
    # The program that will run carries the byte-exact prepared preamble.
    assert header + INJECTED_HEADER_END_MARKER + "\n" == prepared.path_header
    assert generated.endswith(CANDIDATE_BODY.strip())
    assert composed_prompts and composed_prompts[0].strip()
    prompt = composed_prompts[0]

    # --- adapter / state / generated-code boundaries -------------------
    for role in ("train", "test", "submission"):
        assert _role_count(info.public_artifacts, role) == 1
    primary = [a for a in info.public_artifacts if a.role != "auxiliary"]
    assert len(info.public_artifacts) == 3
    assert sorted(a.role for a in primary) == ["submission", "test", "train"]
    assert all(a.layout == "rectangular_table" for a in primary)
    assert len({a.path for a in primary}) == 3

    workspace_root = workspace.resolve()
    for artifact in info.public_artifacts:
        assert artifact.path.resolve().is_relative_to(workspace_root)
        if artifact.source_archive is not None:
            assert artifact.source_archive.resolve().is_relative_to(
                workspace_root
            )
    for key in ("train_data_path", "test_data_path", "sample_submission_path"):
        assert Path(state_paths[key]).resolve().is_relative_to(workspace_root)
    # Every path the adapter serialized into state, including the typed
    # records it derived them from, resolves inside the run workspace.
    serialized_paths = [
        value
        for key, value in state_paths["data_files"].items()
        if key not in {"data_type", "label_files", "public_artifacts"} and value
    ]
    for record in state_paths["data_files"]["public_artifacts"]:
        serialized_paths.append(record["path"])
        if record["source_archive"]:
            serialized_paths.append(record["source_archive"])
    assert serialized_paths
    for serialized_path in serialized_paths:
        assert Path(serialized_path).resolve().is_relative_to(workspace_root)

    serialized_data_files = json.dumps(state_paths["data_files"])
    manifest_blob = json.dumps(
        [item.to_dict() for item in target_source.protected_inputs]
    )
    for forbidden in (
        str(cache_root),
        str(cache_root.resolve()),
        str(public_dir),
        str(public_dir.resolve()),
        str(private_test),
        str(private_test.parent),
        str(private_test.parent.resolve()),
    ):
        assert forbidden not in serialized_data_files
        assert forbidden not in manifest_blob
        assert forbidden not in prompt
        assert forbidden not in generated
    # The cache root is a controller field, and only a controller field: it
    # appears exactly once in the whole state, under `mlebench_cache_path`.
    assert state["mlebench_cache_path"] == str(cache_root)
    assert json.dumps(state_paths).count(str(cache_root)) == 0
    whole_state = json.dumps(state, default=str)
    assert whole_state.count(str(cache_root)) == 1
    assert str(private_test) not in whole_state
    assert str(private_test.parent) not in whole_state
    assert str(public_dir) not in whole_state

    assert info.label_files == []
    assert state_paths["data_files"]["label_files"] == []
    # The generator-owned preamble never names the sparse-preload lane. The
    # candidate body does, on purpose, to prove the name is undefined at
    # runtime - so this is asserted on the rendered header, not the program.
    for forbidden_name in (
        "_PRELOADED_TARGETS_DF",
        "_load_targets_from_files",
        "PRE-LOADING TARGETS",
    ):
        assert forbidden_name not in prepared.path_header

    # The public test table has no ID column, so the only identity it has is
    # positional - and that is NOT the submission identity.
    header_namespace: dict = {}
    exec(compile(header, "<generated-preamble>", "exec"), header_namespace)
    assert header_namespace["TEST_IDS_ARE_POSITIONAL"] is True
    assert header_namespace["CANONICAL_TEST_IDS"].tolist() == ["0", "1"]

    # 8: the real executor, on the real staged workspace.
    executor = CodeExecutor(
        timeout=30,
        run_mode="mlebench",
        mlebench_cache_path=str(cache_root),
    )
    result = execute_generated_candidate(
        executor,
        generated,
        working_dir=str(workspace),
        expected_artifacts=["submission.csv"],
        component_type="model",
    )

    assert result.errors == []
    assert result.success is True
    assert result.exit_code == 0
    assert result.candidate_body_reached is True
    assert BODY_SENTINEL in result.stdout
    assert result.header_sha256 == prepared.header_sha256
    assert result.contract_fingerprint == prepared.contract_fingerprint

    submission = pd.read_csv(workspace / "submission.csv", dtype=str)
    assert list(submission.columns) == ["id", "after"]
    template_ids = pd.read_csv(
        Path(state["data_files"]["sample_submission"]), dtype=str
    )["id"].tolist()
    assert submission["id"].tolist() == template_ids
    assert submission["id"].tolist() == ["5_0", "5_1"]
    # Template identity, preserved by row order - never the canonical row IDs.
    assert submission["id"].tolist() != canonical_test_ids.tolist()
    assert submission["after"].tolist() == ["two", "dogs"]
