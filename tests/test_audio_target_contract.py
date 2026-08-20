"""Regression tests for benchmark-neutral audio target handling."""

from types import SimpleNamespace

import numpy as np
import pytest

from kaggle_agents.agents.developer.code_generator import CodeGeneratorMixin
from kaggle_agents.agents.planner.fallback_plans.audio import (
    create_audio_fallback_plan,
)
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.mlebench.data_adapter.detection import DetectionMixin
from kaggle_agents.prompts.templates.audio_template import get_audio_config
from kaggle_agents.prompts.templates.builders.context import (
    _build_audio_context,
)
from kaggle_agents.prompts.templates.builders.model import (
    _build_audio_domain_instructions,
)
from kaggle_agents.utils.data_audit import (
    AuditFailedError,
    audit_audio_competition,
)
from kaggle_agents.utils.label_parser import (
    infer_filename_label_table,
    read_id_mapping,
)
from kaggle_agents.workflow.nodes.canonical_data import (
    _filename_label_pattern_from_state,
    canonical_data_preparation_node,
)


def _files(tmp_path, names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)
    return paths


def test_filename_target_inference_uses_unique_repeated_structure(tmp_path) -> None:
    paths = _files(
        tmp_path,
        ["clip_a_red.wav", "clip_b_red.wav", "clip_c_blue.wav", "clip_d_blue.wav"],
    )

    table = infer_filename_label_table(paths)

    assert list(table.columns) == ["record_id", "file_path", "target"]
    assert table["target"].tolist() == ["red", "red", "blue", "blue"]
    assert table.attrs["target_inference"]["mode"] == "unique_filename_structure"


def test_filename_target_inference_rejects_unique_identifier_suffixes(tmp_path) -> None:
    paths = _files(
        tmp_path,
        ["clip_001.wav", "clip_002.wav", "clip_003.wav", "clip_004.wav"],
    )

    with pytest.raises(ValueError, match="not uniquely supported"):
        infer_filename_label_table(paths)


def test_ambiguous_partitions_resolved_by_expected_class_count(tmp_path) -> None:
    """Template-backed class counts break leading/terminal token ties.

    Capture timestamps repeat (several clips per session), so the leading
    stem token forms a second "viable" partition next to the terminal binary
    label and conservative inference fails closed. The public submission
    template proves how many classes the task has; that evidence selects the
    matching partition without lowering the repeated-structure bar.
    """
    names = [
        "20240101_000000_recA_1.wav",
        "20240101_000130_recB_0.wav",
        "20240102_000000_recC_1.wav",
        "20240102_000130_recD_0.wav",
        "20240103_000000_recE_1.wav",
        "20240103_000130_recF_0.wav",
    ]
    paths = _files(tmp_path, names)

    with pytest.raises(ValueError, match="not uniquely supported"):
        infer_filename_label_table(paths)

    table = infer_filename_label_table(paths, expected_num_classes=2)
    assert table["target"].tolist() == ["1", "0", "1", "0", "1", "0"]
    assert "expected_class_count" in table.attrs["target_inference"]["evidence"]


def test_expected_class_count_cannot_pick_between_equal_size_partitions(
    tmp_path,
) -> None:
    paths = _files(
        tmp_path,
        [
            "group_one/a_cat.wav",
            "group_one/b_cat.wav",
            "group_one/c_dog.wav",
            "group_one/d_dog.wav",
            "group_two/e_cat.wav",
            "group_two/f_cat.wav",
            "group_two/g_dog.wav",
            "group_two/h_dog.wav",
        ],
    )

    with pytest.raises(ValueError, match="viable interpretations"):
        infer_filename_label_table(paths, expected_num_classes=2)


def test_audio_fallback_disambiguates_with_binary_probability_template(
    tmp_path,
) -> None:
    """End to end: timestamped stems + binary template still yield a contract."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _files(
        train_dir,
        [
            "20240101_000000_recA_1.wav",
            "20240101_000130_recB_0.wav",
            "20240102_000000_recC_1.wav",
            "20240102_000130_recD_0.wav",
            "20240103_000000_recE_1.wav",
            "20240103_000130_recF_0.wav",
        ],
    )
    _files(test_dir, ["t_1.wav", "t_2.wav"])
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("clip,probability\nt_1,0\nt_2,0\n", encoding="utf-8")

    updates = canonical_data_preparation_node(
        {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "train": str(train_dir),
                "test": str(test_dir),
                "sample_submission": str(sample),
            },
            "submission_contract": {
                "id_col": "clip",
                "target_cols": ["probability"],
            },
            "sample_submission_path": str(sample),
        }
    )

    assert updates["canonical_data_prepared"] is True
    y = np.load(tmp_path / "canonical" / "y.npy", allow_pickle=True)
    assert sorted({str(value) for value in y}) == ["0", "1"]


def test_mlebench_audio_header_defines_canonical_flag_when_contract_absent(
    tmp_path,
) -> None:
    """The injected preamble must define the flag its own contract references.

    In mlebench mode without a canonical contract the header tells candidates
    to check ``CANONICAL_FOLDS_AVAILABLE`` before touching CANONICAL_*, but the
    flag itself was only injected outside mlebench mode — so every candidate
    died on NameError instead of the intended clear data-contract error.
    """

    class _HeaderGen(CodeGeneratorMixin):
        use_dspy = False
        config = SimpleNamespace()
        llm = SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content=""))

        @staticmethod
        def _get_dataset_info(*_args, **_kwargs) -> str:
            return ""

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _files(train_dir, ["clip_a.wav"])
    _files(test_dir, ["clip_b.wav"])
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("clip,probability\nclip_b,0\n", encoding="utf-8")

    state = {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "current_train_path": str(train_dir),
        "current_test_path": str(test_dir),
        "sample_submission_path": str(sample),
        "data_files": {
            "data_type": "audio",
            "train": str(train_dir),
            "test": str(test_dir),
            "sample_submission": str(sample),
        },
        "submission_contract": {"id_col": "clip", "target_cols": ["probability"]},
        "failed_contract_fingerprints": {},
    }
    component = AblationComponent("model_a", "model", "train a model")
    info = CompetitionInfo("demo", "", "auc", "binary_classification")

    prepared = _HeaderGen()._prepare_generated_contract(
        component,
        info,
        tmp_path,
        "audio",
        state,
    )

    header = prepared.path_header
    assert header.count("CANONICAL_FOLDS_AVAILABLE = False") == 1
    assert "CANONICAL_FOLDS_AVAILABLE = True" not in header
    # ensure_folds is the live-mode escape hatch; mlebench stays fail-closed.
    assert "def ensure_folds" not in header
    compile(header, "<injected-header>", "exec")


def test_filename_target_inference_rejects_ambiguous_partitions(tmp_path) -> None:
    paths = _files(
        tmp_path,
        [
            "group_one/a_cat.wav",
            "group_one/b_cat.wav",
            "group_one/c_dog.wav",
            "group_one/d_dog.wav",
            "group_two/e_cat.wav",
            "group_two/f_cat.wav",
            "group_two/g_dog.wav",
            "group_two/h_dog.wav",
        ],
    )

    with pytest.raises(ValueError, match="viable interpretations"):
        infer_filename_label_table(paths)


def test_explicit_filename_pattern_requires_full_coverage(tmp_path) -> None:
    paths = _files(
        tmp_path,
        ["sample-A-001.wav", "sample-B-002.wav"],
    )

    table = infer_filename_label_table(
        paths,
        explicit_pattern=r"sample-(?P<target>[A-Z])-\d+\.wav$",
    )
    assert table["target"].tolist() == ["A", "B"]

    with pytest.raises(ValueError, match="does not match every file"):
        infer_filename_label_table(paths, explicit_pattern=r"sample-A-(\d+)")


def test_audio_canonical_folds_adapt_to_observed_class_support(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    paths = _files(
        audio_dir,
        [
            "sample_01_alpha.wav",
            "sample_02_alpha.wav",
            "sample_03_alpha.wav",
            "sample_04_beta.wav",
            "sample_05_beta.wav",
            "sample_06_beta.wav",
        ],
    )
    assert len(paths) == 6

    result = DetectionMixin().create_canonical_from_audio_filenames(
        audio_dir,
        tmp_path / "canonical",
        n_folds=5,
    )

    assert result["success"] is True
    assert result["metadata"]["n_folds"] == 3
    assert result["metadata"]["target_source"] == "unique_filename_structure"
    assert result["metadata"]["id_col"] == "record_id"


def test_audio_fallback_state_carries_a_gradeable_contract(tmp_path) -> None:
    """The audio fallback must return the same contract the tabular path does.

    Its success dict used to omit ``canonical_contract`` (and never passed
    test IDs through), so the trusted scorer — which reads
    ``canonical_contract["train_ids_path"]`` with no directory fallback —
    rejected every model with "Canonical/model train IDs are unavailable".
    Both audio competitions in the benchmark suite zeroed by construction.
    """
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _files(
        train_dir,
        ["clip_a_red.wav", "clip_b_red.wav", "clip_c_blue.wav", "clip_d_blue.wav"],
    )
    _files(test_dir, ["t_1.wav", "t_2.wav"])
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("clip,probability\nt_1,0\nt_2,0\n", encoding="utf-8")

    updates = canonical_data_preparation_node(
        {
            "working_directory": str(tmp_path),
            "data_files": {
                "data_type": "audio",
                "train": str(train_dir),
                "test": str(test_dir),
                "sample_submission": str(sample),
            },
            "submission_contract": {"id_col": "clip"},
            "sample_submission_path": str(sample),
        }
    )

    assert updates["canonical_data_prepared"] is True
    contract = updates["canonical_contract"]
    # The exact key the trusted scorer refuses to work without:
    train_ids_path = contract["train_ids_path"]
    assert train_ids_path
    assert (tmp_path / "canonical" / "train_ids.npy").is_file()
    assert set(np.load(train_ids_path, allow_pickle=True).tolist()) == {
        "clip_a_red",
        "clip_b_red",
        "clip_c_blue",
        "clip_d_blue",
    }
    # Test IDs, in submission-template order, so components and the ensemble
    # agree on prediction alignment.
    test_ids = np.load(
        tmp_path / "canonical" / "test_ids.npy", allow_pickle=False
    )
    assert [str(v) for v in test_ids] == ["t_1", "t_2"]
    assert updates["expected_test_rows"] == 2


def test_failed_media_canonical_leaves_no_directory_behind(tmp_path) -> None:
    """A refused fallback must not leave a canonical/ shell on disk.

    The single-samples check runs AFTER mkdir, so a refusal used to leave an
    empty canonical/ — which the executor's integrity gate reads as a
    declared-but-incomplete contract and then refuses ALL generated-code
    execution (the orphan-directory class of bug).
    """
    audio_dir = tmp_path / "audio"
    # One sample per class: stratified CV is impossible, producer refuses.
    _files(audio_dir, ["clip_a_red.wav", "clip_b_blue.wav"])

    result = DetectionMixin().create_canonical_from_audio_filenames(
        audio_dir,
        tmp_path / "canonical",
        n_folds=5,
    )

    assert result["success"] is False
    assert not (tmp_path / "canonical").exists()


def test_media_canonical_rebuild_removes_stale_files(tmp_path) -> None:
    """The fallback owns the whole directory it writes.

    It is invoked precisely because no usable contract exists, yet it used to
    overwrite only its own five files — stale artifacts from an earlier
    failed prep (a temporal mask, old test IDs) survived next to the fresh
    arrays and every component then failed shape checks against them.
    """
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "oof_eligible_mask.npy").write_bytes(b"stale")
    (canonical / "temporal_splits.npz").write_bytes(b"stale")
    audio_dir = tmp_path / "audio"
    _files(
        audio_dir,
        ["clip_a_red.wav", "clip_b_red.wav", "clip_c_blue.wav", "clip_d_blue.wav"],
    )

    result = DetectionMixin().create_canonical_from_audio_filenames(
        audio_dir,
        canonical,
        n_folds=2,
        test_ids=["t_1", "t_2"],
    )

    assert result["success"] is True
    assert not (canonical / "oof_eligible_mask.npy").exists()
    assert not (canonical / "temporal_splits.npz").exists()
    assert (canonical / "metadata.json").is_file()


def test_id_mapping_defaults_to_generic_contract_columns(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text(
        "source_key,asset_name\nrow-a,clip-a.wav\nrow-b,clip-b.wav\n",
        encoding="utf-8",
    )

    table = read_id_mapping(mapping_path, resolve_extensions=False)

    assert list(table.columns[:2]) == ["record_id", "file_path"]


def test_id_mapping_resolves_real_full_paths(tmp_path) -> None:
    audio_dir = tmp_path / "media"
    audio_dir.mkdir()
    audio_path = audio_dir / "clip-a.wav"
    audio_path.touch()
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text(
        "source_key,asset_name\nrow-a,clip-a\n",
        encoding="utf-8",
    )

    table = read_id_mapping(mapping_path, audio_dir=audio_dir)

    assert table.loc[0, "file_path"] == str(audio_path)


def test_audio_audit_uses_nonempty_and_contract_coverage(tmp_path) -> None:
    audio_path = tmp_path / "only-record.wav"
    audio_path.touch()

    result = audit_audio_competition(
        working_dir=tmp_path,
        audio_source_dir=tmp_path,
        strict=True,
    )
    assert result.audio_files_found == 1

    with pytest.raises(AuditFailedError, match="INCOMPLETE AUDIO COVERAGE"):
        audit_audio_competition(
            working_dir=tmp_path,
            audio_source_dir=tmp_path,
            expected_file_paths=[tmp_path / "missing.wav"],
            strict=True,
        )


def test_multiple_classes_do_not_imply_multilabel_audio() -> None:
    ordinary_state = {
        "competition_info": SimpleNamespace(problem_type="classification"),
        "submission_format_info": {"num_classes": 4},
    }
    multilabel_state = {
        "competition_info": SimpleNamespace(problem_type="multi_label"),
        "submission_format_info": {"num_classes": 4},
    }

    ordinary = "\n".join(_build_audio_domain_instructions(ordinary_state))
    multilabel = "\n".join(_build_audio_domain_instructions(multilabel_state))

    assert "do not by themselves prove multi-label" in ordinary
    assert "BCEWithLogitsLoss" not in ordinary
    assert "BCEWithLogitsLoss" in multilabel


def test_long_audio_submission_prompt_casts_observed_record_ids_explicitly() -> None:
    state = {
        "domain_type": "audio",
        "submission_format_info": {
            "format_type": "long",
            "id_column": "Id",
            "id_pattern": "record * 10 + class",
            "id_multiplier": 10,
            "num_classes": 4,
            "target_columns": ["Probability"],
        },
    }

    context = _build_audio_context(state)
    model = "\n".join(_build_audio_domain_instructions(state))

    for prompt in (context, model):
        assert "record_id_int = int(record_id)" in prompt
        assert "submission_id = record_id *" not in prompt


def test_audio_config_is_derived_from_observations() -> None:
    config = get_audio_config(
        observed_sample_rates=[16_000, 16_000, 22_050],
        observed_durations=[1.0, 2.0, 3.0],
    )

    assert config["sample_rate"] == 16_000
    assert config["duration"] == 2.0
    assert config["fmax"] == 8_000

    with pytest.raises(ValueError, match="requires observed"):
        get_audio_config([], [])


def test_filename_pattern_requires_recorded_public_evidence() -> None:
    assert (
        _filename_label_pattern_from_state(
            {
                "parsing_info": {
                    "filename_labels": {
                        "pattern": r"-(?P<target>[A-Z])-",
                        "evidence": "public data description",
                    }
                }
            }
        )
        == r"-(?P<target>[A-Z])-"
    )
    assert (
        _filename_label_pattern_from_state(
            {
                "parsing_info": {
                    "filename_labels": {
                        "pattern": r"-(?P<target>[A-Z])-",
                    }
                }
            }
        )
        is None
    )


def test_audio_fallback_has_no_fixed_domain_or_target_defaults() -> None:
    plan = create_audio_fallback_plan("audio_classification", {})
    text = "\n".join(str(component) for component in plan).lower()

    assert "22050" not in text
    assert "32000" not in text
    assert "efficientnet" not in text
    assert "resnet" not in text
    assert "bcewithlogitsloss" not in text
    assert "len(audio_files) < 10" not in text
    assert "target_sample_rate" in text
    assert "infer single-label versus multi-label" in text
