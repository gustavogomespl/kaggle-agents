"""Executable regressions for image-classification numeric targets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import kaggle_agents.agents.developer.code_generator as code_generator_module
from kaggle_agents.agents.developer.code_generator import CodeGeneratorMixin
from kaggle_agents.agents.developer.retry import _maybe_add_encoding_hint
from kaggle_agents.agents.developer.target_source import (
    CanonicalTargetContractError,
    reset_target_source_caches,
)
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.prompts.templates.constraints.image import IMAGE_CONSTRAINTS
from kaggle_agents.workflow.nodes.canonical_data import (
    canonical_data_preparation_node,
)


@pytest.fixture(autouse=True)
def _clean_target_source_caches():
    reset_target_source_caches()
    yield
    reset_target_source_caches()


class _HeaderGenerator(CodeGeneratorMixin):
    """Small real-generator harness that replaces only the LLM response."""

    use_dspy = False
    config = SimpleNamespace()
    llm = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content="pass")
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


def _write_canonical_workspace(tmp_path: Path, *, raw_labels: list[object]) -> dict:
    """Build a COMPLETE canonical contract with the real producer.

    The fixture used to hand-write ``{"y_path": ...}`` as the whole contract.
    That partial claim is now corruption: the target-source selector fails
    closed on it before generation, and weakening the rule to keep the stub
    would remove exactly the protection this file exercises.
    """
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "image_id": [f"train-{index}" for index in range(len(raw_labels))],
            "width": [float(index % 3) for index in range(len(raw_labels))],
            "label": raw_labels,
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        {
            "image_id": ["test-0", "test-1"],
            "width": [0.0, 1.0],
        }
    ).to_csv(test_csv, index=False)

    sample_submission_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"image_id": ["test-0", "test-1"], "label": [0, 0]}).to_csv(
        sample_submission_path,
        index=False,
    )

    state = {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "target_col": "label",
        "target_cols": ["label"],
        "current_train_path": str(train_dir),
        "current_test_path": str(test_dir),
        "sample_submission_path": str(sample_submission_path),
        "submission_contract": {
            "id_col": "image_id",
            "target_cols": ["label"],
        },
        "data_files": {
            "data_type": "image",
            "train": str(train_dir),
            "test": str(test_dir),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "sample_submission": str(sample_submission_path),
        },
    }
    state.update(canonical_data_preparation_node(state))
    assert state["canonical_data_prepared"] is True
    return state


def _tamper_class_order(state: dict, class_order: list[str]) -> None:
    metadata_path = Path(state["working_directory"]) / "canonical" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["class_order"] = class_order
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _generate_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: dict,
) -> str:
    monkeypatch.setattr(
        code_generator_module,
        "build_dynamic_instructions",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        code_generator_module,
        "build_context",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        code_generator_module,
        "compose_generate_prompt",
        lambda **_kwargs: "",
    )

    generated = _HeaderGenerator()._generate_code(
        AblationComponent("image_model", "model", "train a classifier"),
        CompetitionInfo(
            "opaque-images",
            "",
            "log_loss",
            "binary_classification",
        ),
        tmp_path,
        "image_classification",
        state,
    )
    return generated.split("# === END PATH CONSTANTS ===", 1)[0]


def _generate_and_execute_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_labels: list[object],
) -> dict:
    state = _write_canonical_workspace(tmp_path, raw_labels=raw_labels)
    header = _generate_header(tmp_path, monkeypatch, state)
    namespace: dict = {}
    exec(compile(header, "<generated-image-header>", "exec"), namespace)
    return namespace


def test_generated_image_header_preserves_raw_labels_and_injects_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_labels = ["dog", "cat"] * 6

    namespace = _generate_and_execute_header(
        tmp_path,
        monkeypatch,
        raw_labels=raw_labels,
    )

    assert namespace["CANONICAL_Y"].tolist() == raw_labels
    assert namespace["CANONICAL_CLASS_ORDER"] == ("cat", "dog")
    assert namespace["CANONICAL_CLASS_INDICES"].dtype == np.int64
    assert namespace["CANONICAL_CLASS_INDICES"].tolist() == [1, 0] * 6


def test_generated_header_maps_numeric_raw_labels_without_mutating_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_labels = [1, 0] * 6

    namespace = _generate_and_execute_header(
        tmp_path,
        monkeypatch,
        raw_labels=raw_labels,
    )

    assert namespace["CANONICAL_Y"].tolist() == raw_labels
    assert namespace["CANONICAL_CLASS_ORDER"] == ("0", "1")
    assert namespace["CANONICAL_CLASS_INDICES"].tolist() == [1, 0] * 6


@pytest.mark.parametrize(
    "class_order",
    [
        ["cat", "cat"],
        ["cat"],
        ["cat", "dog", "fox"],
    ],
)
def test_tampered_class_order_fails_before_any_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    class_order: list[str],
) -> None:
    """A class order that cannot describe the real labels never reaches an LLM."""
    state = _write_canonical_workspace(
        tmp_path,
        raw_labels=["cat", "dog"] * 6,
    )
    _tamper_class_order(state, class_order)

    with pytest.raises(CanonicalTargetContractError, match="class_order"):
        _generate_header(tmp_path, monkeypatch, state)


@pytest.mark.parametrize(
    ("class_order", "expected"),
    [
        (["cat", "cat"], r"class order.*unique"),
        (["cat"], r"class order.*exactly cover"),
        (["cat", "dog", "fox"], r"class order.*exactly cover"),
    ],
)
def test_injected_header_still_rejects_a_bad_class_order_at_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    class_order: list[str],
    expected: str,
) -> None:
    """Defence in depth: the header validates what it loads, not what it was told.

    Generation happens against a valid contract; the metadata is corrupted
    afterwards, exactly as a concurrent component would corrupt it.
    """
    state = _write_canonical_workspace(
        tmp_path,
        raw_labels=["cat", "dog"] * 6,
    )
    header = _generate_header(tmp_path, monkeypatch, state)
    _tamper_class_order(state, class_order)

    with pytest.raises(ValueError, match=expected):
        exec(compile(header, "<generated-image-header>", "exec"), {})


def test_image_prompt_uses_injected_indices_with_loss_specific_dtypes() -> None:
    assert "Never cast `CANONICAL_Y` directly" in IMAGE_CONSTRAINTS
    assert "CANONICAL_CLASS_INDICES" in IMAGE_CONSTRAINTS
    assert "BCEWithLogitsLoss" in IMAGE_CONSTRAINTS
    assert "torch.float32" in IMAGE_CONSTRAINTS
    assert "CrossEntropyLoss" in IMAGE_CONSTRAINTS
    assert "torch.long" in IMAGE_CONSTRAINTS


@pytest.mark.parametrize(
    "error",
    [
        "TypeError: invalid data type 'numpy.str_'",
        "ValueError: could not convert string to float: np.str_('cat')",
    ],
)
def test_target_conversion_errors_use_numeric_target_hint(error: str) -> None:
    hint = _maybe_add_encoding_hint(error)

    assert "CANONICAL_CLASS_INDICES" in hint
    assert "CANONICAL_Y" in hint
    assert "TF-IDF" not in hint


def test_free_form_feature_conversion_error_keeps_feature_hint() -> None:
    hint = _maybe_add_encoding_hint(
        "ValueError: could not convert string to float: 'hello world'"
    )

    assert "TF-IDF" in hint
    assert "text_feature_cols" in hint
    assert "CANONICAL_CLASS_INDICES" not in hint
