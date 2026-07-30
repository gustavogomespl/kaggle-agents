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
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.prompts.templates.constraints.image import IMAGE_CONSTRAINTS


class _HeaderGenerator(CodeGeneratorMixin):
    """Small real-generator harness that replaces only the LLM response."""

    use_dspy = False
    config = SimpleNamespace()
    llm = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content="pass")
    )

    @staticmethod
    def _get_dataset_info(_working_dir: Path, _state: dict | None = None) -> str:
        return ""

    @staticmethod
    def _get_domain_template(_domain: str, _component_type: str) -> str:
        return ""

    @staticmethod
    def _extract_code_from_response(response: str) -> str:
        return response.strip()


def _write_canonical_workspace(
    tmp_path: Path,
    *,
    raw_labels: list[object],
    class_order: list[str],
) -> dict:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    canonical_dir = tmp_path / "canonical"
    train_dir.mkdir()
    test_dir.mkdir()
    canonical_dir.mkdir()

    train_ids = np.asarray(
        [f"train-{index}" for index in range(len(raw_labels))],
        dtype=str,
    )
    np.save(canonical_dir / "train_ids.npy", train_ids, allow_pickle=False)
    np.save(
        canonical_dir / "y.npy",
        np.asarray(raw_labels),
        allow_pickle=False,
    )
    np.save(
        canonical_dir / "folds.npy",
        np.arange(len(raw_labels), dtype=np.int64) % 2,
        allow_pickle=False,
    )
    (canonical_dir / "feature_cols.json").write_text("[]", encoding="utf-8")
    (canonical_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_folds": 2,
                "id_col": "image_id",
                "target_col": "label",
                "target_cols": ["label"],
                "target_type": "single",
                "n_targets": 1,
                "is_classification": True,
                "class_order": class_order,
                "canonical_rows": len(raw_labels),
            }
        ),
        encoding="utf-8",
    )
    sample_submission_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"image_id": ["test-0"], "label": [0]}).to_csv(
        sample_submission_path,
        index=False,
    )

    return {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "canonical_data_prepared": True,
        "current_train_path": str(train_dir),
        "current_test_path": str(test_dir),
        "sample_submission_path": str(sample_submission_path),
        "canonical_contract": {"y_path": str(canonical_dir / "y.npy")},
        "submission_contract": {
            "id_col": "image_id",
            "target_cols": ["label"],
        },
        "data_files": {
            "data_type": "image",
            "train": str(train_dir),
            "test": str(test_dir),
            "sample_submission": str(sample_submission_path),
        },
    }


def _generate_and_execute_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_labels: list[object],
    class_order: list[str],
) -> dict:
    state = _write_canonical_workspace(
        tmp_path,
        raw_labels=raw_labels,
        class_order=class_order,
    )
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
    header = generated.split("# === END PATH CONSTANTS ===", 1)[0]
    namespace: dict = {}
    exec(compile(header, "<generated-image-header>", "exec"), namespace)
    return namespace


def test_generated_image_header_preserves_raw_labels_and_injects_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_labels = ["dog", "cat", "dog", "cat"]

    namespace = _generate_and_execute_header(
        tmp_path,
        monkeypatch,
        raw_labels=raw_labels,
        class_order=["cat", "dog"],
    )

    assert namespace["CANONICAL_Y"].tolist() == raw_labels
    assert namespace["CANONICAL_CLASS_ORDER"] == ("cat", "dog")
    assert namespace["CANONICAL_CLASS_INDICES"].dtype == np.int64
    assert namespace["CANONICAL_CLASS_INDICES"].tolist() == [1, 0, 1, 0]


def test_generated_header_maps_numeric_raw_labels_without_mutating_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_labels = [1, 0, 1, 0]

    namespace = _generate_and_execute_header(
        tmp_path,
        monkeypatch,
        raw_labels=raw_labels,
        class_order=["0", "1"],
    )

    assert namespace["CANONICAL_Y"].tolist() == raw_labels
    assert namespace["CANONICAL_CLASS_ORDER"] == ("0", "1")
    assert namespace["CANONICAL_CLASS_INDICES"].tolist() == [1, 0, 1, 0]


def test_generated_image_header_rejects_duplicate_class_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"class order.*unique"):
        _generate_and_execute_header(
            tmp_path,
            monkeypatch,
            raw_labels=["cat", "dog"],
            class_order=["cat", "cat"],
        )


@pytest.mark.parametrize(
    "class_order",
    [
        ["cat"],
        ["cat", "dog", "fox"],
    ],
)
def test_generated_image_header_requires_exact_class_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    class_order: list[str],
) -> None:
    with pytest.raises(ValueError, match=r"class order.*exactly cover"):
        _generate_and_execute_header(
            tmp_path,
            monkeypatch,
            raw_labels=["cat", "dog"],
            class_order=class_order,
        )


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
