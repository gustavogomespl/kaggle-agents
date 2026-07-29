"""The packed image writer must bind submission bytes to saved evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.agents.developer.code_contracts import (
    missing_submission_helper_call,
    untrusted_contract_helper_import,
)
from kaggle_agents.agents.developer.code_generator import (
    _IMAGE_SUBMISSION_HELPER,
    _SUBMISSION_HELPER,
    CodeGeneratorMixin,
    _submission_helper_for_contract,
)
from kaggle_agents.utils.image_to_image_contract import save_packed_images


def _helper_namespace(
    tmp_path: Path,
    *,
    target_cols: list[str],
    id_col: str = "id",
) -> dict[str, object]:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    namespace = {
        "MODELS_DIR": models,
        "COMPONENT_NAME": "candidate",
        "SAMPLE_SUBMISSION_PATH": tmp_path / "sample_submission.csv",
        "SUBMISSION_PATH": tmp_path / "submission.csv",
        "SUBMISSION_ID_COL": id_col,
        "SUBMISSION_TARGET_COLS": target_cols,
    }
    exec(
        compile(_IMAGE_SUBMISSION_HELPER, "<image-submission-helper>", "exec"),
        namespace,
    )
    return namespace


def _generic_helper_namespace(
    tmp_path: Path,
    *,
    target_cols: list[str],
) -> dict[str, object]:
    namespace = {
        "SAMPLE_SUBMISSION_PATH": tmp_path / "sample_submission.csv",
        "SUBMISSION_PATH": tmp_path / "submission.csv",
        "SUBMISSION_TARGET_COLS": target_cols,
    }
    exec(
        compile(_SUBMISSION_HELPER, "<submission-helper>", "exec"),
        namespace,
    )
    return namespace


def _save_test_artifact(
    tmp_path: Path,
    images: list[np.ndarray],
    image_ids: list[str],
) -> Path:
    return save_packed_images(
        tmp_path / "models" / "test_candidate.npz",
        images,
        image_ids=image_ids,
    )


def test_helper_selection_is_specific_to_packed_image_contract() -> None:
    assert _submission_helper_for_contract(False) == _SUBMISSION_HELPER
    assert _submission_helper_for_contract(True) == _IMAGE_SUBMISSION_HELPER


def test_writer_uses_packed_artifact_and_preserves_literal_echo_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample_submission.csv"
    sample.write_text(
        "value,id,echo\n"
        "0,page_1_1,NA\n"
        "0,page_1_2,NULL\n"
        "0,page_2_1,N/A\n"
        "0,page_2_2,text\n",
        encoding="utf-8",
    )
    namespace = _helper_namespace(tmp_path, target_cols=["value"])
    _save_test_artifact(
        tmp_path,
        [np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)],
        ["page.png"],
    )
    real_read_csv = pd.read_csv
    calls: list[dict[str, object]] = []

    def guarded_read_csv(*args, **kwargs):
        calls.append(dict(kwargs))
        assert "chunksize" in kwargs or "nrows" in kwargs
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setenv("KAGGLE_AGENTS_SUBMISSION_CHUNK_ROWS", "2")

    # These values are deliberately wrong: submission must come from the
    # validated packed artifact, not a second unrelated prediction vector.
    namespace["write_submission"](np.full(4, 0.99))

    written = real_read_csv(
        tmp_path / "submission.csv",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    assert written["value"].astype(float).tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert written["id"].tolist() == [
        "page_1_1",
        "page_1_2",
        "page_2_1",
        "page_2_2",
    ]
    assert written["echo"].tolist() == ["NA", "NULL", "N/A", "text"]
    assert any(call.get("chunksize") == 2 for call in calls)


def test_packed_writer_reads_semicolon_template_and_emits_csv(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "id;value\npage_1_1;0\npage_1_2;0\n",
        encoding="utf-8",
    )
    namespace = _helper_namespace(tmp_path, target_cols=["value"])
    _save_test_artifact(
        tmp_path,
        [np.asarray([[0.1, 0.2]], dtype=np.float32)],
        ["page.png"],
    )

    namespace["write_submission"](None)

    assert (tmp_path / "submission.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0] == "id,value"
    written = pd.read_csv(tmp_path / "submission.csv")
    assert written["value"].tolist() == pytest.approx([0.1, 0.2])


def test_generic_writer_reads_semicolon_template_and_emits_csv(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "id;value\nrow-a;0\nrow-b;0\n",
        encoding="utf-8",
    )
    namespace = _generic_helper_namespace(
        tmp_path,
        target_cols=["value"],
    )

    namespace["write_submission"](
        np.asarray([0.25, 0.75]),
        test_ids=["row-a", "row-b"],
    )

    assert (tmp_path / "submission.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0] == "id,value"
    written = pd.read_csv(tmp_path / "submission.csv")
    assert written["value"].tolist() == [0.25, 0.75]


def test_generation_sanitizer_keeps_trusted_packed_writer_call() -> None:
    sanitized, removals = CodeGeneratorMixin()._strip_nrows_param(
        _IMAGE_SUBMISSION_HELPER
    )

    assert removals == 0
    assert "write_packed_image_submission" in sanitized


def test_rejects_multiple_pixel_prediction_columns(tmp_path: Path) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "left,id,right\n0,page_1_1,0\n",
        encoding="utf-8",
    )
    namespace = _helper_namespace(
        tmp_path,
        target_cols=["left", "right"],
    )
    _save_test_artifact(
        tmp_path,
        [np.asarray([[0.5]], dtype=np.float32)],
        ["page.png"],
    )

    with pytest.raises(ValueError, match="exactly one prediction column"):
        namespace["write_submission"](None)


def test_rejects_test_ids_because_pixel_ids_come_from_template(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "value,id\n0,page_1_1\n",
        encoding="utf-8",
    )
    namespace = _helper_namespace(tmp_path, target_cols=["value"])

    with pytest.raises(ValueError, match="test_ids is not supported"):
        namespace["write_submission"](None, test_ids=["page_1_1"])


def test_invalid_packed_values_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "value,id\n0,page_1_1\n",
        encoding="utf-8",
    )
    namespace = _helper_namespace(tmp_path, target_cols=["value"])
    np.savez(
        tmp_path / "models" / "test_candidate.npz",
        values=np.asarray([np.nan], dtype=np.float32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        shapes=np.asarray([[1, 1]], dtype=np.int32),
        image_ids=np.asarray(["page.png"], dtype=str),
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        namespace["write_submission"](None)


def test_row_mismatch_does_not_replace_existing_submission(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "value,id\n0,page_1_1\n0,page_1_2\n0,page_1_3\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    submission.write_text("previous-valid-result\n", encoding="utf-8")
    namespace = _helper_namespace(tmp_path, target_cols=["value"])
    _save_test_artifact(
        tmp_path,
        [np.asarray([[0.1, 0.2]], dtype=np.float32)],
        ["page.png"],
    )

    with pytest.raises(ValueError, match="outside its packed image"):
        namespace["write_submission"](None)

    assert submission.read_text(encoding="utf-8") == "previous-valid-result\n"
    assert not list(tmp_path.glob(".submission.csv.*.tmp"))


def test_missing_packed_artifact_fails_before_submission(tmp_path: Path) -> None:
    (tmp_path / "sample_submission.csv").write_text(
        "value,id\n0,page_1_1\n",
        encoding="utf-8",
    )
    namespace = _helper_namespace(tmp_path, target_cols=["value"])

    with pytest.raises(ValueError, match="Packed test evidence is missing"):
        namespace["write_submission"](None)

    assert not (tmp_path / "submission.csv").exists()


def test_static_contract_checker_accepts_packed_helper_definition_and_call() -> None:
    code = _IMAGE_SUBMISSION_HELPER + "\nwrite_submission(None)\n"

    assert missing_submission_helper_call(code) is False
    assert untrusted_contract_helper_import(code) is None
