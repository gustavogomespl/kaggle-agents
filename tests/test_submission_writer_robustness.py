"""Regressions for the injected submission writer's parsing and memory bounds.

Two failure modes are covered here, both of which pass every column-level
check while producing a wrong or impossible outcome:

1. The template is materialized as one DataFrame, so a pixel-level template
   exhausts memory in the same place the rest of the pipeline streams.
2. A protected path constant used as a dict key is mistaken for a
   redefinition, and the line is deleted from an otherwise correct program.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_agents.agents.developer.code_generator import (
    _SUBMISSION_HELPER,
    CodeGeneratorMixin,
    _protected_assignment_nodes,
)


def _writer_namespace(tmp_path: Path, target_cols: list[str]) -> dict:
    namespace = {
        "SAMPLE_SUBMISSION_PATH": tmp_path / "sample_submission.csv",
        "SUBMISSION_PATH": tmp_path / "submission.csv",
        "SUBMISSION_TARGET_COLS": target_cols,
    }
    # Executing our own shipped header text is exactly what is under test.
    exec(
        compile(_SUBMISSION_HELPER, "<helper>", "exec"), namespace
    )
    return namespace


class TestWriterStreamsTheTemplate:
    """The template must never be materialized as one DataFrame."""

    def test_template_is_read_in_bounded_chunks(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {"id": [f"r{index}" for index in range(6)], "target": [0] * 6}
        ).to_csv(sample, index=False)

        calls: list[dict] = []
        real_read_csv = pd.read_csv

        def recording_read_csv(*args, **kwargs):
            calls.append(dict(kwargs))
            return real_read_csv(*args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", recording_read_csv)
        namespace = _writer_namespace(tmp_path, ["target"])
        namespace["write_submission"](np.arange(6, dtype=float))

        assert any(call.get("chunksize") for call in calls)
        # No full-file read: every content read is either the header or chunked.
        assert not any(
            call.get("chunksize") is None and call.get("nrows") is None
            for call in calls
        )
        written = pd.read_csv(tmp_path / "submission.csv")
        assert written["target"].tolist() == list(range(6))
        assert written["id"].tolist() == [f"r{index}" for index in range(6)]

    def test_chunk_boundaries_do_not_shift_predictions(
        self, tmp_path: Path
    ) -> None:
        rows = 250
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame(
            {"id": [f"r{index}" for index in range(rows)], "target": [0] * rows}
        ).to_csv(sample, index=False)
        namespace = _writer_namespace(tmp_path, ["target"])

        predictions = np.arange(rows, dtype=float) / rows
        namespace["write_submission"](predictions)

        written = pd.read_csv(tmp_path / "submission.csv")
        assert written["target"].tolist() == predictions.tolist()

    def test_partial_file_is_not_left_behind_on_failure(
        self, tmp_path: Path
    ) -> None:
        sample = tmp_path / "sample_submission.csv"
        pd.DataFrame({"id": ["a", "b"], "target": [0, 0]}).to_csv(
            sample, index=False
        )
        namespace = _writer_namespace(tmp_path, ["target"])

        try:
            namespace["write_submission"](np.asarray([0.1, 0.2, 0.3]))
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("row-count mismatch must be rejected")

        assert not list(tmp_path.glob("*.partial"))


class TestProtectedConstantsAreNotOverStripped:
    """Only a rebinding is a redefinition."""

    def test_protected_constant_used_as_a_dict_key_is_left_alone(self) -> None:
        code = "cache = {}\ncache[DATA_DIR] = 1\n"

        assert _protected_assignment_nodes(code, {"DATA_DIR"}) == []
        assert CodeGeneratorMixin._strip_path_redefinitions(
            None,
            f"# === END PATH CONSTANTS ===\n{code}",
        ).endswith(code)

    def test_attribute_assignment_on_a_constant_is_not_a_rebinding(self) -> None:
        # The name still points at the same object afterwards, so stripping
        # this line changes behaviour without protecting anything.
        code = "TRAIN_PATH.suffix = '.csv'\n"

        assert _protected_assignment_nodes(code, {"TRAIN_PATH"}) == []

    def test_real_rebinding_is_still_detected(self) -> None:
        for code in (
            "DATA_DIR = Path('/tmp')\n",
            "DATA_DIR, other = Path('/tmp'), 1\n",
            "globals()['DATA_DIR'] = Path('/tmp')\n",
        ):
            nodes = _protected_assignment_nodes(code, {"DATA_DIR"})
            assert nodes, code
            assert nodes[0][1] == {"DATA_DIR"}
