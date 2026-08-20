"""Blank free-text predictions are scoreable answers, not format errors.

Text-to-text targets legitimately contain empty strings — the competition's
own training target does — so a blank cell in the prediction column is a
(possibly wrong) prediction the grader will score, not a corrupt file. The
validator used to hard-reject the whole submission on a single blank, which
quarantined components whose trusted OOF score had already been verified and
zeroed the run. Only an entirely blank column still fails, because that is a
broken pipeline rather than predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kaggle_agents.tools.code_executor.submission import (
    SubmissionValidationMixin,
)


class _Validator(SubmissionValidationMixin):
    pass


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"{row}_0" for row in range(8)],
            "after": ["token"] * 8,
        }
    )


def _validate(
    tmp_path: Path,
    submission: pd.DataFrame,
    problem_type: str,
) -> tuple[bool, str]:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample_frame().to_csv(sample_path, index=False)
    submission.to_csv(submission_path, index=False)
    return _Validator().validate_submission_format(
        submission_path=submission_path,
        sample_submission_path=sample_path,
        component_type="model",
        problem_type=problem_type,
        target_cols=["after"],
    )


def test_seq2seq_submission_accepts_sparse_blank_predictions(
    tmp_path: Path,
) -> None:
    submission = _sample_frame()
    submission["after"] = ["one", "", "three", "four", "five", "six", "7", ""]

    valid, message = _validate(tmp_path, submission, "seq2seq")

    assert valid is True, message


def test_non_seq2seq_label_submission_still_rejects_blanks(
    tmp_path: Path,
) -> None:
    submission = _sample_frame()
    submission["after"] = ["one", "", "three", "four", "five", "six", "7", "8"]

    valid, message = _validate(tmp_path, submission, "classification")

    assert valid is False
    assert "Blank label values" in message


def test_seq2seq_submission_rejects_entirely_blank_column(
    tmp_path: Path,
) -> None:
    submission = _sample_frame()
    submission["after"] = [""] * 8

    valid, message = _validate(tmp_path, submission, "seq2seq")

    assert valid is False
    assert "blank" in message.lower()
