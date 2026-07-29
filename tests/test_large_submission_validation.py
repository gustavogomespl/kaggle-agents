"""Large pixel submissions must be validated, never accepted by a size gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.tools.code_executor.submission import (
    SubmissionValidationMixin,
)


class _Validator(SubmissionValidationMixin):
    pass


N_ROWS = 100_001


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"image_{index // 1000}:{index}" for index in range(N_ROWS)],
            "channel": np.arange(N_ROWS, dtype=np.int64) % 3,
            "prediction": np.zeros(N_ROWS, dtype=np.float32),
        }
    )


def _validate(
    tmp_path: Path,
    submission: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[bool, str]:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)
    submission.to_csv(submission_path, index=False)
    return _Validator().validate_submission_format(
        submission_path=submission_path,
        sample_submission_path=sample_path,
        component_type="model",
        problem_type="regression",
        target_cols=["prediction"],
    )


def test_large_submission_is_actually_validated(tmp_path: Path) -> None:
    sample = _sample_frame()
    submission = sample.copy()
    submission["prediction"] = np.linspace(0.1, 0.9, N_ROWS)

    valid, message = _validate(tmp_path, submission, sample)

    assert valid is True
    assert "validated" in message.lower()
    assert "skipped" not in message.lower()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_row", "Row count mismatch"),
        ("swapped_ids", "WRONG ORDER"),
        ("nan_prediction", "NaN"),
        ("template_copy", "unchanged from sample_submission"),
    ],
)
def test_large_submission_rejects_invalid_content(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    sample = _sample_frame()
    submission = sample.copy()
    submission["prediction"] = np.linspace(0.1, 0.9, N_ROWS)
    if mutation == "missing_row":
        submission = submission.iloc[:-1].copy()
    elif mutation == "swapped_ids":
        submission.loc[[0, 1], "id"] = submission.loc[[1, 0], "id"].to_numpy()
    elif mutation == "nan_prediction":
        submission.loc[50_000, "prediction"] = np.nan
    elif mutation == "template_copy":
        submission["prediction"] = sample["prediction"]

    valid, validation_message = _validate(tmp_path, submission, sample)

    assert valid is False
    assert message in validation_message
