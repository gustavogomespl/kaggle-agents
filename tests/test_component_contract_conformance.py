"""One rule, checked across every problem shape the benchmark contains.

A component that produced exactly the artifacts it was told to produce must
never be discarded by a later check. Every gate in the pipeline can veto a
candidate on its own, and each expresses its own idea of what a valid component
looks like. When any of those ideas drifts from what the developer actually
demanded, a model that did everything asked of it is destroyed -- and the run
reports "no valid submission" after hours of GPU time.

That is not a hypothesis. It is what happened, once per shape:

- a wide multilabel template: the gate wanted a class-order file that multilabel
  components are never asked to write, so no such competition could be graded;
- a wide multiclass template: probabilities that do not sum to 1 failed two
  separate checks even though the graded metric scores each column
  independently and accepts them;
- a table whose key repeats: canonical prep accepted it as the row identity, so
  every component raised before it trained.

Each cost a full run to find. This file finds the next one at test speed: for
each shape, stage exactly the demanded artifacts, then assert every gate
accepts them. A gate that wants more than the developer asks for fails here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.agents.developer.agent import (
    _expected_model_artifacts,
    _model_validation_problem_type,
    _requires_class_order_artifact,
    _validation_class_order_for_state,
)
from kaggle_agents.core.config import metric_reads_rows_as_distribution
from kaggle_agents.core.state import AblationComponent
from kaggle_agents.tools.code_executor.submission import SubmissionValidationMixin
from kaggle_agents.utils.strict_validation import (
    StrictValidationConfig,
    validate_model_artifacts,
)
from kaggle_agents.workflow.nodes.robustness_gate import _mle_evidence_failures


N_TRAIN = 8
N_TEST = 4


@dataclass(frozen=True)
class Shape:
    """A competition shape, described only by what the public data reveals."""

    name: str
    problem_type: str
    metric: str
    id_col: str
    target_cols: tuple[str, ...]
    # Column count of the OOF/test matrices a component writes. Differs from
    # len(target_cols) for label-format multiclass, where one submission column
    # carries a class name but the model emits one probability per class.
    prediction_width: int
    class_order: tuple[str, ...] | None = None


SHAPES = [
    Shape(
        name="regression",
        problem_type="regression",
        metric="rmse",
        id_col="key",
        target_cols=("fare_amount",),
        prediction_width=1,
    ),
    Shape(
        name="binary",
        problem_type="binary_classification",
        metric="log_loss",
        id_col="id",
        target_cols=("label",),
        prediction_width=1,
    ),
    Shape(
        name="multiclass_wide_ranking_metric",
        problem_type="multiclass_classification",
        metric="auc",
        id_col="image_id",
        target_cols=("healthy", "multiple_diseases", "rust", "scab"),
        prediction_width=4,
        class_order=("healthy", "multiple_diseases", "rust", "scab"),
    ),
    Shape(
        name="multiclass_wide_likelihood_metric",
        problem_type="multiclass_classification",
        metric="multi class log loss",
        id_col="id",
        target_cols=("class_a", "class_b", "class_c"),
        prediction_width=3,
        class_order=("class_a", "class_b", "class_c"),
    ),
    Shape(
        name="multilabel_wide",
        problem_type="multilabel_classification",
        metric="auc",
        id_col="id",
        target_cols=(
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ),
        prediction_width=6,
    ),
    Shape(
        name="multi_target_regression",
        problem_type="regression",
        metric="rmsle",
        id_col="id",
        target_cols=("formation_energy", "bandgap_energy"),
        prediction_width=2,
    ),
]


def _predictions(shape: Shape, rows: int) -> np.ndarray:
    """Well-formed predictions for this shape: in range, varied, never constant.

    Multiclass rows are left unnormalized unless the graded metric reads a row
    as a probability vector. That is what a per-class sigmoid head produces,
    the grader accepts it under a ranking metric, and treating it as fatal is
    exactly the defect this file guards against.
    """
    grid = np.linspace(0.1, 0.9, rows * shape.prediction_width)
    values = grid.reshape(rows, shape.prediction_width)
    if shape.problem_type == "regression":
        return values * 100.0
    if metric_reads_rows_as_distribution(shape.metric) and shape.prediction_width > 1:
        return values / values.sum(axis=1, keepdims=True)
    return values


def _stage(tmp_path: Path, shape: Shape, component_name: str) -> dict:
    """Write exactly the artifacts a conforming component produces."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    train_ids = np.asarray([f"tr{index}" for index in range(N_TRAIN)])
    test_ids = np.asarray([f"te{index}" for index in range(N_TEST)])
    np.save(canonical_dir / "train_ids.npy", train_ids, allow_pickle=False)
    np.save(canonical_dir / "test_ids.npy", test_ids, allow_pickle=False)
    (canonical_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_folds": 2,
                "id_col": shape.id_col,
                "target_col": shape.target_cols[0],
                "target_cols": list(shape.target_cols),
                "target_type": "single" if len(shape.target_cols) == 1 else "multi_label",
                "n_targets": len(shape.target_cols),
                "is_classification": "regression" not in shape.problem_type,
                "class_order": list(shape.class_order) if shape.class_order else None,
            }
        ),
        encoding="utf-8",
    )

    oof = _predictions(shape, N_TRAIN)
    test = _predictions(shape, N_TEST)
    if shape.prediction_width == 1:
        oof, test = oof.reshape(-1), test.reshape(-1)
    np.save(models_dir / f"oof_{component_name}.npy", oof)
    np.save(models_dir / f"test_{component_name}.npy", test)
    np.save(models_dir / f"train_ids_{component_name}.npy", train_ids, allow_pickle=False)
    np.save(models_dir / f"test_ids_{component_name}.npy", test_ids, allow_pickle=False)

    template = pd.DataFrame({shape.id_col: test_ids})
    for column in shape.target_cols:
        template[column] = 0.0
    template.to_csv(tmp_path / "sample_submission.csv", index=False)

    submission = pd.DataFrame({shape.id_col: test_ids})
    for position, column in enumerate(shape.target_cols):
        source = test if test.ndim > 1 else test.reshape(-1, 1)
        submission[column] = source[:, min(position, source.shape[1] - 1)]
    submission.to_csv(tmp_path / "submission.csv", index=False)

    return {
        "run_mode": "mlebench",
        "working_directory": str(tmp_path),
        "problem_type": shape.problem_type,
        "oof_availability": {component_name: True},
        "trusted_component_scores": {component_name: 0.5},
        "canonical_contract": {
            "train_ids_path": str(canonical_dir / "train_ids.npy"),
            "test_ids_path": str(canonical_dir / "test_ids.npy"),
            "metadata_path": str(canonical_dir / "metadata.json"),
        },
        "canonical_metadata": {
            "class_order": list(shape.class_order) if shape.class_order else None
        },
        "submission_contract": {
            "id_col": shape.id_col,
            "target_cols": list(shape.target_cols),
            "class_order": (
                list(shape.target_cols) if len(shape.target_cols) > 1 else None
            ),
        },
    }


def _write_demanded_class_order(
    tmp_path: Path, state: dict, shape: Shape, component_name: str
) -> None:
    """Write the class-order file only when the developer demands one."""
    problem_type = _model_validation_problem_type(state)
    if not _requires_class_order_artifact(state, problem_type):
        return
    order = _validation_class_order_for_state(state, problem_type)
    assert order is not None, "a demanded class order must be resolvable"
    np.save(
        tmp_path / "models" / f"class_order_{component_name}.npy",
        np.asarray(order, dtype=str),
        allow_pickle=False,
    )


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.name)
class TestAConformingComponentSurvivesEveryGate:
    """The producer's contract and every consumer's contract must agree."""

    COMPONENT = "conforming_component"

    def _conforming(self, tmp_path: Path, shape: Shape) -> dict:
        state = _stage(tmp_path, shape, self.COMPONENT)
        _write_demanded_class_order(tmp_path, state, shape, self.COMPONENT)
        return state

    def test_the_demanded_artifacts_are_the_ones_written(
        self, tmp_path: Path, shape: Shape
    ) -> None:
        state = self._conforming(tmp_path, shape)
        component = AblationComponent(
            name=self.COMPONENT,
            component_type="model",
            code=shape.name,
        )
        demanded = _expected_model_artifacts(component, tmp_path, "mlebench") or []
        problem_type = _model_validation_problem_type(state)
        if _requires_class_order_artifact(state, problem_type):
            demanded.append(f"models/class_order_{self.COMPONENT}.npy")

        missing = [path for path in demanded if not (tmp_path / path).is_file()]
        assert missing == []

    def test_the_robustness_gate_accepts_it(
        self, tmp_path: Path, shape: Shape
    ) -> None:
        state = self._conforming(tmp_path, shape)

        assert _mle_evidence_failures(state) == {}

    def test_strict_artifact_validation_accepts_it(
        self, tmp_path: Path, shape: Shape
    ) -> None:
        state = self._conforming(tmp_path, shape)
        problem_type = _model_validation_problem_type(state)
        config = StrictValidationConfig(
            strict_mode=True,
            require_train_ids=True,
            require_test_ids=True,
            require_normalized_rows=metric_reads_rows_as_distribution(shape.metric),
        )
        expected_class_order = _validation_class_order_for_state(state, problem_type)
        if _requires_class_order_artifact(state, problem_type):
            config.require_class_order = True
            config.require_component_class_order = True

        result = validate_model_artifacts(
            working_dir=tmp_path,
            component_name=self.COMPONENT,
            expected_n_train=N_TRAIN,
            expected_n_test=N_TEST,
            expected_class_order=expected_class_order,
            expected_train_ids=[f"tr{index}" for index in range(N_TRAIN)],
            expected_test_ids=[f"te{index}" for index in range(N_TEST)],
            problem_type=problem_type,
            config=config,
        )

        assert result.errors == []

    def test_submission_format_validation_accepts_it(
        self, tmp_path: Path, shape: Shape
    ) -> None:
        self._conforming(tmp_path, shape)

        is_valid, message = SubmissionValidationMixin().validate_submission_format(
            submission_path=tmp_path / "submission.csv",
            sample_submission_path=tmp_path / "sample_submission.csv",
            component_type="model",
            problem_type=shape.problem_type,
            target_cols=list(shape.target_cols),
            require_normalized_rows=metric_reads_rows_as_distribution(shape.metric),
        )

        assert is_valid, message


class TestTheGuardStillCatchesRealDefects:
    """A conformance suite that accepts anything would protect nothing."""

    SHAPE = SHAPES[2]  # multiclass wide, ranking metric
    COMPONENT = "defective_component"

    def test_a_missing_evidence_artifact_is_still_rejected(
        self, tmp_path: Path
    ) -> None:
        state = _stage(tmp_path, self.SHAPE, self.COMPONENT)
        _write_demanded_class_order(tmp_path, state, self.SHAPE, self.COMPONENT)
        (tmp_path / "models" / f"test_{self.COMPONENT}.npy").unlink()

        assert _mle_evidence_failures(state)

    def test_an_unscored_component_is_still_rejected(self, tmp_path: Path) -> None:
        state = _stage(tmp_path, self.SHAPE, self.COMPONENT)
        _write_demanded_class_order(tmp_path, state, self.SHAPE, self.COMPONENT)
        state["trusted_component_scores"] = {self.COMPONENT: float("nan")}

        assert _mle_evidence_failures(state)

    def test_unnormalized_rows_still_fail_a_likelihood_metric(
        self, tmp_path: Path
    ) -> None:
        shape = SHAPES[3]  # multiclass wide, log loss
        _stage(tmp_path, shape, self.COMPONENT)
        # Break the one property this metric actually needs.
        submission = pd.read_csv(tmp_path / "submission.csv")
        submission[list(shape.target_cols)] *= 2.0
        submission.to_csv(tmp_path / "submission.csv", index=False)

        is_valid, _ = SubmissionValidationMixin().validate_submission_format(
            submission_path=tmp_path / "submission.csv",
            sample_submission_path=tmp_path / "sample_submission.csv",
            component_type="model",
            problem_type=shape.problem_type,
            target_cols=list(shape.target_cols),
            require_normalized_rows=metric_reads_rows_as_distribution(shape.metric),
        )

        assert is_valid is False
