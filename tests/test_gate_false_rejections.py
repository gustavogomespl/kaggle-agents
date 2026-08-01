"""Regressions for two gates that rejected correct work and blocked grading.

A run that produced a valid 0.039 log_loss submission finished with
`Valid Submission: No` because:

1. The static leakage rule read `X_train, X_val = X[train_idx], X[val_idx]` as
   fitting on held-out data — it tainted every target of the statement, so a
   sibling naming the validation split condemned the training side. Any failed
   module fails the whole robustness gate, and MLE-bench grading is fail-closed
   on that gate.
2. The OOF alignment check compared saved train IDs to canonical train IDs with
   np.array_equal. The injected helper always writes IDs as text while the
   canonical file keeps the public column's dtype, so identical IDs compared
   unequal and every component was dropped from the ensemble.

Three later runs were blocked the same way, each by a gate holding a component
to a contract nothing had asked it to satisfy:

3. The robustness gate demanded a class-order artifact from every submission
   with more than two prediction columns. A wide multilabel template has six,
   its columns are independent labels with no order to record, and the
   developer therefore never asks for the file. No multilabel competition could
   be graded, and the gate printed nothing.
4. Canonical prep accepted a repeating column as the row key, so every
   component raised on `align_train_to_canonical` before it trained and no
   repair could succeed: the defect was in the contract, not in the code.
5. Multiclass rows that did not sum to 1 failed a candidate outright even when
   the graded metric scores each column independently and would have accepted
   the very same predictions.
6. Fixing (5) at the developer gate was not enough: the SubmissionAgent's
   terminal validation and the ensemble's snapshot restore re-applied the
   unconditional row-sum rule to the very same bytes, so the run was accepted
   in the middle and rejected at the end.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_agents.agents.ensemble.submission import safe_restore_submission
from kaggle_agents.agents.robustness_agent import RobustnessAgent
from kaggle_agents.agents.submission_agent import SubmissionAgent
from kaggle_agents.core.config import metric_reads_rows_as_distribution
from kaggle_agents.utils.data_contract import _ensure_id_column
from kaggle_agents.utils.oof_validation import assert_oof_sanity
from kaggle_agents.utils.strict_validation import (
    StrictValidationConfig,
    validate_model_artifacts,
)
from kaggle_agents.utils.submission_artifacts import sha256_file
from kaggle_agents.workflow.nodes.robustness_gate import _mle_evidence_failures


class TestFoldLocalPreprocessingIsNotLeakage:
    """The idiom the constraints mandate must not be reported as leakage."""

    @staticmethod
    def _findings(code: str) -> list[str]:
        return [
            finding["description"]
            for finding in RobustnessAgent._find_direct_leakage(code)
        ]

    def test_tuple_unpacked_fold_split_is_clean(self) -> None:
        code = (
            "for fold, train_idx, val_idx in iter_canonical_cv_splits():\n"
            "    X_train, X_val = X[train_idx], X[val_idx]\n"
            "    X_train_scaled = scaler.fit_transform(X_train)\n"
            "    model.fit(X_train_scaled, y[train_idx])\n"
        )

        assert self._findings(code) == []

    def test_train_test_split_outputs_are_not_all_tainted(self) -> None:
        code = (
            "X_train, X_val, y_train, y_val = train_test_split(X, y)\n"
            "model.fit(X_train, y_train)\n"
        )

        assert self._findings(code) == []

    def test_log_regression_indexing_train_outputs_is_clean(self) -> None:
        code = (
            "X_train_f, X_val_f, y_train_f, y_val_f = "
            "train_test_split(X, y)\n"
            "model.fit(X[t_idx], y_train_f[t_idx])\n"
        )

        assert self._findings(code) == []

    def test_log_regression_training_indices_are_clean(self) -> None:
        code = (
            "train_idx, val_idx = train_test_split(indices)\n"
            "model.fit(X[train_idx])\n"
        )

        assert self._findings(code) == []

    def test_fold_local_vectorizer_is_clean(self) -> None:
        code = (
            "for fold, train_idx, val_idx in iter_canonical_cv_splits():\n"
            "    train_texts, val_texts = texts[train_idx], texts[val_idx]\n"
            "    features = vectorizer.fit_transform(train_texts)\n"
        )

        assert self._findings(code) == []


class TestRealLeakageIsStillCaught:
    """The relaxation must not blind the rule to the cases it exists for."""

    @staticmethod
    def _findings(code: str) -> list[str]:
        return [
            finding["description"]
            for finding in RobustnessAgent._find_direct_leakage(code)
        ]

    def test_fit_on_concatenated_train_and_test(self) -> None:
        code = (
            "full = pd.concat([train_df, test_df])\n"
            "scaler.fit_transform(full)\n"
        )

        assert self._findings(code)

    def test_fit_directly_on_validation(self) -> None:
        assert self._findings("scaler.fit(X_val)\n")

    def test_training_name_bound_to_a_validation_slice(self) -> None:
        """A single target named 'train' is still tainted by its own value."""
        code = "X_train = X[val_idx]\nscaler.fit_transform(X_train)\n"

        assert self._findings(code)

    def test_alias_of_validation_data(self) -> None:
        code = (
            "X_val = X[val_idx]\n"
            "X_train = X_val.copy()\n"
            "model.fit(X_train)\n"
        )

        assert self._findings(code)

    def test_scaler_fitted_on_the_full_matrix(self) -> None:
        code = (
            "X_all = np.vstack([train_features, test_features])\n"
            "X_train = X_all[:n_train]\n"
            "scaler.fit_transform(X_train)\n"
        )

        assert self._findings(code)

    def test_uninformative_multi_target_stays_conservative(self) -> None:
        """With no name saying which side is which, taint every target."""
        code = "first, second = build_matrices(test_frame)\nmodel.fit(first)\n"

        assert self._findings(code)


class TestOofIdentityIgnoresDtype:
    """Identity of IDs, not their numpy dtype, decides row-order agreement."""

    @staticmethod
    def _models_dir(tmp_path: Path, saved_ids: np.ndarray) -> Path:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        np.save(models_dir / "oof_candidate.npy", np.zeros((3, 2)))
        np.save(models_dir / "train_ids_candidate.npy", saved_ids, allow_pickle=False)
        return models_dir

    def test_text_ids_match_numeric_canonical_ids(self, tmp_path: Path) -> None:
        # What the injected helper writes vs what canonical prep keeps.
        models_dir = self._models_dir(tmp_path, np.asarray(["1", "3", "5"]))

        result = assert_oof_sanity(
            oof_path=models_dir / "oof_candidate.npy",
            models_dir=models_dir,
            expected_train_ids=np.asarray([1, 3, 5]),
        )

        assert "Train IDs mismatch - row order inconsistent" not in result.errors

    def test_a_real_reordering_is_still_rejected(self, tmp_path: Path) -> None:
        models_dir = self._models_dir(tmp_path, np.asarray(["3", "1", "5"]))

        result = assert_oof_sanity(
            oof_path=models_dir / "oof_candidate.npy",
            models_dir=models_dir,
            expected_train_ids=np.asarray([1, 3, 5]),
        )

        assert "Train IDs mismatch - row order inconsistent" in result.errors

    def test_different_ids_are_still_rejected(self, tmp_path: Path) -> None:
        models_dir = self._models_dir(tmp_path, np.asarray(["1", "3", "9"]))

        result = assert_oof_sanity(
            oof_path=models_dir / "oof_candidate.npy",
            models_dir=models_dir,
            expected_train_ids=np.asarray([1, 3, 5]),
        )

        assert "Train IDs mismatch - row order inconsistent" in result.errors

    def test_row_count_mismatch_is_still_rejected(self, tmp_path: Path) -> None:
        models_dir = self._models_dir(tmp_path, np.asarray(["1", "3"]))

        result = assert_oof_sanity(
            oof_path=models_dir / "oof_candidate.npy",
            models_dir=models_dir,
            expected_train_ids=np.asarray([1, 3, 5]),
        )

        assert "Train IDs mismatch - row order inconsistent" in result.errors


@pytest.mark.parametrize(
    ("saved", "canonical", "identical"),
    [
        (np.asarray(["1", "3", "5"]), np.asarray([1, 3, 5]), True),
        (np.asarray(["a", "b"]), np.asarray(["a", "b"]), True),
        (np.asarray(["1", "5", "3"]), np.asarray([1, 3, 5]), False),
    ],
)
def test_dtype_blind_comparison_matches_intent(saved, canonical, identical):
    """Documents why np.array_equal was the wrong comparison here."""
    normalized = [str(value) for value in saved] == [
        str(value) for value in canonical
    ]

    assert normalized is identical


class TestExclusionIsNotLeakage:
    """The training partition is routinely defined by removing validation."""

    @staticmethod
    def _findings(code: str) -> list[dict]:
        return RobustnessAgent._find_direct_leakage(code)

    def test_numpy_delete_of_validation_rows(self) -> None:
        code = (
            "y_tr = np.delete(y, val_idx, axis=0)\n"
            "model.fit(X[t_idx], y_tr[t_idx])\n"
        )

        assert self._findings(code) == []

    def test_negated_validation_mask(self) -> None:
        code = (
            "val_mask = folds == fold\n"
            "y_tr = y[~val_mask]\n"
            "model.fit(X[t_idx], y_tr[t_idx])\n"
        )

        assert self._findings(code) == []

    def test_setdiff_against_validation_indices(self) -> None:
        code = "t_idx = np.setdiff1d(all_idx, val_idx)\nmodel.fit(X[t_idx])\n"

        assert self._findings(code) == []

    def test_inequality_against_the_validation_fold(self) -> None:
        code = "t_idx = np.flatnonzero(folds != val_fold)\nmodel.fit(X[t_idx])\n"

        assert self._findings(code) == []

    def test_dropping_validation_rows(self) -> None:
        code = "train_df = df.drop(val_idx)\nmodel.fit(train_df)\n"

        assert self._findings(code) == []

    def test_selecting_the_validation_mask_is_still_leakage(self) -> None:
        """One character apart from the excluded form, and the opposite thing."""
        code = "model.fit(X[val_mask], y[val_mask])\n"

        assert self._findings(code)


class TestLeakageSeveritySeparatesFactFromInference:
    """Only evidence in the code's shape may fail a candidate by itself."""

    @staticmethod
    def _severity(code: str) -> str | None:
        findings = RobustnessAgent._find_direct_leakage(code)
        return findings[0]["severity"] if findings else None

    def test_concatenated_train_and_test_is_direct(self) -> None:
        code = "full = pd.concat([train_df, test_df])\nscaler.fit_transform(full)\n"

        assert self._severity(code) == "direct"

    def test_structural_taint_survives_intermediate_aliases(self) -> None:
        code = (
            "X_all = np.vstack([train_features, test_features])\n"
            "X_tr = X_all[:n_train]\n"
            "scaler.fit(X_tr)\n"
        )

        assert self._severity(code) == "direct"

    def test_fitting_the_held_out_name_is_direct(self) -> None:
        assert self._severity("scaler.fit(X_val)\n") == "direct"

    def test_name_chain_only_is_derived(self) -> None:
        code = "X_train = X[val_idx]\nscaler.fit_transform(X_train)\n"

        assert self._severity(code) == "derived"

    def test_alias_chain_is_derived(self) -> None:
        code = "X_val = X[vi]\nX_tr = X_val.copy()\nmodel.fit(X_tr)\n"

        assert self._severity(code) == "derived"


def _evidence_state(
    tmp_path: Path,
    *,
    name: str,
    problem_type: str,
    class_order: list[str],
    width: int,
    write_class_order: list[str] | None = None,
) -> dict:
    """Stage exactly the artifacts the injected helper writes for a component."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(exist_ok=True)

    train_ids = np.asarray(["a", "b", "c"])
    test_ids = np.asarray(["x", "y"])
    np.save(canonical_dir / "train_ids.npy", train_ids, allow_pickle=False)
    np.save(models_dir / f"oof_{name}.npy", np.zeros((3, width)))
    np.save(models_dir / f"test_{name}.npy", np.zeros((2, width)))
    np.save(models_dir / f"train_ids_{name}.npy", train_ids, allow_pickle=False)
    np.save(models_dir / f"test_ids_{name}.npy", test_ids, allow_pickle=False)
    if write_class_order is not None:
        np.save(
            models_dir / f"class_order_{name}.npy",
            np.asarray(write_class_order, dtype=str),
            allow_pickle=False,
        )

    return {
        "run_mode": "mlebench",
        "working_directory": str(tmp_path),
        "oof_availability": {name: True},
        "trusted_component_scores": {name: 0.9847},
        "canonical_contract": {
            "train_ids_path": str(canonical_dir / "train_ids.npy")
        },
        "submission_contract": {"class_order": class_order},
        "problem_type": problem_type,
    }


# Six independent labels vs four mutually exclusive classes: both are wide
# templates, only the second one has an order worth recording.
TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
DISEASE_CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]


class TestMultilabelEvidenceNeedsNoClassOrder:
    """Independent labels have no order, so none can be demanded of them."""

    def test_wide_multilabel_component_is_complete_evidence(self, tmp_path) -> None:
        state = _evidence_state(
            tmp_path,
            name="tfidf_linear_baseline",
            problem_type="multilabel_classification",
            class_order=TOXICITY_LABELS,
            width=6,
        )

        assert _mle_evidence_failures(state) == {}

    def test_multiclass_still_must_declare_its_column_order(self, tmp_path) -> None:
        state = _evidence_state(
            tmp_path,
            name="efficientnet_b0",
            problem_type="multiclass_classification",
            class_order=DISEASE_CLASSES,
            width=4,
        )

        assert _mle_evidence_failures(state) == {
            "efficientnet_b0": ["missing component-specific multiclass class order"]
        }

    def test_multiclass_with_matching_order_is_complete(self, tmp_path) -> None:
        state = _evidence_state(
            tmp_path,
            name="efficientnet_b0",
            problem_type="multiclass_classification",
            class_order=DISEASE_CLASSES,
            width=4,
            write_class_order=DISEASE_CLASSES,
        )

        assert _mle_evidence_failures(state) == {}

    def test_multiclass_with_permuted_order_is_still_caught(self, tmp_path) -> None:
        state = _evidence_state(
            tmp_path,
            name="efficientnet_b0",
            problem_type="multiclass_classification",
            class_order=DISEASE_CLASSES,
            width=4,
            write_class_order=["scab", "rust", "healthy", "multiple_diseases"],
        )

        assert _mle_evidence_failures(state) == {
            "efficientnet_b0": [
                "component class order does not match submission contract"
            ]
        }


class TestContractViolationsAreReportedTogether:
    """A repair that trades one contract for another must not cost an attempt.

    Observed on every multiclass component of one run: attempt 1 failed on
    ``omits class_order=``, attempt 2 on ``shadows an injected helper``, and
    only attempt 3 ran. Three generations for one program, because the
    pre-execution checks reported one violation at a time.
    """

    @staticmethod
    def _pre_execution_source() -> str:
        import inspect

        from kaggle_agents.agents.developer.agent import DeveloperAgent

        return inspect.getsource(DeveloperAgent)

    def test_every_violation_is_collected_before_reporting(self) -> None:
        source = self._pre_execution_source()

        assert "contract_violations: list[tuple[str, str]] = []" in source
        # An elif chain here means only the first violation is ever reported.
        assert "elif missing_class_order:" not in source
        assert "elif untrusted_helper_import:" not in source

    def test_violations_reach_the_fixer_as_one_message(self) -> None:
        """Consumers read errors[0]; a list would drop all but the first."""
        source = self._pre_execution_source()

        assert "errors=[combined_error]" in source
        assert "Fix all of them" in source


class TestRepeatingKeyCannotNameRows:
    """The canonical contract names each training row exactly once."""

    def test_duplicate_key_falls_back_to_row_position(self) -> None:
        # This competition's key is a pickup timestamp, and timestamps collide.
        frame = pd.DataFrame(
            {
                "key": [
                    "2014-03-30 12:14:00.000000128",
                    "2014-03-30 12:14:00.000000128",
                    "2011-08-18 00:35:00.00000049",
                ],
                "fare_amount": [7.5, 9.0, 5.7],
            }
        )

        aligned, id_col, synthetic = _ensure_id_column(frame, "key")

        assert (id_col, synthetic) == ("_row_id", True)
        assert aligned[id_col].is_unique

    def test_unique_key_is_kept(self) -> None:
        frame = pd.DataFrame({"key": ["a", "b", "c"], "y": [1, 2, 3]})

        _, id_col, synthetic = _ensure_id_column(frame, "key")

        assert (id_col, synthetic) == ("key", False)

    def test_absent_column_still_falls_back(self) -> None:
        frame = pd.DataFrame({"y": [1, 2, 3]})

        _, id_col, synthetic = _ensure_id_column(frame, "id")

        assert (id_col, synthetic) == ("_row_id", True)


class TestTerminalGatesFollowTheGradedMetric:
    """The terminal gates must apply the same metric-aware row-sum rule.

    The developer gate learned that rows are a distribution only under a
    likelihood metric; the SubmissionAgent's final validation and the
    ensemble's hash-verified snapshot restore kept the unconditional rule, so
    bytes accepted (and snapshotted) mid-run were rejected at the very end and
    the run graded as nothing. Shape: four one-hot prediction columns scored
    by column-wise AUC, sigmoid rows.
    """

    WIDE_COLS = ("label_a", "label_b", "label_c", "label_d")

    @classmethod
    def _unnormalized_submission(cls, tmp_path: Path) -> tuple[Path, Path]:
        header = "row_id," + ",".join(cls.WIDE_COLS)
        sample = tmp_path / "sample_submission.csv"
        sample.write_text(
            f"{header}\nr0,0.25,0.25,0.25,0.25\nr1,0.25,0.25,0.25,0.25\n",
            encoding="utf-8",
        )
        submission = tmp_path / "submission.csv"
        submission.write_text(
            f"{header}\nr0,0.9,0.8,0.1,0.2\nr1,0.3,0.9,0.4,0.8\n",
            encoding="utf-8",
        )
        return submission, sample

    def test_submission_agent_accepts_sigmoid_rows_under_ranking_metric(
        self, tmp_path
    ) -> None:
        submission, sample = self._unnormalized_submission(tmp_path)

        is_valid, message = SubmissionAgent()._validate_submission(
            submission,
            sample,
            problem_type="multiclass_classification",
            metric_name="auc",
            target_cols=list(self.WIDE_COLS),
        )

        assert is_valid, message

    def test_submission_agent_still_rejects_under_likelihood_metric(
        self, tmp_path
    ) -> None:
        submission, sample = self._unnormalized_submission(tmp_path)

        is_valid, message = SubmissionAgent()._validate_submission(
            submission,
            sample,
            problem_type="multiclass_classification",
            metric_name="multi class log loss",
            target_cols=list(self.WIDE_COLS),
        )

        assert is_valid is False
        assert "sum to 1.0" in message

    def test_snapshot_restore_does_not_relitigate_row_sums(
        self, tmp_path
    ) -> None:
        """A hash-verified snapshot was validated when it was accepted; the
        restore boundary checks structure and bytes, not quality rules."""
        submission, sample = self._unnormalized_submission(tmp_path)
        destination = tmp_path / "restored" / "submission.csv"

        restored = safe_restore_submission(
            submission,
            destination,
            sample,
            target_cols=list(self.WIDE_COLS),
            problem_type="multiclass",
            expected_sha256=sha256_file(submission),
            require_hash=True,
        )

        assert restored is True
        assert destination.read_bytes() == submission.read_bytes()


class TestRowSumContractFollowsTheGradedMetric:
    """Rows are a distribution only when the metric reads them as one."""

    @staticmethod
    def _artifacts(tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Per-class sigmoid outputs: valid predictions, rows do not sum to 1.
        np.save(
            models_dir / "oof_model.npy",
            np.array([[0.8, 0.3], [0.4, 0.6], [0.1, 0.9]]),
        )
        np.save(
            models_dir / "test_model.npy",
            np.array([[0.7, 0.3], [0.9, 0.8]]),
        )

    @pytest.mark.parametrize(
        ("metric", "reads_rows"),
        [
            ("multi class log loss", True),
            ("log_loss", True),
            ("cross_entropy", True),
            ("auc", False),
            ("roc_auc", False),
            ("accuracy", False),
            ("rmse", False),
            ("", False),
        ],
    )
    def test_metric_classification(self, metric: str, reads_rows: bool) -> None:
        assert metric_reads_rows_as_distribution(metric) is reads_rows

    def test_ranking_metric_reports_but_does_not_reject(self, tmp_path) -> None:
        self._artifacts(tmp_path)

        result = validate_model_artifacts(
            tmp_path,
            "model",
            expected_n_train=3,
            expected_n_test=2,
            expected_class_order=["negative", "positive"],
            problem_type="multiclass_classification",
            config=StrictValidationConfig(
                strict_mode=True, require_normalized_rows=False
            ),
        )

        assert result.is_valid is True
        assert any("do not sum to 1.0" in warning for warning in result.warnings)

    def test_likelihood_metric_still_rejects(self, tmp_path) -> None:
        self._artifacts(tmp_path)

        result = validate_model_artifacts(
            tmp_path,
            "model",
            expected_n_train=3,
            expected_n_test=2,
            expected_class_order=["negative", "positive"],
            problem_type="multiclass_classification",
            config=StrictValidationConfig(
                strict_mode=True, require_normalized_rows=True
            ),
        )

        assert result.is_valid is False
        assert any(
            "probability-sum contract" in error for error in result.errors
        )
