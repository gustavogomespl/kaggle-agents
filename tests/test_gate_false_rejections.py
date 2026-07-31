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
"""

from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.agents.robustness_agent import RobustnessAgent
from kaggle_agents.utils.oof_validation import assert_oof_sanity


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
