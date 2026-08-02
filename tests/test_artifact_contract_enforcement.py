"""Execution-time artifact contract must match the promotion-time contract.

The trusted-OOF gate needs ``train_ids_<name>.npy`` to verify row alignment;
if the executor does not enforce it, components train successfully and then
die silently at promotion with no retry loop (aerial-cactus smoke run).
"""

from pathlib import Path

import numpy as np
import pytest

from kaggle_agents.agents.developer.agent import (
    _expected_model_artifacts,
    _has_combinable_model_predictions,
    _model_validation_problem_type,
    _oof_artifact_digest,
    _requires_class_order_artifact,
    _validation_class_order_for_state,
)
from kaggle_agents.agents.developer.code_contracts import (
    missing_class_order_helper_argument,
    untrusted_contract_helper_import,
)
from kaggle_agents.agents.developer.code_generator import (
    _PROBABILITY_VALIDATION_HELPER,
    _probability_validation_helper_for_component,
)
from kaggle_agents.agents.developer.retry import (
    RetryMixin,
    _maybe_add_artifact_hint,
)
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    DevelopmentResult,
)
from kaggle_agents.prompts.templates.constraints.base import BASE_CONSTRAINTS


class _Retry(RetryMixin):
    pass


@pytest.fixture(autouse=True)
def _default_oof_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_AGENTS_REQUIRE_OOF", raising=False)


@pytest.fixture
def dense_header() -> str:
    return (
        _probability_validation_helper_for_component("model", False)
        + "\n# === END PATH CONSTANTS ==="
    )


def test_injected_probability_helper_accepts_finite_binary_predictions(
    dense_header: str,
) -> None:
    namespace: dict = {}
    exec(compile(dense_header, "<dense-header>", "exec"), namespace)

    validated = namespace["validate_probabilities"](
        np.asarray([0.0, 0.25, 1.0]),
        expected_rows=3,
        is_multiclass=False,
        name="Binary",
    )

    np.testing.assert_allclose(validated, [0.0, 0.25, 1.0])
    assert validated.dtype == np.float64


@pytest.mark.parametrize(
    ("predictions", "expected_rows", "error"),
    [
        ([0.2, 0.8], 3, "row mismatch"),
        ([0.2, np.nan], 2, "NaN/Inf values; candidate is invalid"),
        ([0.2, np.inf], 2, "NaN/Inf values; candidate is invalid"),
    ],
)
def test_injected_probability_helper_rejects_invalid_predictions(
    dense_header: str,
    predictions: list[float],
    expected_rows: int,
    error: str,
) -> None:
    namespace: dict = {}
    exec(compile(dense_header, "<dense-header>", "exec"), namespace)

    with pytest.raises(ValueError, match=error):
        namespace["validate_probabilities"](
            predictions,
            expected_rows=expected_rows,
            is_multiclass=False,
            name="Candidate",
        )


class TestTemporalWarmupNaNContract:
    """The helper must accept the NaN the temporal contract REQUIRES.

    The injected header mandates that temporal warm-up OOF rows remain NaN
    and every host validator rejects anything else — while this helper's
    docstring says to call it on BOTH OOF and test and it raised on any NaN.
    A compliant temporal component therefore had no passing path. Temporal
    handling activates only for a full-length array that already carries NaN
    under a masked contract, so test predictions (all finite) and
    non-temporal runs (all-True mask) keep the exact old behavior.
    """

    @staticmethod
    def _helper(mask: np.ndarray | None):
        header = (
            _probability_validation_helper_for_component("model", False)
            + "\n# === END PATH CONSTANTS ==="
        )
        namespace: dict = {}
        exec(compile(header, "<dense-header>", "exec"), namespace)
        if mask is not None:
            namespace["CANONICAL_OOF_ELIGIBLE_MASK"] = mask
        return namespace["validate_probabilities"]

    def test_compliant_temporal_oof_is_accepted(self) -> None:
        mask = np.asarray([False, False, True, True, True])
        oof = np.asarray([np.nan, np.nan, 0.2, 0.7, 0.9])

        validated = self._helper(mask)(
            oof, expected_rows=5, is_multiclass=False, name="OOF"
        )

        assert np.isnan(validated[:2]).all()
        np.testing.assert_allclose(validated[2:], [0.2, 0.7, 0.9])

    def test_partially_filled_warmup_is_rejected_with_the_contract(self) -> None:
        mask = np.asarray([False, False, True, True, True])
        oof = np.asarray([0.5, np.nan, 0.2, 0.7, 0.9])

        with pytest.raises(ValueError, match="warm-up rows must remain NaN"):
            self._helper(mask)(
                oof, expected_rows=5, is_multiclass=False, name="OOF"
            )

    def test_nan_on_eligible_rows_is_still_invalid(self) -> None:
        mask = np.asarray([False, True, True])
        oof = np.asarray([np.nan, np.nan, 0.9])

        with pytest.raises(ValueError, match="NaN/Inf values"):
            self._helper(mask)(
                oof, expected_rows=3, is_multiclass=False, name="OOF"
            )

    def test_multiclass_normalization_skips_warmup_rows(self) -> None:
        mask = np.asarray([False, True, True])
        oof = np.asarray(
            [[np.nan, np.nan], [0.6, 0.6], [0.2, 0.2]]
        )

        validated = self._helper(mask)(
            oof, expected_rows=3, is_multiclass=True, name="OOF"
        )

        assert np.isnan(validated[0]).all()
        np.testing.assert_allclose(validated[1:].sum(axis=1), [1.0, 1.0])

    def test_all_true_mask_keeps_the_old_behavior(self) -> None:
        mask = np.ones(3, dtype=bool)

        with pytest.raises(ValueError, match="NaN/Inf values"):
            self._helper(mask)(
                np.asarray([0.1, np.nan, 0.9]),
                expected_rows=3,
                is_multiclass=False,
                name="OOF",
            )

    def test_finite_test_predictions_pass_under_a_temporal_mask(self) -> None:
        # n_test can coincide with n_train; all-finite predictions must not
        # be mistaken for a non-compliant OOF.
        mask = np.asarray([False, True, True])

        validated = self._helper(mask)(
            np.asarray([0.1, 0.5, 0.9]),
            expected_rows=3,
            is_multiclass=False,
            name="Test",
        )

        np.testing.assert_allclose(validated, [0.1, 0.5, 0.9])


@pytest.mark.parametrize("component_type", ["model", "ensemble"])
def test_dense_model_and_ensemble_headers_receive_probability_helper(
    component_type: str,
) -> None:
    assert (
        _probability_validation_helper_for_component(component_type, False)
        == _PROBABILITY_VALIDATION_HELPER
    )


_PROBABILITY_SHADOWING_FORMS = [
    "from candidate_validators import validate_probabilities\n",
    (
        "def validate_probabilities(*args, **kwargs):\n"
        "    return args[0]\n"
    ),
    "validate_probabilities = arbitrary_validator\n",
    "globals()['validate_probabilities'] = arbitrary_validator\n",
]


@pytest.mark.parametrize("shadow", _PROBABILITY_SHADOWING_FORMS)
def test_probability_helper_shadowing_is_rejected_only_when_injected(
    dense_header: str,
    shadow: str,
) -> None:
    finding = untrusted_contract_helper_import(dense_header + "\n" + shadow)

    assert finding is not None
    assert "validate_probabilities" in finding


@pytest.mark.parametrize(
    "harmless_import",
    [
        "from __main__ import validate_probabilities\n",
        "from __main__ import write_submission, save_component_artifacts\n",
        "from __main__ import write_submission as ws\n",
    ],
)
def test_importing_injected_helpers_from_main_is_not_shadowing(
    dense_header: str,
    harmless_import: str,
) -> None:
    """``from __main__ import <helper>`` rebinds the SAME injected objects.

    The detector checked only the imported names, never the module, so this
    no-op — a natural move for a model told the helpers "are already defined"
    — killed the attempt before execution. Observed burning an attempt in
    four separate benchmark runs.
    """
    finding = untrusted_contract_helper_import(
        dense_header + "\n" + harmless_import
    )

    assert finding is None


def test_importing_a_helper_from_any_other_module_is_still_shadowing(
    dense_header: str,
) -> None:
    finding = untrusted_contract_helper_import(
        dense_header + "\nfrom my_utils import write_submission\n"
    )

    assert finding is not None


@pytest.mark.parametrize("candidate_code", _PROBABILITY_SHADOWING_FORMS)
def test_packed_image_header_does_not_protect_absent_probability_helper(
    candidate_code: str,
) -> None:
    header = (
        _probability_validation_helper_for_component("model", True)
        + "\n# === END PATH CONSTANTS ==="
    )

    assert "def validate_probabilities(" not in header
    assert untrusted_contract_helper_import(header + "\n" + candidate_code) is None


def test_non_model_header_does_not_receive_probability_helper() -> None:
    assert (
        _probability_validation_helper_for_component("preprocessing", False)
        == ""
    )


def test_probability_prompt_calls_injected_helper_without_defining_it() -> None:
    assert "def validate_probabilities" not in BASE_CONSTRAINTS
    assert "oof_preds = validate_probabilities(" in BASE_CONSTRAINTS
    assert "test_preds = validate_probabilities(" in BASE_CONSTRAINTS
    assert "NaN/Inf values; candidate is invalid" in BASE_CONSTRAINTS
    assert "packed image-to-image" in BASE_CONSTRAINTS.lower()


def test_model_artifacts_require_train_ids_when_canonical_exists(
    tmp_path: Path,
) -> None:
    component = AblationComponent("candidate", "model", "train")

    expected = _expected_model_artifacts(component, tmp_path)
    assert expected == [
        "models/oof_candidate.npy",
        "models/test_candidate.npy",
    ]

    (tmp_path / "canonical").mkdir()
    (tmp_path / "canonical" / "metadata.json").write_text("{}", encoding="utf-8")
    expected = _expected_model_artifacts(component, tmp_path)
    assert "models/train_ids_candidate.npy" in expected
    assert "models/test_ids_candidate.npy" not in expected


def test_mlebench_model_artifacts_match_strict_validation(tmp_path: Path) -> None:
    # Strict post-acceptance validation requires train_ids AND test_ids in
    # mlebench mode regardless of canonical presence; the executor must
    # enforce the same set or components die after training with no retry.
    component = AblationComponent("candidate", "model", "train")

    expected = _expected_model_artifacts(component, tmp_path, "mlebench")
    assert expected == [
        "models/oof_candidate.npy",
        "models/test_candidate.npy",
        "models/train_ids_candidate.npy",
        "models/test_ids_candidate.npy",
    ]


@pytest.mark.parametrize(
    ("call", "missing"),
    [
        ("save_component_artifacts(oof, test, train_ids, test_ids)", True),
        (
            "save_component_artifacts("
            "oof, test, train_ids, test_ids, class_order=None)",
            True,
        ),
        (
            "save_component_artifacts("
            "oof, test, train_ids, test_ids, class_order=class_order)",
            False,
        ),
        (
            "save_component_artifacts("
            "oof, test, train_ids, test_ids, class_order)",
            False,
        ),
    ],
)
def test_multiclass_preflight_requires_concrete_class_order(
    call: str,
    missing: bool,
) -> None:
    code = (
        "def save_component_artifacts(*args, **kwargs):\n"
        "    pass\n\n"
        f"{call}\n"
    )

    assert missing_class_order_helper_argument(code) is missing


def test_mlebench_image_model_expects_only_packed_artifacts(tmp_path: Path) -> None:
    component = AblationComponent("candidate", "model", "train")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "metadata.json").write_text(
        '{"task_type": "image_to_image", "packed_image_contract": true}',
        encoding="utf-8",
    )

    assert _expected_model_artifacts(component, tmp_path, "mlebench") == [
        "models/oof_candidate.npz",
        "models/test_candidate.npz",
    ]


def test_image_problem_type_wins_over_generic_regression_declaration() -> None:
    state = {
        "domain_detected": "image_to_image",
        "problem_type": "regression",
        "competition_info": CompetitionInfo("demo", "", "rmse", "regression"),
    }

    assert _model_validation_problem_type(state) == "image_to_image"


def test_binary_single_probability_does_not_require_class_order_artifact() -> None:
    state = {
        "canonical_metadata": {"class_order": ["0", "1"]},
        "submission_contract": {"class_order": None},
    }

    assert (
        _validation_class_order_for_state(
            state,
            "binary_classification",
        )
        is None
    )
    assert _validation_class_order_for_state(
        state,
        "multiclass_classification",
    ) == ["0", "1"]

    wide_state = {
        "submission_contract": {
            "class_order": ["negative", "positive"],
        },
        "canonical_metadata": {"class_order": ["0", "1"]},
    }
    assert _validation_class_order_for_state(
        wide_state,
        "binary_classification",
    ) == ["negative", "positive"]
    assert _requires_class_order_artifact(
        wide_state,
        "binary_classification",
    )
    assert (
        _validation_class_order_for_state(
            wide_state,
            "multi_target_regression",
        )
        is None
    )
    assert not _requires_class_order_artifact(
        wide_state,
        "multi_target_regression",
    )


def test_multi_label_uses_wide_target_order_without_class_order_artifact() -> None:
    state = {
        "submission_contract": {
            "class_order": ["toxic", "insult", "threat"],
        },
        "canonical_metadata": {
            "class_order": ["toxic", "insult", "threat"],
        },
    }

    for problem_type in ("multi_label", "multi_label_classification"):
        assert _validation_class_order_for_state(
            state,
            problem_type,
        ) == ["toxic", "insult", "threat"]
        assert not _requires_class_order_artifact(
            state,
            problem_type,
        )


def test_packed_oof_digest_and_cached_reuse(tmp_path: Path) -> None:
    component = AblationComponent("candidate", "model", "train")
    canonical = tmp_path / "canonical"
    models = tmp_path / "models"
    canonical.mkdir()
    models.mkdir()
    (canonical / "metadata.json").write_text(
        '{"task_type": "image_to_image", "packed_image_contract": true}',
        encoding="utf-8",
    )
    np.savez(
        models / "oof_candidate.npz",
        values=np.array([0.0], dtype=np.float32),
        offsets=np.array([0, 1], dtype=np.int64),
        shapes=np.array([[1, 1]], dtype=np.int32),
        image_ids=np.array(["a.png"], dtype=str),
    )
    np.save(canonical / "train_ids.npy", np.array(["a.png"], dtype=str))
    np.save(canonical / "test_ids.npy", np.array(["b.png"], dtype=str))
    state = _cached_state(tmp_path, "print('candidate')")
    state["domain_detected"] = "image_to_image"

    assert _oof_artifact_digest(tmp_path, component.name) is not None
    assert _Retry()._should_skip_component(component, state) is None

    np.savez(
        models / "test_candidate.npz",
        values=np.array([0.5], dtype=np.float32),
        offsets=np.array([0, 1], dtype=np.int64),
        shapes=np.array([[1, 1]], dtype=np.int32),
        image_ids=np.array(["b.png"], dtype=str),
    )
    reused = _Retry()._should_skip_component(component, state)
    assert reused is not None
    assert reused.success is True


def test_non_model_components_have_no_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessing = AblationComponent("cache_images", "preprocessing", "resize")
    assert _expected_model_artifacts(preprocessing, tmp_path) is None

    monkeypatch.setenv("KAGGLE_AGENTS_REQUIRE_OOF", "0")
    model = AblationComponent("candidate", "model", "train")
    assert _expected_model_artifacts(model, tmp_path) is None


def _cached_state(tmp_path: Path, code: str) -> dict:
    return {
        "working_directory": str(tmp_path),
        "run_mode": "mlebench",
        "development_results": [
            DevelopmentResult(code=code, success=True, execution_time=1.0)
        ],
    }


def test_cached_model_reuse_requires_oof_artifact_on_disk(tmp_path: Path) -> None:
    # The aerial-cactus run re-planned "image_augmentation_preprocessing" as a
    # MODEL component; the name-keyed cache reused the old preprocessing run
    # (no OOF evidence) and the candidate died at promotion.
    component = AblationComponent("candidate", "model", "train")
    state = _cached_state(tmp_path, "print('candidate')")

    assert _Retry()._should_skip_component(component, state) is None

    (tmp_path / "models").mkdir()
    np.save(tmp_path / "models" / "oof_candidate.npy", np.array([0.1, 0.9]))
    assert _Retry()._should_skip_component(component, state) is None

    np.save(tmp_path / "models" / "test_candidate.npy", np.array([0.2]))
    np.save(
        tmp_path / "models" / "train_ids_candidate.npy",
        np.array(["a", "b"], dtype=str),
    )
    np.save(
        tmp_path / "models" / "test_ids_candidate.npy",
        np.array(["c"], dtype=str),
    )
    reused = _Retry()._should_skip_component(component, state)
    assert reused is not None
    assert reused.success is True


def test_cached_non_model_reuse_is_unchanged(tmp_path: Path) -> None:
    component = AblationComponent("cache_images", "preprocessing", "resize")
    state = _cached_state(tmp_path, "print('cache_images')")

    reused = _Retry()._should_skip_component(component, state)
    assert reused is not None


def test_ensemble_component_needs_combinable_predictions(tmp_path: Path) -> None:
    assert _has_combinable_model_predictions({}, tmp_path) is False
    assert (
        _has_combinable_model_predictions({"oof_availability": {"m": False}}, tmp_path)
        is False
    )
    assert (
        _has_combinable_model_predictions({"oof_availability": {"m": True}}, tmp_path)
        is True
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    np.save(models_dir / "rejected_oof_bad.npy", np.array([0.5]))
    assert _has_combinable_model_predictions({}, tmp_path) is False

    np.save(models_dir / "oof_good.npy", np.array([0.5]))
    assert _has_combinable_model_predictions({}, tmp_path) is True


def test_generic_ensemble_is_not_offered_packed_image_artifacts(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "oof_image_model.npz").write_bytes(b"packed")

    assert (
        _has_combinable_model_predictions(
            {
                "domain_detected": "image_to_image",
                "oof_availability": {"image_model": True},
            },
            tmp_path,
        )
        is False
    )


def test_missing_artifact_error_gets_reuse_hint() -> None:
    hinted = _maybe_add_artifact_hint(
        "Missing expected artifacts: models/test_candidate.npy"
    )
    assert "Do NOT retrain from scratch" in hinted
    assert "allow_pickle=False" in hinted
    assert "save_component_artifacts(" in hinted
    assert "np.save(path" not in hinted

    untouched = _maybe_add_artifact_hint("ValueError: shapes do not match")
    assert untouched == "ValueError: shapes do not match"


def test_fallback_prompt_receives_dynamic_contract_requirements() -> None:
    from kaggle_agents.core.state import CompetitionInfo
    from kaggle_agents.prompts.templates.builders.context import DynamicContext
    from kaggle_agents.prompts.templates.developer.prompt_composition import (
        compose_generate_prompt,
    )

    component = AblationComponent("candidate", "model", "train")
    prompt = compose_generate_prompt(
        component=component,
        competition_info=CompetitionInfo("demo", "", "auc", "classification"),
        paths={"output_dir": "."},
        context=DynamicContext(),
        requirements="DYNAMIC CONTRACT: use the injected helpers only.",
    )

    assert "DYNAMIC CONTRACT: use the injected helpers only." in prompt


def test_stacking_prompt_uses_declared_submission_target_roles() -> None:
    from kaggle_agents.prompts.templates.builders.cv import (
        build_stacking_oof_instructions,
    )

    prompt = "\n".join(build_stacking_oof_instructions(".", "candidate"))

    assert "SUBMISSION_TARGET_COLS" in prompt
    assert "columns[1:]" not in prompt
    assert "columns[1:4]" not in prompt


def test_tabular_constraints_never_derive_class_order_by_position() -> None:
    from kaggle_agents.prompts.templates.constraints.tabular import (
        TABULAR_CONSTRAINTS,
    )

    assert "submission_targets = list(SUBMISSION_TARGET_COLS)" in TABULAR_CONSTRAINTS
    assert "if len(submission_targets) > 1" in TABULAR_CONSTRAINTS
    assert "class_order = sample_sub.columns[1:]" not in TABULAR_CONSTRAINTS


def _write_csv(path: Path, ids: list, target: list | None = None) -> None:
    import pandas as pd

    data = {"id": ids}
    if target is not None:
        data["has_cactus"] = target
    pd.DataFrame(data).to_csv(path, index=False)


def test_canonical_string_train_ids_survive_unpickled_resave(tmp_path: Path) -> None:
    # String IDs from pandas arrive as object dtype; stored raw they poison
    # the whole artifact chain (np.save(..., allow_pickle=False) crashes and
    # pickled saves are refused by the trusted scorer).
    from kaggle_agents.utils.data_contract import prepare_canonical_data

    _write_csv(
        tmp_path / "train.csv",
        [f"img_{i}.jpg" for i in range(10)],
        [i % 2 for i in range(10)],
    )
    _write_csv(tmp_path / "test.csv", ["t_0.jpg", "t_1.jpg"])

    prepare_canonical_data(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        target_col="has_cactus",
        output_dir=tmp_path,
        id_col="id",
        n_folds=2,
    )

    saved = np.load(tmp_path / "canonical" / "train_ids.npy", allow_pickle=False)
    assert saved.dtype.kind == "U"
    # The exact operation generated code performs with the loaded IDs.
    np.save(tmp_path / "resaved.npy", saved, allow_pickle=False)


def test_canonical_integer_train_ids_keep_numeric_dtype(tmp_path: Path) -> None:
    from kaggle_agents.utils.data_contract import prepare_canonical_data

    _write_csv(tmp_path / "train.csv", list(range(10)), [i % 2 for i in range(10)])
    _write_csv(tmp_path / "test.csv", [100, 101])

    prepare_canonical_data(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        target_col="has_cactus",
        output_dir=tmp_path,
        id_col="id",
        n_folds=2,
    )

    saved = np.load(tmp_path / "canonical" / "train_ids.npy", allow_pickle=False)
    assert saved.dtype.kind in "iu"


def test_preamble_normalizes_legacy_object_train_ids() -> None:
    import inspect

    from kaggle_agents.agents.developer import code_generator

    src = inspect.getsource(code_generator)
    assert "CANONICAL_TRAIN_IDS.dtype == object" in src
