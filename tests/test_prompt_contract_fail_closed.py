"""Regression tests for fail-closed developer prompt contracts."""

import inspect

from kaggle_agents.prompts.templates.builders.cv import (
    build_cv_instructions,
    build_multi_seed_instructions,
    build_oof_hygiene_instructions,
    build_stacking_oof_instructions,
)
from kaggle_agents.prompts.templates.builders.image_model import (
    build_image_model_instructions,
)
from kaggle_agents.prompts.templates.builders.model import (
    build_model_component_instructions,
)
from kaggle_agents.prompts.templates.builders.optuna import (
    build_optuna_tuning_instructions,
)
from kaggle_agents.prompts.templates.constraints.base import BASE_CONSTRAINTS
from kaggle_agents.prompts.templates.constraints.audio import AUDIO_CONSTRAINTS
from kaggle_agents.prompts.templates.constraints.image import IMAGE_CONSTRAINTS
from kaggle_agents.prompts.templates.constraints.loader import (
    get_constraints_for_domain,
)
from kaggle_agents.prompts.templates.constraints.nlp import NLP_CONSTRAINTS
from kaggle_agents.prompts.templates.constraints.tabular import TABULAR_CONSTRAINTS
from kaggle_agents.prompts.templates.developer.component_guidance import (
    COMPONENT_GUIDANCE,
)


def test_base_requires_canonical_folds_instead_of_a_global_splitter() -> None:
    assert "CANONICAL_FOLDS_PATH" in BASE_CONSTRAINTS
    assert "CANONICAL_TRAIN_IDS_PATH" in BASE_CONSTRAINTS
    assert "NEVER create a new KFold/StratifiedKFold/GroupKFold" in BASE_CONSTRAINTS
    assert "Use StratifiedKFold:" not in BASE_CONSTRAINTS


def test_base_rejects_nonfinite_predictions_instead_of_imputing_them() -> None:
    assert "np.nan_to_num" not in BASE_CONSTRAINTS
    assert "candidate is invalid" in BASE_CONSTRAINTS
    assert "replacing with 0.5" not in BASE_CONSTRAINTS
    assert "raise ValueError" in BASE_CONSTRAINTS


def test_component_guidance_uses_canonical_target_and_submission_roles() -> None:
    guidance = COMPONENT_GUIDANCE["model"]

    assert "TARGET_COL" in guidance
    assert "submission_format_info" in guidance
    assert "train_df.columns[0]" not in guidance
    assert "sample_sub.columns[0]" not in guidance
    assert "sample_sub.columns[1:]" not in guidance
    assert "StratifiedKFold CV" not in guidance


def test_tabular_prompt_requires_exact_rows_and_wide_prediction_shape() -> None:
    assert "len(train_engineered) != len(train_original)" in TABULAR_CONSTRAINTS
    assert "changed canonical row order" in TABULAR_CONSTRAINTS
    assert "len(train_original) * 0.95" not in TABULAR_CONSTRAINTS
    assert "predictions.shape !=" in TABULAR_CONSTRAINTS
    assert "replicate for each column" not in TABULAR_CONSTRAINTS


def test_tabular_regression_bounds_require_an_explicit_contract() -> None:
    assert "explicit_bounds=None" in TABULAR_CONSTRAINTS
    assert "Never derive clipping bounds from a target" in TABULAR_CONSTRAINTS
    assert "taxi fares" not in TABULAR_CONSTRAINTS
    assert "prices" not in TABULAR_CONSTRAINTS


def test_optuna_objective_errors_are_not_converted_to_scores() -> None:
    text = "\n".join(build_optuna_tuning_instructions())

    assert "NEVER return 0.0" in text
    assert "raise optuna.TrialPruned()" in text
    assert "marks the trial FAIL" in text
    assert "if none exist, raise RuntimeError" in text
    assert "on exception log and return 0.0" not in text


def test_image_builder_preserves_every_canonical_record_and_fold() -> None:
    text = "\n".join(
        build_image_model_instructions(
            is_image_to_image=True,
            data_files={},
            suggested_epochs=10,
        )
    )

    assert "CANONICAL_FOLDS" in text
    assert "every canonical fold" in text
    assert "Unresolved canonical image IDs" in text
    assert "Missing paired targets" in text
    assert "model.state_dict()" in text
    assert "Dataset.ignore_errors()," not in text
    assert "df = df[df['image_path'].notna()]" not in text
    assert "prefer 1 holdout split" not in text
    assert "Use 1-2 CV folds" not in text
    assert "torch.save(model" not in text


def test_image_constraints_fail_on_missing_paths_or_decodes() -> None:
    assert "raise FileNotFoundError" in IMAGE_CONSTRAINTS
    assert "assert_cardinality" in IMAGE_CONSTRAINTS
    assert "model.state_dict()" in IMAGE_CONSTRAINTS
    assert "load_state_dict" in IMAGE_CONSTRAINTS
    assert "dataset.apply(tf.data.Dataset.ignore_errors())" not in IMAGE_CONSTRAINTS
    assert "return candidates[0]" not in IMAGE_CONSTRAINTS


def test_image_to_image_constraints_do_not_include_resize_batch_conflicts() -> None:
    text = get_constraints_for_domain("image_to_image")

    assert "For validation/inference use\n`batch_size=1`" in text
    assert "Use the SAME fixed-size Resize" not in text
    assert "NEVER use batch_size=1" not in text
    assert "batch_size=128" not in text


def test_nlp_naive_bayes_examples_remain_fold_local() -> None:
    assert "fit_transform(texts)" not in NLP_CONSTRAINTS
    assert "model.fit(X_tfidf, y)" not in NLP_CONSTRAINTS
    assert "fit_transform(X_tfidf)" not in NLP_CONSTRAINTS
    assert "iter_canonical_cv_splits()" in NLP_CONSTRAINTS
    assert "CANONICAL_Y[train_idx]" in NLP_CONSTRAINTS


def test_audio_constraints_use_only_injected_output_helpers() -> None:
    assert "save_component_artifacts(" in AUDIO_CONSTRAINTS
    assert "write_submission(" in AUDIO_CONSTRAINTS
    assert "class_order=" in AUDIO_CONSTRAINTS
    assert "oof_ids_{COMPONENT_NAME}" not in AUDIO_CONSTRAINTS
    assert "`models/oof_{COMPONENT_NAME}.npy`" not in AUDIO_CONSTRAINTS
    assert "`submission.csv` with" not in AUDIO_CONSTRAINTS


def test_multiseed_and_stacking_prompts_require_multiclass_order() -> None:
    multiseed = "\n".join(build_multi_seed_instructions("/tmp/run", "model"))
    stacking = "\n".join(build_stacking_oof_instructions("/tmp/run", "model"))

    assert "class_order=class_order" in multiseed
    assert "TARGET_TYPE == 'single'" in multiseed
    assert "class_order file not found" not in stacking
    assert "raise ValueError" in stacking


def test_fallbacks_call_submission_helper_instead_of_describing_manual_write() -> None:
    from kaggle_agents.agents.planner_agent import (
        PlannerAgent as LegacyPlannerAgent,
    )
    from kaggle_agents.agents.planner.fallback_plans.tabular import (
        _FAST_COMPONENT_DEFS,
    )

    image_source = inspect.getsource(
        LegacyPlannerAgent._create_image_fallback_plan
    )
    stacking_outline = _FAST_COMPONENT_DEFS["stacking_ensemble"]["code_outline"]

    assert "Write submission.csv" not in image_source
    assert "write submission.csv" not in image_source
    assert "write_submission(" in image_source
    assert "Write submission.csv" not in stacking_outline
    assert "write_submission(" in stacking_outline


def test_base_deadline_examples_keep_canonical_folds_and_state_dicts() -> None:
    assert "iter_canonical_cv_splits()" in BASE_CONSTRAINTS
    assert "CANONICAL_FOLDS != fold_idx" in BASE_CONSTRAINTS
    assert "cannot be reconstructed" in BASE_CONSTRAINTS
    assert "model.state_dict()" in BASE_CONSTRAINTS
    assert "kfold.split" not in BASE_CONSTRAINTS
    assert "torch.save(model," not in BASE_CONSTRAINTS
    assert "until every row selected by" in BASE_CONSTRAINTS
    assert "CANONICAL_OOF_ELIGIBLE_MASK" in BASE_CONSTRAINTS


def test_model_builder_has_no_positional_submission_id_pattern() -> None:
    source = inspect.getsource(build_model_component_instructions)

    assert "sample_sub.iloc[:, 0]" not in source
    assert "sample_sub.columns[0]" not in source
    assert "sample_sub[sample_sub.columns[1]]" not in source
    assert "submission_format_info" in source


def test_image_path_header_falls_back_to_existing_image_directory() -> None:
    from kaggle_agents.agents.developer.code_generator import CodeGeneratorMixin

    source = inspect.getsource(CodeGeneratorMixin)

    assert (
        "TRAIN_PATH = TRAIN_CSV_PATH if TRAIN_CSV_PATH.exists() else TRAIN_IMG_DIR"
        in source
    )
    assert (
        'TRAIN_PATH = TRAIN_CSV_PATH if TRAIN_CSV_PATH.exists() else Path("{working_dir}/train.csv")'
        not in source
    )


def test_cv_prompts_reject_partial_oof_without_misclassifying_zero() -> None:
    cv_text = "\n".join(build_cv_instructions("/work", "candidate"))
    stacking_text = "\n".join(
        build_stacking_oof_instructions("/work", "candidate")
    )
    hygiene_text = "\n".join(
        build_oof_hygiene_instructions("/work", "candidate")
    )
    seed_text = "\n".join(build_multi_seed_instructions("/work", "candidate"))

    assert "Initialize OOF with NaN" in cv_text
    assert "compute log_loss on rows with sum>0" not in cv_text
    assert "Zero is a" in stacking_text
    assert "raise ValueError('OOF-eligible rows contain NaN or Inf')" in stacking_text
    assert "Temporal warm-up OOF rows must remain NaN" in stacking_text
    assert "never drop, duplicate, or reorder" in hygiene_text.lower()
    assert "folds.csv" not in hygiene_text
    assert "canonical/folds.npy" not in seed_text
    assert "canonical_dir / 'folds.npy'" in seed_text
