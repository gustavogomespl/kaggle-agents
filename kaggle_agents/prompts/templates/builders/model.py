"""
Model component and dynamic instruction builders.
"""

from __future__ import annotations

import os

from ....core.config import is_metric_minimization
from .budget import build_budget_instructions, build_mlebench_objective_instructions
from .cv import build_cv_instructions, build_stacking_oof_instructions
from .ensemble import build_ensemble_instructions
from .feature_eng import build_feature_engineering_instructions
from .image_model import build_image_model_instructions
from .optuna import build_optuna_tuning_instructions


def build_iteration_context(current_iteration: int, refinement_guidance: dict) -> list[str]:
    """Build iteration context instructions."""
    instructions = []

    if current_iteration > 0:
        instructions.append(f"\n⚡ REFINEMENT ITERATION {current_iteration}")
        instructions.append("Focus on improvements that address previous shortcomings.")

    if refinement_guidance and refinement_guidance.get("developer_guidance"):
        instructions.append("\nMETA-EVALUATOR GUIDANCE:")
        instructions.append(f"  {refinement_guidance['developer_guidance']}")

    if refinement_guidance and refinement_guidance.get("priority_fixes"):
        instructions.append("\nAVOID THESE ERROR PATTERNS:")
        for error in refinement_guidance["priority_fixes"][:3]:
            instructions.append(f"  - {error}")

    return instructions


def build_previous_results_context(dev_results: list) -> list[str]:
    """Build context from previous development results."""
    if not dev_results:
        return []

    instructions = []
    successful_components = [r for r in dev_results if r.success]
    failed_components = [r for r in dev_results if not r.success]

    if successful_components:
        instructions.append("\n✅ SUCCESSFUL PATTERNS FROM PREVIOUS COMPONENTS:")
        for result in successful_components[-2:]:
            if "LightGBM" in result.code:
                instructions.append("  - LightGBM implementation worked well")
            if "StratifiedKFold" in result.code:
                instructions.append("  - StratifiedKFold cross-validation successful")
            if "predict_proba" in result.code:
                instructions.append("  - predict_proba() for probabilities confirmed working")

    if failed_components:
        instructions.append("\nAVOID THESE ERRORS FROM PREVIOUS ATTEMPTS:")
        for result in failed_components[-2:]:
            if result.errors:
                error_msg = result.errors[0][:300]
                instructions.append(f"  - {error_msg}")

    return instructions


def build_performance_gap_instructions(
    current_score: float,
    target_score: float | None,
    metric_name: str,
) -> list[str]:
    """Build performance gap instructions."""
    if current_score <= 0 or target_score is None:
        return []

    minimize = is_metric_minimization(metric_name)
    gap = (
        (float(current_score) - float(target_score))
        if minimize
        else (float(target_score) - float(current_score))
    )
    if gap <= 0:
        return []
    instructions = [
        f"\nPERFORMANCE GAP: {gap:.4f} to reach target ({float(target_score):.4f}, {'minimize' if minimize else 'maximize'})"
    ]

    if gap < 0.01:
        instructions.append("  - Small gap: Focus on fine-tuning hyperparameters")
    elif gap < 0.05:
        instructions.append("  - Medium gap: Consider feature engineering or ensemble methods")
    else:
        instructions.append("  - Large gap: May need different model architecture or approach")

    return instructions


def build_standard_requirements() -> list[str]:
    """Build standard requirements."""
    return [
        "\nSTANDARD REQUIREMENTS:",
        "  - Save models to models/ directory",
        "  - Print progress and metrics throughout execution",
        "  - NO sys.exit() or exit() calls",
        "  - CRITICAL: Do NOT use deprecated 'pandas.append()'. Use 'pd.concat()' instead.",
        "  - Complete, executable single-file Python program",
    ]


def build_model_component_instructions(
    component,
    state: dict,
    working_dir: str,
    is_image: bool,
    is_audio: bool,
    is_image_to_image: bool,
    is_classification: bool,
    sample_integer_labels: bool,
    target_col: str = "target",
    suggested_epochs: int = 600,
    early_stopping_patience: int = 30,
) -> list[str]:
    """Build model component instructions with adaptive epoch budget and patience (SOTA pattern)."""
    instructions = [
        "\nMODEL COMPONENT REQUIREMENTS:",
        "  - MUST train a model and generate predictions",
    ]

    if is_image_to_image:
        instructions.extend(
            [
                "  - MUST train on (noisy -> clean) image pairs and output FULL images (H x W), NOT a single scalar.",
                "  - MUST write pixel-level submission.csv matching sample_submission (id format: image_row_col).",
                "  - Use an encoder-decoder (U-Net/autoencoder). DO NOT use a classifier head or global pooling.",
            ]
        )
    elif is_classification:
        if sample_integer_labels:
            instructions.append(
                "  - MUST create submission.csv with integer class labels (0..K-1) matching sample_submission"
            )
        else:
            instructions.append(
                "  - MUST create submission.csv with probability predictions (0.0-1.0)"
            )
        instructions.extend(
            [
                "  - If sample_submission has >2 target columns, determine multi-class vs multi-label using train labels:",
                "    - Multi-class: exactly one positive per row -> softmax",
                "    - Multi-label: multiple positives per row -> sigmoid per class (no normalization)",
                "  - For multi-class log_loss: probabilities must sum to 1 per row (clip to [1e-15, 1-1e-15] then renormalize)",
                "  - If using logits (TF/PyTorch), apply softmax BEFORE log_loss and submission",
                "  - For log_loss: avoid overconfidence (label_smoothing / calibration) and clip probabilities",
                "  - Map class index order to sample_submission columns (do NOT sort labels independently)",
            ]
        )
    else:
        instructions.append("  - MUST create submission.csv with numeric predictions (regression)")

    instructions.append(
        "  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target')"
    )

    if not is_image:
        instructions.extend(
            [
                f"  - CRITICAL: Use target_col from dataset info (target_col='{target_col}' if available)",
                "  - CRITICAL: MUST encode categorical features (object/category dtypes) using ColumnTransformer + OneHotEncoder",
                "  - CRITICAL: Never pass raw categorical strings to LightGBM/XGBoost/sklearn (will fail with 'could not convert string to float')",
                "  - CatBoost is the ONLY exception that handles categorical features natively",
                "  - Use OneHotEncoder(handle_unknown='ignore', sparse_output=False) (NOT sparse=...)",
                "  - If X has 0 real features after preprocessing, STOP with a clear error (do NOT create dummy features)",
            ]
        )
    else:
        data_files = state.get("data_files", {}) if state else {}
        instructions.extend(
            build_image_model_instructions(
                is_image_to_image, data_files, suggested_epochs, early_stopping_patience
            )
        )
        if is_audio:
            instructions.extend(
                [
                    "\n🔊 AUDIO MODELLING (CLASSIFICATION/REGRESSION):",
                    "  - Convert audio to log-mel spectrograms (librosa) and treat as image inputs",
                    "  - Use fixed duration: pad/trim to a consistent length per sample",
                    "  - Ensure consistent sample rate (e.g., 32k or 44.1k) for all files",
                    "  - Cache spectrograms to disk if repeated epochs to avoid recompute",
                ]
            )
        if not is_image_to_image:
            train_csv_path = data_files.get("train_csv", "") if isinstance(data_files, dict) else ""
            instructions.extend(
                [
                    "  - CRITICAL: This is an image competition. Do NOT use tabular models unless you have real numeric features.",
                    "    - If train.csv only has id+label (<=2 cols), you MUST train an image model (CNN/transformer) or add an embedding extractor first.",
                ]
            )
            if train_csv_path:
                instructions.append(f"  - Train CSV path (check columns): {train_csv_path}")

    # Add CV and OOF instructions
    component_name = getattr(component, "name", "component")
    instructions.extend(build_cv_instructions(working_dir, component_name))
    instructions.extend(build_stacking_oof_instructions(working_dir, component_name))

    # Add submission format instructions (CRITICAL for CV vs public score match)
    instructions.extend(
        [
            "\n⚠️ SUBMISSION FORMAT (CRITICAL - SEE HARD_CONSTRAINTS):",
            "  - ALWAYS read sample_submission.csv and use its columns and order",
            "  - If sample has 2 columns: fill sample.columns[1] only",
            "  - If sample has >2 columns: fill ALL target columns (columns[1:]) in order",
            "  - Keep ID column values and order exactly as in sample_submission",
            "  - DO NOT add/drop columns or reorder rows",
        ]
    )

    return instructions


def build_dynamic_instructions(
    component,
    state: dict,
    config,
    working_dir: str,
) -> str:
    """
    Build dynamic instructions based on current state (MLE-STAR pattern).

    Creates context-aware guidance by analyzing:
    - Previous component results (what worked/failed)
    - Current iteration number (more specific in later iterations)
    - Performance trends
    - Common error patterns

    Args:
        component: Component being implemented
        state: Current workflow state
        config: Agent configuration
        working_dir: Working directory path

    Returns:
        Dynamic instructions string
    """
    instructions = []

    instructions.append(f"Implement {component.component_type}: {component.name}")

    run_mode = str(state.get("run_mode", "")).lower()
    objective = str(state.get("objective", "")).lower()
    domain = str(state.get("domain_detected", state.get("domain", "tabular"))).lower()
    submission_format_type = str(state.get("submission_format_type") or "").lower()
    is_audio = domain.startswith("audio")
    is_image = domain.startswith("image") or domain in {"computer_vision", "vision"} or is_audio
    is_image_to_image = domain == "image_to_image" or submission_format_type == "pixel_level"

    # Detect problem type
    problem_type = ""
    try:
        comp_info = state.get("competition_info")
        problem_type = comp_info.problem_type if comp_info else ""
    except Exception:
        problem_type = ""
    is_classification = "class" in str(problem_type).lower()

    # Check sample submission for integer labels
    sample_integer_labels = False
    sample_submission_path = state.get("sample_submission_path")
    if sample_submission_path:
        try:
            import numpy as np
            import pandas as pd

            sample_sub = pd.read_csv(sample_submission_path)
            if sample_sub.shape[1] >= 2:
                sample_vals = sample_sub.iloc[:, 1]
                if pd.api.types.is_numeric_dtype(sample_vals):
                    vals = sample_vals.to_numpy()
                    if vals.size:
                        sample_integer_labels = np.allclose(vals, np.round(vals))
        except Exception:
            sample_integer_labels = False

    # Get timeout hint
    timeout_hint = state.get("timeout_per_component")
    if not isinstance(timeout_hint, (int, float)):
        try:
            timeout_hint = int(timeout_hint) if timeout_hint is not None else None
        except Exception:
            timeout_hint = None

    target_col = state.get("target_col", "target")
    current_iteration = state.get("current_iteration", 0)
    refinement_guidance = state.get("refinement_guidance", {})
    dev_results = state.get("development_results", [])
    current_score = state.get("current_performance_score", 0.0)
    target_score = state.get("target_score")

    if isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = None

    competition_info = state.get("competition_info")
    metric_name = competition_info.evaluation_metric if competition_info else ""

    if metric_name:
        metric_lower = str(metric_name).lower()
        instructions.extend(
            [
                "\n📏 METRIC REQUIREMENT (CONSISTENT EVALUATION):",
                f"  - Use competition metric '{metric_name}' for Final Validation Performance",
            ]
        )
        if is_classification and ("log" in metric_lower or "loss" in metric_lower):
            instructions.extend([
                "  - For log_loss metrics: compute log_loss on OOF predictions (clip + renormalize)",
                "  - CRITICAL: Final Validation Performance MUST be log_loss value (NOT accuracy/AUC)",
                "  - Lower is better: 0.02 = excellent, 0.7+ = nearly random for multiclass",
                "  - Use: `from sklearn.metrics import log_loss; score = log_loss(y_true, oof_preds)`",
            ])

    # Build budget instructions
    instructions.extend(build_budget_instructions(timeout_hint))

    # Build MLE-bench instructions if applicable
    if run_mode == "mlebench" or "medal" in objective:
        instructions.extend(build_mlebench_objective_instructions())

    # Build iteration context
    instructions.extend(build_iteration_context(current_iteration, refinement_guidance))

    # Build refinement guidance
    if refinement_guidance and "component_type_guidance" in refinement_guidance:
        comp_guidance = refinement_guidance["component_type_guidance"].get(component.component_type)
        if comp_guidance:
            instructions.append(f"\n🎯 {component.component_type.upper()} SPECIFIC GUIDANCE:")
            instructions.append(f"  {comp_guidance}")

    # Build previous results context
    instructions.extend(build_previous_results_context(dev_results))

    # Build performance gap instructions
    instructions.extend(
        build_performance_gap_instructions(current_score, target_score, metric_name)
    )

    # Get adaptive epoch budget and patience from state (SOTA pattern)
    epoch_budget = int(state.get("epoch_budget", 600))  # SOTA uses 600
    early_stopping_patience = int(state.get("early_stopping_patience", 30))  # SOTA uses 30
    min_epochs = int(os.getenv("KAGGLE_AGENTS_MIN_EPOCHS", "5"))

    # Check if last run timed out and reduce epochs
    suggested_epochs = epoch_budget
    if dev_results:
        last_result = dev_results[-1]
        last_stdout = str(getattr(last_result, "stdout", "") or "").lower()
        last_stderr = str(getattr(last_result, "stderr", "") or "").lower()
        last_exec_time = getattr(last_result, "execution_time", 0) or 0
        timeout_component = timeout_hint or 3600

        timed_out = (
            "timeout" in last_stderr
            or "deadline" in last_stdout
            or "[timeout]" in last_stdout
            or last_exec_time >= timeout_component * 0.95
        )
        if timed_out:
            reduction_factor = float(os.getenv("KAGGLE_AGENTS_EPOCH_REDUCTION", "0.5"))
            suggested_epochs = max(min_epochs, int(epoch_budget * reduction_factor))

    # Component-type specific instructions
    if component.component_type == "model":
        instructions.extend(
            build_model_component_instructions(
                component=component,
                state=state,
                working_dir=working_dir,
                is_image=is_image,
                is_audio=is_audio,
                is_image_to_image=is_image_to_image,
                is_classification=is_classification,
                sample_integer_labels=sample_integer_labels,
                target_col=target_col,
                suggested_epochs=suggested_epochs,
                early_stopping_patience=early_stopping_patience,
            )
        )

        # Optuna instructions if component name suggests tuning
        name_lower = component.name.lower()
        if "optuna" in name_lower or "tuned" in name_lower or "optimized" in name_lower:
            n_trials = (
                getattr(getattr(config, "ablation", None), "optuna_trials", 5) if config else 5
            )
            timeout = (
                getattr(getattr(config, "ablation", None), "testing_timeout", 600)
                if config
                else 600
            ) - 60
            instructions.extend(build_optuna_tuning_instructions(n_trials, timeout))

    elif component.component_type == "feature_engineering":
        instructions.extend(build_feature_engineering_instructions())

    elif component.component_type == "ensemble":
        instructions.extend(build_ensemble_instructions(target_col))

    # Standard requirements
    instructions.extend(build_standard_requirements())

    return "\n".join(instructions)
