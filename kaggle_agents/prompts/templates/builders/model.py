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


# Constants for task type detection
CLASSIFICATION_METRICS = {
    "auc", "roc_auc", "roc-auc", "log_loss", "logloss",
    "accuracy", "f1", "precision", "recall", "mcc",
    "balanced_accuracy", "cohen_kappa", "gini", "f1_weighted",
    "f1_macro", "quadratic_weighted_kappa", "qwk",
}
REGRESSION_METRICS = {
    "rmse", "mse", "mae", "mape", "r2", "rmsle", "smape",
    "mean_squared", "mean_absolute", "medae", "msle",
}
CLASSIFICATION_KEYWORDS = {
    "classification", "classifier", "multiclass", "binary",
    "multi-class", "multi-label", "categorical",
}
REGRESSION_KEYWORDS = {
    "regression", "regressor", "continuous", "forecasting",
    "prediction", "estimation",
}


def _detect_is_classification(state: dict | None) -> bool | None:
    """
    Detect if task is classification using multiple sources.

    Priority order:
    1. canonical_metadata from state
    2. Load metadata.json from canonical path if not in state
    3. evaluation_metric (reliable signal)
    4. submission_format_type (single_col_regression vs proba_df)
    5. domain_detected (tabular_classification vs tabular_regression)
    6. problem_type string (expanded patterns)
    7. Return None (caller must handle - no unsafe default)

    Args:
        state: Workflow state dictionary

    Returns:
        True for classification, False for regression, None if undetermined
    """
    if state is None:
        return None

    # Step 1: Try canonical_metadata from state (most authoritative)
    canonical_metadata = state.get("canonical_metadata", {})
    if canonical_metadata:
        is_classification = canonical_metadata.get("is_classification")
        if is_classification is not None:
            return bool(is_classification)

    # Step 1.5: Load from canonical directory using working_directory
    # This bypasses state timing issues by reading directly from disk
    try:
        import json
        from pathlib import Path

        working_dir = state.get("working_directory")
        if working_dir:
            canonical_metadata_path = Path(working_dir) / "canonical" / "metadata.json"
            if canonical_metadata_path.exists():
                with open(canonical_metadata_path) as f:
                    metadata = json.load(f)
                    is_classification = metadata.get("is_classification")
                    if is_classification is not None:
                        print(f"[DEBUG] is_classification={is_classification} (from canonical/metadata.json)")
                        return bool(is_classification)
    except Exception:
        pass

    # Step 2: Load metadata.json from canonical path if available
    try:
        import json
        from pathlib import Path

        comp_info = state.get("competition_info")
        if comp_info and hasattr(comp_info, "data_files") and comp_info.data_files:
            for data_file in comp_info.data_files:
                data_path = Path(data_file)
                metadata_path = data_path.parent / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        is_classification = metadata.get("is_classification")
                        if is_classification is not None:
                            return bool(is_classification)
                    break
    except Exception:
        pass

    # Step 3: Try evaluation_metric (reliable signal)
    eval_metric = ""
    try:
        comp_info = state.get("competition_info")
        eval_metric = (comp_info.evaluation_metric or "").lower() if comp_info else ""
    except Exception:
        pass

    # Debug: Show what metric is being checked
    if eval_metric:
        print(f"[DEBUG] eval_metric='{eval_metric}' for classification detection")
        if any(m in eval_metric for m in CLASSIFICATION_METRICS):
            print(f"[DEBUG] Metric '{eval_metric}' matched CLASSIFICATION_METRICS -> True")
            return True
        if any(m in eval_metric for m in REGRESSION_METRICS):
            print(f"[DEBUG] Metric '{eval_metric}' matched REGRESSION_METRICS -> False")
            return False
    else:
        print("[DEBUG] eval_metric is empty, skipping metric-based detection")

    # Step 4: Try submission_format_type
    try:
        comp_info = state.get("competition_info")
        fmt_type = comp_info.submission_format_type if comp_info else None
        if fmt_type:
            fmt_str = str(fmt_type).lower()
            if "proba" in fmt_str or "multi" in fmt_str:
                return True
            if "regression" in fmt_str or "single_col" in fmt_str:
                return False
    except Exception:
        pass

    # Step 4.5: Try target column name heuristic
    # Target names like "Cover_Type", "class", "label" suggest classification
    # NOTE: We use ONLY specific patterns here to avoid misclassifying regression tasks
    # that happen to have generic names like "target"
    try:
        target_col = state.get("target_col", "")
        if target_col:
            target_lower = target_col.lower()
            # Only exact matches for unambiguous classification terms
            # DO NOT include "target" - it's too generic (used in regression too)
            exact_classification_targets = {"class", "label", "category"}

            # Suffix patterns that strongly indicate classification
            # e.g., "cover_type", "product_class", "spam_label"
            suffix_classification_targets = {"_class", "_label", "_category"}

            # Check exact match (very high confidence)
            if target_lower in exact_classification_targets:
                print(f"[DEBUG] Target column '{target_col}' exact match -> classification")
                return True

            # Check suffix match (high confidence)
            if any(target_lower.endswith(suffix) for suffix in suffix_classification_targets):
                print(f"[DEBUG] Target column '{target_col}' suffix match -> classification")
                return True

            # Special case: "_type" suffix only if it's a known classification pattern
            # This catches "cover_type", "soil_type" but not generic "data_type"
            if target_lower.endswith("_type"):
                # Check if it's a likely classification target by looking for known patterns
                classification_type_patterns = {"cover_type", "soil_type", "species_type", "class_type"}
                if target_lower in classification_type_patterns:
                    print(f"[DEBUG] Target column '{target_col}' is known classification type -> True")
                    return True
    except Exception:
        pass

    # Step 5: Try domain_detected
    domain = state.get("domain_detected", "")
    if domain:
        domain_lower = str(domain).lower()
        print(f"[DEBUG] domain_detected='{domain_lower}' for classification detection")
        if "classification" in domain_lower:
            return True
        if "regression" in domain_lower:
            print(f"[DEBUG] domain_detected contains 'regression' -> False")
            return False

    # Step 6: Try problem_type string (expanded patterns)
    problem_type = ""
    try:
        comp_info = state.get("competition_info")
        problem_type = (comp_info.problem_type or "").lower() if comp_info else ""
    except Exception:
        pass

    if problem_type:
        if any(kw in problem_type for kw in CLASSIFICATION_KEYWORDS):
            return True
        if any(kw in problem_type for kw in REGRESSION_KEYWORDS):
            return False

    # NO DEFAULT - return None if undetermined
    return None


def _infer_from_sample_submission(state: dict | None) -> bool:
    """
    Infer task type from sample_submission.csv structure.

    Heuristics:
    - If >2 columns (id + multiple targets) -> likely classification proba
    - If 2 columns with float values in [0,1] -> likely classification proba
    - If 2 columns with values outside [0,1] -> likely regression
    - Default to True (classification is more common in Kaggle)

    Args:
        state: Workflow state dictionary

    Returns:
        True for classification (default), False for regression
    """
    if state is None:
        return True  # Default to classification

    sample_submission_path = state.get("sample_submission_path")
    if not sample_submission_path:
        return True  # Default

    try:
        import pandas as pd

        sample_df = pd.read_csv(sample_submission_path)
        n_cols = len(sample_df.columns)

        # Multiple target columns -> classification proba
        if n_cols > 2:
            print("[LOG:INFO] sample_submission has >2 columns -> inferring classification")
            return True

        # Single target column -> check value range
        if n_cols == 2:
            target_col = sample_df.columns[1]
            values = sample_df[target_col]

            if pd.api.types.is_numeric_dtype(values):
                min_val, max_val = values.min(), values.max()

                # All zeros (placeholder) - check column name
                if min_val == 0 and max_val == 0:
                    col_lower = str(target_col).lower()
                    if any(kw in col_lower for kw in ["proba", "prob", "class", "target"]):
                        return True
                    if any(kw in col_lower for kw in ["price", "fare", "amount", "sales"]):
                        return False
                    return True  # Default to classification

                # Values in [0, 1] range -> likely probabilities
                if 0 <= min_val <= max_val <= 1:
                    return True

                # Values outside [0, 1] -> likely regression
                if min_val < 0 or max_val > 1:
                    print(f"[LOG:INFO] sample_submission values outside [0,1]: [{min_val}, {max_val}] -> inferring regression")
                    return False

    except Exception as e:
        print(f"[LOG:WARNING] Could not infer from sample_submission: {e}")

    # Default to classification (more common in Kaggle)
    return True


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
                "  - **CRITICAL: TARGET COLUMN COUNT VALIDATION (MUST DO FIRST)**:",
                "    ```python",
                "    # STEP 0: ALWAYS count target columns from sample_submission FIRST",
                "    sample_sub = pd.read_csv(sample_submission_path)",
                "    target_cols = sample_sub.columns[1:].tolist()  # All columns except ID",
                "    N_CLASSES = len(target_cols)  # THIS IS THE REQUIRED OUTPUT SIZE",
                "    print(f'CRITICAL: Competition requires {N_CLASSES} target columns: {target_cols}')",
                "    ",
                "    # YOUR MODEL MUST OUTPUT EXACTLY N_CLASSES PREDICTIONS!",
                "    # For PyTorch: nn.Linear(..., N_CLASSES)",
                "    # For Keras: Dense(N_CLASSES, activation=...)",
                "    # For sklearn: Ensure predict_proba returns N_CLASSES columns",
                "    ```",
                "  - **MULTI-CLASS vs MULTI-LABEL DETECTION (CRITICAL - WRONG CHOICE = AUC ~0.5)**:",
                "    ```python",
                "    # MANDATORY: Detect problem type from sample_submission structure",
                "    # (N_CLASSES and target_cols already defined above)",
                "    ",
                "    if N_CLASSES == 1:",
                "        # Single target column: binary classification or regression",
                "        problem_type = 'binary'",
                "        activation = 'sigmoid'",
                "        loss = 'binary_crossentropy'  # or BCEWithLogitsLoss in PyTorch",
                "    elif N_CLASSES > 1:",
                "        # Multiple columns: need to determine multi-class vs multi-label",
                "        if all(col in train_df.columns for col in target_cols):",
                "            # Target columns exist in train -> likely multi-label",
                "            label_sums = train_df[target_cols].sum(axis=1)",
                "            is_multilabel = (label_sums > 1).any()  # Any row with >1 positive?",
                "            ",
                "            if is_multilabel:",
                "                problem_type = 'multilabel'",
                "                activation = 'sigmoid'  # Independent sigmoid per class",
                "                loss = 'binary_crossentropy'  # BCEWithLogitsLoss per class",
                "                # DO NOT normalize rows - each class is independent",
                "                print(f'Detected MULTI-LABEL: {N_CLASSES} independent labels')",
                "            else:",
                "                problem_type = 'multiclass'",
                "                activation = 'softmax'",
                "                loss = 'categorical_crossentropy'  # or CrossEntropyLoss",
                "                # MUST normalize rows to sum=1",
                "                print(f'Detected MULTI-CLASS: {N_CLASSES} mutually exclusive classes')",
                "        else:",
                "            # Target columns NOT in train -> definitely multi-class",
                "            # (train has single target column with class names/indices)",
                "            problem_type = 'multiclass'",
                "            activation = 'softmax'",
                "            loss = 'categorical_crossentropy'",
                "            print(f'Detected MULTI-CLASS: {N_CLASSES} classes from single target column')",
                "    ",
                "    print(f'Problem type: {problem_type}, N_CLASSES: {N_CLASSES}, Activation: {activation}')",
                "    ```",
                "  - For multi-class log_loss: probabilities MUST sum to 1 per row (clip to [1e-15, 1-1e-15] then renormalize)",
                "  - For multi-label: DO NOT normalize rows - predictions are independent probabilities",
                "  - If using logits (TF/PyTorch), apply activation BEFORE saving (softmax for multiclass, sigmoid for multilabel/binary)",
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
                "  - CRITICAL: Encode categorical features BASED ON CARDINALITY (prevents OOM):",
                "    ```python",
                "    HIGH_CARDINALITY_THRESHOLD = 50  # Use label encoding above this",
                "    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()",
                "    # Exclude ID and target columns",
                "    exclude_cols = {'id', 'target'}  # Add your actual ID/target column names if different",
                "    cat_cols = [c for c in cat_cols if c.lower() not in exclude_cols]",
                "    ",
                "    low_card_cols = [c for c in cat_cols if X[c].nunique() <= HIGH_CARDINALITY_THRESHOLD]",
                "    high_card_cols = [c for c in cat_cols if X[c].nunique() > HIGH_CARDINALITY_THRESHOLD]",
                "    print(f'Low cardinality (OHE): {low_card_cols}')",
                "    print(f'High cardinality (Label): {high_card_cols}')",
                "    ",
                "    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder",
                "    from sklearn.compose import ColumnTransformer",
                "    transformers = []",
                "    if low_card_cols:",
                "        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cols))",
                "    if high_card_cols:",
                "        transformers.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_card_cols))",
                "    preprocessor = ColumnTransformer(transformers, remainder='passthrough') if transformers else None",
                "    ```",
                "  - CRITICAL: NEVER use OneHotEncoder on columns with >50 unique values (causes OOM/memory crash)",
                "  - For LightGBM: Can also convert high-cardinality cols to 'category' dtype for native handling",
                "  - CatBoost handles ALL categorical features natively (no encoding needed)",
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
                    "\n🔊 AUDIO MODELLING (ROBUST LOADING REQUIRED):",
                    "  **CRITICAL: ROBUST FILE PATH MAPPING** (IDs in CSV often don't match filenames):",
                    "    - DO NOT assume `path = dir / f'{id}.wav'` - this pattern frequently fails",
                    "    - INSTEAD: Scan directory first, build id_to_path dict:",
                    "      ```python",
                    "      from pathlib import Path",
                    "      audio_dir = Path('essential_data') if Path('essential_data').exists() else Path('train')",
                    "      all_audio = list(audio_dir.rglob('*.wav')) + list(audio_dir.rglob('*.flac')) + list(audio_dir.rglob('*.mp3'))",
                    "      id_to_path = {f.stem: f for f in all_audio}",
                    "      df['audio_path'] = df['id_col'].astype(str).map(id_to_path)",
                    "      df = df[df['audio_path'].notna()]  # Filter to existing files only",
                    "      print(f'Loaded {len(df)} samples with valid audio paths')",
                    "      ```",
                    "  **AUDIO LOADING:**",
                    "    - Use librosa.load(path, sr=target_sr) with consistent sample rate (32000 or 44100)",
                    "    - Handle loading errors gracefully: wrap in try/except, skip bad files",
                    "    - Convert to log-mel spectrograms and treat as image inputs (CNN/ViT)",
                    "  **PREPROCESSING:**",
                    "    - Use fixed duration: pad short clips, trim long clips to consistent length",
                    "    - Normalize spectrograms per-sample or use dataset-wide mean/std",
                    "    - Cache spectrograms to disk (.npy) if training multiple epochs",
                    "  **EXTENSION DETECTION:**",
                    "    - Detect audio extension by scanning directory (not all datasets use .wav)",
                    "    - Common extensions: .wav, .flac, .mp3, .ogg",
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

    # Detect problem type - CRITICAL: Use robust multi-source detection
    is_classification = _detect_is_classification(state)
    if is_classification is None:
        # Final fallback: check sample_submission structure
        is_classification = _infer_from_sample_submission(state)
        print(f"[DEBUG] is_classification={is_classification} (from sample_submission inference)")
    else:
        print(f"[DEBUG] is_classification={is_classification} (from detection chain)")

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
        is_minimize = is_metric_minimization(metric_name)
        direction = "LOWER is better" if is_minimize else "HIGHER is better"

        instructions.extend(
            [
                "\n📏 METRIC REQUIREMENT (CRITICAL - MUST FOLLOW):",
                f"  - Competition metric: '{metric_name}' ({direction})",
                f"  - ⚠️ Final Validation Performance MUST report {metric_name} ONLY",
                "  - DO NOT report a different metric (e.g., don't report LogLoss if metric is Accuracy)",
            ]
        )

        # Specific instructions based on metric type
        if is_classification and ("log" in metric_lower or "loss" in metric_lower):
            instructions.extend([
                "  - For log_loss metrics: compute log_loss on OOF predictions (clip + renormalize)",
                "  - Lower is better: 0.02 = excellent, 0.7+ = nearly random for multiclass",
                "  - Use: `from sklearn.metrics import log_loss; score = log_loss(y_true, oof_preds)`",
            ])
        elif is_classification and ("accuracy" in metric_lower or "acc" in metric_lower):
            instructions.extend([
                "  - For accuracy metrics: compute accuracy_score on predicted classes",
                "  - Higher is better: 1.0 = perfect, 0.5 = random for binary",
                "  - Use: `from sklearn.metrics import accuracy_score; score = accuracy_score(y_true, y_pred)`",
                "  - ⚠️ DO NOT report log_loss or AUC - only report Accuracy",
            ])
        elif is_classification and ("auc" in metric_lower or "roc" in metric_lower):
            instructions.extend([
                "  - For AUC metrics: compute roc_auc_score on probability predictions",
                "  - Higher is better: 1.0 = perfect, 0.5 = random",
                "  - Use: `from sklearn.metrics import roc_auc_score; score = roc_auc_score(y_true, y_proba)`",
            ])
        elif "rmse" in metric_lower or "mse" in metric_lower:
            instructions.extend([
                "  - For RMSE/MSE metrics: compute mean_squared_error then sqrt for RMSE",
                "  - Lower is better: 0 = perfect",
                "  - Use: `from sklearn.metrics import mean_squared_error; score = np.sqrt(mean_squared_error(y_true, y_pred))`",
            ])

    # Explicit model type requirement based on is_classification
    if not is_image:
        if is_classification:
            instructions.extend([
                "\n⚠️ CLASSIFICATION MODEL REQUIREMENT (CRITICAL):",
                "  - IS_CLASSIFICATION = True (from canonical metadata)",
                "  - MUST use CLASSIFIER models: MLPClassifier, LGBMClassifier, XGBClassifier, CatBoostClassifier",
                "  - DO NOT use REGRESSOR models: MLPRegressor, LGBMRegressor, XGBRegressor will produce INVALID predictions",
                "  - Predictions MUST be probabilities in range [0.0, 1.0]",
                "  - For sklearn: use predict_proba()[:, 1] for binary classification (2 classes)",
                "  - For AUC metric: probability predictions are REQUIRED (not class labels)",
                "  ```python",
                "  # MANDATORY CHECK: Validate predictions are probabilities",
                "  assert 0 <= oof_preds.min() <= oof_preds.max() <= 1, 'Predictions must be probabilities [0,1]'",
                "  if oof_preds.min() < 0 or oof_preds.max() > 1:",
                "      raise ValueError(f'INVALID: predictions outside [0,1]: min={oof_preds.min()}, max={oof_preds.max()}')",
                "  ```",
            ])
        else:
            instructions.extend([
                "\n📊 REGRESSION MODEL REQUIREMENT:",
                "  - IS_CLASSIFICATION = False (from canonical metadata)",
                "  - MUST use REGRESSOR models: MLPRegressor, LGBMRegressor, XGBRegressor",
                "  - DO NOT use CLASSIFIER models",
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

    # Check if last run timed out and reduce epochs (LIMIT: max 1 reduction to prevent cascade)
    suggested_epochs = epoch_budget
    epochs_already_reduced = state.get("epochs_already_reduced", False)
    max_reductions = int(os.getenv("KAGGLE_AGENTS_MAX_EPOCH_REDUCTIONS", "1"))
    reduction_count = state.get("epoch_reduction_count", 0)

    if dev_results and reduction_count < max_reductions:
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
            # Also reduce early_stopping_patience proportionally
            early_stopping_patience = max(5, int(early_stopping_patience * reduction_factor))
            # Track reduction to prevent cascade
            state["epochs_already_reduced"] = True
            state["epoch_reduction_count"] = reduction_count + 1
            state["early_stopping_patience"] = early_stopping_patience

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

    # Audio-specific context injection (CRITICAL for MLSP-style competitions)
    if is_audio:
        instructions.extend(_build_audio_domain_instructions(state))

    # Regression post-processing (CRITICAL for valid predictions)
    if not is_classification:
        instructions.extend(_build_regression_postprocessing_instructions(state))

    # Standard requirements
    instructions.extend(build_standard_requirements())

    return "\n".join(instructions)


def _build_regression_postprocessing_instructions(state: dict) -> list[str]:
    """
    Build regression post-processing instructions for clipping predictions.

    Many regression targets have natural bounds (e.g., taxi fares >= 0).
    Negative predictions cause large RMSE penalties and are often invalid.

    Args:
        state: Workflow state dictionary

    Returns:
        List of instruction strings
    """
    # Only for regression problems
    problem_type = ""
    try:
        comp_info = state.get("competition_info")
        problem_type = comp_info.problem_type if comp_info else ""
    except Exception:
        problem_type = ""

    if "regression" not in str(problem_type).lower():
        return []

    target_col = str(state.get("target_col", "target")).lower()

    # Infer bounds from common regression targets
    bounds_hints = {
        "fare": (0, 500),  # Taxi fare: $0-$500
        "price": (0, None),  # Prices: non-negative
        "amount": (0, None),  # Amounts: non-negative
        "count": (0, None),  # Counts: non-negative integers
        "duration": (0, None),  # Duration: non-negative
        "age": (0, 120),  # Age: 0-120
        "distance": (0, None),  # Distance: non-negative
        "time": (0, None),  # Time: non-negative
        "sales": (0, None),  # Sales: non-negative
        "revenue": (0, None),  # Revenue: non-negative
    }

    lower_bound, upper_bound = None, None
    for keyword, bounds in bounds_hints.items():
        if keyword in target_col:
            lower_bound, upper_bound = bounds
            break

    instructions = []
    if lower_bound is not None or upper_bound is not None:
        instructions = [
            "\n📊 REGRESSION POST-PROCESSING (MANDATORY):",
            f"  - Target column '{target_col}' should be clipped to valid bounds",
            f"  - Lower bound: {lower_bound if lower_bound is not None else 'None (no lower limit)'}",
            f"  - Upper bound: {upper_bound if upper_bound is not None else 'None (no upper limit)'}",
            "  - **Apply clipping AFTER predictions to avoid invalid values:**",
            "    ```python",
            "    # Clip predictions to valid range",
            f"    oof_preds = np.clip(oof_preds, {lower_bound}, {upper_bound})",
            f"    test_preds = np.clip(test_preds, {lower_bound}, {upper_bound})",
            "    print(f'[LOG:INFO] Clipped predictions: min={{test_preds.min():.2f}}, max={{test_preds.max():.2f}}')",
            "    ```",
            "  - WHY: Negative predictions for non-negative targets cause large RMSE penalties",
        ]
    else:
        # Generic regression guidance
        instructions = [
            "\n📊 REGRESSION VALIDATION:",
            "  - Check if target values are always non-negative (y >= 0)",
            "  - If so, clip predictions: `preds = np.clip(preds, 0, None)`",
            "  - Check for outliers and clip to reasonable bounds if needed",
        ]

    return instructions


def _build_audio_domain_instructions(state: dict) -> list[str]:
    """
    Build audio domain-specific instructions from state.

    Injects critical audio competition context:
    - Submission format (Wide vs Long with ID pattern)
    - CVfolds train/test split
    - Precomputed features

    Args:
        state: Workflow state dictionary

    Returns:
        List of instruction strings
    """
    instructions = []

    # Submission format info (CRITICAL for MLSP-style)
    submission_format = state.get("submission_format_info")
    if submission_format and isinstance(submission_format, dict):
        format_type = submission_format.get("format_type", "unknown")
        id_multiplier = submission_format.get("id_multiplier")
        num_classes = submission_format.get("num_classes")
        id_column = submission_format.get("id_column", "Id")
        target_columns = submission_format.get("target_columns", [])

        instructions.append("\n🎯 AUDIO SUBMISSION FORMAT (DETECTED FROM sample_submission.csv):")
        instructions.append(f"  - **Format Type:** {format_type.upper()}")

        if format_type == "long" and id_multiplier:
            instructions.extend([
                f"  - **ID Pattern:** Id = rec_id * {id_multiplier} + class_id",
                f"  - **Number of Classes:** {num_classes}",
                "  - **CRITICAL SUBMISSION CODE:**",
                "    ```python",
                "    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)",
                "    pred_map = {}",
                "    for i, rec_id in enumerate(test_rec_ids):",
                f"        for class_id in range({num_classes}):",
                f"            submission_id = rec_id * {id_multiplier} + class_id",
                "            pred_map[submission_id] = predictions[i, class_id]",
                f"    submission['{target_columns[0] if target_columns else 'Probability'}'] = submission['{id_column}'].map(pred_map)",
                "    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)",
                "    ```",
            ])
        elif format_type == "wide":
            instructions.extend([
                f"  - **Target Columns:** {target_columns}",
                "  - **WIDE FORMAT:** One column per class, one row per sample",
                "    ```python",
                "    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)",
                f"    for i, col in enumerate({target_columns}):",
                "        submission[col] = predictions[:, i]",
                "    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)",
                "    ```",
            ])

    # CVfolds train/test split
    if state.get("cv_folds_used"):
        train_ids = state.get("train_rec_ids", [])
        test_ids = state.get("test_rec_ids", [])
        instructions.extend([
            "\n📊 TRAIN/TEST SPLIT (FROM CVfolds - DO NOT INFER FROM sample_submission):",
            f"  - Train samples: {len(train_ids)} rec_ids (use train_rec_ids from state)",
            f"  - Test samples: {len(test_ids)} rec_ids (use test_rec_ids from state)",
            "  - Filter audio files by rec_id membership in these lists",
        ])

    # Precomputed features
    precomputed = state.get("precomputed_features_info")
    if precomputed and isinstance(precomputed, dict):
        features_found = precomputed.get("features_found", {})
        feature_features = {k: v for k, v in features_found.items() if k not in ("cv_folds", "id_mapping")}
        if feature_features:
            instructions.append("\n📁 PRECOMPUTED FEATURES (USE THESE INSTEAD OF RE-EXTRACTING):")
            for feature_type, file_path in feature_features.items():
                shape = precomputed.get("feature_shapes", {}).get(feature_type, "unknown")
                instructions.append(f"  - {feature_type}: {file_path} (shape: {shape})")
            instructions.append("  - Load with pd.read_csv() for .txt/.csv, np.load() for .npy")

    # Label parsing guidance for multi-label
    if state.get("submission_format_info", {}).get("num_classes", 0) > 1:
        instructions.extend([
            "\n⚠️ MULTI-LABEL CLASSIFICATION (MLSP-STYLE):",
            "  - Use BCEWithLogitsLoss (NOT CrossEntropyLoss)",
            "  - Use sigmoid activation (NOT softmax)",
            "  - For sparse label format (e.g., 'rec_id,class1,class5,class12'):",
            "    from kaggle_agents.utils.label_parser import parse_mlsp_multilabel",
            "    rec_ids, label_matrix = parse_mlsp_multilabel(label_path, num_classes=N)",
            "  - DO NOT use np.zeros() as fallback - if label parsing fails, FIX IT",
        ])

    return instructions
