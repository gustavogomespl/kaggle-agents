"""Plan validation and enhancement for the planner agent."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import AblationComponent, KaggleState


def validate_plan(
    plan: list[AblationComponent],
    *,
    state: KaggleState | None = None,
    coerce_components_fn=None,
    is_image_competition_without_features_fn=None,
) -> list[AblationComponent]:
    """
    Validate and enhance the ablation plan.

    Args:
        plan: Initial plan
        state: Current workflow state (optional, for fast-mode constraints)
        coerce_components_fn: Function to normalize components
        is_image_competition_without_features_fn: Function to detect image competitions

    Returns:
        Validated plan
    """
    from ...core.state import AblationComponent

    # Normalize any raw dict entries before applying validation rules.
    if coerce_components_fn:
        plan = coerce_components_fn(plan)

    run_mode = str((state or {}).get("run_mode", "")).lower()
    domain = str((state or {}).get("domain_detected", "tabular")).lower()
    timeout_cap = (state or {}).get("timeout_per_component")

    if isinstance(timeout_cap, str):
        try:
            timeout_cap = int(timeout_cap)
        except ValueError:
            timeout_cap = None

    fast_mode = bool((state or {}).get("fast_mode"))
    if isinstance(timeout_cap, int) and timeout_cap <= 1200:
        fast_mode = True

    # `estimated_impact` is an uncalibrated planner field. Keep every candidate
    # here and let trusted canonical-fold results decide what survives later.
    valid_plan = list(plan)

    # Guardrail: block tabular models for image competitions without features.
    if is_image_competition_without_features_fn and is_image_competition_without_features_fn(state):
        tabular_signals = [
            "lightgbm",
            "lgbm",
            "xgboost",
            "catboost",
            "randomforest",
            "logistic",
            "svm",
            "naive",
            "optuna",
            "stacking",
            "ridge",
        ]
        filtered_plan = []
        removed = []
        for comp in valid_plan:
            text = f"{comp.name} {comp.code}".lower()
            if any(sig in text for sig in tabular_signals):
                removed.append(comp.name)
                continue
            filtered_plan.append(comp)
        if removed:
            print(
                f"  ⚠️  Removed tabular components for image competition without features: {', '.join(removed)}"
            )
            valid_plan = filtered_plan

    # Limit components (quality over quantity)
    default_max_components = 3 if run_mode == "mlebench" else 3 if fast_mode else 6
    max_components = max(
        1,
        int((state or {}).get("max_components") or default_max_components),
    )
    override = os.getenv("KAGGLE_AGENTS_MAX_COMPONENTS")
    if override:
        try:
            override_val = int(override)
            if override_val >= 1:
                max_components = override_val
        except ValueError:
            print(f"  ⚠️ Invalid KAGGLE_AGENTS_MAX_COMPONENTS='{override}', using default")
    if len(valid_plan) > max_components:
        print(
            f"  ⚠️  Plan has {len(valid_plan)} components - preserving the first "
            f"{max_components} proposed components"
        )
    uncapped_plan = valid_plan
    valid_plan = uncapped_plan[:max_components]

    # Ensure one executable model without forcing an unmeasured diversity arm.
    model_count = sum(1 for c in valid_plan if c.component_type == "model")
    if model_count == 0:
        print("  ⚠️  No 'model' component in the capped plan - ensuring one model")
        deferred_model = next(
            (
                component
                for component in uncapped_plan[max_components:]
                if component.component_type == "model"
            ),
            None,
        )
        if deferred_model is not None:
            baseline = deferred_model
        elif domain == "image_to_image" or domain == "image_segmentation":
            baseline = AblationComponent(
                name="baseline_unet_encoder_decoder",
                component_type="model",
                code="U-Net encoder-decoder for pixel-level prediction. Output must be same size as input. Flatten to pixel-level CSV format.",
                estimated_impact=0.0,
                tested=False,
                actual_impact=None,
            )
        elif domain.startswith("image_"):
            baseline = AblationComponent(
                name="baseline_resnet18",
                component_type="model",
                code="",
                estimated_impact=0.0,
                tested=False,
                actual_impact=None,
            )
        else:
            baseline = AblationComponent(
                name="baseline_lightgbm",
                component_type="model",
                code="",
                estimated_impact=0.0,
                tested=False,
                actual_impact=None,
            )
        if len(valid_plan) < max_components:
            valid_plan.append(baseline)
        else:
            valid_plan[-1] = baseline
        print(f"     Ensured: {baseline.name}")

    # Debug log: Show final plan composition
    preprocessing_count = sum(
        c.component_type in ["preprocessing", "feature_engineering"]
        for c in valid_plan
    )
    model_count = sum(c.component_type == "model" for c in valid_plan)
    other_count = len(valid_plan) - preprocessing_count - model_count
    print(
        f"  📊 Final plan: {preprocessing_count} FE + {model_count} models + "
        f"{other_count} ensemble = {len(valid_plan)} total"
    )

    return valid_plan


def is_image_competition_without_features(state: KaggleState | None) -> bool:
    """
    Detect if competition is image-based but has no tabular features.

    This catches cases where domain detection fails but the competition
    is clearly image-based (has image files and minimal train.csv columns).

    Args:
        state: Current workflow state

    Returns:
        True if this appears to be an image competition without tabular features
    """
    if state is None:
        return False

    from pathlib import Path

    # Check for image files in data directory
    data_dir = state.get("data_dir", "")
    has_images = False
    if data_dir:
        data_path = Path(data_dir)
        if data_path.exists():
            # Check for common image directories (train/, test/, images/)
            for subdir in ["train", "test", "images", "train_images", "test_images"]:
                subdir_path = data_path / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    # Check if directory contains image files
                    image_extensions = {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".bmp",
                        ".tiff",
                        ".webp",
                    }
                    for f in subdir_path.iterdir():
                        if f.suffix.lower() in image_extensions:
                            has_images = True
                            break
                if has_images:
                    break

    # Check if train.csv has minimal columns (only id + label)
    train_csv_minimal = False
    train_csv_path = state.get("train_csv_path", "")
    if train_csv_path:
        import pandas as pd

        train_path = Path(train_csv_path)
        if train_path.exists():
            try:
                train_df = pd.read_csv(train_path, nrows=5)  # Only read header
                # If train.csv has 2 or fewer columns, it's likely just id + label
                train_csv_minimal = len(train_df.columns) <= 2
            except Exception:
                pass

    if has_images and train_csv_minimal:
        print("  [WARNING] Detected IMAGE competition without tabular features!")
        print(f"            - Has image files: {has_images}")
        print(f"            - train.csv minimal (<=2 cols): {train_csv_minimal}")
        return True

    return False


def detect_multimodal_competition(state: KaggleState | None) -> dict[str, Any]:
    """
    Detect if competition has both images AND rich tabular features.

    Multi-modal datasets have:
    - Images in train/ or test/ directories
    - Rich tabular features in train.csv (>10 columns)

    Returns guidance for hybrid model strategies.

    Args:
        state: Current workflow state

    Returns:
        Dictionary with detection results and strategy recommendations
    """
    if state is None:
        return {"type": "unknown", "is_multimodal": False}

    from pathlib import Path

    # Check for image files
    data_dir = state.get("data_dir", "")
    has_images = False
    image_count = 0

    if data_dir:
        data_path = Path(data_dir)
        if data_path.exists():
            for subdir in ["train", "test", "images", "train_images", "test_images"]:
                subdir_path = data_path / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
                    for f in subdir_path.iterdir():
                        if f.suffix.lower() in image_extensions:
                            has_images = True
                            image_count += 1
                            if image_count > 10:  # Found enough images
                                break
                if has_images:
                    break

    # Check if train.csv has rich tabular features
    has_rich_tabular = False
    tabular_feature_count = 0
    train_csv_path = state.get("train_csv_path", "") or state.get("train_data_path", "")

    if train_csv_path:
        import pandas as pd

        train_path = Path(train_csv_path)
        if train_path.exists():
            try:
                train_df = pd.read_csv(train_path, nrows=5)
                # Count features from the explicit state contract. Column-name
                # recipes silently encode benchmark-shaped priors and can
                # misclassify arbitrary targets as usable tabular features.
                canonical_contract = state.get("canonical_contract") or {}
                contract_targets = canonical_contract.get("target_cols") or []
                exclude_cols = {
                    str(value).lower()
                    for value in (
                        state.get("target_col"),
                        canonical_contract.get("target_col"),
                        canonical_contract.get("id_col"),
                        *contract_targets,
                    )
                    if isinstance(value, str) and value
                }
                feature_cols = [
                    c for c in train_df.columns if c.lower() not in exclude_cols
                ]
                tabular_feature_count = len(feature_cols)
                has_rich_tabular = tabular_feature_count >= 10
            except Exception:
                pass

    # Determine competition type and strategy
    if has_images and has_rich_tabular:
        print("\n  🔍 MULTI-MODAL COMPETITION DETECTED:")
        print(f"      - Has image files: {has_images}")
        print(f"      - Tabular features: {tabular_feature_count}")

        return {
            "type": "multi_modal",
            "is_multimodal": True,
            "has_images": True,
            "has_rich_tabular": True,
            "tabular_features": tabular_feature_count,
            "strategy": "hybrid_cnn_tabular",
            "recommendation": (
                "Use Keras Functional API with multi-input model: "
                "CNN branch (EfficientNet) for images + MLP branch for tabular features. "
                "Alternatively, the pre-extracted tabular features may be sufficient "
                "for competitive performance with LightGBM/XGBoost alone."
            ),
            "priority_models": [
                "LightGBM with all tabular features (fast, often competitive)",
                "XGBoost with all tabular features",
                "Hybrid CNN+Tabular (best but slower)",
            ],
        }
    if has_images:
        return {
            "type": "image_only",
            "is_multimodal": False,
            "has_images": True,
            "has_rich_tabular": False,
            "strategy": "efficientnet",
            "recommendation": "Use transfer learning with EfficientNet or ResNet.",
        }
    return {
        "type": "tabular_only",
        "is_multimodal": False,
        "has_images": False,
        "has_rich_tabular": has_rich_tabular,
        "tabular_features": tabular_feature_count,
        "strategy": "lightgbm_xgboost",
        "recommendation": "Use gradient boosting (LightGBM, XGBoost, CatBoost).",
    }
