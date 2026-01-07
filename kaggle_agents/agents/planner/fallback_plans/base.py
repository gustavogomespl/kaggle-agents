"""
Base utilities and router for fallback plans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....core.state import KaggleState
from .audio import create_audio_fallback_plan
from .image import create_image_fallback_plan, create_image_to_image_fallback_plan
from .tabular import create_tabular_fallback_plan
from .text import create_text_fallback_plan


# Domain sets for routing
IMAGE_DOMAINS = {
    "image_classification",
    "image_regression",
    "object_detection",
    "image_to_image",
    "image_segmentation",
}
TEXT_DOMAINS = {"text_classification", "text_regression", "seq_to_seq"}
AUDIO_DOMAINS = {"audio_classification", "audio_regression"}


def is_image_competition_without_features(state: KaggleState | None) -> bool:
    """
    Detect image competition masquerading as tabular due to detection failure.

    Checks:
    1. train.csv exists but has <= 2 columns (id + label only)
    2. Image directories exist (train/, test/, train_images/, etc.)

    This prevents LightGBM/XGBoost from being used when there are no features.

    Args:
        state: Current workflow state

    Returns:
        True if this appears to be an image competition without tabular features
    """
    if state is None:
        return False

    working_dir = state.get("working_directory")
    if not working_dir:
        return False
    working_dir = Path(working_dir)

    # Check for image directories
    image_dirs = ["train", "test", "train_images", "test_images", "images"]
    has_image_dir = any((working_dir / d).is_dir() for d in image_dirs)

    if not has_image_dir:
        return False

    # Check train.csv columns
    train_csv = working_dir / "train.csv"
    if not train_csv.exists():
        # No train.csv but has image dirs = likely image competition
        return True

    try:
        import pandas as pd

        df = pd.read_csv(train_csv, nrows=5)
        # If only id + label columns, this is image competition
        return len(df.columns) <= 2
    except Exception:
        return False


def create_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    curriculum_insights: str = "",
    *,
    state: KaggleState | None = None,
) -> list[dict[str, Any]]:
    """
    Create domain-specific fallback plan when LLM parsing fails.

    Routes to appropriate domain-specific fallback method based on domain type.

    Args:
        domain: Competition domain (e.g., 'image_classification', 'text_classification', 'tabular')
        sota_analysis: SOTA analysis results
        curriculum_insights: Insights from previous iterations (optional)
        state: Current workflow state (optional)

    Returns:
        List of component dictionaries (3-5 components depending on domain)
    """
    print(f"  [DEBUG] Creating fallback plan for domain: '{domain}'")

    run_mode = str((state or {}).get("run_mode", "")).lower()
    objective = str((state or {}).get("objective", "")).lower()
    timeout_cap = (state or {}).get("timeout_per_component")
    if isinstance(timeout_cap, str):
        try:
            timeout_cap = int(timeout_cap)
        except ValueError:
            timeout_cap = None

    # Speed-first when optimizing for MLE-bench medals or tight component caps.
    fast_mode = (
        run_mode == "mlebench"
        or "medal" in objective
        or (isinstance(timeout_cap, int) and timeout_cap <= 1200)
    )

    # Get competition name for domain-specific settings (MUST be before safety check)
    competition_name = ""
    if state:
        comp_info = state.get("competition_info")
        if comp_info:
            competition_name = getattr(comp_info, "name", "") or getattr(comp_info, "id", "") or ""
        if not competition_name:
            competition_name = str(state.get("competition_id", "") or state.get("competition_name", ""))

    # SAFETY CHECK: Prevent tabular models for image competitions
    if is_image_competition_without_features(state):
        print(
            "  [WARNING] Forcing IMAGE fallback plan (detected image competition without features)"
        )
        print("            Tree models (LightGBM/XGBoost) require tabular features!")
        return create_image_fallback_plan(
            "image_classification", sota_analysis, fast_mode=fast_mode, competition_name=competition_name
        )

    # Route to domain-specific fallback method
    if domain in ("image_to_image", "image_segmentation"):
        return create_image_to_image_fallback_plan(domain, sota_analysis, fast_mode=fast_mode)
    if domain in IMAGE_DOMAINS or domain.startswith("image_"):
        return create_image_fallback_plan(domain, sota_analysis, fast_mode=fast_mode, competition_name=competition_name)
    if domain in TEXT_DOMAINS or domain.startswith("text_"):
        return create_text_fallback_plan(domain, sota_analysis)
    if domain in AUDIO_DOMAINS or domain.startswith("audio_"):
        return create_audio_fallback_plan(domain, sota_analysis)
    # Tabular (default)
    return create_tabular_fallback_plan(
        domain,
        sota_analysis,
        curriculum_insights,
        fast_mode=fast_mode,
    )
