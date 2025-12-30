"""
Time budget and MLE-bench objective instruction builders.
"""

from __future__ import annotations


def build_budget_instructions(timeout_hint: int | None) -> list[str]:
    """Build time budget instructions."""
    if not isinstance(timeout_hint, (int, float)):
        return []

    return [
        "\n⏱️ TIME BUDGET (CRITICAL):",
        f"  - Component must complete within ~{int(timeout_hint)}s (env: KAGGLE_AGENTS_COMPONENT_TIMEOUT_S).",
        "  - Implement a soft-deadline (e.g., budget-45s): if exceeded, stop training, save best artifacts, and still print the final metric line.",
        "  - Read env vars: KAGGLE_AGENTS_FAST_MODE and KAGGLE_AGENTS_CV_FOLDS to reduce compute when needed.",
    ]


def build_mlebench_objective_instructions() -> list[str]:
    """Build MLE-bench objective instructions."""
    return [
        "\n🏁 MLE-BENCH OBJECTIVE:",
        "  - Optimize for MLE-bench medal: prioritize fast end-to-end runtime + robust valid submission.",
        "  - Prefer cheaper training (fewer folds/epochs) and inference-time tricks (TTA) over expensive CV sweeps.",
    ]
