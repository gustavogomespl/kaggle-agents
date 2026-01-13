"""Debug loop handling for the planner agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import KaggleState


def handle_debug_loop_trigger(state: KaggleState) -> dict[str, Any] | None:
    """
    Handle debug loop trigger from Meta-Evaluator.

    When a performance gap is detected, create a focused debug plan
    for the underperforming model instead of the normal ablation plan.

    Args:
        state: Current workflow state

    Returns:
        Debug plan if triggered, None otherwise
    """
    from ...core.state import AblationComponent

    trigger_debug = state.get("trigger_debug_loop", False)
    if not trigger_debug:
        return None

    debug_target = state.get("debug_target_model", "unknown")
    debug_hints = state.get("debug_hints", [])
    performance_gap = state.get("performance_gap", 0.0)

    print(f"\n  🔧 DEBUG LOOP TRIGGERED for: {debug_target}")
    print(f"      Performance gap: {performance_gap:.2f}")
    print("      Debug hints:")
    for hint in debug_hints[:3]:
        print(f"        - {hint}")

    # Create focused debug plan
    debug_components = [
        AblationComponent(
            name=f"debug_{debug_target}_labelencoder",
            component_type="debug",
            code=(
                f"# Debug {debug_target}: Verify LabelEncoder class order\n"
                "# Compare with other models' class_order.npy files\n"
                "# Ensure consistent encoding across all models"
            ),
            estimated_impact=0.8,
        ),
        AblationComponent(
            name=f"debug_{debug_target}_preprocessing",
            component_type="debug",
            code=(
                f"# Debug {debug_target}: Check preprocessing pipeline\n"
                "# Verify train/val splits use same random_state\n"
                "# Compare feature scaling and encoding"
            ),
            estimated_impact=0.7,
        ),
        AblationComponent(
            name=f"debug_{debug_target}_hyperparams",
            component_type="debug",
            code=(
                f"# Debug {debug_target}: Review hyperparameters\n"
                "# Check class_weight setting\n"
                "# Verify objective function matches metric"
            ),
            estimated_impact=0.6,
        ),
    ]

    return {
        "ablation_plan": debug_components,
        "is_debug_iteration": True,
        "debug_target": debug_target,
        "debug_hints": debug_hints,
        "skip_normal_planning": True,
    }
