"""Run-level wall-clock budget helpers.

Component timeouts alone cannot bound a run: three iterations of three
components, each with its own retry ladder, can exceed a day of wall clock while
every individual timeout is respected. A benchmark sweep budgeted in GPU-hours
needs a single deadline that every stage checks against, so that the cost of a
run is bounded by design rather than by how quickly the generated code happens
to converge.

The deadline covers the agent only: it starts once public data is staged and
ends before grading, matching the budget definition in the experimental
protocol. Runs without a configured budget are unbudgeted, and every helper here
degrades to "no constraint" for them.

Enforcement is **cooperative**, not policed: stages call these helpers before
starting new work, and component timeouts are clamped to the remaining budget,
but nothing kills a component that is already running. A run can therefore
overrun the deadline by at most one clamped component timeout. Reporting the
budget as a hard limit would overstate what this provides; ``deadline_reached``
on the run result records when it actually happened.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any


# Wall clock held back for the closing stages (ensemble, submission validation,
# snapshotting) once the last iteration is done.
FINALIZATION_RESERVE_S = 600.0

# Floor for a clamped component timeout: below this, execution cannot even
# import its dependencies, so returning less would only manufacture failures.
MIN_COMPONENT_TIMEOUT_S = 60


def run_deadline(state: Mapping[str, Any]) -> float | None:
    """Absolute epoch deadline for this run, or ``None`` when unbudgeted."""
    try:
        deadline = float(state.get("run_deadline_ts"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return deadline if deadline > 0 else None


def remaining_budget_s(
    state: Mapping[str, Any],
    now: float | None = None,
) -> float | None:
    """Seconds left before the run deadline, or ``None`` when unbudgeted."""
    deadline = run_deadline(state)
    if deadline is None:
        return None
    return deadline - (time.time() if now is None else now)


def budget_exhausted(
    state: Mapping[str, Any],
    reserve_s: float = 0.0,
    now: float | None = None,
) -> bool:
    """Whether less than ``reserve_s`` remains. Unbudgeted runs never exhaust."""
    remaining = remaining_budget_s(state, now)
    return remaining is not None and remaining <= reserve_s


def clamp_timeout_to_budget(
    state: Mapping[str, Any],
    timeout_s: float,
    *,
    reserve_s: float = 0.0,
    minimum_s: int = MIN_COMPONENT_TIMEOUT_S,
    now: float | None = None,
) -> int:
    """Shrink a component timeout so it cannot outlive the run deadline.

    Callers should gate on :func:`budget_exhausted` first: when nothing usable
    remains this returns ``minimum_s`` rather than zero, because a non-positive
    timeout would break the executor instead of stopping the run cleanly.
    """
    remaining = remaining_budget_s(state, now)
    if remaining is None:
        return int(timeout_s)
    usable = remaining - reserve_s
    if usable <= 0:
        return int(minimum_s)
    return int(max(minimum_s, min(float(timeout_s), usable)))


def format_remaining(state: Mapping[str, Any], now: float | None = None) -> str:
    """Human-readable remaining budget for logs."""
    remaining = remaining_budget_s(state, now)
    if remaining is None:
        return "unbudgeted"
    if remaining <= 0:
        return "exhausted"
    return f"{remaining / 60:.1f} min"
