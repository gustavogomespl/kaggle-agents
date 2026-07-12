"""Enforce robustness validation before ensemble and submission."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.config import is_metric_minimization
from ...core.state import KaggleState
from ...utils.telemetry import make_event


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _restore_best_valid_submission(state: KaggleState) -> Path | None:
    """Restore the best previously accepted submission, if one exists."""
    candidates = []
    for submission in state.get("submissions", []) or []:
        if not bool(_field(submission, "valid", True)):
            continue
        file_path = _field(submission, "file_path")
        if not file_path:
            continue
        path = Path(file_path)
        if path.exists() and path.is_file():
            candidates.append((submission, path))

    if not candidates:
        return None

    metric_name = ""
    competition_info = state.get("competition_info")
    if competition_info is not None:
        metric_name = str(getattr(competition_info, "evaluation_metric", "") or "")

    scored = [
        (submission, path)
        for submission, path in candidates
        if isinstance(_field(submission, "public_score"), (int, float))
    ]
    if scored:
        choose = min if is_metric_minimization(metric_name) else max
        _, source = choose(scored, key=lambda item: float(_field(item[0], "public_score")))
    else:
        _, source = candidates[-1]

    destination = Path(state["working_directory"]) / "submission.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def robustness_gate_node(state: KaggleState) -> dict[str, Any]:
    """Turn robustness results into pass, one bounded recovery, or safe stop."""
    if state.get("robustness_abstained", False):
        return {
            "robustness_gate_action": "pass",
            "robustness_recovery_count": 0,
            "current_candidate_valid": True,
            "workflow_valid": True,
            "force_refinement": False,
            "last_updated": datetime.now(),
        }

    if state.get("robustness_passed") is True:
        return {
            "robustness_gate_action": "pass",
            "robustness_recovery_count": 0,
            "current_candidate_valid": True,
            "workflow_valid": True,
            "force_refinement": False,
            "last_updated": datetime.now(),
        }

    try:
        recovery_count = max(0, int(state.get("robustness_recovery_count", 0)))
    except (TypeError, ValueError):
        recovery_count = 0
    try:
        max_recoveries = max(0, int(state.get("max_robustness_recoveries", 1)))
    except (TypeError, ValueError):
        max_recoveries = 1

    details = state.get("robustness_failure_details", {}) or {}
    failed_modules = list(details.get("failed_modules", []) or [])
    issues = list(details.get("issues", []) or [])
    suggestions = list(details.get("suggestions", []) or [])

    if recovery_count < max_recoveries:
        issue_text = "; ".join(str(issue) for issue in issues[:8]) or "robustness validation failed"
        suggestion_text = "; ".join(str(item) for item in suggestions[:8])
        guidance = dict(state.get("refinement_guidance", {}) or {})
        guidance["planner_guidance"] = (
            "Create a targeted correction plan for failed robustness modules "
            f"{failed_modules}: {issue_text}"
        )
        guidance["developer_guidance"] = (
            f"Fix these guardrail failures before ensemble: {issue_text}. {suggestion_text}"
        ).strip()
        guidance["priority_fixes"] = issues[:8]

        return {
            "robustness_gate_action": "recover",
            "robustness_recovery_count": recovery_count + 1,
            "current_component_index": 0,
            "skip_remaining_components": False,
            "needs_refinement": True,
            "force_refinement": True,
            "current_candidate_valid": False,
            "refinement_guidance": guidance,
            "telemetry_events": [
                make_event(
                    "guardrails",
                    "recovery_requested",
                    iteration=state.get("current_iteration", 0),
                    attempt=recovery_count + 1,
                    failed_modules=failed_modules,
                )
            ],
            "last_updated": datetime.now(),
        }

    restored = _restore_best_valid_submission(state)
    preserved = restored is not None
    reason = (
        "robustness_failed_preserved_best_submission"
        if preserved
        else "robustness_failed_no_valid_submission"
    )
    error = None if preserved else "Robustness validation failed; candidate blocked"

    return {
        "robustness_gate_action": "fail",
        "current_candidate_valid": False,
        "workflow_valid": preserved,
        "should_continue": False,
        "needs_refinement": False,
        "force_refinement": False,
        "termination_reason": reason,
        "submission_validation_error": error,
        "telemetry_events": [
            make_event(
                "guardrails",
                "candidate_blocked",
                iteration=state.get("current_iteration", 0),
                preserved_previous_submission=preserved,
                failed_modules=failed_modules,
            )
        ],
        "last_updated": datetime.now(),
    }

