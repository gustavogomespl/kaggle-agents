"""Enforce robustness validation before ensemble and submission."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...agents.developer.validation import (
    _model_validation_problem_type,
    _requires_class_order_artifact,
    quarantine_component_artifacts,
)
from ...core.state import KaggleState
from ...utils.image_to_image_contract import load_packed_images
from ...utils.submission_artifacts import (
    restore_accepted_submission,
    restore_best_candidate_submission,
)
from ...utils.telemetry import make_event


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _restore_best_valid_submission(
    state: KaggleState,
    rejected_components: list[str],
) -> Path | None:
    """Restore an accepted snapshot, or an unaffected verified best candidate."""
    working_dir = Path(state["working_directory"])
    accepted_owner = str(
        state.get("accepted_submission_score_owner") or ""
    )
    if not accepted_owner or accepted_owner not in rejected_components:
        restored = restore_accepted_submission(state, working_dir)
        if restored is not None:
            return restored
    owner = str(state.get("best_candidate_submission_component_name") or "")
    if owner and owner not in rejected_components:
        return restore_best_candidate_submission(state, working_dir)
    return None


def _mle_evidence_failures(state: KaggleState) -> dict[str, list[str]]:
    """Fail closed when an MLE component lacks canonical, scored OOF identity.

    Robustness can be disabled as an ablation, but that must not turn an
    unscored or row-unaligned prediction file into benchmark evidence.
    """
    if str(state.get("run_mode", "")).strip().lower() != "mlebench":
        return {}

    eligible = sorted(
        str(name)
        for name, available in (state.get("oof_availability", {}) or {}).items()
        if available is True and str(name)
    )
    if not eligible:
        return {}

    contract = state.get("canonical_contract") or {}
    if hasattr(contract, "to_dict"):
        contract = contract.to_dict()
    domain = str(state.get("domain_detected", "") or "").lower().replace("-", "_")
    packed_image_contract = bool(
        isinstance(contract, dict)
        and contract.get("packed_image_contract")
    ) or domain == "image_to_image"
    canonical_ids_path = (
        Path(str(contract.get("train_ids_path")))
        if isinstance(contract, dict) and contract.get("train_ids_path")
        else None
    )
    canonical_test_ids_path = (
        Path(str(contract.get("test_ids_path")))
        if isinstance(contract, dict) and contract.get("test_ids_path")
        else None
    )
    canonical_ids: np.ndarray | None = None
    canonical_test_ids: np.ndarray | None = None
    if canonical_ids_path is not None and canonical_ids_path.is_file():
        try:
            canonical_ids = np.asarray(
                np.load(
                    canonical_ids_path,
                    allow_pickle=not packed_image_contract,
                )
            ).reshape(-1)
        except Exception:
            canonical_ids = None
    if (
        packed_image_contract
        and canonical_test_ids_path is not None
        and canonical_test_ids_path.is_file()
    ):
        try:
            canonical_test_ids = np.asarray(
                np.load(canonical_test_ids_path, allow_pickle=False)
            ).reshape(-1)
        except Exception:
            canonical_test_ids = None

    trusted_scores = state.get("trusted_component_scores") or {}
    submission_contract = state.get("submission_contract") or {}
    expected_class_order = (
        submission_contract.get("class_order")
        if isinstance(submission_contract, dict)
        else None
    )
    if not isinstance(expected_class_order, (list, tuple)):
        expected_class_order = None
    # Ask the same question the developer asked when it decided which artifacts
    # to demand. A wide multilabel template also has more than two prediction
    # columns, but its columns are independent labels: no component is ever
    # told to save an order for them, so requiring the file here made every
    # multilabel competition permanently ungradeable.
    requires_class_order = (
        not packed_image_contract
        and bool(expected_class_order)
        and _requires_class_order_artifact(
            state,
            _model_validation_problem_type(state),
        )
    )
    working_dir = Path(state["working_directory"])
    failures: dict[str, list[str]] = {}
    for name in eligible:
        issues: list[str] = []
        raw_score: Any = (
            trusted_scores.get(name)
            if isinstance(trusted_scores, dict)
            else None
        )
        if isinstance(raw_score, dict):
            raw_score = raw_score.get("score", raw_score.get("cv_score"))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = math.nan
        if not math.isfinite(score):
            issues.append("missing independently recomputed finite OOF score")

        if packed_image_contract:
            packed_artifacts = {}
            for artifact_kind in ("oof", "test"):
                artifact_path = (
                    working_dir / "models" / f"{artifact_kind}_{name}.npz"
                )
                if not artifact_path.is_file():
                    issues.append(
                        f"missing {artifact_kind} packed image artifact"
                    )
                    continue
                try:
                    packed_artifacts[artifact_kind] = load_packed_images(
                        artifact_path
                    )
                except Exception as exc:
                    issues.append(
                        f"{artifact_kind} packed image artifact cannot be "
                        f"verified: {exc}"
                    )

            if canonical_ids is None:
                issues.append("canonical train image IDs are unavailable")
            elif "oof" in packed_artifacts and (
                packed_artifacts["oof"].image_ids.tolist()
                != [str(value) for value in canonical_ids]
            ):
                issues.append(
                    "packed OOF image IDs do not match canonical OOF image order"
                )
            if canonical_test_ids is None:
                issues.append("canonical test image IDs are unavailable")
            elif "test" in packed_artifacts and (
                packed_artifacts["test"].image_ids.tolist()
                != [str(value) for value in canonical_test_ids]
            ):
                issues.append(
                    "packed test image IDs do not match canonical test image order"
                )
        else:
            for artifact_kind in ("oof", "test"):
                artifact_path = (
                    working_dir / "models" / f"{artifact_kind}_{name}.npy"
                )
                if not artifact_path.is_file():
                    issues.append(
                        f"missing {artifact_kind} prediction artifact"
                    )

        if requires_class_order:
            class_order_path = (
                working_dir / "models" / f"class_order_{name}.npy"
            )
            if not class_order_path.is_file():
                issues.append(
                    "missing component-specific multiclass class order"
                )
            else:
                try:
                    model_class_order = np.asarray(
                        np.load(class_order_path, allow_pickle=False)
                    ).reshape(-1)
                    if [str(value) for value in model_class_order] != [
                        str(value) for value in expected_class_order
                    ]:
                        issues.append(
                            "component class order does not match submission contract"
                        )
                except Exception as exc:
                    issues.append(
                        f"component class order cannot be verified: {exc}"
                    )

        if not packed_image_contract:
            model_ids_path = working_dir / "models" / f"train_ids_{name}.npy"
            if canonical_ids is None:
                issues.append("canonical train IDs are unavailable")
            elif not model_ids_path.is_file():
                issues.append("model train IDs are unavailable")
            else:
                try:
                    model_ids = np.asarray(
                        np.load(model_ids_path, allow_pickle=False)
                    ).reshape(-1)
                    if [str(value) for value in model_ids] != [
                        str(value) for value in canonical_ids
                    ]:
                        issues.append(
                            "model train IDs do not match canonical OOF row order"
                        )
                except Exception as exc:
                    issues.append(
                        f"model train IDs cannot be verified: {exc}"
                    )

        if issues:
            failures[name] = issues
    return failures


def _flagged_component_name(state: KaggleState) -> str:
    """Recover the latest component name for legacy failure details.

    Current robustness results include explicit per-component decisions. This
    parser remains only as a fail-closed fallback for old/resumed states.
    """
    dev_results = state.get("development_results", []) or []
    if not dev_results:
        return ""
    code = str(_field(dev_results[-1], "code", "") or "")
    match = re.search(r"COMPONENT_NAME\s*=\s*[\"']([\w.-]+)[\"']", code)
    return match.group(1) if match else ""


def _rejected_component_names(
    state: KaggleState,
    explicit_rejections: set[str] | None = None,
) -> list[str]:
    """Resolve every component explicitly rejected by robustness.

    The flagged/all-eligible fallbacks exist only for legacy states where
    nothing names a component. They must never fire when the caller already
    holds explicit rejections — otherwise an innocent, fully evidenced
    component gets quarantined alongside the real one.
    """
    details = state.get("robustness_failure_details", {}) or {}
    names = {
        str(name)
        for name in details.get("failed_components", []) or []
        if str(name)
    }
    approvals = dict(state.get("robustness_approved_components", {}) or {})
    names.update(
        str(name)
        for name, approved in approvals.items()
        if approved is False and str(name)
    )
    names.update(
        str(name) for name in (explicit_rejections or set()) if str(name)
    )
    if not names:
        flagged = _flagged_component_name(state)
        if flagged:
            names.add(flagged)
    if not names:
        names.update(
            str(name)
            for name, available in (
                state.get("oof_availability", {}) or {}
            ).items()
            if available is True
        )
    return sorted(names)


def _quarantine_rejected_candidate(
    state: KaggleState,
    component_names: list[str],
) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    """Quarantine mutable candidate files and restore a verified prior result."""
    working_dir = Path(state["working_directory"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    quarantine_root = (
        working_dir / ".rejected_candidates" / f"robustness_{timestamp}"
    )
    quarantined_submission = False
    submission = working_dir / "submission.csv"
    if submission.is_file():
        quarantine_root.mkdir(parents=True, exist_ok=True)
        submission.replace(quarantine_root / "submission.csv")
        quarantined_submission = True

    moved_by_component: dict[str, list[str]] = {}
    for component_name in component_names:
        safe_component = re.sub(r"[^A-Za-z0-9_.-]", "_", component_name)
        while ".." in safe_component:
            safe_component = safe_component.replace("..", "_")
        safe_component = safe_component.strip(".") or "unknown_component"
        moved = quarantine_component_artifacts(
            working_dir / "models",
            component_name,
            quarantine_dir=quarantine_root / "models" / safe_component,
        )
        if moved:
            moved_by_component[component_name] = moved

    restored = _restore_best_valid_submission(state, component_names)
    oof_availability = dict(state.get("oof_availability", {}) or {})
    component_results = dict(state.get("component_results", {}) or {})
    trusted_scores = dict(state.get("trusted_component_scores", {}) or {})
    approvals = dict(state.get("robustness_approved_components", {}) or {})
    for component_name in component_names:
        oof_availability[component_name] = False
        component_results.pop(component_name, None)
        trusted_scores.pop(component_name, None)
        approvals[component_name] = False

    existing_failed = set(state.get("failed_component_names", []) or [])
    new_failed = [
        name for name in component_names if name not in existing_failed
    ]
    updates: dict[str, Any] = {
        "oof_availability": oof_availability,
        "component_results": component_results,
        "trusted_component_scores": trusted_scores,
        "robustness_approved_components": approvals,
    }
    snapshot_owner = str(
        state.get("best_candidate_submission_component_name") or ""
    )
    accepted_owner = str(
        state.get("accepted_submission_score_owner") or ""
    )
    if accepted_owner in component_names:
        updates.update(
            {
                "accepted_submission_path": None,
                "accepted_submission_snapshot_path": None,
                "accepted_submission_sha256": None,
                "accepted_submission_cv_score": None,
                "accepted_submission_score_owner": None,
                "accepted_submission_score_source": None,
            }
        )
    best_model_name = str(state.get("best_single_model_name") or "")
    rejected_best_snapshot = snapshot_owner in component_names or (
        not snapshot_owner and best_model_name in component_names
    )
    if rejected_best_snapshot:
        updates.update(
            {
                "best_candidate_submission_snapshot_path": None,
                "best_candidate_submission_sha256": None,
                "best_candidate_submission_component_name": None,
            }
        )
    if best_model_name in component_names:
        updates.update(
            {
                "best_single_model_name": None,
                "best_single_model_score": None,
                "baseline_cv_score": None,
                # Declared `float`: None here would crash downstream readers
                # that use `state.get(key, 0.0)`, since the default does not
                # apply to a key present with value None.
                "current_performance_score": 0.0,
            }
        )
    if new_failed:
        updates["failed_component_names"] = new_failed

    audit = {
        "components": component_names,
        "quarantine_directory": (
            str(quarantine_root.relative_to(working_dir))
            if quarantine_root.exists()
            else None
        ),
        "submission_quarantined": quarantined_submission,
        "artifacts": moved_by_component,
        "restored_accepted_submission": restored is not None,
    }
    return updates, restored, audit


def _print_withheld_reason(
    state: KaggleState,
    evidence_failures: dict[str, list[str]],
    unapproved_components: list[str],
) -> None:
    """Say which constraint withheld the candidate.

    The gate applies its own evidence checks on top of the robustness agent's
    verdict, so it can withhold a candidate the agent just approved. Printing
    nothing made that combination read as "validation passed, then grading was
    refused for no stated reason" — the most expensive way to report a block,
    because the run gives no clue which check to look at.
    """
    print("\n" + "=" * 60)
    print("=  ROBUSTNESS GATE: Candidate withheld")
    print("=" * 60)
    if state.get("robustness_passed") is not True:
        print("   Robustness validation did not pass")
    for component_name, component_issues in sorted(evidence_failures.items()):
        print(f"   {component_name}: {'; '.join(component_issues)}")
    approval_only = sorted(set(unapproved_components) - set(evidence_failures))
    if approval_only:
        print(f"   Approval missing for: {', '.join(approval_only)}")


def robustness_gate_node(state: KaggleState) -> dict[str, Any]:
    """Turn robustness results into pass, one bounded recovery, or safe stop."""
    evidence_failures = _mle_evidence_failures(state)
    if state.get("robustness_abstained", False) and not evidence_failures:
        return {
            "robustness_gate_action": "pass",
            "robustness_recovery_count": 0,
            "current_candidate_valid": True,
            "workflow_valid": True,
            "force_refinement": False,
            "last_updated": datetime.now(),
        }

    approvals = dict(state.get("robustness_approved_components", {}) or {})
    for component_name in evidence_failures:
        approvals[component_name] = False
    eligible_components = {
        str(name)
        for name, available in (
            state.get("oof_availability", {}) or {}
        ).items()
        if available is True
    }
    unapproved_components = sorted(
        name for name in eligible_components if approvals.get(name) is not True
    )

    if state.get("robustness_passed") is True and not unapproved_components:
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
    if unapproved_components:
        failed_modules.append("component_approval")
        issues.append(
            "Missing explicit robustness approval for: "
            + ", ".join(unapproved_components)
        )
        suggestions.append("Validate every eligible prediction pair before ensemble")
    if evidence_failures:
        failed_modules.append("trusted_oof_evidence")
        for component_name, component_issues in sorted(evidence_failures.items()):
            issues.append(
                f"{component_name}: " + "; ".join(component_issues)
            )
        suggestions.append(
            "Regenerate canonical OOF/test artifacts and independently recompute "
            "the public metric before robustness approval"
        )

    _print_withheld_reason(state, evidence_failures, unapproved_components)

    rejected_components = _rejected_component_names(
        state,
        explicit_rejections=(
            set(unapproved_components) | set(evidence_failures.keys())
        ),
    )
    rejection_updates, restored, quarantine_audit = _quarantine_rejected_candidate(
        state,
        rejected_components,
    )

    if recovery_count < max_recoveries:
        print(
            f"   Requesting recovery attempt {recovery_count + 1}/{max_recoveries}"
        )
        issue_text = "; ".join(str(issue) for issue in issues[:8]) or "robustness validation failed"
        suggestion_text = "; ".join(str(item) for item in suggestions[:8])
        # Naming every rejected component does double duty: the planner is told
        # to re-plan them, and the developer skip-cache invalidates on name
        # mention. Without the explicit names stale code can be revalidated and
        # recovery cannot converge.
        rejected_text = ", ".join(repr(name) for name in rejected_components)
        regen_planner = (
            f" The plan MUST reimplement every rejected component: {rejected_text} "
            "without these issues."
            if rejected_text
            else ""
        )
        regen_developer = (
            f" Regenerate every rejected component from scratch: {rejected_text}."
            if rejected_text
            else ""
        )
        guidance = dict(state.get("refinement_guidance", {}) or {})
        guidance["planner_guidance"] = (
            "Create a targeted correction plan for failed robustness modules "
            f"{failed_modules}: {issue_text}.{regen_planner}"
        )
        guidance["developer_guidance"] = (
            f"Fix these guardrail failures before ensemble: {issue_text}. "
            f"{suggestion_text}{regen_developer}"
        ).strip()
        guidance["priority_fixes"] = issues[:8]

        return {
            **rejection_updates,
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
                    rejected_components=rejected_components,
                    quarantine=quarantine_audit,
                )
            ],
            "last_updated": datetime.now(),
        }

    preserved = restored is not None
    reason = (
        "robustness_failed_preserved_best_submission"
        if preserved
        else "robustness_failed_no_valid_submission"
    )
    error = None if preserved else "Robustness validation failed; candidate blocked"
    print(
        f"   Recovery budget exhausted ({recovery_count}/{max_recoveries}); "
        + (
            "grading an earlier verified snapshot"
            if preserved
            else "no verified snapshot survives, grading is blocked"
        )
    )

    return {
        **rejection_updates,
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
                rejected_components=rejected_components,
                quarantine=quarantine_audit,
            )
        ],
        "last_updated": datetime.now(),
    }
