"""
Run telemetry for paper instrumentation.

Collects lightweight, JSON-serializable events during a workflow run
(state channel: ``telemetry_events``) and summarizes the final state into
per-run measurements: guardrail interventions, recovery-route activations,
contamination-filter decisions, and learning-system activity.

The summary is written as ``telemetry.json`` next to the run artifacts so
ablation studies can compare component contributions across runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def make_event(
    category: str,
    name: str,
    iteration: int | None = None,
    **detail: Any,
) -> dict[str, Any]:
    """
    Create a telemetry event dict for the ``telemetry_events`` state channel.

    Args:
        category: Event group (e.g. "search", "guardrails", "recovery", "ablation")
        name: Event name within the category (e.g. "contamination_filtered")
        iteration: Workflow iteration the event belongs to
        **detail: Extra JSON-serializable context

    Returns:
        Event dict ready to be appended to state["telemetry_events"]
    """
    event: dict[str, Any] = {
        "category": category,
        "event": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if iteration is not None:
        event["iteration"] = iteration
    if detail:
        event["detail"] = _to_jsonable(detail)
    return event


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception:
            value = str(value)
        return _to_jsonable(value)
    return str(value)


def _get(obj: Any, field: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or a dict."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def collect_run_provenance(
    config: Any,
    repo_root: str | Path,
    **workflow_settings: Any,
) -> dict[str, Any]:
    """Collect reproducibility metadata without recording credentials or secrets."""
    root = Path(repo_root)
    commit: str | None = os.getenv("GITHUB_SHA") or os.getenv("KAGGLE_AGENTS_GIT_COMMIT")
    dirty: bool | None = None
    worktree_sha256: str | None = None
    try:
        if not commit:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if revision.returncode == 0:
                commit = revision.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())

        # A commit id is insufficient when the run uses uncommitted code.
        # Hash the tracked diff plus names/content of untracked files without
        # storing the source itself in telemetry.
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        )
        if diff.returncode == 0 and untracked.returncode == 0:
            fingerprint = hashlib.sha256(diff.stdout)
            for raw_name in sorted(name for name in untracked.stdout.split(b"\0") if name):
                relative = raw_name.decode("utf-8", errors="surrogateescape")
                path = root / relative
                fingerprint.update(b"\0" + raw_name + b"\0")
                try:
                    if path.is_symlink():
                        fingerprint.update(os.readlink(path).encode("utf-8"))
                    elif path.is_file():
                        fingerprint.update(path.read_bytes())
                except OSError:
                    fingerprint.update(b"<unreadable>")
            worktree_sha256 = fingerprint.hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass

    lock_path = root / "uv.lock"
    lock_sha256: str | None = None
    try:
        if lock_path.is_file():
            lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except OSError:
        pass

    llm = _get(config, "llm")
    gpu: str | None = None
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if gpu_query.returncode == 0:
            gpu = gpu_query.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": {
            "commit": commit,
            "dirty": dirty,
            "worktree_sha256": worktree_sha256,
        },
        "dependencies": {"lockfile": "uv.lock", "sha256": lock_sha256},
        "llm": {
            "provider": _get(llm, "provider"),
            "model": _get(llm, "model"),
            "temperature": _get(llm, "temperature"),
            "max_tokens": _get(llm, "max_tokens"),
            "planner_provider": _get(llm, "planner_provider"),
            "planner_model": _get(llm, "planner_model"),
            "developer_provider": _get(llm, "developer_provider"),
            "developer_model": _get(llm, "developer_model"),
            "evaluator_provider": _get(llm, "evaluator_provider"),
            "evaluator_model": _get(llm, "evaluator_model"),
            "dynamic_temperature_policy": {
                "initial_generation": 0.1,
                "feature_engineering": 0.2,
                "ensemble": 0.3,
                "refinement": 0.35,
                "error_fixing": [0.25, 0.4, 0.5],
                "debug": 0.45,
            },
        },
        "randomness": {"run_seed": os.getenv("RUN_SEED")},
        "workflow": _to_jsonable(workflow_settings),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "gpu": gpu,
            "kaggle_accelerator": os.getenv("KAGGLE_ACCELERATOR"),
            "colab_gpu": os.getenv("COLAB_GPU"),
        },
    }


def summarize_run_telemetry(state: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate the final workflow state into per-run measurements.

    Everything here is derived from state channels that already exist plus the
    append-only ``telemetry_events`` log, so it works for any run mode.

    Args:
        state: Final workflow state (dict-like)

    Returns:
        JSON-serializable telemetry summary
    """
    events = list(state.get("telemetry_events", []) or [])
    event_counts: dict[str, int] = {}
    for ev in events:
        key = f"{ev.get('category', '?')}.{ev.get('event', '?')}"
        event_counts[key] = event_counts.get(key, 0) + 1

    # Guardrails: per-module stats from accumulated ValidationResults
    guardrails: dict[str, dict[str, int]] = {}
    for vr in state.get("validation_results", []) or []:
        module = str(_get(vr, "module", "unknown"))
        entry = guardrails.setdefault(module, {"runs": 0, "passed": 0, "failed": 0, "issues": 0})
        entry["runs"] += 1
        if _get(vr, "passed", False):
            entry["passed"] += 1
        else:
            entry["failed"] += 1
        entry["issues"] += len(_get(vr, "issues", []) or [])

    # Development: component attempts and outcomes
    dev_results = state.get("development_results", []) or []
    dev_succeeded = sum(1 for r in dev_results if _get(r, "success", False))
    attempts_by_stage: dict[str, dict[str, int]] = {}
    for attempt in state.get("code_attempts", []) or []:
        stage = str(_get(attempt, "stage", "unknown"))
        entry = attempts_by_stage.setdefault(stage, {"attempts": 0, "succeeded": 0})
        entry["attempts"] += 1
        if _get(attempt, "success", False):
            entry["succeeded"] += 1

    # Search: retrieval volume + contamination-filter decisions
    search_audit = list(state.get("search_audit", []) or [])
    contamination_filtered = sum(
        1 for rec in search_audit if rec.get("filtered") and rec.get("same_competition")
    )
    search_excluded = sum(1 for rec in search_audit if rec.get("filtered"))
    # Per-reason rejection breakdown: without it a run cannot distinguish
    # "rejected as target contamination" (guard working) from "rejected for
    # unverifiable provenance" (over-filtering starving Search-First).
    search_rejection_reasons: dict[str, int] = {}
    for rec in search_audit:
        if rec.get("filtered"):
            reason = str(rec.get("filter_reason") or "unspecified")
            search_rejection_reasons[reason] = (
                search_rejection_reasons.get(reason, 0) + 1
            )
    search_rejection_reasons = dict(
        sorted(
            search_rejection_reasons.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    query_audit = [rec for rec in search_audit if rec.get("stage") == "query"]
    source_stages = {
        "metadata",
        "download",
        "provenance",
        "source_read",
        "source_parse",
        "code_scan",
        "selection",
    }
    source_audit = [rec for rec in search_audit if rec.get("stage") in source_stages]
    provider_candidates = [rec for rec in search_audit if rec.get("stage") == "provider_candidate"]
    accepted_source_records = [
        rec for rec in source_audit if rec.get("stage") == "code_scan" and not rec.get("filtered")
    ]
    accepted_source_identities = {
        str(rec.get("source_sha256") or rec.get("ref") or "").strip()
        for rec in accepted_source_records
        if rec.get("source_sha256") or rec.get("ref")
    }
    search_attempt_ids = sorted(
        {str(rec["search_attempt_id"]) for rec in search_audit if rec.get("search_attempt_id")}
    )

    eligible_retrieved = bool(
        state.get(
            "search_eligible_retrieved",
            state.get("search_effective"),
        )
    )
    eligibility_reason = state.get(
        "search_eligibility_reason",
        state.get("search_failure_reason"),
    )
    downstream_gain = state.get("search_downstream_gain")
    downstream_gain_status = state.get("search_downstream_gain_status")
    if not downstream_gain_status:
        downstream_gain_status = (
            "unknown_not_measured" if eligible_retrieved else "not_applicable_no_eligible_sources"
        )

    competition_info = state.get("competition_info")
    competition_name = _get(competition_info, "name", "") if competition_info else ""
    identity_aliases = (
        list(_get(competition_info, "identity_aliases", []) or []) if competition_info else []
    )
    identity_alias_evidence = (
        list(_get(competition_info, "identity_alias_evidence", []) or [])
        if competition_info
        else []
    )

    # Materialize the declared retrieval lineage in the persisted run artifact.
    # This joins opaque source IDs to final components and their independently
    # trusted evaluation status. It is an audit trail of declared inspiration,
    # never an estimate of retrieval's causal effect.
    def external_source_id(solution: Any) -> str | None:
        source = str(_get(solution, "source", "") or "").strip()
        if not source or source.lower().startswith(("fallback/", "internal/")):
            return None
        source_sha256 = str(
            _get(solution, "source_sha256", "") or ""
        ).strip().lower()
        identity = (
            f"content-sha256:{source_sha256}"
            if source_sha256
            else f"private-source-ref:{source}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"extsrc_{digest}"

    def finite_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    source_lineage: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for solution in state.get("sota_solutions", []) or []:
        source_id = external_source_id(solution)
        if not source_id or source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        source_lineage.append(
            {
                "external_source_id": source_id,
                "source_ref": _get(solution, "source"),
                "source_sha256": _get(solution, "source_sha256"),
                "eligibility_status": "retrieved_external_source",
            }
        )

    trusted_scores = state.get("trusted_component_scores") or {}
    if not isinstance(trusted_scores, dict):
        trusted_scores = {}
    oof_availability = state.get("oof_availability") or {}
    if not isinstance(oof_availability, dict):
        oof_availability = {}
    robustness_approved = state.get("robustness_approved_components") or {}
    if not isinstance(robustness_approved, dict):
        robustness_approved = {}
    component_results = state.get("component_results") or {}
    if not isinstance(component_results, dict):
        component_results = {}

    component_lineage: list[dict[str, Any]] = []
    for index, component in enumerate(state.get("ablation_plan", []) or []):
        name = str(_get(component, "name", "") or f"component_{index + 1}")
        declared_ids = [
            str(source_id)
            for source_id in list(
                _get(component, "external_source_ids", []) or []
            )
            if isinstance(source_id, str)
        ]
        known_ids = [
            source_id
            for source_id in dict.fromkeys(declared_ids)
            if source_id in seen_source_ids
        ]
        result = component_results.get(name)
        execution_success = _get(result, "success")
        oof_available = oof_availability.get(name) is True
        robustness_status = robustness_approved.get(name)
        score = finite_float(trusted_scores.get(name))
        trusted = (
            score is not None
            and execution_success is True
            and oof_available
            and robustness_status is True
        )
        if trusted:
            evidence_status = "trusted_canonical_oof"
        elif execution_success is False:
            evidence_status = "execution_failed"
        elif robustness_status is False:
            evidence_status = "robustness_rejected"
        elif not oof_available:
            evidence_status = "trusted_oof_unavailable"
        elif score is None:
            evidence_status = "trusted_score_missing"
        else:
            evidence_status = "inconsistent_trusted_score_state"

        component_lineage.append(
            {
                "component": name,
                "component_type": _get(component, "component_type"),
                "external_source_ids": known_ids,
                "declared_external_inspiration": bool(known_ids),
                "unknown_declared_source_ids": [
                    source_id
                    for source_id in dict.fromkeys(declared_ids)
                    if source_id not in seen_source_ids
                ],
                "execution_success": execution_success,
                "oof_available": oof_available,
                "robustness_approved": robustness_status,
                "trusted_oof_score": score if trusted else None,
                "evidence_status": evidence_status,
            }
        )

    stagnation = state.get("stagnation_detection") or {}

    summary: dict[str, Any] = {
        "competition": competition_name,
        "run_mode": state.get("run_mode"),
        "iterations": state.get("current_iteration", 0),
        "events": event_counts,
        "guardrails": {
            "by_module": guardrails,
            "overall_validation_score": state.get("overall_validation_score"),
        },
        "development": {
            "components_attempted": len(dev_results),
            "components_succeeded": dev_succeeded,
            "code_attempts_by_stage": attempts_by_stage,
        },
        # A run that stopped on a failure no candidate could repair must say so
        # in its own artifact: the sweep needs to tell an invalid harness
        # attempt apart from a real agent-quality outcome without re-reading
        # logs, and the fingerprints name which contracts were suppressed.
        "terminal_failure": {
            "origin": state.get("terminal_failure_origin"),
            "detail": _to_jsonable(state.get("terminal_failure_detail")),
            "failed_contract_fingerprints": sorted(
                str(fingerprint)
                for fingerprint in (state.get("failed_contract_fingerprints") or {})
            ),
            "workflow_valid": bool(state.get("workflow_valid", True)),
        },
        "search": {
            "target_identity": {
                "aliases": _to_jsonable(identity_aliases),
                "evidence": _to_jsonable(identity_alias_evidence),
            },
            "sota_solutions": len(state.get("sota_solutions", []) or []),
            "attempted": bool(state.get("search_attempted")),
            "eligible_retrieved": eligible_retrieved,
            "retrieval_treatment_eligible": bool(
                state.get("search_attempted") and eligible_retrieved
            ),
            "eligibility_reason": eligibility_reason,
            "last_attempt_eligible_retrieved": bool(
                state.get(
                    "search_last_attempt_eligible_retrieved",
                    eligible_retrieved,
                )
            ),
            "last_attempt_reason": state.get("search_last_attempt_reason"),
            "downstream_gain": downstream_gain,
            "downstream_gain_status": downstream_gain_status,
            "causal_effect_estimated": bool(
                downstream_gain is not None
                and downstream_gain_status == "measured_from_trusted_evaluation"
            ),
            "audit_records": len(search_audit),
            "contamination_filtered": contamination_filtered,
            "excluded": search_excluded,
            "rejection_reasons": search_rejection_reasons,
            "queries_audited": len(query_audit),
            "queries_filtered": sum(1 for rec in query_audit if rec.get("filtered")),
            "sources_audited": len(source_audit),
            "sources_filtered": sum(1 for rec in source_audit if rec.get("filtered")),
            "retrieval_errors": sum(1 for rec in search_audit if rec.get("error")),
            "provider_candidates_audited": len(provider_candidates),
            "provider_candidate_context_complete": (
                all(
                    rec.get("query")
                    and rec.get("iteration") is not None
                    and rec.get("search_attempt_id")
                    for rec in provider_candidates
                )
                if provider_candidates
                else None
            ),
            "provider_duplicates": sum(
                1 for rec in provider_candidates if rec.get("provider_decision") == "duplicate"
            ),
            "provider_parse_errors": sum(
                1 for rec in provider_candidates if rec.get("provider_decision") == "parse_error"
            ),
            "provider_below_min_votes": sum(
                1
                for rec in provider_candidates
                if rec.get("provider_decision") == "below_min_votes"
            ),
            "external_source_acceptance_records": len(accepted_source_records),
            "eligible_external_sources_unique": len(accepted_source_identities),
            "search_attempt_ids": search_attempt_ids,
            "queries_used": len(state.get("search_queries_used", []) or []),
            "records": [_to_jsonable(record) for record in search_audit],
        },
        "recovery_routes": {
            "sota_search_executions": event_counts.get("recovery.sota_search_executed", 0),
            "curriculum_activations": event_counts.get("recovery.curriculum_executed", 0),
            "prompt_refinement_runs": event_counts.get("recovery.prompt_refinement_executed", 0),
        },
        "learning_systems": {
            "preference_pairs": len(state.get("preference_pairs", []) or []),
            "reasoning_traces": len(state.get("reasoning_traces", []) or []),
            "self_evaluations": len(state.get("self_evaluations", []) or []),
            "curriculum_subtasks": len(state.get("curriculum_subtasks", []) or []),
            "optimized_prompts": sorted((state.get("optimized_prompts") or {}).keys()),
        },
        "ablation": {
            "disabled_components": sorted(
                {
                    ev.get("detail", {})
                    .get("component", ev.get("event", ""))
                    .replace("_skipped", "")
                    for ev in events
                    if ev.get("category") == "ablation"
                }
            ),
        },
        "retrieval_lineage": {
            "interpretation": "declared_inspiration_not_causal_effect",
            "eligible_sources": source_lineage,
            "components": component_lineage,
            "components_with_declared_external_inspiration": sum(
                1
                for component in component_lineage
                if component["declared_external_inspiration"]
            ),
            "components_with_trusted_oof_and_external_inspiration": sum(
                1
                for component in component_lineage
                if component["declared_external_inspiration"]
                and component["evidence_status"] == "trusted_canonical_oof"
            ),
        },
        "stagnation_detection": _to_jsonable(stagnation),
        "event_log": [_to_jsonable(ev) for ev in events],
    }
    return summary


def write_run_telemetry(state: dict[str, Any], output_dir: str | Path) -> Path | None:
    """
    Summarize telemetry from the final state and write ``telemetry.json``.

    Never raises: telemetry must not break a run.

    Args:
        state: Final workflow state
        output_dir: Directory to write telemetry.json into

    Returns:
        Path to the written file, or None on failure
    """
    try:
        summary = summarize_run_telemetry(state)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / "telemetry.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return target
    except Exception as e:  # pragma: no cover - defensive
        print(f"  Telemetry write failed (non-fatal): {e}")
        return None
