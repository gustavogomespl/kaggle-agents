"""Tests for run telemetry (utils/telemetry.py)."""

from datetime import datetime
from types import SimpleNamespace

from kaggle_agents.agents.planner.sota_analysis import stable_external_source_id
from kaggle_agents.core.state import (
    AblationComponent,
    CodeAttempt,
    DevelopmentResult,
    SOTASolution,
    ValidationResult,
)
from kaggle_agents.utils.telemetry import (
    collect_run_provenance,
    make_event,
    summarize_run_telemetry,
    write_run_telemetry,
)


class TestMakeEvent:
    def test_basic_fields(self):
        event = make_event("search", "fallback_used", iteration=2)
        assert event["category"] == "search"
        assert event["event"] == "fallback_used"
        assert event["iteration"] == 2
        assert "timestamp" in event

    def test_detail_is_jsonable(self):
        event = make_event(
            "guardrails",
            "validation_completed",
            iteration=1,
            when=datetime(2026, 1, 1),
            modules={"leakage": {"passed": False}},
        )
        assert event["detail"]["when"] == "2026-01-01T00:00:00"
        assert event["detail"]["modules"]["leakage"]["passed"] is False

    def test_no_iteration_omits_key(self):
        event = make_event("ablation", "search_skipped", component="search")
        assert "iteration" not in event
        assert event["detail"]["component"] == "search"


def _make_state() -> dict:
    return {
        "competition_info": SimpleNamespace(
            name="test-comp",
            identity_aliases=["test-comp", "Official Test Competition"],
            identity_alias_evidence=[
                {
                    "alias": "test-comp",
                    "source": "competition_slug",
                },
                {
                    "alias": "Official Test Competition",
                    "source": "public_description_markdown_h1",
                    "line": 1,
                },
            ],
        ),
        "run_mode": "mlebench",
        "current_iteration": 3,
        "telemetry_events": [
            make_event("recovery", "sota_search_executed", iteration=2, found=3),
            make_event("recovery", "curriculum_executed", iteration=2, subtasks=2),
            make_event("ablation", "ensemble_skipped", iteration=1, component="ensemble"),
        ],
        "validation_results": [
            ValidationResult(module="leakage", passed=False, score=0.5, issues=["target leak"]),
            ValidationResult(module="leakage", passed=True, score=1.0),
            ValidationResult(module="format", passed=True, score=1.0),
        ],
        "overall_validation_score": 0.83,
        "development_results": [
            DevelopmentResult(code="x", success=True),
            DevelopmentResult(code="y", success=False),
        ],
        "code_attempts": [
            CodeAttempt(
                component_name="model_lgbm",
                component_type="model",
                stage="generate",
                attempt=1,
                success=False,
            ),
            CodeAttempt(
                component_name="model_lgbm",
                component_type="model",
                stage="debug",
                attempt=2,
                success=True,
            ),
        ],
        "search_audit": [
            {
                "ref": "a/b",
                "filtered": True,
                "same_competition": True,
                "stage": "metadata",
            },
            {
                "ref": "c/d",
                "filtered": False,
                "same_competition": False,
                "stage": "code_scan",
            },
        ],
        "sota_solutions": [object(), object()],
        "search_queries_used": ["q1", "q2"],
        "search_attempted": True,
        "search_eligible_retrieved": True,
        "search_last_attempt_eligible_retrieved": True,
        "search_eligibility_reason": None,
        "search_downstream_gain": None,
        "search_downstream_gain_status": "unknown_not_measured",
        "search_effective": True,
        "search_failure_reason": None,
        "preference_pairs": [object()],
        "reasoning_traces": [],
        "self_evaluations": [object(), object()],
        "curriculum_subtasks": [{"task": "fix"}],
        "optimized_prompts": {"developer": "..."},
        "stagnation_detection": {"stagnated": True, "trigger_sota_search": True},
    }


class TestSummarizeRunTelemetry:
    def test_event_counts(self):
        summary = summarize_run_telemetry(_make_state())
        assert summary["events"]["recovery.sota_search_executed"] == 1
        assert summary["events"]["recovery.curriculum_executed"] == 1
        assert summary["events"]["ablation.ensemble_skipped"] == 1

    def test_guardrails_by_module(self):
        summary = summarize_run_telemetry(_make_state())
        leakage = summary["guardrails"]["by_module"]["leakage"]
        assert leakage["runs"] == 2
        assert leakage["passed"] == 1
        assert leakage["failed"] == 1
        assert leakage["issues"] == 1
        assert summary["guardrails"]["overall_validation_score"] == 0.83

    def test_development_and_attempts(self):
        summary = summarize_run_telemetry(_make_state())
        assert summary["development"]["components_attempted"] == 2
        assert summary["development"]["components_succeeded"] == 1
        assert summary["development"]["code_attempts_by_stage"]["debug"]["succeeded"] == 1

    def test_search_and_contamination(self):
        summary = summarize_run_telemetry(_make_state())
        assert summary["search"]["sota_solutions"] == 2
        assert summary["search"]["attempted"] is True
        assert summary["search"]["eligible_retrieved"] is True
        assert summary["search"]["retrieval_treatment_eligible"] is True
        assert summary["search"]["eligibility_reason"] is None
        assert summary["search"]["downstream_gain"] is None
        assert summary["search"]["downstream_gain_status"] == "unknown_not_measured"
        assert summary["search"]["causal_effect_estimated"] is False
        assert summary["search"]["audit_records"] == 2
        assert summary["search"]["contamination_filtered"] == 1
        assert summary["search"]["excluded"] == 1
        assert summary["search"]["queries_audited"] == 0
        assert summary["search"]["sources_audited"] == 2
        assert summary["search"]["sources_filtered"] == 1
        assert summary["search"]["external_source_acceptance_records"] == 1
        assert summary["search"]["eligible_external_sources_unique"] == 1
        assert summary["search"]["retrieval_errors"] == 0
        assert summary["search"]["records"][0]["ref"] == "a/b"
        assert summary["search"]["target_identity"]["aliases"] == [
            "test-comp",
            "Official Test Competition",
        ]
        assert summary["search"]["target_identity"]["evidence"][1]["source"] == (
            "public_description_markdown_h1"
        )

    def test_search_audit_separates_queries_from_sources(self):
        state = _make_state()
        state["search_audit"].insert(
            0,
            {
                "query": "target competition solution",
                "stage": "query",
                "same_competition": True,
                "filtered": True,
            },
        )

        summary = summarize_run_telemetry(state)

        assert summary["search"]["queries_audited"] == 1
        assert summary["search"]["queries_filtered"] == 1
        assert summary["search"]["sources_audited"] == 2

    def test_operational_errors_are_not_mislabeled_as_sources(self):
        state = _make_state()
        state["search_audit"].append(
            {
                "stage": "initialization",
                "filtered": False,
                "error": "credentials unavailable",
            }
        )

        summary = summarize_run_telemetry(state)

        assert summary["search"]["audit_records"] == 3
        assert summary["search"]["sources_audited"] == 2
        assert summary["search"]["retrieval_errors"] == 1

    def test_empty_search_is_not_retrieval_treatment_eligible(self):
        state = _make_state()
        state["search_eligible_retrieved"] = False
        state["search_effective"] = False
        state["search_eligibility_reason"] = "retrieval_error:initialization"
        state["search_failure_reason"] = "retrieval_error:initialization"

        summary = summarize_run_telemetry(state)

        assert summary["search"]["retrieval_treatment_eligible"] is False

    def test_recovery_routes_and_ablation(self):
        summary = summarize_run_telemetry(_make_state())
        assert summary["recovery_routes"]["sota_search_executions"] == 1
        assert summary["recovery_routes"]["curriculum_activations"] == 1
        assert "ensemble" in summary["ablation"]["disabled_components"]

    def test_learning_systems(self):
        summary = summarize_run_telemetry(_make_state())
        assert summary["learning_systems"]["preference_pairs"] == 1
        assert summary["learning_systems"]["self_evaluations"] == 2
        assert summary["learning_systems"]["optimized_prompts"] == ["developer"]

    def test_declared_source_component_trusted_oof_lineage_is_persisted(self):
        state = _make_state()
        solution = SOTASolution(
            source="owner/public-other-task",
            title="not exposed to planner",
            score=0.0,
            votes=10,
            source_sha256="a" * 64,
        )
        source_id = stable_external_source_id(solution)
        assert source_id is not None
        state.update(
            {
                "sota_solutions": [solution],
                "ablation_plan": [
                    AblationComponent(
                        name="retrieved_candidate",
                        component_type="model",
                        code="fit model",
                        external_source_ids=[source_id],
                    )
                ],
                "component_results": {
                    "retrieved_candidate": DevelopmentResult(
                        code="fit model",
                        success=True,
                    )
                },
                "oof_availability": {"retrieved_candidate": True},
                "robustness_approved_components": {
                    "retrieved_candidate": True
                },
                "trusted_component_scores": {
                    "retrieved_candidate": 0.8123
                },
            }
        )

        lineage = summarize_run_telemetry(state)["retrieval_lineage"]

        assert lineage["interpretation"] == (
            "declared_inspiration_not_causal_effect"
        )
        assert lineage["eligible_sources"] == [
            {
                "external_source_id": source_id,
                "source_ref": "owner/public-other-task",
                "source_sha256": "a" * 64,
                "eligibility_status": "retrieved_external_source",
            }
        ]
        assert lineage["components"] == [
            {
                "component": "retrieved_candidate",
                "component_type": "model",
                "external_source_ids": [source_id],
                "declared_external_inspiration": True,
                "unknown_declared_source_ids": [],
                "execution_success": True,
                "oof_available": True,
                "robustness_approved": True,
                "trusted_oof_score": 0.8123,
                "evidence_status": "trusted_canonical_oof",
            }
        ]
        assert (
            lineage[
                "components_with_trusted_oof_and_external_inspiration"
            ]
            == 1
        )

    def test_empty_state_does_not_crash(self):
        summary = summarize_run_telemetry({})
        assert summary["iterations"] == 0
        assert summary["events"] == {}
        assert summary["search"]["contamination_filtered"] == 0


class TestWriteRunTelemetry:
    def test_writes_json_file(self, temp_data_dir):
        path = write_run_telemetry(_make_state(), temp_data_dir)
        assert path is not None
        assert path.name == "telemetry.json"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "recovery.sota_search_executed" in content


def test_collect_run_provenance_records_reproducibility_without_secrets(temp_data_dir, monkeypatch):
    monkeypatch.setenv("RUN_SEED", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-appear")
    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            model="gpt-5",
            temperature=0.1,
            max_tokens=8192,
            planner_provider=None,
            planner_model=None,
            developer_provider=None,
            developer_model=None,
            evaluator_provider=None,
            evaluator_model=None,
        )
    )

    provenance = collect_run_provenance(config, temp_data_dir, cv_folds=5)

    assert provenance["randomness"]["run_seed"] == "2"
    assert provenance["workflow"]["cv_folds"] == 5
    assert provenance["llm"]["model"] == "gpt-5"
    assert "must-not-appear" not in str(provenance)


def test_search_summary_breaks_down_rejections_by_reason():
    # Without the per-reason breakdown, "guard working" (target rejections)
    # and "over-filtering" (unverified provenance) are indistinguishable.
    state = _make_state()
    state["search_audit"] = [
        {"stage": "provenance", "filtered": True,
         "filter_reason": "unverified_source_competition"},
        {"stage": "provenance", "filtered": True,
         "filter_reason": "unverified_source_competition"},
        {"stage": "code_scan", "filtered": True,
         "filter_reason": "target_competition_source_reference",
         "same_competition": True},
        {"stage": "code_scan", "filtered": False, "filter_reason": None},
    ]

    summary = summarize_run_telemetry(state)

    assert summary["search"]["rejection_reasons"] == {
        "unverified_source_competition": 2,
        "target_competition_source_reference": 1,
    }
    assert summary["search"]["excluded"] == 3
