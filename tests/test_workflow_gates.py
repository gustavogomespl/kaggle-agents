"""Integration-level unit tests for workflow gates and causal ablations."""

from types import SimpleNamespace

from kaggle_agents.agents.developer.agent import DeveloperAgent
from kaggle_agents.agents.meta_evaluator.agent import MetaEvaluatorAgent
from kaggle_agents.agents.planner.agent import PlannerAgent, determine_planning_mode
from kaggle_agents.agents.planner.plan_refinement import refine_ablation_plan
from kaggle_agents.agents.robustness_agent import RobustnessAgent
from kaggle_agents.core.config import AgentConfig, reset_config, set_config
from kaggle_agents.core.state import (
    AblationComponent,
    CompetitionInfo,
    DevelopmentResult,
    SubmissionResult,
)
from kaggle_agents.workflow.nodes.robustness_gate import robustness_gate_node
from kaggle_agents.workflow.routing import (
    route_after_meta_evaluator,
    route_after_robustness_gate,
)


def _toggles(**values):
    defaults = {
        "disable_search": False,
        "disable_robustness": False,
        "disable_meta_evaluator": False,
        "disable_ensemble": False,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _failure_state(tmp_path, **overrides):
    state = {
        "working_directory": str(tmp_path),
        "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
        "robustness_passed": False,
        "robustness_abstained": False,
        "robustness_recovery_count": 0,
        "max_robustness_recoveries": 1,
        "robustness_failure_details": {
            "failed_modules": ["leakage"],
            "issues": ["target leakage"],
            "suggestions": ["fit transforms inside each fold"],
        },
        "refinement_guidance": {},
        "submissions": [],
        "current_iteration": 0,
    }
    state.update(overrides)
    return state


class TestRobustnessGate:
    def test_pass_reaches_ensemble(self, tmp_path):
        state = _failure_state(tmp_path, robustness_passed=True)
        updates = robustness_gate_node(state)
        assert updates["robustness_gate_action"] == "pass"
        assert updates["workflow_valid"] is True
        assert route_after_robustness_gate(updates) == "pass"

    def test_first_failure_requests_one_targeted_recovery(self, tmp_path):
        updates = robustness_gate_node(_failure_state(tmp_path))
        assert updates["robustness_gate_action"] == "recover"
        assert updates["robustness_recovery_count"] == 1
        assert updates["force_refinement"] is True
        assert updates["current_component_index"] == 0
        assert "target leakage" in updates["refinement_guidance"]["developer_guidance"]

    def test_second_failure_stops_invalid_without_submission(self, tmp_path):
        state = _failure_state(tmp_path, robustness_recovery_count=1)
        updates = robustness_gate_node(state)
        assert updates["robustness_gate_action"] == "fail"
        assert updates["workflow_valid"] is False
        assert updates["should_continue"] is False
        assert updates["termination_reason"] == "robustness_failed_no_valid_submission"

    def test_second_failure_restores_previously_valid_submission(self, tmp_path):
        previous = tmp_path / "submission_iter_0.csv"
        previous.write_text("id,target\n1,0.8\n", encoding="utf-8")
        submission = SubmissionResult(
            submission_id=None,
            public_score=0.8,
            file_path=str(previous),
            valid=True,
        )
        state = _failure_state(
            tmp_path,
            robustness_recovery_count=1,
            submissions=[submission],
        )
        updates = robustness_gate_node(state)
        assert updates["workflow_valid"] is True
        assert updates["termination_reason"] == "robustness_failed_preserved_best_submission"
        assert (tmp_path / "submission.csv").read_text(encoding="utf-8") == previous.read_text(
            encoding="utf-8"
        )


class TestCausalAblations:
    def test_robustness_ablation_abstains_without_perfect_score(self, tmp_path):
        agent = object.__new__(RobustnessAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_robustness=True))
        updates = agent(
            {
                "working_directory": str(tmp_path),
                "current_iteration": 0,
                "development_results": [],
            }
        )
        assert updates["robustness_abstained"] is True
        assert updates["robustness_passed"] is True
        assert updates["overall_validation_score"] is None

    def test_meta_ablation_clears_stale_recovery_signals(self):
        agent = object.__new__(MetaEvaluatorAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_meta_evaluator=True))
        updates = agent(
            {
                "current_iteration": 3,
                "stagnation_detection": {"trigger_sota_search": True},
                "failure_analysis": {"error_patterns": ["memory_error"]},
            }
        )
        assert updates["stagnation_detection"] == {}
        assert updates["failure_analysis"] == {}
        assert updates["refinement_guidance"] == {}
        assert updates["trigger_debug_loop"] is False

    def test_meta_ablation_routes_directly_to_iteration_control(self):
        config = AgentConfig()
        config.ablation_toggles.disable_meta_evaluator = True
        set_config(config)
        try:
            route = route_after_meta_evaluator(
                {
                    "stagnation_detection": {"trigger_sota_search": True},
                    "failure_analysis": {"error_patterns": ["memory_error"]},
                }
            )
            assert route == "skip_recovery"
        finally:
            reset_config()

    def test_planner_removes_ensemble_components(self):
        agent = object.__new__(PlannerAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_ensemble=True))
        plan = [
            AblationComponent("model", "model", "train"),
            AblationComponent("stack", "ensemble", "blend"),
        ]
        finalized, hashes = agent._finalize_plan(plan, {"previous_plan_hashes": []})
        assert [component.name for component in finalized] == ["model"]
        assert len(hashes) == 1

    def test_developer_skips_stale_ensemble_component(self, tmp_path):
        agent = object.__new__(DeveloperAgent)
        agent.config = SimpleNamespace(ablation_toggles=_toggles(disable_ensemble=True))
        updates = agent(
            {
                "working_directory": str(tmp_path),
                "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
                "ablation_plan": [AblationComponent("stack", "ensemble", "blend")],
                "current_component_index": 0,
                "current_iteration": 0,
            }
        )
        assert updates["current_component_index"] == 1
        assert updates["telemetry_events"][0]["event"] == "developer_ensemble_component_skipped"


class TestPlannerRefinementMode:
    def test_first_completed_cycle_uses_targeted_refinement(self):
        is_refinement, use_eureka = determine_planning_mode(
            {
                "current_iteration": 1,
                "crossover_guidance": {"preserve_components": ["model"]},
                "evolutionary_generation": 1,
            }
        )
        assert is_refinement is True
        assert use_eureka is False

    def test_guardrail_can_force_same_iteration_refinement(self):
        is_refinement, use_eureka = determine_planning_mode(
            {"current_iteration": 0, "force_refinement": True}
        )
        assert is_refinement is True
        assert use_eureka is False

    def test_eureka_remains_explicitly_available(self):
        is_refinement, use_eureka = determine_planning_mode(
            {
                "current_iteration": 2,
                "force_refinement": True,
                "force_eureka_planning": True,
            }
        )
        assert is_refinement is True
        assert use_eureka is True

    def test_duplicate_tabular_plan_uses_live_rotation(self):
        repeated = AblationComponent("repeat_model", "model", "train")
        repeated_hash = hash((("repeat_model", "model"),))
        state = {
            "ablation_plan": [repeated],
            "development_results": [DevelopmentResult(code="x", success=True)],
            "best_score": 0.0,
            "current_performance_score": 0.0,
            "competition_info": CompetitionInfo("demo", "", "auc", "classification"),
            "domain_detected": "tabular_classification",
            "run_mode": "mlebench",
            "previous_plan_hashes": [repeated_hash],
            "refinement_guidance": {},
            "failure_analysis": {},
            "failed_component_names": [],
        }

        plan = refine_ablation_plan(
            state=state,
            sota_analysis={},
            llm=None,
            use_dspy=True,
            refine_ablation_plan_prompt="{gap_analysis}{previous_plan}{test_results}{current_score}{memory_summary}",
            analyze_gaps_fn=lambda **_kwargs: {},
            create_refined_fallback_plan_fn=lambda *_args: [
                {
                    "name": "repeat_model",
                    "component_type": "model",
                    "code_outline": "train",
                    "estimated_impact": 0.1,
                }
            ],
            create_diversified_fallback_plan_fn=lambda *_args: [],
            get_memory_summary_for_planning_fn=lambda _state: "",
        )

        names = {component.name for component in plan}
        assert "catboost_fast_cv" in names
        assert "lightgbm_tuned_cv" in names
