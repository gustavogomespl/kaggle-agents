"""Tests for planner agent stagnation handling and exploration plans."""

import importlib.util
from unittest.mock import patch

import pytest


# Check if dspy is available for planner tests
DSPY_AVAILABLE = importlib.util.find_spec("dspy") is not None


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
class TestCreateTabularFallbackPlan:
    """Tests for create_tabular_fallback_plan with stagnation rotation (live path)."""

    @pytest.fixture
    def fallback_fn(self):
        """Live fallback-plan function (import deferred so skipif applies first)."""
        from kaggle_agents.agents.planner.fallback_plans.tabular import (
            create_tabular_fallback_plan,
        )

        return create_tabular_fallback_plan

    @pytest.fixture
    def base_state(self):
        """Create base state for testing."""
        return {
            "domain_detected": "tabular_classification",
            "competition_name": "test-competition",
            "working_directory": "/tmp/test",
            "previous_plan_hashes": [],
            "development_history": [],
            "failed_component_names": [],
            "refinement_guidance": {},
        }

    def test_rotation_0_returns_lgbm_xgb(self, fallback_fn, base_state):
        """Iteration 0 should return lightgbm + xgboost."""
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=0,
        )

        component_names = [c["name"] for c in plan]
        assert "lightgbm_fast_cv" in component_names
        assert "xgboost_fast_cv" in component_names

    def test_rotation_1_returns_catboost_lgbm_tuned(self, fallback_fn, base_state):
        """Iteration 1 should return catboost + lgbm_tuned."""
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=1,
        )

        component_names = [c["name"] for c in plan]
        assert "catboost_fast_cv" in component_names
        assert "lightgbm_tuned_cv" in component_names

    def test_rotation_2_returns_mlp_rf(self, fallback_fn, base_state):
        """Iteration 2 should return neural_network_mlp + random_forest."""
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=2,
        )

        component_names = [c["name"] for c in plan]
        assert "neural_network_mlp" in component_names
        assert "random_forest_fast" in component_names

    def test_rotation_3_returns_target_encoding(self, fallback_fn, base_state):
        """Iteration 3 should return target_encoding_fe + catboost."""
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=3,
        )

        component_names = [c["name"] for c in plan]
        assert "target_encoding_fe" in component_names
        assert "catboost_fast_cv" in component_names

    def test_rotation_4_returns_intensive_lgbm(self, fallback_fn, base_state):
        """Iteration 4 should return lightgbm_intensive + catboost."""
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=4,
        )

        component_names = [c["name"] for c in plan]
        assert "lightgbm_intensive" in component_names
        assert "catboost_fast_cv" in component_names

    def test_rotation_wraps_around(self, fallback_fn, base_state):
        """Iteration 5 should wrap to rotation 0."""
        plan_iter_0 = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=0,
        )
        plan_iter_5 = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=5,
        )

        names_0 = {c["name"] for c in plan_iter_0 if c["component_type"] == "model"}
        names_5 = {c["name"] for c in plan_iter_5 if c["component_type"] == "model"}
        assert names_0 == names_5

    def test_all_rotations_different(self, fallback_fn, base_state):
        """All 5 rotations should produce different component sets."""
        component_sets = []
        for i in range(5):
            plan = fallback_fn(
                domain="tabular_classification",
                sota_analysis={},
                curriculum_insights="",
                fast_mode=True,
                state=base_state,
                stagnation_iteration=i,
            )
            names = frozenset(
                c["name"] for c in plan if c["component_type"] in ("model", "feature_engineering")
            )
            component_sets.append(names)

        # All 5 should be unique
        assert len(set(component_sets)) == 5, f"Expected 5 unique rotations, got: {component_sets}"

    def test_failed_components_are_filtered(self, fallback_fn, base_state):
        """Previously failed components must not be re-planned."""
        base_state["failed_component_names"] = ["lightgbm_fast_cv"]
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=True,
            state=base_state,
            stagnation_iteration=0,
        )

        component_names = [c["name"] for c in plan]
        assert "lightgbm_fast_cv" not in component_names
        # Still delivers 2 models + ensemble
        assert len(plan) == 3
        assert plan[-1]["component_type"] == "ensemble"

    def test_non_fast_mode_keeps_full_plan(self, fallback_fn, base_state):
        """Without fast_mode the full multi-model plan is returned (no rotation)."""
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            fast_mode=False,
            state=base_state,
            stagnation_iteration=3,
        )

        component_names = [c["name"] for c in plan]
        assert "lightgbm_optuna_tuned" in component_names
        assert "stacking_ensemble" in component_names


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
class TestSOTAOverride:
    """Tests for SOTA guidance override in fallback plans (live path)."""

    @pytest.fixture
    def fallback_fn(self):
        from kaggle_agents.agents.planner.fallback_plans.tabular import (
            create_tabular_fallback_plan,
        )

        return create_tabular_fallback_plan

    @pytest.fixture
    def state_with_sota(self):
        """Create state with SOTA guidance."""
        return {
            "domain_detected": "tabular_classification",
            "competition_name": "test-competition",
            "working_directory": "/tmp/test",
            "previous_plan_hashes": [],
            "development_history": [],
            "failed_component_names": [],
            "refinement_guidance": {
                "sota_guidance": {
                    "recommended_models": ["catboost", "feature_engineering"],
                    "techniques": ["target encoding", "gradient boosting"],
                }
            },
        }

    def test_sota_catboost_override(self, fallback_fn, state_with_sota):
        """SOTA recommending catboost should influence rotation."""
        # With SOTA guidance for catboost, iteration 1+ should prioritize it
        plan = fallback_fn(
            domain="tabular_classification",
            sota_analysis={"recommendations": "catboost works best"},
            curriculum_insights="",
            fast_mode=True,
            state=state_with_sota,
            stagnation_iteration=1,
        )

        component_names = [c["name"] for c in plan]
        # CatBoost should be present due to SOTA override
        assert any("catboost" in name for name in component_names)

    def test_extract_sota_recommendations(self):
        from kaggle_agents.agents.planner.fallback_plans.tabular import (
            extract_sota_recommendations,
        )

        recs = extract_sota_recommendations(
            {"models": "CatBoost and LightGBM", "notes": "use target encoding"}
        )
        assert "catboost" in recs
        assert "lightgbm" in recs
        assert "feature_engineering" in recs
        assert extract_sota_recommendations(None) == []


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
class TestCreateExplorationPlan:
    """Tests for _create_exploration_plan method."""

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner agent for testing."""
        from kaggle_agents.agents.planner_agent import PlannerAgent

        with patch.object(PlannerAgent, "__init__", lambda _self: None):
            return PlannerAgent()

    @pytest.fixture
    def base_state(self):
        """Create base state for testing."""
        return {
            "domain_detected": "tabular_classification",
            "competition_name": "test-competition",
            "working_directory": "/tmp/test",
            "previous_plan_hashes": [],
            "development_history": [],
            "ablation_plan": [],
            "failed_components": [],
            "refinement_guidance": {},
        }

    def test_exploration_avoids_used_components(self, mock_planner, base_state):
        """Exploration should avoid recently used components."""
        base_state["development_history"] = [
            {"component_name": "catboost_optuna_intensive"},
            {"component_name": "lgbm_intensive_7fold"},
        ]

        plan = mock_planner._create_exploration_plan(
            state=base_state,
            sota_analysis={},
        )

        component_names = [c["name"] for c in plan]
        # Should not include components from development_history
        assert "catboost_optuna_intensive" not in component_names
        assert "lgbm_intensive_7fold" not in component_names

    def test_exploration_avoids_planned_components(self, mock_planner, base_state):
        """Exploration should avoid currently planned components."""
        base_state["ablation_plan"] = [
            {"name": "sota_feature_engineering"},
        ]

        plan = mock_planner._create_exploration_plan(
            state=base_state,
            sota_analysis={},
        )

        component_names = [c["name"] for c in plan]
        # Should not include components from current plan
        assert "sota_feature_engineering" not in component_names

    def test_exploration_avoids_failed_components(self, mock_planner, base_state):
        """Exploration should avoid failed components."""
        base_state["failed_components"] = ["catboost_optuna_intensive"]

        plan = mock_planner._create_exploration_plan(
            state=base_state,
            sota_analysis={},
        )

        component_names = [c["name"] for c in plan]
        assert "catboost_optuna_intensive" not in component_names

    def test_exploration_responds_to_sota(self, mock_planner, base_state):
        """Exploration should include SOTA-recommended approaches."""
        plan = mock_planner._create_exploration_plan(
            state=base_state,
            sota_analysis={"text": "Use feature engineering and target encoding"},
        )

        component_names = [c["name"] for c in plan]
        # Should include feature engineering based on SOTA
        assert any("feature" in name.lower() for name in component_names)

    def test_exploration_returns_max_3_components(self, mock_planner, base_state):
        """Exploration should return at most 3 components."""
        plan = mock_planner._create_exploration_plan(
            state=base_state,
            sota_analysis={"text": "feature engineering, neural network, optuna, catboost"},
        )

        assert len(plan) <= 3

    def test_exploration_for_image_domain(self, mock_planner, base_state):
        """Exploration should return image-specific components for image domain."""
        base_state["domain_detected"] = "image_classification"

        plan = mock_planner._create_exploration_plan(
            state=base_state,
            sota_analysis={},
        )

        component_names = [c["name"] for c in plan]
        # Should include image-specific models
        assert any("efficientnet" in name.lower() or "convnext" in name.lower() for name in component_names)
