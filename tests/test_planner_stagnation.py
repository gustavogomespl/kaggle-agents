"""Tests for planner agent stagnation handling and exploration plans."""

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# Check if dspy is available for planner tests
DSPY_AVAILABLE = importlib.util.find_spec("dspy") is not None


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
class TestCreateTabularFallbackPlan:
    """Tests for _create_tabular_fallback_plan with stagnation rotation."""

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner agent for testing."""
        from kaggle_agents.agents.planner_agent import PlannerAgent

        # Create planner with mocked LLM
        with patch.object(PlannerAgent, "__init__", lambda x: None):
            planner = PlannerAgent()
            planner.fast_mode = True
            planner.config = MagicMock()
            planner.config.fast_mode = True
            return planner

    @pytest.fixture
    def base_state(self):
        """Create base state for testing."""
        return {
            "domain_detected": "tabular_classification",
            "competition_name": "test-competition",
            "working_directory": "/tmp/test",
            "previous_plan_hashes": [],
            "development_history": [],
            "failed_components": [],
            "refinement_guidance": {},
        }

    def test_rotation_0_returns_lgbm_xgb(self, mock_planner, base_state):
        """Iteration 0 should return lightgbm + xgboost."""
        plan = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=0,
        )

        component_names = [c["name"] for c in plan]
        assert "lightgbm_fast_cv" in component_names
        assert "xgboost_fast_cv" in component_names

    def test_rotation_1_returns_catboost_lgbm_tuned(self, mock_planner, base_state):
        """Iteration 1 should return catboost + lgbm_tuned."""
        plan = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=1,
        )

        component_names = [c["name"] for c in plan]
        assert "catboost_fast_cv" in component_names
        assert "lightgbm_tuned_cv" in component_names

    def test_rotation_2_returns_mlp_rf(self, mock_planner, base_state):
        """Iteration 2 should return neural_network_mlp + random_forest."""
        plan = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=2,
        )

        component_names = [c["name"] for c in plan]
        assert "neural_network_mlp" in component_names
        assert "random_forest_fast" in component_names

    def test_rotation_3_returns_target_encoding(self, mock_planner, base_state):
        """Iteration 3 should return target_encoding_fe + catboost."""
        plan = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=3,
        )

        component_names = [c["name"] for c in plan]
        assert "target_encoding_fe" in component_names
        assert "catboost_fast_cv" in component_names

    def test_rotation_4_returns_intensive_lgbm(self, mock_planner, base_state):
        """Iteration 4 should return lightgbm_intensive + catboost."""
        plan = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=4,
        )

        component_names = [c["name"] for c in plan]
        assert "lightgbm_intensive" in component_names
        assert "catboost_fast_cv" in component_names

    def test_rotation_wraps_around(self, mock_planner, base_state):
        """Iteration 5 should wrap to rotation 0."""
        plan_iter_0 = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=0,
        )
        plan_iter_5 = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={},
            curriculum_insights="",
            state=base_state,
            stagnation_iteration=5,
        )

        names_0 = set(c["name"] for c in plan_iter_0 if c["component_type"] == "model")
        names_5 = set(c["name"] for c in plan_iter_5 if c["component_type"] == "model")
        assert names_0 == names_5

    def test_all_rotations_different(self, mock_planner, base_state):
        """All 5 rotations should produce different model sets."""
        model_sets = []
        for i in range(5):
            plan = mock_planner._create_tabular_fallback_plan(
                domain="tabular_classification",
                sota_analysis={},
                curriculum_insights="",
                state=base_state,
                stagnation_iteration=i,
            )
            models = frozenset(c["name"] for c in plan if c["component_type"] == "model")
            model_sets.append(models)

        # All 5 should be unique
        assert len(set(model_sets)) == 5, f"Expected 5 unique rotations, got: {model_sets}"


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
class TestSOTAOverride:
    """Tests for SOTA guidance override in fallback plans."""

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner agent for testing."""
        from kaggle_agents.agents.planner_agent import PlannerAgent

        with patch.object(PlannerAgent, "__init__", lambda x: None):
            planner = PlannerAgent()
            planner.fast_mode = True
            planner.config = MagicMock()
            planner.config.fast_mode = True
            return planner

    @pytest.fixture
    def state_with_sota(self):
        """Create state with SOTA guidance."""
        return {
            "domain_detected": "tabular_classification",
            "competition_name": "test-competition",
            "working_directory": "/tmp/test",
            "previous_plan_hashes": [],
            "development_history": [],
            "failed_components": [],
            "refinement_guidance": {
                "sota_guidance": {
                    "recommended_models": ["catboost", "feature_engineering"],
                    "techniques": ["target encoding", "gradient boosting"],
                }
            },
        }

    def test_sota_catboost_override(self, mock_planner, state_with_sota):
        """SOTA recommending catboost should influence rotation."""
        # With SOTA guidance for catboost, iteration 1+ should prioritize it
        plan = mock_planner._create_tabular_fallback_plan(
            domain="tabular_classification",
            sota_analysis={"recommendations": "catboost works best"},
            curriculum_insights="",
            state=state_with_sota,
            stagnation_iteration=1,
        )

        component_names = [c["name"] for c in plan]
        # CatBoost should be present due to SOTA override
        assert any("catboost" in name for name in component_names)


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
class TestCreateExplorationPlan:
    """Tests for _create_exploration_plan method."""

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner agent for testing."""
        from kaggle_agents.agents.planner_agent import PlannerAgent

        with patch.object(PlannerAgent, "__init__", lambda x: None):
            planner = PlannerAgent()
            return planner

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


class TestDataExplorationLogic:
    """Tests for EDA logic (imbalance detection, cardinality, missing values).

    These tests verify the EDA computation logic directly without importing
    the full workflow module (which requires dspy and other heavy dependencies).
    """

    def test_imbalance_detection_single_class(self):
        """Single class target should NOT cause division by zero."""
        # Simulate value_counts for single class
        value_counts = pd.Series([1.0], index=[0])
        min_count = value_counts.min()
        n_classes = len(value_counts)

        # This is the guard we added
        is_imbalanced = False
        imbalance_ratio = None
        if min_count > 0 and n_classes >= 2:
            imbalance_ratio = float(value_counts.max() / min_count)
            is_imbalanced = imbalance_ratio > 3.0

        # Should not crash, should be False
        assert is_imbalanced is False
        assert imbalance_ratio is None

    def test_imbalance_detection_binary_balanced(self):
        """Balanced binary should NOT be imbalanced."""
        value_counts = pd.Series([0.5, 0.5], index=[0, 1])
        min_count = value_counts.min()
        n_classes = len(value_counts)

        is_imbalanced = False
        imbalance_ratio = None
        if min_count > 0 and n_classes >= 2:
            imbalance_ratio = float(value_counts.max() / min_count)
            is_imbalanced = imbalance_ratio > 3.0

        assert is_imbalanced is False
        assert imbalance_ratio == 1.0

    def test_imbalance_detection_binary_imbalanced(self):
        """90/10 split should be detected as imbalanced."""
        value_counts = pd.Series([0.9, 0.1], index=[0, 1])
        min_count = value_counts.min()
        n_classes = len(value_counts)

        is_imbalanced = False
        imbalance_ratio = None
        if min_count > 0 and n_classes >= 2:
            imbalance_ratio = float(value_counts.max() / min_count)
            is_imbalanced = imbalance_ratio > 3.0

        assert is_imbalanced is True
        assert imbalance_ratio == 9.0

    def test_imbalance_detection_multiclass_balanced(self):
        """Balanced multiclass should NOT be imbalanced (threshold=5 for multiclass)."""
        value_counts = pd.Series([0.25, 0.25, 0.25, 0.25], index=[0, 1, 2, 3])
        min_count = value_counts.min()
        n_classes = len(value_counts)

        is_imbalanced = False
        imbalance_ratio = None
        if min_count > 0 and n_classes >= 2:
            imbalance_ratio = float(value_counts.max() / min_count)
            if n_classes == 2:
                is_imbalanced = imbalance_ratio > 3.0
            elif n_classes <= 100:
                is_imbalanced = imbalance_ratio > 5.0

        assert is_imbalanced is False
        assert imbalance_ratio == 1.0

    def test_imbalance_detection_multiclass_imbalanced(self):
        """Multiclass with 6:1 ratio should be detected as imbalanced."""
        value_counts = pd.Series([0.6, 0.3, 0.1], index=[0, 1, 2])
        min_count = value_counts.min()
        n_classes = len(value_counts)

        is_imbalanced = False
        imbalance_ratio = None
        if min_count > 0 and n_classes >= 2:
            imbalance_ratio = float(value_counts.max() / min_count)
            if n_classes == 2:
                is_imbalanced = imbalance_ratio > 3.0
            elif n_classes <= 100:
                is_imbalanced = imbalance_ratio > 5.0

        assert is_imbalanced is True
        assert abs(imbalance_ratio - 6.0) < 0.01  # Floating point tolerance

    def test_high_cardinality_detection(self):
        """Columns with >50 unique values should be flagged as high cardinality."""
        df = pd.DataFrame({
            "low_card": ["A", "B", "C"] * 100,
            "high_card": [f"user_{i}" for i in range(300)],
        })
        categorical_cols = ["low_card", "high_card"]

        high_cardinality_cols = []
        for col in categorical_cols:
            nunique = df[col].nunique()
            if nunique > 50:
                high_cardinality_cols.append(col)

        assert "high_card" in high_cardinality_cols
        assert "low_card" not in high_cardinality_cols

    def test_missing_values_detection(self):
        """Columns with missing values should be detected with correct ratio."""
        df = pd.DataFrame({
            "complete": [1, 2, 3, 4, 5],
            "partial": [1, np.nan, 3, np.nan, 5],  # 40% missing
        })

        missing_value_cols = {}
        for col in df.columns:
            missing_ratio = df[col].isna().mean()
            if missing_ratio > 0:
                missing_value_cols[col] = float(missing_ratio)

        assert "complete" not in missing_value_cols
        assert "partial" in missing_value_cols
        assert abs(missing_value_cols["partial"] - 0.4) < 0.01

    def test_constant_column_detection(self):
        """Columns with single unique value should be flagged as constant."""
        df = pd.DataFrame({
            "varying": [1, 2, 3, 4, 5],
            "constant": [42, 42, 42, 42, 42],
            "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        assert "constant" in constant_cols
        assert "all_null" in constant_cols
        assert "varying" not in constant_cols

    def test_correlation_detection(self):
        """Highly correlated pairs (>0.9) should be detected."""
        # Create perfectly correlated columns
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({
            "a": x,
            "b": x * 2 + 1,  # Perfectly correlated with a
            "c": np.random.randn(100),  # Independent
        })

        numeric_cols = ["a", "b", "c"]
        highly_correlated_pairs = []
        corr_matrix = df[numeric_cols].corr()
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.9:
                        highly_correlated_pairs.append((col1, col2, float(corr)))

        assert len(highly_correlated_pairs) == 1
        assert highly_correlated_pairs[0][0] == "a"
        assert highly_correlated_pairs[0][1] == "b"
        assert abs(highly_correlated_pairs[0][2] - 1.0) < 0.01
