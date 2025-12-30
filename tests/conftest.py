"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.choice(["A", "B", "C"], n_samples),
        "feature_4": np.random.randint(0, 100, n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 1000

    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 2 * X1 + 3 * X2 + np.random.randn(n_samples) * 0.1

    data = {
        "feature_1": X1,
        "feature_2": X2,
        "feature_3": np.random.choice(["Low", "Medium", "High"], n_samples),
        "target": y,
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_kaggle_state():
    """Create mock KaggleState for testing."""
    from kaggle_agents.utils.state import KaggleState

    return KaggleState(
        messages=[],
        competition_name="test-competition",
        competition_type="classification",
        metric="accuracy",
        train_data_path="/tmp/train.csv",
        test_data_path="/tmp/test.csv",
        sample_submission_path="/tmp/submission.csv",
        eda_summary={},
        data_insights=[],
        features_engineered=[],
        feature_importance={},
        models_trained=[],
        best_model={},
        cv_scores=[],
        submission_path="",
        submission_score=0.0,
        leaderboard_rank=0,
        iteration=0,
        max_iterations=5,
        errors=[],
    )
