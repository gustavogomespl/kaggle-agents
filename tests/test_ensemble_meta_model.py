"""Second-level ensemble selection must be evaluated out of sample."""

import numpy as np

from kaggle_agents.agents.ensemble.meta_model import (
    cross_fitted_meta_predictions,
    cross_validated_constrained_blend,
)
from kaggle_agents.agents.ensemble.scoring import score_predictions


def test_constrained_blend_returns_cross_fitted_full_coverage():
    y = np.linspace(0.0, 1.0, 18)
    oof_stack = np.stack(
        [
            y + np.sin(np.arange(18)) * 0.05,
            y + np.cos(np.arange(18)) * 0.08,
        ]
    )

    weights, cross_fitted, score = cross_validated_constrained_blend(
        oof_stack,
        y,
        "regression",
        "rmse",
    )

    assert cross_fitted is not None
    assert cross_fitted.shape == y.shape
    assert np.isfinite(cross_fitted).all()
    assert np.isclose(weights.sum(), 1.0)
    assert score == score_predictions(cross_fitted, y, "regression", "rmse")


def test_meta_predictions_use_nested_cv_and_cover_every_row():
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 15)
    meta_X = np.column_stack(
        [
            np.clip(y * 0.7 + rng.normal(0.15, 0.08, len(y)), 0, 1),
            np.clip(y * 0.6 + rng.normal(0.20, 0.10, len(y)), 0, 1),
        ]
    )

    model, predictions = cross_fitted_meta_predictions(
        meta_X,
        y,
        "classification",
        2,
        binary_single_col=True,
    )

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert hasattr(model, "fit")
