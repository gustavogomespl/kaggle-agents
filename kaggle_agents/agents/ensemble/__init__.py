"""Ensemble module for model stacking and blending.

This module provides functionality for creating model ensembles
using various strategies including stacking, blending, and rank averaging.
"""

from .agent import EnsembleAgent, ensemble_agent_node
from .alignment import (
    align_oof_by_canonical_ids,
    load_and_align_oof,
    stack_with_alignment,
    validate_oof_alignment,
)
from .fallback import (
    create_ensemble_with_fallback,
    fallback_to_best_single_model,
    recover_from_checkpoints,
)
from .meta_model import (
    constrained_meta_learner,
    diagnose_stacking_issues,
    dirichlet_weight_search,
    tune_meta_model,
)
from .prediction_pairs import find_prediction_pairs, validate_prediction_artifacts_contract
from .scoring import compute_oof_score, filter_by_score_threshold, score_predictions
from .stacking import load_cv_folds, stack_from_prediction_pairs
from .submission import safe_restore_submission, validate_and_align_submission
from .utils import class_orders_match, encode_labels, normalize_class_order


__all__ = [
    # Main exports
    "EnsembleAgent",
    "ensemble_agent_node",
    # Alignment functions
    "validate_oof_alignment",
    "align_oof_by_canonical_ids",
    "load_and_align_oof",
    "stack_with_alignment",
    # Scoring functions
    "score_predictions",
    "compute_oof_score",
    "filter_by_score_threshold",
    # Meta-model functions
    "tune_meta_model",
    "diagnose_stacking_issues",
    "constrained_meta_learner",
    "dirichlet_weight_search",
    # Prediction pair functions
    "find_prediction_pairs",
    "validate_prediction_artifacts_contract",
    # Stacking functions
    "stack_from_prediction_pairs",
    "load_cv_folds",
    # Submission functions
    "validate_and_align_submission",
    "safe_restore_submission",
    # Fallback functions
    "fallback_to_best_single_model",
    "recover_from_checkpoints",
    "create_ensemble_with_fallback",
    # Utility functions
    "normalize_class_order",
    "class_orders_match",
    "encode_labels",
]
