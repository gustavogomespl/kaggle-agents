"""
Model Registry for Kaggle Agents.

Provides first-class tracking of trained models with:
- OOF and test prediction paths
- Performance metrics per fold
- Provenance tracking (feature set, preprocessing)
- Validation status

MLE-STAR uses model registry to:
- Select models for ensembling
- Track which models are compatible for blending
- Avoid retraining models that already exist
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class RegisteredModel:
    """First-class model with OOF/test predictions.

    Attributes:
        model_id: Unique identifier (hash or name)
        model_name: Human-readable name (e.g., "lightgbm_fast_cv")
        model_type: Type of model (lightgbm, xgboost, etc.)

        oof_path: Path to OOF predictions (.npy)
        test_path: Path to test predictions (.npy)
        train_ids_path: Path to train IDs used
        folds_path: Path to folds used

        metric: Metric name (e.g., "auc", "rmse")
        direction: Whether to maximize or minimize
        score_mean: Mean CV score
        score_std: Standard deviation of CV scores
        fold_scores: Individual fold scores

        feature_set_id: Which feature set was used (optional)
        preprocess_id: Which preprocessing was used (optional)
        component_name: Which ablation component created this

        validated: Whether the model passed validation
        validation_errors: List of validation errors
        created_at: When the model was registered
    """

    model_id: str
    model_name: str
    model_type: Literal["lightgbm", "xgboost", "catboost", "neural_net", "linear", "ensemble", "other"]

    # Prediction paths
    oof_path: str
    test_path: str
    train_ids_path: str
    folds_path: str

    # Performance
    metric: str
    direction: Literal["maximize", "minimize"]
    score_mean: float
    score_std: float
    fold_scores: list[float]

    # Provenance
    feature_set_id: str | None = None
    preprocess_id: str | None = None
    component_name: str = ""

    # Validation
    validated: bool = False
    validation_errors: list[str] = field(default_factory=list)

    # Metadata
    created_at: str = ""
    hyperparameters: dict = field(default_factory=dict)

    def __post_init__(self):
        """Set created_at if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def is_usable(self) -> bool:
        """Check if model is usable for ensembling."""
        return self.validated and len(self.validation_errors) == 0

    @property
    def comparable_score(self) -> float:
        """Return score normalized so higher is always better.

        This simplifies sorting logic across the codebase.
        For maximize metrics, returns score_mean directly.
        For minimize metrics, returns -score_mean.
        """
        return self.score_mean if self.direction == "maximize" else -self.score_mean

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        status = "OK" if self.is_usable else "INVALID"
        return (
            f"RegisteredModel({self.model_name}, {self.model_type}, "
            f"score={self.score_mean:.4f}±{self.score_std:.4f}, {status})"
        )

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "oof_path": self.oof_path,
            "test_path": self.test_path,
            "train_ids_path": self.train_ids_path,
            "folds_path": self.folds_path,
            "metric": self.metric,
            "direction": self.direction,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "fold_scores": self.fold_scores,
            "feature_set_id": self.feature_set_id,
            "preprocess_id": self.preprocess_id,
            "component_name": self.component_name,
            "validated": self.validated,
            "validation_errors": self.validation_errors,
            "created_at": self.created_at,
            "hyperparameters": self.hyperparameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegisteredModel:
        """Deserialize from checkpoint."""
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            model_type=data["model_type"],
            oof_path=data["oof_path"],
            test_path=data["test_path"],
            train_ids_path=data["train_ids_path"],
            folds_path=data["folds_path"],
            metric=data["metric"],
            direction=data["direction"],
            score_mean=data["score_mean"],
            score_std=data["score_std"],
            fold_scores=data["fold_scores"],
            feature_set_id=data.get("feature_set_id"),
            preprocess_id=data.get("preprocess_id"),
            component_name=data.get("component_name", ""),
            validated=data.get("validated", False),
            validation_errors=data.get("validation_errors", []),
            created_at=data.get("created_at", ""),
            hyperparameters=data.get("hyperparameters", {}),
        )


@dataclass
class ModelRegistry:
    """Registry of all trained models.

    Provides:
    - Registration of trained models with metadata
    - Lookup by type, component, or performance
    - Selection of ensemble candidates
    """

    models: dict[str, RegisteredModel] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        n_valid = len([m for m in self.models.values() if m.is_usable])
        return f"ModelRegistry({len(self.models)} models, {n_valid} validated)"

    def register(self, model: RegisteredModel) -> str:
        """Register a model.

        Args:
            model: The RegisteredModel to add

        Returns:
            The model_id
        """
        self.models[model.model_id] = model
        return model.model_id

    def get(self, model_id: str) -> RegisteredModel | None:
        """Get a model by ID."""
        return self.models.get(model_id)

    def get_by_type(self, model_type: str) -> list[RegisteredModel]:
        """Get all models of a specific type."""
        return [m for m in self.models.values() if m.model_type == model_type]

    def get_by_component(self, component_name: str) -> list[RegisteredModel]:
        """Get all models created by a specific component."""
        return [m for m in self.models.values() if m.component_name == component_name]

    def get_best_by_type(self, model_type: str) -> RegisteredModel | None:
        """Get the best model of a specific type.

        Best is determined by score_mean, respecting the direction.
        """
        models = self.get_by_type(model_type)
        if not models:
            return None

        return max(models, key=lambda m: m.comparable_score)

    def get_best_overall(self) -> RegisteredModel | None:
        """Get the best model across all types."""
        if not self.models:
            return None

        return max(self.models.values(), key=lambda m: m.comparable_score)

    def get_validated_models(self) -> list[RegisteredModel]:
        """Get all validated models."""
        return [m for m in self.models.values() if m.is_usable]

    def get_ensemble_candidates(
        self,
        min_score: float | None = None,
        model_types: list[str] | None = None,
    ) -> list[RegisteredModel]:
        """Get models suitable for ensembling.

        Args:
            min_score: Minimum score threshold (optional)
            model_types: Filter by model types (optional)

        Returns:
            List of validated models meeting criteria
        """
        candidates = self.get_validated_models()

        if model_types:
            candidates = [m for m in candidates if m.model_type in model_types]

        if min_score is not None:
            candidates = [
                m for m in candidates
                if (m.score_mean >= min_score if m.direction == "maximize" else m.score_mean <= min_score)
            ]

        # Sort by score (best first) - higher comparable_score is always better
        return sorted(candidates, key=lambda m: m.comparable_score, reverse=True)

    def get_diversity_candidates(self, top_n: int = 5) -> list[RegisteredModel]:
        """Get diverse models for ensembling.

        Selects the best model from each type to maximize diversity.
        """
        best_by_type = {}
        for model_type in ["lightgbm", "xgboost", "catboost", "neural_net", "linear"]:
            best = self.get_best_by_type(model_type)
            if best and best.is_usable:
                best_by_type[model_type] = best

        # Sort by score (best first) and take top_n
        models = list(best_by_type.values())
        return sorted(models, key=lambda m: m.comparable_score, reverse=True)[:top_n]

    def get_summary(self) -> str:
        """Get a summary of registered models."""
        lines = [f"## Model Registry ({len(self.models)} models)"]

        by_type: dict[str, list[RegisteredModel]] = {}
        for m in self.models.values():
            by_type.setdefault(m.model_type, []).append(m)

        for model_type, models in sorted(by_type.items()):
            lines.append(f"\n### {model_type} ({len(models)} models)")
            # Sort by comparable_score (best first)
            for m in sorted(models, key=lambda x: x.comparable_score, reverse=True)[:3]:
                status = "OK" if m.is_usable else "INVALID"
                lines.append(f"- {m.model_name}: {m.score_mean:.4f} +/- {m.score_std:.4f} [{status}]")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {"models": {k: v.to_dict() for k, v in self.models.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> ModelRegistry:
        """Deserialize from checkpoint."""
        registry = cls()
        for k, v in data.get("models", {}).items():
            registry.models[k] = RegisteredModel.from_dict(v)
        return registry
