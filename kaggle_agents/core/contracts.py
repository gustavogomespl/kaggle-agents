"""
Pydantic Contracts for Inter-Agent Communication.

Provides validated data structures for passing information between
Kaggle agents (Developer, Ensemble, Evaluator).

Research: Agent K achieves 92.5% success rate partly due to explicit inter-agent contracts.

Pydantic >=2.0.0 is already a dependency.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentContractError(Exception):
    """Raised when agent output violates its contract."""

    pass


# === Developer Agent Contracts ===


class DeveloperOutput(BaseModel):
    """Contract for Developer agent output artifacts.

    Validates that code generation produced the expected artifacts
    with correct naming and structure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str = Field(..., min_length=10, description="Generated Python code")
    component_name: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Component name (lowercase, underscores)",
    )
    oof_path: Path | None = Field(default=None, description="Path to OOF predictions")
    test_path: Path | None = Field(default=None, description="Path to test predictions")
    train_ids_path: Path | None = Field(default=None, description="Path to train IDs")
    cv_score: float | None = Field(default=None, description="Cross-validation score")
    n_features: int = Field(default=0, ge=0, description="Number of features used")

    @field_validator("component_name")
    @classmethod
    def valid_component_name(cls, v: str) -> str:
        """Ensure component name is not a reserved word."""
        reserved = {"train", "test", "submission", "sample", "models", "data"}
        if v in reserved:
            raise ValueError(f"Component name '{v}' is reserved")
        return v


class PredictionArtifact(BaseModel):
    """Contract for saved prediction artifacts.

    Validates that OOF and test predictions have compatible shapes
    and are saved to expected locations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Model/component name")
    oof_path: Path = Field(..., description="Path to OOF predictions (.npy)")
    test_path: Path = Field(..., description="Path to test predictions (.npy)")
    oof_shape: tuple[int, ...] = Field(
        ..., min_length=1, max_length=2, description="Shape of OOF array"
    )
    test_shape: tuple[int, ...] = Field(
        ..., min_length=1, max_length=2, description="Shape of test array"
    )
    cv_score: float | None = Field(default=None, description="CV score for this model")
    class_order: list[str] | None = Field(
        default=None, description="Class order for multi-class"
    )

    @model_validator(mode="after")
    def validate_shapes_compatible(self) -> PredictionArtifact:
        """Ensure OOF and test have compatible column counts."""
        # OOF and test should have same number of columns
        if len(self.oof_shape) == 2 and len(self.test_shape) == 2:
            if self.oof_shape[1] != self.test_shape[1]:
                raise ValueError(
                    f"OOF columns ({self.oof_shape[1]}) != test columns ({self.test_shape[1]})"
                )
        return self


# === Ensemble Agent Contracts ===


class EnsembleInput(BaseModel):
    """Contract for Ensemble agent input.

    Validates that all required inputs for ensemble creation
    are present and valid.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    models_dir: Path = Field(..., description="Directory containing model artifacts")
    canonical_train_ids_path: Path | None = Field(
        default=None, description="Path to canonical train IDs"
    )
    problem_type: Literal["classification", "regression", "multi_label"] = Field(
        ..., description="Type of ML problem"
    )
    metric_name: str = Field(..., description="Evaluation metric name")
    n_train_samples: int = Field(..., gt=0, description="Number of training samples")
    n_test_samples: int = Field(..., gt=0, description="Number of test samples")
    n_classes: int = Field(default=1, ge=1, description="Number of classes")
    min_models: int = Field(default=2, ge=1, description="Minimum models for ensemble")

    @field_validator("models_dir")
    @classmethod
    def models_dir_exists(cls, v: Path) -> Path:
        """Validate that models directory exists."""
        if not v.exists():
            raise ValueError(f"Models directory does not exist: {v}")
        return v


class EnsembleResult(BaseModel):
    """Contract for Ensemble agent output.

    Validates that ensemble creation produced valid results
    with consistent weights and model counts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: Literal[
        "stacking", "blending", "rank_averaging", "caruana", "single_model"
    ] = Field(..., description="Ensemble strategy used")
    n_models_used: int = Field(..., ge=1, description="Number of models in ensemble")
    model_names: list[str] = Field(..., description="Names of models used")
    weights: list[float] | None = Field(
        default=None, description="Weights for each model"
    )
    oof_score: float | None = Field(default=None, description="Ensemble OOF score")
    submission_path: Path | None = Field(
        default=None, description="Path to submission file"
    )

    @model_validator(mode="after")
    def validate_weights_match_models(self) -> EnsembleResult:
        """Ensure weights count matches model count."""
        if self.weights is not None:
            if len(self.weights) != self.n_models_used:
                raise ValueError(
                    f"Weights count ({len(self.weights)}) != models count ({self.n_models_used})"
                )
        return self

    @model_validator(mode="after")
    def validate_model_names_match_count(self) -> EnsembleResult:
        """Ensure model names count matches n_models_used."""
        if len(self.model_names) != self.n_models_used:
            raise ValueError(
                f"Model names count ({len(self.model_names)}) != n_models_used ({self.n_models_used})"
            )
        return self


# === Validation Contracts ===


class ValidationInput(BaseModel):
    """Contract for validation/robustness agent input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code_path: Path = Field(..., description="Path to code file to validate")
    working_dir: Path = Field(..., description="Working directory")
    component_name: str = Field(..., description="Component being validated")
    timeout_seconds: int = Field(
        default=600, ge=60, le=7200, description="Timeout in seconds"
    )


class ValidationResult(BaseModel):
    """Contract for validation agent output."""

    success: bool = Field(..., description="Whether validation passed")
    component_name: str = Field(..., description="Component that was validated")
    cv_score: float | None = Field(default=None, description="CV score if successful")
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    execution_time: float = Field(default=0.0, ge=0, description="Execution time")
    artifacts_created: list[str] = Field(
        default_factory=list, description="List of artifacts created"
    )


# === Validation Decorator ===


def validate_contract(contract_class: type[BaseModel]):
    """Decorator to validate function output against a contract.

    Usage:
        @validate_contract(EnsembleResult)
        def create_ensemble(...) -> dict:
            return {...}

    Args:
        contract_class: Pydantic model class to validate against

    Returns:
        Decorated function that validates output
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                try:
                    validated = contract_class(**result)
                    return validated.model_dump()
                except Exception as e:
                    raise AgentContractError(
                        f"Contract violation in {func.__name__}: {e}"
                    )
            return result

        return wrapper

    return decorator


def validate_input(contract_class: type[BaseModel]):
    """Decorator to validate function input against a contract.

    Usage:
        @validate_input(EnsembleInput)
        def create_ensemble(input_data: dict) -> dict:
            ...

    Args:
        contract_class: Pydantic model class to validate against

    Returns:
        Decorated function that validates input
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Assume first positional arg or 'input_data' kwarg contains the dict
            input_data = kwargs.get("input_data") or (args[0] if args else None)
            if isinstance(input_data, dict):
                try:
                    contract_class(**input_data)
                except Exception as e:
                    raise AgentContractError(
                        f"Input contract violation in {func.__name__}: {e}"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# === Helper Functions ===


def create_prediction_artifact(
    name: str,
    oof_path: Path,
    test_path: Path,
    cv_score: float | None = None,
) -> PredictionArtifact | None:
    """Create a validated PredictionArtifact from file paths.

    Args:
        name: Model/component name
        oof_path: Path to OOF predictions
        test_path: Path to test predictions
        cv_score: Optional CV score

    Returns:
        Validated PredictionArtifact or None if validation fails
    """
    import numpy as np

    try:
        if not oof_path.exists() or not test_path.exists():
            return None

        oof = np.load(oof_path)
        test = np.load(test_path)

        return PredictionArtifact(
            name=name,
            oof_path=oof_path,
            test_path=test_path,
            oof_shape=oof.shape,
            test_shape=test.shape,
            cv_score=cv_score,
        )
    except Exception as e:
        print(f"[WARNING] Failed to create artifact for {name}: {e}")
        return None


def validate_prediction_artifacts(
    prediction_pairs: dict[str, tuple[Path, Path]],
) -> list[PredictionArtifact]:
    """Validate discovered prediction pairs against contract.

    Args:
        prediction_pairs: Dict mapping name to (oof_path, test_path)

    Returns:
        List of validated PredictionArtifact objects
    """
    import numpy as np

    validated: list[PredictionArtifact] = []

    for name, (oof_path, test_path) in prediction_pairs.items():
        try:
            oof = np.load(oof_path)
            test = np.load(test_path)

            artifact = PredictionArtifact(
                name=name,
                oof_path=oof_path,
                test_path=test_path,
                oof_shape=oof.shape,
                test_shape=test.shape,
            )
            validated.append(artifact)
        except Exception as e:
            print(f"   [SKIP] {name}: contract violation - {e}")

    return validated
