"""Functions for discovering and validating prediction pairs."""

from pathlib import Path

from ...core.contracts import (
    PredictionArtifact,
    validate_prediction_artifacts,
)


def find_prediction_pairs(models_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Find matching OOF/Test prediction pairs under models/.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary mapping model name to (oof_path, test_path)
    """
    oof_files = sorted(models_dir.glob("oof_*.npy"))
    pairs: dict[str, tuple[Path, Path]] = {}
    for oof_path in oof_files:
        name = oof_path.stem.replace("oof_", "", 1)
        test_path = models_dir / f"test_{name}.npy"
        if test_path.exists():
            pairs[name] = (oof_path, test_path)
    return pairs


def validate_prediction_artifacts_contract(
    prediction_pairs: dict[str, tuple[Path, Path]],
) -> list[PredictionArtifact]:
    """Validate discovered prediction pairs against Pydantic contract.

    Uses the PredictionArtifact contract to ensure OOF and test predictions
    have compatible shapes and valid data.

    Args:
        prediction_pairs: Dict mapping name to (oof_path, test_path)

    Returns:
        List of validated PredictionArtifact objects
    """
    validated = validate_prediction_artifacts(prediction_pairs)
    print(f"      Contract validation: {len(validated)}/{len(prediction_pairs)} artifacts valid")
    return validated
