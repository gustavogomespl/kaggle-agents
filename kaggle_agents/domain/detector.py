"""
Domain Detection Module for Kaggle Competitions.

Uses LLM to classify competition domain based on description and file metadata.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from ..core.state import CompetitionInfo, DomainType


class DomainDetector:
    """
    Detects the domain/type of a Kaggle competition using LLM.

    Supports granular domain classification for various competition types.
    """

    DOMAINS = [
        # Image-based
        "image_classification",
        "image_regression",
        "image_to_image",
        "image_segmentation",
        "object_detection",
        # Text-based
        "text_classification",
        "seq_to_seq",
        "text_regression",
        # Audio-based
        "audio_classification",
        "audio_regression",
        # Tabular
        "tabular_classification",
        "tabular_regression",
        # Time series
        "time_series_forecasting",
        # Multi-modal
        "multi_modal",
    ]

    PROMPT = """Classify this Kaggle competition into exactly ONE category.

Categories:
- image_classification: Classify images into categories (dog breeds, cancer detection, plant diseases)
- image_regression: Predict continuous values from images (age estimation, severity scores)
- image_to_image: Transform images (denoising, super-resolution, style transfer)
- image_segmentation: Pixel-wise classification of images
- object_detection: Locate and classify objects in images
- text_classification: Classify text (sentiment, toxicity, spam, author identification)
- seq_to_seq: Sequence to sequence (translation, text normalization, summarization)
- text_regression: Predict continuous values from text
- audio_classification: Classify audio signals (speaker, music genre, species by sound)
- audio_regression: Predict continuous values from audio
- tabular_classification: Classify rows in structured CSV data
- tabular_regression: Predict continuous values from structured CSV data
- time_series_forecasting: Predict future values from temporal sequences
- multi_modal: Combination of multiple data types (images + text + tabular)

Competition Name: {name}
Description: {description}
Data Files: {files}

IMPORTANT CLUES FOR DETECTION:
- Directories ending with "/" containing .jpg/.png files → image_* domain
- Directories with .wav/.mp3 files → audio_* domain
- Directories with .txt files → text_* domain
- Only .csv/.parquet files with no directories → tabular_* domain

Respond with ONLY the category name, nothing else. Example: image_classification"""

    DESCRIPTIONS = {
        "image_classification": "Classify images into discrete categories",
        "image_regression": "Predict continuous values from images",
        "image_to_image": "Transform images (denoising, super-resolution, style transfer)",
        "image_segmentation": "Pixel-wise classification of images",
        "object_detection": "Locate and classify objects in images",
        "text_classification": "Classify text into categories",
        "seq_to_seq": "Sequence to sequence transformation (translation, normalization)",
        "text_regression": "Predict continuous values from text",
        "audio_classification": "Classify audio signals",
        "audio_regression": "Predict continuous values from audio",
        "tabular_classification": "Classify rows in structured tabular data",
        "tabular_regression": "Predict continuous values from tabular data",
        "time_series_forecasting": "Predict future values from temporal data",
        "multi_modal": "Combination of multiple data types",
    }

    def __init__(self, llm: "BaseChatModel | None" = None):
        """
        Initialize the domain detector.

        Args:
            llm: LangChain LLM client. If None, defaults to tabular domain.
        """
        self.llm = llm

    def detect(
        self,
        competition_info: CompetitionInfo,
        data_directory: Path | str,
    ) -> tuple[DomainType, float]:
        """
        Detect the domain type of a competition using LLM.

        Args:
            competition_info: Competition metadata
            data_directory: Path to competition data files

        Returns:
            Tuple of (detected_domain, confidence_score)
        """
        if self.llm is None:
            return "tabular_classification", 0.5

        data_dir = Path(data_directory) if isinstance(data_directory, str) else data_directory

        # Heuristic fast-path: Use metadata from data pipeline if available
        metadata_type = None
        if hasattr(competition_info, 'data_files_metadata'):
            metadata_type = competition_info.data_files_metadata.get('data_type')

        if metadata_type == "image":
            desc_lower = (competition_info.description or "").lower()
            if "segment" in desc_lower:
                return "image_segmentation", 0.95
            elif "detect" in desc_lower or "object" in desc_lower:
                return "object_detection", 0.95
            elif "regression" in desc_lower:
                return "image_regression", 0.90
            else:
                return "image_classification", 0.90

        elif metadata_type == "audio":
            desc_lower = (competition_info.description or "").lower()
            if "regression" in desc_lower:
                return "audio_regression", 0.90
            else:
                return "audio_classification", 0.90

        elif metadata_type == "text":
            desc_lower = (competition_info.description or "").lower()
            if "translate" in desc_lower or "normalize" in desc_lower or "summarize" in desc_lower:
                return "seq_to_seq", 0.90
            elif "regression" in desc_lower:
                return "text_regression", 0.90
            else:
                return "text_classification", 0.90

        # No fast-path match, continue with LLM detection

        # Scan both files AND directories to get full picture
        files = []
        if data_dir.exists():
            for path in data_dir.glob("*"):
                if path.is_file():
                    files.append(path.name)
                elif path.is_dir():
                    # Analyze directory contents
                    contents = list(path.glob("*"))[:100]
                    if contents:
                        extensions: dict[str, int] = {}
                        for item in contents:
                            ext = item.suffix.lower()
                            extensions[ext] = extensions.get(ext, 0) + 1
                        if extensions:
                            dominant = max(extensions.items(), key=lambda x: x[1])
                            files.append(f"{path.name}/ ({len(contents)} files, mostly {dominant[0]})")
            files = files[:20]  # Limit to 20 entries

        prompt = self.PROMPT.format(
            name=competition_info.name,
            description=(competition_info.description or "")[:500],
            files=files if files else ["No files found"],
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            domain = content.strip().lower().replace(" ", "_")

            if domain in self.DOMAINS:
                return domain, 0.9  # type: ignore
            else:
                return "tabular_classification", 0.6

        except Exception:
            return "tabular_classification", 0.5

    def get_domain_description(self, domain: DomainType) -> str:
        """Get a human-readable description of a domain."""
        return self.DESCRIPTIONS.get(domain, "Unknown domain type")


# ==================== Convenience Function ====================


def detect_competition_domain(
    competition_info: CompetitionInfo,
    data_directory: Path | str,
    llm: "BaseChatModel | None" = None,
) -> tuple[DomainType, float]:
    """
    Convenience function to detect competition domain.

    Args:
        competition_info: Competition metadata
        data_directory: Path to competition data
        llm: Optional LLM client

    Returns:
        Tuple of (domain_type, confidence)
    """
    detector = DomainDetector(llm=llm)
    return detector.detect(competition_info, data_directory)
