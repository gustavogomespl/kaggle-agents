"""
Audio competition fallback plan.

Converts audio to spectrograms, then uses image models.
"""

from typing import Any


def create_audio_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create fallback plan for audio competitions (mel-spectrograms + CNNs).

    Converts audio to spectrograms, then uses image models.

    Args:
        domain: Competition domain (audio_classification, audio_regression)
        sota_analysis: SOTA analysis results

    Returns:
        List of component dictionaries (4 components: 1 preprocessing + 2 models + 1 ensemble)
    """
    return [
        {
            "name": "mel_spectrogram_preprocessing",
            "component_type": "preprocessing",
            "description": "Convert audio files to mel-spectrograms using librosa. Save as PNG images for CNN input.",
            "estimated_impact": 0.20,
            "rationale": "Mel-spectrograms are the standard time-frequency representation for audio. Convert audio problem to computer vision problem, enabling use of powerful pre-trained image models.",
            "code_outline": "Use librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128), convert to dB scale with librosa.power_to_db(), normalize to [0, 255], save as 3-channel PNG to spectrograms/ directory"
        },
        {
            "name": "efficientnet_audio",
            "component_type": "model",
            "description": "EfficientNet-B0 trained on mel-spectrogram images. Transfer learning from ImageNet.",
            "estimated_impact": 0.25,
            "rationale": "CNNs excel at recognizing patterns in spectrograms (frequency bands, temporal patterns). EfficientNet provides excellent accuracy with computational efficiency.",
            "code_outline": "Load mel-spectrogram images with PyTorch DataLoader, torchvision.models.efficientnet_b0(pretrained=True), replace classifier, train with data augmentation on spectrograms"
        },
        {
            "name": "resnet_audio",
            "component_type": "model",
            "description": "ResNet50 for architectural diversity in ensemble.",
            "estimated_impact": 0.20,
            "rationale": "ResNet learns different features than EfficientNet due to different architecture (residual connections). Ensemble benefits from this diversity.",
            "code_outline": "Similar pipeline to EfficientNet but with torchvision.models.resnet50(pretrained=True)"
        },
        {
            "name": "audio_ensemble",
            "component_type": "ensemble",
            "description": "Weighted average of EfficientNet and ResNet predictions.",
            "estimated_impact": 0.12,
            "rationale": "Ensemble reduces overfitting to specific architecture biases and improves generalization.",
            "code_outline": "Load OOF predictions, compute weights by CV score, weighted average for test predictions"
        }
    ]
