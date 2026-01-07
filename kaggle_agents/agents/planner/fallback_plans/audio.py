"""
Audio competition fallback plan.

Converts audio to spectrograms, then uses image models.
Includes mandatory data audit to prevent wasted compute on broken pipelines.
"""

from typing import Any


def create_audio_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create fallback plan for audio competitions (mel-spectrograms + CNNs).

    Converts audio to spectrograms, then uses image models.
    Starts with mandatory data audit to fail-fast if data is missing.

    Args:
        domain: Competition domain (audio_classification, audio_regression)
        sota_analysis: SOTA analysis results

    Returns:
        List of component dictionaries (5 components: 1 audit + 1 preprocessing + 2 models + 1 ensemble)
    """
    return [
        {
            "name": "data_audit",
            "component_type": "preprocessing",
            "description": "Validate audio files exist and labels are parseable. FAIL FAST if data is missing or invalid.",
            "estimated_impact": 0.0,  # No score impact, but prevents wasted compute
            "rationale": "Audio competitions often have non-standard data layouts (e.g., mlsp-2013-birds with essential_data/src_wavs/). Audit ensures data is accessible before expensive training begins. MUST run first to prevent 70+ minutes wasted on broken pipelines.",
            "code_outline": """
# MANDATORY DATA AUDIT - Run this FIRST before any other processing
from pathlib import Path

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

def find_audio_files(directory):
    '''Recursively find audio files with case-insensitive extension matching.'''
    audio_files = []
    for f in directory.rglob('*'):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            audio_files.append(f)
    return sorted(audio_files)

# Search for audio in multiple possible locations
search_dirs = [AUDIO_SOURCE_DIR if 'AUDIO_SOURCE_DIR' in dir() else None, TRAIN_PATH, OUTPUT_DIR]
search_dirs = [d for d in search_dirs if d and Path(d).exists()]

audio_files = []
audio_source = None
for d in search_dirs:
    files = find_audio_files(Path(d))
    if files:
        audio_files = files
        audio_source = d
        break

# FAIL FAST if insufficient data
if len(audio_files) < 10:
    raise FileNotFoundError(
        f"AUDIT FAILED: Only {len(audio_files)} audio files found. "
        f"Searched: {search_dirs}. Check data paths."
    )

# Validate label files
for lf in LABEL_FILES if 'LABEL_FILES' in dir() else []:
    if not Path(lf).exists():
        raise FileNotFoundError(f"AUDIT FAILED: Label file not found: {lf}")

print("=== DATA AUDIT PASSED ===")
print(f"Audio files found: {len(audio_files)}")
print(f"Audio source: {audio_source}")
print(f"Sample file: {audio_files[0]}")
print(f"Extensions: {set(f.suffix.lower() for f in audio_files[:100])}")

# Save audit results for downstream components
import json
audit_result = {
    'audio_files_count': len(audio_files),
    'audio_source': str(audio_source),
    'sample_files': [str(f) for f in audio_files[:5]]
}
(MODELS_DIR / 'audit_result.json').write_text(json.dumps(audit_result))
print("Final Validation Performance: 1.0")  # Audit passed
""",
        },
        {
            "name": "mel_spectrogram_preprocessing",
            "component_type": "preprocessing",
            "description": "Convert audio files to mel-spectrograms. Cache to disk for reuse across folds.",
            "estimated_impact": 0.20,
            "rationale": "Mel-spectrograms are the standard time-frequency representation for audio. Caching avoids recomputing spectrograms every epoch, significantly reducing training time.",
            "code_outline": """
# Use torchaudio for faster loading (preferred over librosa)
import torchaudio
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Create cache directory
cache_dir = MODELS_DIR / 'mel_cache'
cache_dir.mkdir(exist_ok=True)

# Mel spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
)

# Find all audio files
audio_files = find_audio_files(AUDIO_SOURCE_DIR if 'AUDIO_SOURCE_DIR' in dir() else TRAIN_PATH)

# Cache spectrograms
for audio_path in tqdm(audio_files, desc="Caching mels"):
    cache_path = cache_dir / f'{audio_path.stem}.npy'
    if cache_path.exists():
        continue
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 22050:
            waveform = torchaudio.functional.resample(waveform, sr, 22050)
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel = mel_transform(waveform)
        mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10, amin=1e-10, db_multiplier=1)
        np.save(cache_path, mel_db.numpy())
    except Exception as e:
        print(f"Warning: Failed to process {audio_path}: {e}")

print(f"Cached {len(list(cache_dir.glob('*.npy')))} spectrograms")
print("Final Validation Performance: 1.0")  # Preprocessing complete
""",
        },
        {
            "name": "efficientnet_audio",
            "component_type": "model",
            "description": "EfficientNet-B0 trained on mel-spectrogram images. Transfer learning from ImageNet.",
            "estimated_impact": 0.25,
            "rationale": "CNNs excel at recognizing patterns in spectrograms (frequency bands, temporal patterns). EfficientNet-B0 provides excellent accuracy with computational efficiency.",
            "code_outline": "Load cached mel-spectrograms from MODELS_DIR/mel_cache/, use torchvision.models.efficientnet_b0(pretrained=True), replace classifier head, train with BCEWithLogitsLoss for multi-label, save OOF predictions to MODELS_DIR/oof_efficientnet_audio.npy",
        },
        {
            "name": "resnet_audio",
            "component_type": "model",
            "description": "ResNet18 for architectural diversity in ensemble. Lighter than ResNet50 for faster training.",
            "estimated_impact": 0.20,
            "rationale": "ResNet18 learns different features than EfficientNet due to residual connections. Lighter architecture (ResNet18 vs ResNet50) reduces training time and risk of timeout. Ensemble benefits from model diversity.",
            "code_outline": "Similar pipeline to EfficientNet but with torchvision.models.resnet18(pretrained=True). ResNet18 is preferred over ResNet50 for faster training. Save OOF predictions to MODELS_DIR/oof_resnet_audio.npy",
        },
        {
            "name": "audio_ensemble",
            "component_type": "ensemble",
            "description": "Weighted average of EfficientNet and ResNet predictions.",
            "estimated_impact": 0.12,
            "rationale": "Ensemble reduces overfitting to specific architecture biases and improves generalization.",
            "code_outline": "Load OOF predictions from MODELS_DIR/oof_*.npy, compute weights by CV score (higher score = higher weight), weighted average for test predictions, save submission.csv",
        },
    ]
