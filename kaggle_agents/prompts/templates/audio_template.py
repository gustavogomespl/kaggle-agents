"""
Audio domain templates for code generation.

Provides reusable patterns for audio classification/detection tasks.
Based on best practices from librosa and common Kaggle audio competition patterns.
"""

# Audio preprocessing configuration constants
AUDIO_CONFIG_TEMPLATE = '''
# === AUDIO CONFIGURATION ===
# Sample rate: 32000 Hz recommended for bird vocalizations (captures up to 16 kHz)
# Use 22050 Hz for general audio (captures up to 11 kHz Nyquist limit)
SR = 32000  # Sample rate in Hz
DURATION = 5  # Clip duration in seconds
N_MELS = 128  # Number of mel frequency bins
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length between frames
FMIN = 20  # Minimum frequency for mel filterbank
FMAX = 16000  # Maximum frequency for mel filterbank (adjust based on SR)
'''

# Core audio loading function
AUDIO_LOAD_TEMPLATE = '''
def load_audio(path, sr=SR, duration=DURATION, offset=0.0):
    """Load and preprocess audio file.

    Args:
        path: Path to audio file
        sr: Target sample rate
        duration: Duration in seconds to load
        offset: Start time in seconds

    Returns:
        Audio waveform as numpy array, padded/trimmed to fixed length
    """
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, offset=offset)
        if len(y) == 0:
            y = np.zeros(sr * duration, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        y = np.zeros(sr * duration, dtype=np.float32)

    # Pad or trim to fixed length
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    return y.astype(np.float32)
'''

# Mel spectrogram conversion
AUDIO_MELSPEC_TEMPLATE = '''
def audio_to_melspec(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                     fmin=FMIN, fmax=FMAX, normalize=True, to_db=True):
    """Convert audio waveform to mel spectrogram.

    Args:
        y: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length between frames
        fmin: Minimum frequency
        fmax: Maximum frequency
        normalize: Whether to normalize to [0, 1]
        to_db: Whether to convert to decibels

    Returns:
        Mel spectrogram as numpy array (n_mels, time_frames)
    """
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )

    # Convert to decibels
    if to_db:
        S = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    if normalize:
        S_min, S_max = S.min(), S.max()
        if S_max - S_min > 1e-6:
            S = (S - S_min) / (S_max - S_min)
        else:
            S = np.zeros_like(S)

    return S.astype(np.float32)
'''

# PyTorch Dataset class
AUDIO_DATASET_TEMPLATE = '''
class AudioDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for audio classification.

    Converts audio files to mel spectrograms on-the-fly.
    Outputs 3-channel images for pretrained CNN compatibility.
    """

    def __init__(self, file_paths, labels=None, sr=SR, duration=DURATION,
                 n_mels=N_MELS, transform=None, mixup_alpha=0.0):
        """
        Args:
            file_paths: List of audio file paths
            labels: Optional labels array (n_samples,) or (n_samples, n_classes)
            sr: Sample rate
            duration: Clip duration in seconds
            n_mels: Number of mel bins
            transform: Optional torchvision transform
            mixup_alpha: Mixup augmentation alpha (0 to disable)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.transform = transform
        self.mixup_alpha = mixup_alpha
        self.is_train = labels is not None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load audio and convert to mel spectrogram
        y = load_audio(path, sr=self.sr, duration=self.duration)
        spec = audio_to_melspec(y, sr=self.sr, n_mels=self.n_mels)

        # Stack to 3 channels for pretrained CNN (e.g., ResNet, EfficientNet)
        # Shape: (3, n_mels, time_frames)
        spec_3ch = np.stack([spec] * 3, axis=0)

        # Apply transform if provided
        if self.transform:
            # Convert to 3-channel PIL Image (RGB) for torchvision transforms
            # Transpose from (3, H, W) to (H, W, 3) for PIL
            spec_uint8 = (spec_3ch * 255).astype(np.uint8)
            spec_pil = Image.fromarray(spec_uint8.transpose(1, 2, 0), mode='RGB')
            spec_3ch = self.transform(spec_pil)
        else:
            spec_3ch = torch.tensor(spec_3ch, dtype=torch.float32)

        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, np.ndarray):
                label = torch.tensor(label, dtype=torch.float32)
            else:
                label = torch.tensor(label, dtype=torch.long)
            return spec_3ch, label

        return spec_3ch
'''

# Complete audio pipeline template
AUDIO_FULL_TEMPLATE = f'''
import librosa
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

{AUDIO_CONFIG_TEMPLATE}

{AUDIO_LOAD_TEMPLATE}

{AUDIO_MELSPEC_TEMPLATE}

{AUDIO_DATASET_TEMPLATE}
'''

# Audio-specific constraints for prompts
AUDIO_CONSTRAINTS = """
## Audio Domain Constraints

1. SAMPLE RATE SELECTION:
   - Bird/wildlife vocalizations: Use SR=32000 Hz or SR=44100 Hz (high-frequency calls)
   - General audio/speech: Use SR=22050 Hz (standard)
   - Music: Use SR=44100 Hz or SR=48000 Hz
   - NEVER use SR=8000 Hz or lower - too much frequency loss

2. MEL SPECTROGRAM PARAMETERS:
   - n_mels=128 is a good default (balance between resolution and compute)
   - n_fft=2048 for high resolution, n_fft=1024 for speed
   - hop_length=512 is standard (n_fft / 4)
   - fmin=20, fmax=SR/2 (Nyquist limit) or fmax=16000 for 32kHz SR

3. DURATION HANDLING:
   - ALWAYS pad short clips to fixed duration
   - ALWAYS trim long clips (or use sliding window for inference)
   - 5-10 seconds is typical for bird/wildlife classification

4. 3-CHANNEL CONVERSION:
   - Pretrained CNNs (ResNet, EfficientNet) expect 3-channel input
   - Stack mel spectrogram 3 times: np.stack([spec] * 3, axis=0)
   - OR use delta features: [spec, delta, delta-delta]

5. NORMALIZATION:
   - Convert power to dB: librosa.power_to_db(S, ref=np.max)
   - Normalize to [0, 1] for CNN input
   - Mean/std normalization with ImageNet stats for pretrained models

6. LABEL FILE PARSING (CRITICAL for audio competitions):
   - Many audio competitions use non-standard label formats (.txt files)
   - ALWAYS use parse_label_file() helper for .txt label files
   - ALWAYS use parse_id_mapping_file() for rec_id to filename mapping
   - NEVER assume labels are in train.csv - check for .txt files first
"""

# Audio model architecture recommendations
AUDIO_MODEL_RECOMMENDATIONS = """
## Audio Model Architecture Recommendations

1. PRETRAINED CNN (Recommended for most tasks):
   - EfficientNet-B0/B2: Best balance of accuracy vs. speed
   - ResNet50: Robust baseline, well-tested
   - ConvNeXt: Modern architecture, good for larger datasets

2. AUDIO-SPECIFIC MODELS:
   - PANNs (Pretrained Audio Neural Networks): Trained on AudioSet
   - AST (Audio Spectrogram Transformer): State-of-the-art but compute-heavy
   - SED models: For sound event detection tasks

3. LOSS FUNCTIONS:
   - Multi-label: BCEWithLogitsLoss
   - Single-label: CrossEntropyLoss
   - Focal loss for imbalanced data

4. AUGMENTATIONS:
   - SpecAugment: Random time/frequency masking
   - Mixup: Blend samples with interpolated labels
   - Time shift: Random offset when loading audio
   - Pitch shift: librosa.effects.pitch_shift (compute-heavy)
"""
