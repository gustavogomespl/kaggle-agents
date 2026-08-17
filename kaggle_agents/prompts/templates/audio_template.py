"""Data-derived audio templates used by the developer prompt."""

from __future__ import annotations

from collections import Counter
from math import log2
from statistics import median


def get_audio_config(
    observed_sample_rates: list[int],
    observed_durations: list[float],
) -> dict[str, int | float]:
    """Derive preprocessing parameters from successfully inspected audio files."""
    sample_rates = [int(value) for value in observed_sample_rates if int(value) > 0]
    durations = [float(value) for value in observed_durations if float(value) > 0]
    if not sample_rates or not durations:
        raise ValueError(
            "Audio configuration requires observed sample rates and durations"
        )

    sample_rate = Counter(sample_rates).most_common(1)[0][0]
    duration = median(durations)
    target_window = max(16, round(sample_rate * 0.025))
    n_fft = 2 ** round(log2(target_window))
    hop_length = max(1, n_fft // 4)
    n_mels = max(32, min(128, n_fft // 8))
    return {
        "sample_rate": sample_rate,
        "duration": duration,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "fmin": 0.0,
        "fmax": sample_rate / 2,
        "power": 2.0,
    }


AUDIO_CONFIG_TEMPLATE = '''
# === AUDIO CONFIGURATION: DERIVE FROM SUPPLIED FILES ===
def infer_audio_config(file_paths):
    """Inspect readable training files; never select parameters by task name."""
    import soundfile as sf
    from collections import Counter

    observations = []
    errors = []
    for path in list(file_paths)[:64]:
        try:
            info = sf.info(str(path))
            if info.samplerate > 0 and info.frames > 0:
                observations.append((int(info.samplerate), info.frames / info.samplerate))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if not observations:
        raise RuntimeError(
            "Could not inspect any training audio file; "
            f"sample errors={errors[:3]}"
        )

    sample_rates = [sample_rate for sample_rate, _ in observations]
    durations = np.asarray([duration for _, duration in observations], dtype=float)
    sample_rate = Counter(sample_rates).most_common(1)[0][0]
    duration = float(np.median(durations))
    target_window = max(16, round(sample_rate * 0.025))
    n_fft = 2 ** round(np.log2(target_window))
    return {
        "sample_rate": sample_rate,
        "duration": duration,
        "n_fft": int(n_fft),
        "hop_length": max(1, int(n_fft) // 4),
        "n_mels": max(32, min(128, int(n_fft) // 8)),
        "fmin": 0.0,
        "fmax": sample_rate / 2,
    }

# Resolve semantic record IDs to real paths first, then call:
# AUDIO_CONFIG = infer_audio_config(train_file_paths)
'''


AUDIO_LOAD_TEMPLATE = '''
def load_audio(path, config, offset=0.0):
    """Load one file using a configuration inferred from supplied audio."""
    sample_rate = int(config["sample_rate"])
    duration = float(config["duration"])
    try:
        waveform, _ = librosa.load(
            path,
            sr=sample_rate,
            duration=duration,
            offset=offset,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file {path}: {exc}") from exc
    if waveform.size == 0:
        raise ValueError(f"Decoded audio is empty: {path}")

    target_len = max(1, int(round(sample_rate * duration)))
    if len(waveform) < target_len:
        waveform = np.pad(
            waveform,
            (0, target_len - len(waveform)),
            mode="constant",
        )
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)
'''


AUDIO_MELSPEC_TEMPLATE = '''
def audio_to_melspec(waveform, config, normalize=True, to_db=True):
    """Convert audio using only parameters inferred from supplied files."""
    spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=int(config["sample_rate"]),
        n_mels=int(config["n_mels"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        fmin=float(config["fmin"]),
        fmax=float(config["fmax"]),
    )
    if to_db:
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    if normalize:
        low, high = spectrogram.min(), spectrogram.max()
        if not np.isfinite([low, high]).all() or high - low <= 1e-6:
            raise ValueError("Degenerate spectrogram; inspect the source audio")
        spectrogram = (spectrogram - low) / (high - low)
    return spectrogram.astype(np.float32)
'''


AUDIO_DATASET_TEMPLATE = '''
class AudioDataset(torch.utils.data.Dataset):
    """Audio dataset with an explicit, data-derived preprocessing contract."""

    def __init__(self, file_paths, config, targets=None, transform=None):
        self.file_paths = list(file_paths)
        self.config = dict(config)
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        waveform = load_audio(self.file_paths[index], self.config)
        spectrogram = audio_to_melspec(waveform, self.config)
        channels = np.stack([spectrogram] * 3, axis=0)
        tensor = torch.tensor(channels, dtype=torch.float32)
        if self.transform is not None:
            tensor = self.transform(tensor)
        if self.targets is None:
            return tensor
        target = self.targets[index]
        target_tensor = torch.as_tensor(target)
        return tensor, target_tensor
'''


AUDIO_FULL_TEMPLATE = f'''
import librosa
import numpy as np
import torch

{AUDIO_CONFIG_TEMPLATE}

{AUDIO_LOAD_TEMPLATE}

{AUDIO_MELSPEC_TEMPLATE}

{AUDIO_DATASET_TEMPLATE}
'''


AUDIO_CONSTRAINTS = """
## Audio Domain Constraints

- Inspect readable training files before choosing sample rate, clip duration,
  FFT size, hop length, or frequency bounds. Record the observed distributions
  and the derived values.
- Preserve semantic record IDs separately from resolved file paths. Use a
  supplied ID-to-file artifact when present; otherwise resolve exact stems and
  extensions without changing the IDs.
- A failed or empty decode is a data error. Raise it with the offending path;
  never replace it with a silent waveform.
- Infer whether targets are single-label, multi-label, or continuous from the
  public target artifact and metric before selecting loss and activation.
- Cache deterministic features when useful, but validate row/ID alignment
  before training and before writing OOF/test artifacts.
"""


AUDIO_MODEL_RECOMMENDATIONS = """
## Audio Model Selection

- Start with a budget-appropriate baseline over data-derived spectrograms.
- Consider pretrained audio or image encoders only when available within the
  runtime budget and compatible with the observed input/target contract.
- Choose loss, output shape, and activation from the verified target structure.
- Evaluate augmentations through the same trusted CV folds; do not assume that
  a domain-named augmentation is beneficial.
"""
