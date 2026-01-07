"""
Audio-specific constraints for audio classification/regression tasks.
"""

AUDIO_CONSTRAINTS = """## AUDIO TASK REQUIREMENTS:

### 0. Data Audit (CRITICAL - MUST DO FIRST)
Before any processing, validate that audio data exists and is sufficient.
NEVER proceed with dummy data - fail fast if data is missing.

```python
from pathlib import Path

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

def find_audio_files(directory: Path) -> list[Path]:
    \"\"\"Recursively find audio files with case-insensitive extension matching.\"\"\"
    audio_files = []
    for f in directory.rglob('*'):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            audio_files.append(f)
    return sorted(audio_files)

# FAIL FAST if insufficient data
audio_files = find_audio_files(AUDIO_SOURCE_DIR if 'AUDIO_SOURCE_DIR' in dir() else TRAIN_PATH)
if len(audio_files) < 10:
    raise FileNotFoundError(
        f"AUDIT FAILED: Only {len(audio_files)} audio files found. "
        f"Expected at least 10. Check paths: TRAIN_PATH={TRAIN_PATH}, "
        f"AUDIO_SOURCE_DIR={'AUDIO_SOURCE_DIR' if 'AUDIO_SOURCE_DIR' in dir() else 'not set'}"
    )

print(f"=== DATA AUDIT ===")
print(f"Audio files found: {len(audio_files)}")
print(f"Extensions: {set(f.suffix.lower() for f in audio_files[:100])}")
print(f"Sample: {audio_files[0] if audio_files else 'NONE'}")
```

### 1. Path Constants (CRITICAL)
NEVER redefine TRAIN_PATH, TEST_PATH, MODELS_DIR, OUTPUT_DIR, or AUDIO_SOURCE_DIR.
These are auto-injected with correct values at the top of the file.

```python
# WRONG - DO NOT DO THIS:
BASE_DIR = Path(os.getcwd())  # NEVER redefine!
MODELS_DIR = BASE_DIR / "models"  # NEVER redefine!

# CORRECT - Use the injected constants directly:
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / "model.pth"
```

### 2. Recursive Audio Discovery with Case-Insensitive Matching
Audio files may be in subdirectories with various extensions and cases (.WAV, .wav, .Mp3).

```python
# WRONG - shallow search, misses nested files:
audio_files = list(TRAIN_PATH.glob("*.wav"))

# CORRECT - recursive with case-insensitive:
audio_files = []
for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
    audio_files.extend(TRAIN_PATH.rglob(ext))
    audio_files.extend(TRAIN_PATH.rglob(ext.upper()))  # Handle .WAV, .MP3
```

### 3. Robust Label File Parsing
Audio competition labels often come in non-standard formats (.txt with varying delimiters).

```python
import csv
from pathlib import Path

def parse_label_file(path: Path) -> pd.DataFrame:
    \"\"\"Parse label file with automatic delimiter detection.\"\"\"
    # Read sample for format detection
    content = path.read_text(encoding='utf-8', errors='ignore')
    sample = '\\n'.join(content.strip().split('\\n')[:20])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',\\t ;|')
        delimiter = dialect.delimiter
    except csv.Error:
        # Fallback: try common delimiters
        for delim in [',', '\\t', ' ', ';']:
            if delim in sample:
                delimiter = delim
                break
        else:
            delimiter = ','

    # For space-delimited, use regex separator
    if delimiter == ' ':
        return pd.read_csv(path, sep=r'\\s+', engine='python', header=None)
    return pd.read_csv(path, sep=delimiter, engine='python')
```

### 4. Mel Spectrogram Processing
Use consistent parameters and normalize properly.

```python
import librosa
import numpy as np

# Standard parameters for audio classification
SR = 22050  # Sample rate
DURATION = 5  # seconds
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

def load_and_process_audio(path: Path) -> np.ndarray:
    # Load with consistent sample rate
    y, sr = librosa.load(path, sr=SR, duration=DURATION)

    # Pad or truncate to fixed length
    target_length = SR * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db
```

### 5. Faster Audio Loading with torchaudio (Preferred)
librosa is CPU-intensive. Use torchaudio for better performance.

```python
import torchaudio
import torch

def load_audio_fast(path: Path, target_sr: int = 22050) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform
```

### 6. Spectrogram Caching (Reduces Training Time)
Cache spectrograms to disk to avoid recomputation every epoch.

```python
cache_dir = MODELS_DIR / 'mel_cache'
cache_dir.mkdir(exist_ok=True)

def get_cached_mel(audio_path: Path) -> np.ndarray:
    cache_path = cache_dir / f'{audio_path.stem}.npy'
    if cache_path.exists():
        return np.load(cache_path)

    mel = load_and_process_audio(audio_path)
    np.save(cache_path, mel)
    return mel
```

### 7. Model Selection (Lighter is Better for Time Limits)
Use smaller models to avoid timeouts.

```python
# PREFERRED - lighter, faster:
from torchvision.models import resnet18, efficientnet_b0
model = resnet18(pretrained=True)  # or efficientnet_b0

# AVOID - heavy, may timeout:
# from torchvision.models import resnet50, efficientnet_b3
```

### 8. Multi-Label vs Multi-Class
Audio competitions often use multi-label format. Use BCEWithLogitsLoss.

```python
# Multi-label: each sample can have multiple classes
criterion = nn.BCEWithLogitsLoss()  # Input: logits, Target: 0/1 for each class

# Single-class: CrossEntropyLoss
criterion = nn.CrossEntropyLoss()  # Target: class index
```

### 9. Print Data Summary at Start
Always verify data loading worked correctly.

```python
print(f"\\n=== DATA SUMMARY ===")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Target columns: {target_cols}")
print(f"Target distribution:\\n{train_df[target_cols].sum()}")
print(f"Audio path sample: {train_df['audio_path'].iloc[0]}")
print(f"===================\\n")
```

### 10. Timeout Handling
Implement soft deadline to prevent hard timeouts.

```python
import os
import time

_TIMEOUT_S = int(os.getenv('KAGGLE_AGENTS_COMPONENT_TIMEOUT_S', 4300))
_SOFT_DEADLINE_S = _TIMEOUT_S - 100  # Reserve 100s for cleanup
_START_TIME = time.time()

def check_timeout():
    return time.time() - _START_TIME > _SOFT_DEADLINE_S

# In training loop:
for epoch in range(EPOCHS):
    if check_timeout():
        print("[TIMEOUT] Soft deadline reached, saving and exiting")
        break
    # ... training code ...
```
"""
