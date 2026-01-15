"""
Audio-specific constraints for audio classification/regression tasks.
"""

AUDIO_CONSTRAINTS = """## AUDIO TASK REQUIREMENTS:

## ⚠️ CRITICAL: USE CANONICAL FOLDS IF AVAILABLE ⚠️

For audio competitions, ALWAYS check if canonical folds exist FIRST:
- **CHECK** `CANONICAL_FOLDS_AVAILABLE` flag (set in path header)
- **DO NOT** create StratifiedKFold if canonical folds are available
- Labels may be embedded in filenames (e.g., `train12345_1.aif` means label=1)

### PRIORITY 1: Use canonical folds (if available)
```python
# ALWAYS check canonical folds first using the flag set in path header!
if CANONICAL_FOLDS_AVAILABLE:
    # Use pre-computed canonical folds for OOF alignment
    folds = CANONICAL_FOLDS  # Already loaded in path header
    train_ids = CANONICAL_TRAIN_IDS
    y = CANONICAL_Y
    print(f"[CANONICAL] Using {N_FOLDS} pre-computed folds")
else:
    # FALLBACK ONLY: Create folds if canonical doesn't exist
    from sklearn.model_selection import StratifiedKFold
    train_df = create_train_df_from_filenames(TRAIN_PATH)
    y = train_df['target'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(train_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, y)):
        folds[val_idx] = fold_idx
    train_ids = train_df['id'].values
```

### How to handle missing train.csv (labels in filenames):
```python
# If filenames contain labels like: train12345_0.aif (label=0), train12345_1.aif (label=1)
# Use the injected create_train_df_from_filenames() function:
train_df = create_train_df_from_filenames(TRAIN_PATH)  # Parses labels from filenames
y = train_df['target'].values
```

### 0. Data Audit (CRITICAL - MUST DO FIRST)
Before any processing, validate that audio data exists and is sufficient.
NEVER proceed with dummy data - fail fast if data is missing.

```python
from pathlib import Path

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff', '.aif'}

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
for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a', '*.aiff', '*.aif']:
    audio_files.extend(TRAIN_PATH.rglob(ext))
    audio_files.extend(TRAIN_PATH.rglob(ext.upper()))  # Handle .WAV, .MP3
```

### 3. Robust Label File Parsing (CRITICAL FOR AUDIO COMPETITIONS)
Audio competition labels often come in VARIABLE-WIDTH formats (.txt with varying number of labels per row).
Example: rec_id,label1 vs rec_id,label1,label2,label3 (multi-label format)

IMPORTANT: Do NOT use parse_label_file() directly - use the injected version at the top of your code.
If LABEL_FILES is defined, those are the label files to use.

```python
# USE THE INJECTED parse_label_file() function - it handles variable-width rows!
# It returns a DataFrame with columns: ['rec_id', 'label'] in long format

# Example usage:
if 'LABEL_FILES' in dir() and LABEL_FILES:
    for lf in LABEL_FILES:
        if lf.exists():
            labels_df = parse_label_file(lf)
            print(f"Loaded {len(labels_df)} label records from {lf.name}")
            # labels_df has columns: rec_id, label
            # Pivot to wide format for multi-label:
            label_matrix = labels_df.pivot_table(
                index='rec_id', columns='label', aggfunc='size', fill_value=0
            )
            label_matrix = (label_matrix > 0).astype(int)  # Binary labels
            break

# For ID to filename mapping:
if 'parse_id_mapping_file' in dir():
    for lf in LABEL_FILES:
        if '2filename' in str(lf) or 'id2file' in str(lf).lower():
            id_map = parse_id_mapping_file(lf)
            print(f"Loaded {len(id_map)} ID mappings from {lf.name}")
            break
```

### Pre-Loaded Labels (AUDIO COMPETITIONS)

When LABEL_FILES is defined, labels are PRE-LOADED as global variables at script start:
- `_PRELOADED_REC_IDS`: List of recording IDs
- `_PRELOADED_LABELS_DF`: DataFrame with ['rec_id', 'label'] columns (long format)
- `_PRELOADED_N_CLASSES`: Number of unique classes

Use these variables directly instead of parsing files yourself.

### 3.5. Filename-Based Label Parsing
Some audio competitions embed labels directly in filenames instead of a CSV.
Examples:
- `train12345_1.aif` → Label = 1 (whale call detected)
- `train12345_0.aif` → Label = 0 (no whale call)
- `sample_42.wav` → Label = 42 (multi-class support)

```python
import re

def create_train_df_from_filenames(audio_dir: Path, label_pattern: str = r'_(\d+)\.'):
    \"\"\"
    Create training DataFrame by parsing labels from filenames.

    Args:
        audio_dir: Directory containing audio files
        label_pattern: Regex pattern to extract label. Default r'_(\d+)\.' matches
                       any digits before the extension (e.g., '_0.', '_1.', '_42.')

    Returns:
        DataFrame with columns: id, path, target
    \"\"\"
    AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    audio_files = [f for f in audio_dir.rglob('*') if f.suffix.lower() in AUDIO_EXTS]

    data = []
    for file_path in audio_files:
        match = re.search(label_pattern, file_path.name)
        if match:
            label = int(match.group(1))
            data.append({
                'id': file_path.stem,
                'path': str(file_path),
                'target': label
            })

    if not data:
        raise ValueError(f"No files with label pattern '{label_pattern}' found in {audio_dir}")

    df = pd.DataFrame(data)
    print(f"Created train_df from filenames: {len(df)} samples")
    print(f"Label distribution: {df['target'].value_counts().to_dict()}")
    return df

# Usage when no train.csv exists:
# First, define where the extracted training data is located
# (adjust path based on where you extracted train.zip or train2.zip)
train_extract_dir = Path("train_extracted")  # or use TRAIN_PATH.parent / "train2" if already extracted

# Check if we need to parse labels from filenames
train_csv_exists = any('train' in f.name.lower() and f.suffix == '.csv'
                       for f in OUTPUT_DIR.glob('*.csv') if f.is_file())
if not train_csv_exists and train_extract_dir.exists():
    print("[INFO] No train.csv found - attempting to parse labels from filenames")
    train_df = create_train_df_from_filenames(train_extract_dir)
    y = train_df['target'].values
    train_ids = train_df['id'].values
```

### 4. Mel Spectrogram Processing
Use consistent parameters and normalize properly.

```python
import librosa
import numpy as np

# Sample rate recommendations:
# - Bird/wildlife calls: SR = 32000 or 44100 (high-frequency vocalizations)
# - General audio/speech: SR = 22050 (standard)
# - Music: SR = 44100 or 48000
SR = 32000  # Higher for bird competitions - captures up to 16kHz
DURATION = 5  # seconds
N_MELS = 128
N_FFT = 2048  # Higher resolution for better frequency discrimination
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

### 5. Audio Loading: Use torchaudio (PRIMARY) with librosa fallback

⚠️ **CRITICAL**: torchaudio is MORE ROBUST than librosa for loading various audio formats.
Logs from librosa often show: "PySoundFile failed. Trying audioread instead."
torchaudio handles .aiff, .wav, .mp3, .flac more reliably and is faster.

```python
import numpy as np
from pathlib import Path

# Try torchaudio first (faster, more robust)
try:
    import torchaudio
    AUDIO_BACKEND = "torchaudio"
except ImportError:
    AUDIO_BACKEND = "librosa"
    import librosa

def load_audio(path: Path | str, target_sr: int = 16000, duration: float | None = None) -> np.ndarray:
    \"\"\"
    Load audio file using torchaudio (primary) with librosa fallback.

    Args:
        path: Path to audio file
        target_sr: Target sample rate for resampling
        duration: Max duration in seconds (None = full audio)

    Returns:
        np.ndarray: Audio waveform (mono, target sample rate)
    \"\"\"
    path = Path(path)

    if AUDIO_BACKEND == "torchaudio":
        try:
            waveform, sr = torchaudio.load(str(path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            # Convert to numpy
            audio = waveform.numpy().squeeze()

            # Trim to duration if specified
            if duration is not None:
                max_samples = int(target_sr * duration)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                elif len(audio) < max_samples:
                    audio = np.pad(audio, (0, max_samples - len(audio)))

            return audio

        except Exception as e:
            print(f"[WARNING] torchaudio failed for {path.name}: {e}")
            # Fall through to librosa

    # Fallback to librosa
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=target_sr, duration=duration)
        return audio
    except Exception as e:
        print(f"[WARNING] librosa also failed for {path.name}: {e}")
        # Return silence as last resort (avoid crashing)
        silence_samples = int(target_sr * (duration or 2.0))
        return np.zeros(silence_samples, dtype=np.float32)

# Usage example:
# audio = load_audio("train/sample_001.aiff", target_sr=16000, duration=5.0)
```

**Why torchaudio over librosa:**
- **Format support**: Native support for .aiff (whale competition), .mp3, .flac without audioread
- **Speed**: ~2-3x faster than librosa for loading
- **Robustness**: Fewer "PySoundFile failed" errors
- **GPU-friendly**: Returns torch tensors, easy to use with PyTorch models

**Fallback chain:**
1. torchaudio (primary)
2. librosa (fallback)
3. Silence (last resort - prevents crash but logs warning)

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

### 8b. MLSP-STYLE MULTI-LABEL AUDIO CLASSIFICATION (CRITICAL)

For competitions with multiple species per recording (e.g., MLSP 2013 Birds), the label
format uses TWO-LEVEL DELIMITERS that standard CSV parsers cannot handle:

**Label file format:**
```
rec_id;label1,label2,label3   (semicolon separates ID from labels, comma separates labels)
0,3,7,12                      (OR comma-only: first is ID, rest are labels)
42,?                          (? marks hidden test labels)
```

**CORRECT parsing approach:**
```python
from kaggle_agents.utils.label_parser import parse_mlsp_multilabel

# Parse multi-label format (rec_id;label1,label2,label3 OR rec_id,label1,label2,label3)
rec_ids, y = parse_mlsp_multilabel(
    label_path,
    outer_delimiter=";",   # Separates rec_id from labels
    inner_delimiter=",",   # Separates individual label indices
    num_classes=19,        # MLSP has 19 species (auto-detected if not set)
    hidden_marker="?",     # Marker for hidden test labels
)
print(f"Detected {y.shape[1]} target columns")  # Should be 19, NOT 1

# Result:
# rec_ids: np.array of record IDs (int)
# y: np.array shape (n_samples, num_classes) with binary indicators (0 or 1)
```

**Loss function (MANDATORY for multi-label):**
```python
import torch.nn as nn

# ALWAYS use BCEWithLogitsLoss for multi-label classification
criterion = nn.BCEWithLogitsLoss()

# NEVER use CrossEntropyLoss - that's for single-label ONLY!
# CrossEntropyLoss expects exactly one class per sample
```

**Evaluation metric (micro-AUC for MLSP):**
```python
from sklearn.metrics import roc_auc_score

# Micro-AUC: evaluate all predictions as a single flattened array
# This is the official metric for MLSP 2013 Birds
micro_auc = roc_auc_score(y_true, y_pred, average='micro')

# DO NOT use average='macro' - that weights each species equally
# which is NOT how MLSP 2013 Birds is evaluated
print(f"Micro-AUC: {micro_auc:.4f}")
```

**Output layer (CRITICAL):**
```python
# Multi-label: output N logits (one per species), NO softmax in the model
self.fc = nn.Linear(hidden_dim, 19)  # 19 species for MLSP

# At inference: apply sigmoid to get independent probabilities
predictions = torch.sigmoid(logits)  # Shape: (batch, 19), values 0-1

# DO NOT use softmax - that forces predictions to sum to 1 (single-label)
```

**Common errors and fixes:**
- "1 target column detected" → Label file parsed incorrectly, use parse_mlsp_multilabel()
- "AUC = 0.5" (random baseline) → Wrong loss function or label alignment issue
- "Predictions sum to 1" → Using softmax instead of sigmoid for multi-label

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

### 11. SUBMISSION FORMAT (CRITICAL)

Audio competitions use two main submission formats. **ALWAYS check sample_submission.csv first!**

**WIDE FORMAT (BirdCLEF style) - One row per sample, one column per class:**
```csv
row_id,species_0,species_1,species_2,...,species_N
audio_0001,0.5,0.5,0.5,...,0.5
audio_0002,0.5,0.5,0.5,...,0.5
```

Code for WIDE format:
```python
# predictions shape: (num_samples, num_classes)
submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
target_cols = [c for c in submission.columns if c != submission.columns[0]]
for i, col in enumerate(target_cols):
    submission[col] = predictions[:, i]
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
```

**LONG FORMAT (MLSP 2013 style) - One row per (sample, class) pair:**
```csv
Id,Probability
100,0.5    # rec_id=1, species_id=0 → Id=1*100+0=100
101,0.5    # rec_id=1, species_id=1 → Id=1*100+1=101
...
118,0.5    # rec_id=1, species_id=18 → Id=1*100+18=118
200,0.5    # rec_id=2, species_id=0 → Id=2*100+0=200
```

Code for LONG format (MLSP pattern: Id = rec_id * 100 + species_id):
```python
# STEP 1: Get test recording IDs from CVfolds (CANONICAL SOURCE)
# TEST_REC_IDS is auto-injected from CVfolds*.txt - ALWAYS use it if available!
if 'TEST_REC_IDS' in dir() and TEST_REC_IDS:
    test_rec_ids = TEST_REC_IDS  # From CVfolds - CANONICAL
    print(f"[CVfolds] Using {len(test_rec_ids)} test recording IDs")
else:
    # FALLBACK: Infer from sample_submission (less reliable!)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    unique_ids = sorted(sample_sub['Id'].apply(lambda x: x // 100).unique())
    test_rec_ids = unique_ids
    print(f"[WARNING] TEST_REC_IDS not found, inferred {len(test_rec_ids)} IDs from submission")

# predictions shape: (len(test_rec_ids), num_classes)
submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Create mapping from submission ID to prediction
pred_map = {}
for i, rec_id in enumerate(test_rec_ids):
    for species_id in range(NUM_CLASSES):
        submission_id = rec_id * 100 + species_id  # MLSP ID pattern
        pred_map[submission_id] = predictions[i, species_id]

# Apply mapping to submission
submission['Probability'] = submission['Id'].map(pred_map)

# Fill any missing with default (0.05 = safe default)
submission['Probability'] = submission['Probability'].fillna(0.05)
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
```

**HOW TO DETECT FORMAT:**
1. Count columns: >2 columns = WIDE, exactly 2 columns = LONG
2. For LONG, check ID pattern:
   - If IDs are like 100, 101, ..., 118, 200, 201, ... → MLSP pattern (Id = rec_id * 100 + class_id)
   - If IDs have underscore (e.g., "1_0", "1_1") → underscore pattern
3. Use submission_format_info from state if available

### 12. PRECOMPUTED FEATURES (OPTIMIZATION)

Some audio competitions provide precomputed features. Check for these files:
- `histogram_features.txt` - Bag-of-words audio features
- `histogram_background.txt` - Background noise features
- `location_features.txt` - Recording location metadata
- `segment_features.txt` - Audio segment information

If precomputed features are available (check precomputed_features_info in state):
```python
# Load precomputed features instead of extracting from audio
if PRECOMPUTED_FEATURES_PATH.exists():
    features_df = pd.read_csv(PRECOMPUTED_FEATURES_PATH)
    print(f"Loaded precomputed features: {features_df.shape}")
    # Use these features for training instead of extracting MFCC/spectrograms
```

This can save significant time on feature extraction.

### 13. TRAIN_PATH vs TRAIN.CSV (CRITICAL - AUDIO COMPETITIONS)

**IMPORTANT**: In audio competitions, `TRAIN_PATH` is a DIRECTORY containing audio files, NOT a CSV file!

```python
# WRONG - Will cause IsADirectoryError:
train_df = pd.read_csv(TRAIN_PATH)  # ERROR! TRAIN_PATH is a directory!

# CORRECT - Use filename-based label parsing:
train_df = create_train_df_from_filenames(TRAIN_PATH)  # Parses labels from filenames
# Returns DataFrame with columns: id, path, target

# OR if canonical data exists:
train_ids = np.load(CANONICAL_TRAIN_IDS_PATH, allow_pickle=True)
y = np.load(CANONICAL_Y_PATH)
```

**Path constants explained:**
- `TRAIN_PATH`: Directory containing training audio files (e.g., `/data/train/`)
- `TEST_PATH`: Directory containing test audio files (e.g., `/data/test/`)
- `SAMPLE_SUBMISSION_PATH`: CSV file with submission format
- `CANONICAL_DIR`: Directory with pre-processed canonical data (folds, IDs, labels)

### 14. TEST PREDICTION ALIGNMENT (CRITICAL - PREVENTS score=0.50 BUG)

**THE #1 CAUSE OF RANDOM PREDICTIONS (score=0.50) IS ID MISALIGNMENT!**

When test predictions are saved in a different order than `sample_submission.csv` expects,
the ensemble creates RANDOM submissions despite having a well-trained model.

**MANDATORY PATTERN for audio models:**

```python
# Audio file extensions constant
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.aif', '.aiff', '.m4a'}

# STEP 1: Get test IDs - PREFER CVfolds injection over sample_submission inference!
# TEST_REC_IDS is auto-injected from CVfolds*.txt if available
if 'TEST_REC_IDS' in dir() and TEST_REC_IDS:
    test_ids = TEST_REC_IDS  # From CVfolds - CANONICAL SOURCE
    n_test = len(test_ids)
    print(f"[CVfolds] Using {n_test} test recording IDs from CVfolds")
else:
    # FALLBACK: Get from sample_submission (may be less reliable for some formats)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test_ids = sample_sub.iloc[:, 0].values  # First column is ID
    n_test = len(test_ids)
    print(f"[FALLBACK] Inferred {n_test} test IDs from sample_submission")

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
ID_COL = sample_sub.columns[0]

# STEP 2: Create ID-to-path mapping for test files
test_path_map = {f.stem: f for f in TEST_PATH.rglob('*') if f.suffix.lower() in AUDIO_EXTS}

# STEP 3: Load test files in SAMPLE_SUBMISSION ORDER
test_paths = []
for test_id in test_ids:
    test_id_str = str(test_id).replace('.aif', '').replace('.aiff', '').replace('.wav', '')
    path = test_path_map.get(test_id_str) or test_path_map.get(test_id)
    test_paths.append(path)

# STEP 4: Generate predictions in SAME ORDER as test_ids
test_preds = np.zeros(n_test)
for i, path in enumerate(test_paths):
    if path:
        test_preds[i] = model.predict(load_audio(path))

# STEP 5: Save test_ids alongside predictions (CRITICAL!)
np.save(MODELS_DIR / f'test_{COMPONENT_NAME}.npy', test_preds)
np.save(MODELS_DIR / f'test_ids_{COMPONENT_NAME}.npy', test_ids)
print(f'[LOG:INFO] Saved test predictions and IDs: {n_test} samples')

# STEP 6: Create submission using ID mapping (NOT row order!)
pred_map = dict(zip(test_ids, test_preds))
sample_sub[sample_sub.columns[1]] = sample_sub[ID_COL].map(pred_map)
assert sample_sub[sample_sub.columns[1]].notna().all(), 'Missing predictions!'
sample_sub.to_csv(SUBMISSION_PATH, index=False)
```

**WHY this matters:**
- Test files loaded via `glob()` return in FILESYSTEM ORDER (arbitrary)
- `sample_submission.csv` has its own ORDER (often alphabetical or by recording date)
- If predictions are assigned by row position instead of ID, they're SHUFFLED
- Result: Good validation AUC, but MLE-bench score = 0.50 (random chance)

**VERIFICATION (add to end of script):**
```python
# Verify test predictions vary (not constant)
test_preds = np.load(MODELS_DIR / f'test_{COMPONENT_NAME}.npy')
if np.std(test_preds) < 1e-6:
    raise ValueError('CRITICAL: Test predictions are constant! Check ID alignment.')
print(f'[VERIFY] Test preds: min={test_preds.min():.4f}, max={test_preds.max():.4f}, std={test_preds.std():.4f}')
```
"""
