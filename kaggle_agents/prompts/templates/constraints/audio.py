"""Audio constraints derived from observed data rather than benchmark schemas."""


AUDIO_CONSTRAINTS = """## AUDIO TASK REQUIREMENTS

Treat every audio layout as unknown until it is supported by the staged public
files, the canonical contract, or `sample_submission.csv`. Do not assume a
species domain, filename convention, sample rate, label delimiter, identifier
name, or submission encoding from prior competitions.

### 1. Use the injected contract first

- Never redefine `TRAIN_PATH`, `TEST_PATH`, `MODELS_DIR`,
  `SAMPLE_SUBMISSION_PATH`, `SUBMISSION_PATH`, or `AUDIO_SOURCE_DIR`.
- If `CANONICAL_FOLDS_AVAILABLE` is true, use `CANONICAL_FOLDS`,
  `CANONICAL_TRAIN_IDS`, and `CANONICAL_Y` exactly as injected.
- If `TRAIN_REC_IDS`, `TEST_REC_IDS`, `TRAIN_FILE_PATHS`, or
  `TEST_FILE_PATHS` exist, keep record identifiers separate from filesystem
  paths. Predictions and OOF rows align by record identifier.
- If preloaded label/path variables exist, use them. They were built from files
  discovered in this run. Do not reparse guessed filenames.
- Use only label/mapping files listed in `LABEL_FILES`; never invent a filename.

```python
if CANONICAL_FOLDS_AVAILABLE:
    folds = np.asarray(CANONICAL_FOLDS)
    train_ids = np.asarray(CANONICAL_TRAIN_IDS).astype(str)
    y = np.asarray(CANONICAL_Y)
else:
    # Build folds only after the observed labels and record IDs are aligned.
    from sklearn.model_selection import KFold, StratifiedKFold
    if y.ndim == 1 and len(np.unique(y)) > 1:
        _, class_counts = np.unique(y, return_counts=True)
        n_splits = min(5, int(class_counts.min()))
        if n_splits < 2:
            raise ValueError("Each class needs at least two records for CV")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RUN_SEED,
        )
        split_targets = y
    else:
        n_splits = min(5, len(y))
        if n_splits < 2:
            raise ValueError("At least two records are required for CV")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RUN_SEED,
        )
        split_targets = None
    folds = np.full(len(y), -1, dtype=int)
    split_iter = splitter.split(np.arange(len(y)), split_targets)
    for fold, (_, valid_idx) in enumerate(split_iter):
        folds[valid_idx] = fold
```

### 2. Audit the supplied audio and schemas

Discover extensions recursively and case-insensitively. Inspect representative
headers, shapes, sampling rates, durations, label cardinality, and submission
columns before selecting a pipeline. Fail if train/test files or alignable
labels are missing; never create dummy examples or synthetic labels.

```python
from pathlib import Path

KNOWN_AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"
}
audio_root = Path(AUDIO_SOURCE_DIR if "AUDIO_SOURCE_DIR" in globals() else TRAIN_PATH)
audio_files = sorted(
    path for path in audio_root.rglob("*")
    if path.is_file() and path.suffix.lower() in KNOWN_AUDIO_EXTS
)
if not audio_files:
    raise FileNotFoundError(f"No audio files found under {audio_root}")

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
id_col = sample_sub.columns[0]
target_cols = sample_sub.columns[1:].tolist()
if not target_cols:
    raise ValueError("sample_submission has no prediction columns")
print({
    "audio_files": len(audio_files),
    "extensions": sorted({path.suffix.lower() for path in audio_files}),
    "submission_shape": sample_sub.shape,
    "target_columns": len(target_cols),
})
```

Label parsing must be conditional on the observed file:

- Fixed-width table: infer delimiter/header and identify ID/target columns from
  its actual columns.
- Variable-width rows: use the injected parser only after the sampled rows
  demonstrate sparse multi-label structure.
- Labels in filenames: use only when no label table/canonical target exists and
  one repeated, unambiguous pattern is verified across the training files.
  Infer the pattern from the filenames; do not assume a numeric suffix.
- Determine single-label versus multi-label from the parsed training targets,
  then verify it against the submission schema.

### 3. Resolve paths without changing semantic IDs

Prefer injected `TRAIN_FILE_PATHS`/`TEST_FILE_PATHS` or the pre-resolved mapping.
Otherwise construct a mapping from observed file stems/relative paths and
require one-to-one coverage. Never use arbitrary glob order as prediction order.

```python
def validate_alignment(record_ids, paths, split_name):
    if len(record_ids) != len(paths):
        raise ValueError(
            f"{split_name}: {len(record_ids)} IDs but {len(paths)} paths"
        )
    missing = [record_id for record_id, path in zip(record_ids, paths) if not path]
    if missing:
        raise ValueError(
            f"{split_name}: unresolved audio for {len(missing)} records; "
            f"sample={missing[:5]}"
        )
    if len(set(map(str, record_ids))) != len(record_ids):
        raise ValueError(f"{split_name}: duplicate semantic record IDs")
```

### 4. Derive preprocessing from the audio

Do not hardcode a domain-specific sample rate or clip duration. Probe a
representative subset with the available backend and derive:

- target sample rate from observed sampling rates and the compute budget;
- clip/crop duration from the observed duration distribution;
- FFT/hop/mel resolution from that target rate and duration.

Record the chosen values in stdout. Start with a cheap representation
(precomputed public features when available, otherwise log-mel/MFCC statistics)
and scale to a CNN only when local CV and remaining time justify it.

Use a loader that preserves failures rather than silently substituting zero
audio. A small number of corrupt training files may be removed before fold
construction with IDs and labels removed together; an unresolved test record
must fail the component.

```python
def load_audio(path, target_sr):
    path = Path(path)
    try:
        import torchaudio
        waveform, source_sr = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0)
        if source_sr != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, source_sr, target_sr
            )
        return waveform.numpy().astype(np.float32)
    except Exception as first_error:
        try:
            import librosa
            waveform, _ = librosa.load(str(path), sr=target_sr, mono=True)
            return np.asarray(waveform, dtype=np.float32)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not decode {path}: {first_error}; {second_error}"
            ) from second_error
```

Cache derived features using a hash of the complete resolved source path plus
the preprocessing configuration, so equal stems or changed parameters cannot
collide.

### 5. Match objective, outputs, and metric to observed targets

- Single-label classification: class-index targets and cross entropy; emit the
  probability/label form required by the metric and sample template.
- Multi-label classification: independent binary targets,
  `BCEWithLogitsLoss`, sigmoid probabilities, and the averaging mode declared
  by the evaluation description.
- Regression: continuous output with the declared regression metric.
- Never choose a loss merely because audio tasks commonly use it.

All preprocessing and model fitting must occur inside each training fold.
Produce complete OOF predictions in canonical training order and compute the
reported score from those OOF predictions.

### 6. Use a budget-aware model ladder

1. Cheap baseline: aggregated log-mel/MFCC features plus a linear/tree model.
2. Compact CNN on cached spectrograms when the baseline is valid.
3. Larger pretrained audio/image backbone or augmentation only if measured CV
   gain and time/memory budget support it.

Use mixed precision for neural models when safe, cast model inputs to
`float32`, apply early stopping on validation folds, and keep a soft deadline
that reserves time for prediction and artifact writing. Save the best
fold checkpoint and also ensure the final selected checkpoint actually exists;
do not replace a failed checkpoint with an unvalidated model.

### 7. Submission and artifact contract

Infer wide, long, or structured output exclusively from
`sample_submission.csv` and injected submission metadata. Map predictions by
the complete sample IDs—never by row position and never by a remembered numeric
multiplier.

Save:

- `models/oof_{COMPONENT_NAME}.npy`
- `models/oof_ids_{COMPONENT_NAME}.npy`
- `models/test_{COMPONENT_NAME}.npy`
- `models/test_ids_{COMPONENT_NAME}.npy`
- `submission.csv` with exactly the sample columns, complete ID set, and order

Before success, verify shapes, finite/non-constant predictions, exact ID
coverage, and class-column order. Print the honest OOF metric using exactly:

`Final Validation Performance: <score>`
"""
