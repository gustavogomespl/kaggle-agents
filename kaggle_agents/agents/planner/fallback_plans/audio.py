"""Benchmark-neutral fallback plan for audio data."""

from typing import Any


def create_audio_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create an audited, data-derived audio fallback plan."""
    del sota_analysis
    target_requirement = (
        "Treat targets as continuous only after validating the public target "
        "artifact and metric."
        if "regression" in domain
        else "Infer single-label versus multi-label structure from the public "
        "target artifact before selecting loss and activation."
    )

    return [
        {
            "name": "data_audit",
            "component_type": "preprocessing",
            "description": (
                "Verify supplied audio coverage, target artifacts, and readable "
                "media metadata before training."
            ),
            "estimated_impact": 0.0,
            "rationale": (
                "A schema- and path-driven audit prevents silent row loss while "
                "remaining independent of any competition-specific layout."
            ),
            "code_outline": """
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torchaudio

AUDIO_EXTS = {
    '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff', '.aif'
}

def find_audio_files(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in AUDIO_EXTS else []
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.rglob('*')
        if item.is_file() and item.suffix.lower() in AUDIO_EXTS
    )

candidate_roots = []
for variable_name in ('AUDIO_SOURCE_DIR', 'TRAIN_PATH', 'TEST_PATH'):
    value = globals().get(variable_name)
    if value and Path(value).exists():
        candidate_roots.append(Path(value))

audio_files = sorted({
    path.resolve()
    for root in candidate_roots
    for path in find_audio_files(root)
})
if not audio_files:
    raise FileNotFoundError(
        f"AUDIT FAILED: no audio files under supplied roots {candidate_roots}"
    )

declared_paths = [
    Path(path)
    for variable_name in ('TRAIN_FILE_PATHS', 'TEST_FILE_PATHS')
    for path in globals().get(variable_name, [])
]
missing_declared = [str(path) for path in declared_paths if not path.is_file()]
if missing_declared:
    raise FileNotFoundError(
        f"AUDIT FAILED: unresolved declared files; sample={missing_declared[:5]}"
    )
if declared_paths and len(audio_files) < len(set(map(str, declared_paths))):
    raise ValueError("AUDIT FAILED: discovered audio does not cover declared records")

for target_path in globals().get('LABEL_FILES', []):
    if not Path(target_path).is_file():
        raise FileNotFoundError(f"AUDIT FAILED: target file missing: {target_path}")

observations = []
metadata_errors = []
for path in audio_files[:64]:
    try:
        info = torchaudio.info(str(path))
        if info.sample_rate > 0 and info.num_frames > 0:
            observations.append({
                'sample_rate': int(info.sample_rate),
                'duration': info.num_frames / info.sample_rate,
            })
    except Exception as exc:
        metadata_errors.append(f"{path}: {exc}")
if not observations:
    raise RuntimeError(
        f"AUDIT FAILED: no readable audio metadata; sample={metadata_errors[:3]}"
    )

sample_rates = [item['sample_rate'] for item in observations]
durations = np.asarray([item['duration'] for item in observations], dtype=float)
target_sample_rate = Counter(sample_rates).most_common(1)[0][0]
audit_result = {
    'audio_files_count': len(audio_files),
    'audio_files': [str(path) for path in audio_files],
    'candidate_roots': [str(path) for path in candidate_roots],
    'observed_sample_rates': dict(Counter(sample_rates)),
    'observed_duration_quantiles': {
        str(q): float(np.quantile(durations, q)) for q in (0.1, 0.5, 0.9)
    },
    'target_sample_rate': target_sample_rate,
}
(MODELS_DIR / 'audio_audit.json').write_text(json.dumps(audit_result, indent=2))
print(json.dumps(audit_result, indent=2))
""",
        },
        {
            "name": "data_derived_spectrogram_preprocessing",
            "component_type": "preprocessing",
            "description": (
                "Cache time-frequency features using sample rate and duration "
                "derived from the audited files."
            ),
            "estimated_impact": 0.18,
            "rationale": (
                "A shared deterministic representation supports fair model "
                "comparison without imposing a domain-specific frequency range."
            ),
            "code_outline": """
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
import torchaudio

audit = json.loads((MODELS_DIR / 'audio_audit.json').read_text())
audio_files = [Path(path) for path in audit['audio_files']]
target_sr = int(audit['target_sample_rate'])
duration = float(audit['observed_duration_quantiles']['0.5'])
window_samples = max(16, round(target_sr * 0.025))
n_fft = int(2 ** round(np.log2(window_samples)))
hop_length = max(1, n_fft // 4)
n_mels = max(32, min(128, n_fft // 8))
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=target_sr,
    n_mels=n_mels,
    n_fft=n_fft,
    hop_length=hop_length,
    f_min=0.0,
    f_max=target_sr / 2,
)

cache_dir = MODELS_DIR / 'audio_feature_cache'
cache_dir.mkdir(exist_ok=True)
failures = []
expected_cache_paths = []
for audio_path in audio_files:
    cache_key = sha256(str(audio_path.resolve()).encode()).hexdigest()
    cache_path = cache_dir / cache_key[:2] / f'{cache_key}.npy'
    expected_cache_paths.append(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        continue
    try:
        waveform, source_sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_sr != target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                source_sr,
                target_sr,
                resampling_method='sinc_interp_hann',
            )
        target_length = max(1, round(duration * target_sr))
        waveform = torch.nn.functional.pad(
            waveform[:, :target_length],
            (0, max(0, target_length - waveform.shape[1])),
        )
        features = mel_transform(waveform)
        features = torchaudio.functional.amplitude_to_DB(
            features,
            multiplier=10,
            amin=1e-10,
            db_multiplier=1,
        )
        np.save(cache_path, features.numpy())
    except Exception as exc:
        failures.append(f"{audio_path}: {exc}")
if failures:
    raise RuntimeError(f"Audio preprocessing failed; sample={failures[:5]}")
if not all(path.is_file() for path in expected_cache_paths):
    raise RuntimeError("Feature cache coverage does not match audited audio")
""",
        },
        {
            "name": "regularized_spectrogram_candidate",
            "component_type": "model",
            "description": (
                "Train a budget-appropriate regularized model over cached "
                "time-frequency features."
            ),
            "estimated_impact": 0.22,
            "rationale": (
                "Select a compact CNN, pretrained encoder, or shallow network "
                "from installed resources, then retain it only by trusted CV."
            ),
            "code_outline": (
                "Load the aligned cached features and canonical folds. "
                f"{target_requirement} Choose output shape, activation, and loss "
                "from that verified contract. Select architecture capacity from "
                "sample count, feature shape, installed weights, and runtime "
                "budget; save honest OOF/test predictions with semantic IDs."
            ),
        },
        {
            "name": "summary_feature_baseline",
            "component_type": "model",
            "description": (
                "Build a cheap diversity baseline from pooled spectral and "
                "temporal statistics."
            ),
            "estimated_impact": 0.14,
            "rationale": (
                "A low-cost feature baseline provides a robust fallback and a "
                "genuinely different error profile from a spectrogram encoder."
            ),
            "code_outline": (
                "Compute deterministic per-record summary features from the "
                "cached representation. "
                f"{target_requirement} Fit a regularized linear or tree model "
                "appropriate to the verified target/metric using canonical "
                "folds; persist aligned OOF/test predictions and IDs."
            ),
        },
        {
            "name": "audio_ensemble",
            "component_type": "ensemble",
            "description": (
                "Blend only valid, aligned candidates when held-out OOF "
                "performance improves."
            ),
            "estimated_impact": 0.08,
            "rationale": (
                "OOF-based gating measures whether diversity helps instead of "
                "assuming that a named architecture pair will ensemble well."
            ),
            "code_outline": (
                "Load trusted prediction artifacts, verify fold/ID/target order, "
                "fit weights using the declared metric and direction, and keep "
                "the blend only if it beats the best constituent under the same "
                "OOF protocol. Otherwise restore the best valid candidate."
            ),
        },
    ]
