"""
Image-to-image task constraints (denoising, segmentation, super-resolution).
"""

IMAGE_TO_IMAGE_CONSTRAINTS = """## IMAGE-TO-IMAGE / PIXEL-LEVEL TASKS (CRITICAL):

### 1. Architecture Requirements
- Output: FULL IMAGE (same HxW as input), NOT a single value
- Use encoder-decoder: U-Net, autoencoder, ResUNet
- NEVER use classifiers (EfficientNet, ResNet with FC head)
- NEVER use global average pooling + dense layers

### 2. Discover paired inputs
Inspect `TRAIN_PATH` and the supplied data manifest. If training assets are
paired directories rather than a table, build pairs from files that exist:
```python
from pathlib import Path

train_dir = Path('/path/to/train')
clean_dir = Path('/path/to/train_cleaned')

noisy_files = sorted(path for path in train_dir.rglob('*') if path.is_file())
pairs = [(f, clean_dir / f.name) for f in noisy_files if (clean_dir / f.name).exists()]
print(f"Found {len(pairs)} paired samples")
```

### 3. Submission Format (CRITICAL)
Use the exact IDs and order in the local template. Do not assume coordinate
indexing or construct an ID pattern that was not observed:
```python
sample_sub = pd.read_csv(sample_submission_path)
id_col, target_col = sample_sub.columns[:2]
template_ids = sample_sub[id_col].astype(str)
predictions_by_template_id = build_predictions_for_observed_ids(
    template_ids.tolist(), test_images, model
)
if set(predictions_by_template_id) != set(template_ids):
    raise ValueError("Prediction IDs do not exactly cover the sample template")
submission = sample_sub.copy()
submission[target_col] = template_ids.map(predictions_by_template_id)
if submission[target_col].isna().any():
    raise ValueError("Missing predictions after template alignment")
submission.to_csv("submission.csv", index=False)
```

### 4. Model Checkpointing for Ensemble
**PREFER TorchScript** (no class definition needed to reload):
```python
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model.pt")

# Loading in ensemble:
model = torch.jit.load("model.pt", map_location=device)
```
"""
