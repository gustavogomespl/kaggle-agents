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

noisy = {
    path.relative_to(train_dir).as_posix(): path
    for path in train_dir.rglob('*') if path.is_file()
}
clean = {
    path.relative_to(clean_dir).as_posix(): path
    for path in clean_dir.rglob('*') if path.is_file()
}
if set(noisy) != set(clean):
    raise ValueError("Noisy/clean relative-path coverage is not exactly 1:1")
pairs = [(image_id, noisy[image_id], clean[image_id]) for image_id in sorted(noisy)]
print(f"Found {len(pairs)} paired samples")
```

For training, sample crop coordinates or padding geometry once per pair and
apply the identical operation to noisy input and clean target. Never compose
independent random transforms for the two images. For validation/inference use
`batch_size=1`; if the network requires divisibility padding, retain a valid
pixel mask and remove padding before metrics, artifacts, and submission.

### 3. Packed OOF/Test Artifacts (CRITICAL)
Variable-sized images do not fit a numeric `N x ...` array and must never use
an object array. Save both component artifacts with the safe helper:
```python
save_component_artifacts(
    oof_images,
    test_images,
    train_ids=CANONICAL_TRAIN_IDS,
    test_ids=CANONICAL_TEST_IDS,
)
```
`save_component_artifacts` is injected and writes component-specific `.npz`
files containing concatenated `float32` values, `int64` offsets, `int32`
shapes, and unicode IDs. Do not import or redefine it, and never use
`dtype=object` or pickle.

### 4. Submission Format (CRITICAL)
Save packed evidence first, then call the injected writer:
```python
save_component_artifacts(
    oof_images,
    test_images,
    train_ids=CANONICAL_TRAIN_IDS,
    test_ids=CANONICAL_TEST_IDS,
)
write_submission(None)
```
`write_submission` loads the saved packed test artifact, proves a unique
zero/one-based mapping against the observed template IDs, verifies exact pixel
coverage, and streams the CSV. Never load the full template, flatten pixels by
an assumed order, write the CSV manually, or choose columns by position.

### 5. Model Checkpointing for Ensemble
**PREFER TorchScript** (no class definition needed to reload):
```python
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model.pt")

# Loading in ensemble:
model = torch.jit.load("model.pt", map_location=device)
```
"""
