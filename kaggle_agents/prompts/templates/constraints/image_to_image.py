"""
Image-to-image task constraints (denoising, segmentation, super-resolution).
"""

IMAGE_TO_IMAGE_CONSTRAINTS = """## IMAGE-TO-IMAGE / PIXEL-LEVEL TASKS (CRITICAL):

### 1. Architecture Requirements
- Output: FULL IMAGE (same HxW as input), NOT a single value
- Use encoder-decoder: U-Net, autoencoder, ResUNet
- NEVER use classifiers (EfficientNet, ResNet with FC head)
- NEVER use global average pooling + dense layers

### 2. No train.csv for Image-to-Image
Many competitions store data in paired directories, not CSV.

**DO NOT**: `pd.read_csv('train.csv')` - Will fail!

**DO THIS**:
```python
from pathlib import Path

train_dir = Path('/path/to/train')
clean_dir = Path('/path/to/train_cleaned')

noisy_files = sorted(train_dir.glob('*.png'))
pairs = [(f, clean_dir / f.name) for f in noisy_files if (clean_dir / f.name).exists()]
print(f"Found {len(pairs)} paired samples")
```

### 3. Submission Format (CRITICAL)
Submission must be FLATTENED to pixel-level format:
```python
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem
    pred = model(preprocess(img))  # Output: HxW image
    H, W = pred.shape
    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"  # 1-indexed
            submission_rows.append({"id": pixel_id, "value": float(pred[row, col])})

assert len(submission_rows) == expected_rows
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
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
