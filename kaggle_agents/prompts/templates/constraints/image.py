"""
Image-specific constraints for computer vision tasks.
"""

IMAGE_CONSTRAINTS = """## IMAGE TASK REQUIREMENTS:

### 0. Image Path Resolution (CRITICAL)
Image datasets may be organized as `train/` or `train/images/` (same for test).
Always resolve the actual image directory before loading files:

```python
def resolve_image_dir(base_dir: Path, split: str) -> Path:
    candidates = [
        base_dir / split,
        base_dir / split / "images",
        base_dir / "images" / split,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]  # fallback

train_dir = resolve_image_dir(base_dir, "train")
test_dir = resolve_image_dir(base_dir, "test")
```

### 1. Variable Image Dimensions (CRITICAL)
Images often have different sizes. DataLoader's `torch.stack()` fails on different sizes.

**TRAINING**: Use fixed-size transforms:
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # OR RandomCrop(256, 256)
    transforms.ToTensor(),
])
```

**VALIDATION/TEST**: Use the SAME fixed-size Resize, then BATCHED inference.
NEVER use batch_size=1 for val/test - it is 10-30x slower and burns the time
budget exactly on large datasets:
```python
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # fixed size -> batching is safe
    transforms.ToTensor(),
])
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True)

model.eval()
preds = []
with torch.no_grad(), torch.cuda.amp.autocast():
    for xb in val_loader:
        preds.append(model(xb.to(device, non_blocking=True)).float().cpu())
```

### 1b. TensorFlow Image Decode Safety (CRITICAL)
`tf.image.decode_image` can return tensors without static shape, causing
`ValueError: 'images' contains no shape` during resize. Use format-specific decoders.

```python
def decode_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # or decode_png
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(img, (224, 224))
    return img

dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.apply(tf.data.Dataset.ignore_errors())
```

If you must use `tf.py_function`, always set shape after:
```python
img.set_shape((224, 224, 3))
```

### 2. Negative Strides (numpy/torch error)
`np.flip()`, `np.rot90()` create negative strides that PyTorch can't handle.

**FIX**: Always call `.copy()` or `np.ascontiguousarray()`:
```python
def apply_augmentation(img: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        img = np.flip(img, axis=1)
    return np.ascontiguousarray(img)  # MANDATORY
```

### 3. Transfer Learning
- Use pretrained backbones: EfficientNet, ResNet, ConvNeXt
- Fine-tune ALL layers for best performance
- Use ImageNet normalization: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

### 4. GPU Utilization
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

### 5. Albumentations v2.x API Changes (CRITICAL)
Several transform APIs changed in Albumentations 2.0+. Use the NEW syntax:

**RandomResizedCrop** (most common error):
```python
# WRONG (v1.x) - causes ValidationError "Input should be a valid tuple":
A.RandomResizedCrop(512, 512, scale=(0.8, 1.0))

# CORRECT (v2.x):
A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0))
```

**CenterCrop / RandomCrop**:
```python
# WRONG (v1.x):
A.CenterCrop(224, 224)
A.RandomCrop(256, 256)

# CORRECT (v2.x):
A.CenterCrop(height=224, width=224)
A.RandomCrop(height=256, width=256)
```

**Resize**:
```python
# Both work, but named params are preferred:
A.Resize(512, 512)  # Still works
A.Resize(height=512, width=512)  # Preferred
```

**Removed/Renamed transforms**:
- `IAASharpen` -> `Sharpen`
- `IAAEmboss` -> `Emboss`
- `IAAAdditiveGaussianNoise` -> `GaussNoise`
- `IAAAffine` -> `Affine`
- `IAAPiecewiseAffine` -> `PiecewiseAffine`
- `IAASuperpixels` -> REMOVED (no replacement)

**Safe v2.x augmentation pipeline example**:
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=(3, 7)),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

- Ensure input is RGB (3 channels) for color transforms

### 6. Keras/TensorFlow Deadline Callback
```python
class DeadlineCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_time, soft_deadline_s):
        super().__init__()
        self.start_time = start_time
        self.soft_deadline_s = soft_deadline_s

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time >= self.soft_deadline_s:
            print(f"[TIMEOUT] Soft deadline reached at epoch {epoch+1}")
            self.model.stop_training = True
```

### 7. PyTorch Loss Function Safety (AMP Compatibility)
NEVER use `nn.BCELoss()` with `torch.cuda.amp.autocast()` - it causes RuntimeError.
ALWAYS use `nn.BCEWithLogitsLoss()` which is AMP-safe:

```python
# WRONG - crashes with autocast:
criterion = nn.BCELoss()
with torch.cuda.amp.autocast():
    loss = criterion(sigmoid(logits), targets)  # RuntimeError!

# CORRECT - AMP-safe:
criterion = nn.BCEWithLogitsLoss()
with torch.cuda.amp.autocast():
    loss = criterion(logits, targets)  # Works! (applies sigmoid internally)
```

Note: With BCEWithLogitsLoss, model output should be RAW LOGITS (no sigmoid layer).
The loss function applies sigmoid internally for numerical stability.

### 8. Submission Format by Metric Type (CRITICAL)
For AUC-ROC, LogLoss, or any probability-based metric:
- ALWAYS submit RAW PROBABILITIES (float between 0 and 1)
- NEVER convert to hard labels - this destroys your score!

### 9. Large-Dataset Throughput (MANDATORY for datasets > 5 GB or DICOM)
Decoding/resizing the original images every epoch is what makes big
competitions time out. Decode ONCE, then train from the cache:

```python
# PREPROCESSING (once): decode + resize/pad every image to a cache.
# Hash the full source path: train/a.jpg and test/a.jpg must not collide.
from hashlib import sha256
import json

CACHE_DIR = Path("models/img_cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_manifest = {}
for path in all_image_paths:  # includes DICOM via pydicom when needed
    cache_key = sha256(str(path.resolve()).encode()).hexdigest()
    is_medical = path.suffix.lower() in {".dcm", ".dicom"}
    base_out = CACHE_DIR / cache_key[:2] / cache_key
    existing = next((p for p in (base_out.with_suffix(".npy"), base_out.with_suffix(".jpg"))
                     if p.exists()), None)
    if existing is not None:
        cache_manifest[str(path.resolve())] = str(existing)
        continue  # resumable
    img = load_any_format(path)          # PIL / pydicom array
    img = resize_and_pad(img, (384, 384))  # exact shape -> safe batching
    # Retain precision for DICOM, TIFF and any >8-bit decoded data. Pixel-level
    # restoration tasks should also force lossless .npy even when inputs are PNG.
    requires_lossless = (is_medical or path.suffix.lower() in {".tif", ".tiff"}
                         or np.asarray(img).dtype.itemsize > 1)
    out = base_out.with_suffix(".npy" if requires_lossless else ".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    if requires_lossless:
        np.save(out, img)                # lossless: retain DICOM/windowed precision
    else:
        Image.fromarray(img).save(out, quality=95)
    cache_manifest[str(path.resolve())] = str(out)

(CACHE_DIR / "manifest.json").write_text(json.dumps(cache_manifest, indent=2))

# MODEL components: read manifest.json and resolve every source through it.
# Never rediscover cached files by basename or stem.
```

Training throughput on GPU is also MANDATORY:
- Mixed precision ALWAYS: `torch.cuda.amp.autocast()` + `GradScaler` (see #7 for loss safety)
- `model = model.to(device, memory_format=torch.channels_last)`
- `DataLoader(num_workers=2-4, pin_memory=True, persistent_workers=True)`
- `torch.backends.cudnn.benchmark = True` (fixed input sizes)

```python
# WRONG for AUC/LogLoss (Score will be ~0.5 - terrible!):
sample_sub[target_col] = (predictions > 0.5).astype(int)

# CORRECT - keep as float probabilities:
sample_sub[target_col] = predictions  # e.g., 0.73, 0.12, 0.89

# Also WRONG - rounding:
sample_sub[target_col] = np.round(predictions)  # Still destroys AUC!
```

WHY: AUC measures ranking ability. Hard labels (0/1) lose all ranking information.
A model with 0.51 confidence and 0.99 confidence both become "1", destroying the metric.

### 9. MULTI-LABEL CLASSIFICATION (CRITICAL - PREVENTS NaN LOSS)
Multi-label tasks (e.g., RANZCR, ChestX-ray14) require DIFFERENT setup than multi-class:

**TRAINING SETUP:**
```python
# ✅ CORRECT for multi-label:
criterion = nn.BCEWithLogitsLoss()  # NOT CrossEntropyLoss!
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # LOWER LR for stability

# Model output: raw logits, NO activation layer
class MultiLabelModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)  # RAW logits, no sigmoid!
```

**LABELS MUST BE FLOAT:**
```python
# ❌ WRONG - causes NaN loss:
labels = torch.tensor([0, 1, 0, 1, 1])  # int tensor
loss = criterion(logits, labels)  # NaN!

# ✅ CORRECT - float tensor:
labels = torch.tensor([0., 1., 0., 1., 1.])  # float tensor
# OR convert explicitly:
labels = labels.float()
loss = criterion(logits, labels)  # Works!
```

**LEARNING RATE MATTERS:**
- Multi-label with pretrained backbone: use `lr=1e-4` or lower
- `lr=1e-3` often causes NaN loss or training instability
- Use warmup: 5% of total epochs with linear warmup

**INFERENCE:**
```python
# Apply sigmoid ONLY during inference, not in model
with torch.no_grad():
    logits = model(images)
    probabilities = torch.sigmoid(logits)  # Now in [0, 1]
    predictions = probabilities.cpu().numpy()
```

**COLUMN ORDER (CRITICAL):**
ALWAYS read target columns from sample_submission.csv, NEVER hardcode:
```python
# ✅ CORRECT - dynamic column reading:
sample_sub = pd.read_csv(sample_submission_path)
TARGET_COLS = sample_sub.columns[1:].tolist()  # Skip ID column
print(f"Target columns from sample_sub: {TARGET_COLS}")

# ❌ WRONG - hardcoded column names may have typos:
TARGET_COLS = ['ETT - Abnormal', 'NGT - Incomplete', ...]  # May not match!
```

### 10. FORBIDDEN: HARDCODED PLACEHOLDER METRICS
NEVER print fake/placeholder performance metrics:

```python
# ❌ ABSOLUTELY FORBIDDEN - will be flagged as failure:
print("Final Validation Performance: 0.9736")  # Hardcoded value!
print(f"Final Validation Performance: {target_score}")  # Using target as score!

# ✅ CORRECT - compute actual metric:
from sklearn.metrics import roc_auc_score
oof_score = roc_auc_score(y_true, oof_preds, average='macro')
print(f"Final Validation Performance: {oof_score:.6f}")  # Actual computed value
```

If you cannot compute the metric (e.g., inference-only component), state it clearly:
```python
print("Final Validation Performance: N/A (inference-only, no ground truth)")
```
"""
