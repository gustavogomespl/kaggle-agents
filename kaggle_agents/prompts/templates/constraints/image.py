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

**VALIDATION/TEST**: Use `batch_size=1`:
```python
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
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

### 11. PYTORCH 2.6+ SAFE MODEL LOADING (CRITICAL)
PyTorch 2.6+ changed `weights_only` default from `False` to `True`, causing:
`_pickle.UnpicklingError: Weights only load failed`

**Use this safe loading pattern:**
```python
import torch
import pickle

def safe_load_model(model_path, device='cpu'):
    '''Load PyTorch model with fallback for older checkpoints.'''
    try:
        # Preferred: weights_only=True (secure)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        print(f"[WARNING] Secure load failed: {e}")
        print("[INFO] Falling back to weights_only=False (legacy mode)")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint

# Usage:
checkpoint = safe_load_model(model_path, device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
elif isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint)
else:
    model = checkpoint  # Full model object
```

**ALWAYS use this pattern when loading models in inference/ensemble components.**

### 12. MEDICAL IMAGING CROSS-VALIDATION (PREVENTS PATIENT LEAKAGE)
For medical datasets with patient_id column, ALWAYS use GroupKFold to prevent data leakage:

```python
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Check for patient column (expanded list for various dataset conventions)
patient_cols = [c for c in train_df.columns if c.lower() in
                ['patient_id', 'patientid', 'subject_id', 'study_id', 'patient',
                 'patient_num', 'subject', 'case_id', 'sid', 'person_id', 'individual_id']]

if patient_cols:
    print(f"[CV] Using GroupKFold on '{patient_cols[0]}' to prevent patient leakage")
    groups = train_df[patient_cols[0]].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        # Same patients NEVER appear in both train and val
        pass
else:
    # Fallback to StratifiedKFold for non-medical data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Why this matters:**
- Patient leakage can inflate validation AUC from 0.85 to 0.98
- But LB score stays at 0.60 because test patients are unseen
- GroupKFold ensures validation is realistic

### 13. CLASS IMBALANCE HANDLING (CRITICAL FOR RARE EVENTS)
For binary classification with imbalanced classes (<10% positive rate):

```python
# Calculate class weight
pos_count = (train_df['target'] == 1).sum()
neg_count = (train_df['target'] == 0).sum()
pos_weight = neg_count / pos_count
print(f"[INFO] Class ratio: 1:{pos_weight:.1f} (pos:{pos_count}, neg:{neg_count})")

# Apply weight to loss function
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight]).to(device)
)
```

**For extreme imbalance (>1:50), consider Focal Loss:**
```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```

**Image Resolution for Medical Imaging:**
- Natural images (ImageNet): 224-256 is sufficient
- Medical imaging (X-rays, dermoscopy): Use 384-512
- Fine structures (catheters, lesion borders): Use 512+
```python
# For medical imaging, use higher resolution:
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),  # NOT 224! Fine details matter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
"""
