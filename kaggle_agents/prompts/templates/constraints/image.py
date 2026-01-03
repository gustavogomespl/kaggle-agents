"""
Image-specific constraints for computer vision tasks.
"""

IMAGE_CONSTRAINTS = """## IMAGE TASK REQUIREMENTS:

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

### 5. Albumentations
- `IAASharpen` is removed, use `Sharpen`
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
"""
