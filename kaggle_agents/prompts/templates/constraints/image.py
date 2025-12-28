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
"""
