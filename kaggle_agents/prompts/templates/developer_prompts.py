"""
Prompt templates for the Developer Agent.

Refactored to be agentic, feedback-driven, and RL-friendly.
Inspired by Claude Code's concise style.

Builder functions are now in the builders/ submodule for better organization.
"""


# Re-export builders for backward compatibility
from .builders import (
    DynamicContext,
    build_context,
    build_dynamic_instructions,
)


# ==================== Core Identity ====================

DEVELOPER_CORE_IDENTITY = """You are a Kaggle Grandmaster implementing ML components.

Style:
- Write minimal, working code - no unnecessary abstractions
- No comments unless logic is non-obvious
- Use proven patterns from SOTA solutions when provided
- Print structured logs for the feedback loop

Output: A single Python code block. No explanations outside the code."""


# ==================== Hard Constraints ====================

HARD_CONSTRAINTS = """## MUST (violations cause failures):
1. predict_proba() for classification (NOT predict())
2. CV folds must respect `KAGGLE_AGENTS_CV_FOLDS` (default 5): StratifiedKFold(n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5")), shuffle=True, random_state=42)
3. Pipeline/ColumnTransformer for preprocessing - fit INSIDE CV folds only
   - If using manual scaling (e.g., Keras/TF), fit scaler on TRAIN fold only, then transform val/test
4. Save OOF predictions: np.save('models/oof_{component_name}.npy', oof_predictions)
   - Save test predictions: np.save('models/test_{component_name}.npy', test_predictions)
   - **CRITICAL CLASS ORDER ALIGNMENT (PREVENTS ENSEMBLE DEGRADATION)**:
     Predictions MUST be reordered to sample_submission column order BEFORE saving:
     ```python
     # Get canonical class order from sample_submission
     sample_sub = pd.read_csv(sample_submission_path)
     class_order = sample_sub.columns[1:].tolist()

     # Get LabelEncoder's class order
     le_classes = label_encoder.classes_.tolist()

     # Compute reorder indices: map from LabelEncoder order to submission order
     reorder_idx = [le_classes.index(c) for c in class_order]

     # Reorder predictions before saving
     oof_preds = oof_preds[:, reorder_idx]
     test_preds = test_preds[:, reorder_idx]

     # Save canonical class order for ensemble validation
     np.save('models/class_order.npy', class_order)
     ```
   - This ensures all models save predictions in the SAME column order, enabling correct ensemble averaging.
5. Clamp predictions: np.clip(predictions, 1e-15, 1 - 1e-15) before saving
   - Multiclass log_loss: re-normalize so each row sums to 1 after clipping
   - Multi-label: DO NOT normalize across classes (sigmoid per class)
6. Match sample_submission.csv exactly: columns, IDs, shape
7. If using logits, apply softmax/sigmoid BEFORE log_loss and submission creation
8. Print "Final Validation Performance: {score:.6f}" at the end (CRITICAL: Meta-Evaluator depends on this exact string)
   **CRITICAL FOR LOG_LOSS METRICS**: The score MUST be log_loss (not accuracy/AUC). For multiclass:
   ```python
   from sklearn.metrics import log_loss
   oof_score = log_loss(y_true, oof_preds_clipped)  # Use this, NOT accuracy
   print(f"Final Validation Performance: {oof_score:.6f}")
   ```
   Lower log_loss is better (0.02 is excellent, 0.7+ is nearly random for 99 classes).
9. Set random_state=42 everywhere for reproducibility
10. MANDATORY SOFT-DEADLINE PATTERN (prevents hard timeout kills):

   For sklearn/manual training loops:
   ```python
   import os, time
   _START_TIME = time.time()
   _TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "600"))
   _SOFT_DEADLINE_S = _TIMEOUT_S - 45  # Reserve 45s for cleanup/save

   def _check_deadline() -> bool:
       '''Return True if deadline exceeded.'''
       return (time.time() - _START_TIME) >= _SOFT_DEADLINE_S

   # Call inside training loops:
   for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
       if _check_deadline():
           print("[LOG:WARNING] Soft deadline reached, stopping early")
           break
       # ... train fold ...

   # ALWAYS print final metric even if stopped early
   print(f"Final Validation Performance: {cv_score:.6f}")
   ```

   For Keras/TensorFlow model.fit() - MUST use callback:
   ```python
   import tensorflow as tf
   import time

   class DeadlineCallback(tf.keras.callbacks.Callback):
       '''Stops training when soft deadline is reached.'''
       def __init__(self, start_time, soft_deadline_s):
           super().__init__()
           self.start_time = start_time
           self.soft_deadline_s = soft_deadline_s

       def on_epoch_end(self, epoch, logs=None):
           if time.time() - self.start_time >= self.soft_deadline_s:
               print(f"[TIMEOUT] Soft deadline reached at epoch {epoch+1}, stopping training")
               self.model.stop_training = True

   # Setup deadline
   _START_TIME = time.time()
   _TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "3300"))
   _SOFT_DEADLINE_S = _TIMEOUT_S - 120  # Reserve 2min for saving

   # Use with model.fit():
   callbacks = [
       DeadlineCallback(_START_TIME, _SOFT_DEADLINE_S),
       EarlyStopping(patience=30, restore_best_weights=True),
       ModelCheckpoint("best_model.h5", save_best_only=True),
   ]
   model.fit(..., callbacks=callbacks)

   # ALWAYS print final metric even if stopped early
   print(f"Final Validation Performance: {val_loss:.6f}")
   ```

   For PyTorch manual training loops - check EVERY BATCH for long epochs:
   ```python
   import time
   _START_TIME = time.time()
   _TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "600"))
   _SOFT_DEADLINE_S = _TIMEOUT_S - 50  # Reserve 50s for cleanup/save

   def _check_deadline() -> bool:
       return (time.time() - _START_TIME) >= _SOFT_DEADLINE_S

   for epoch in range(max_epochs):
       for batch_idx, batch in enumerate(dataloader):
           if batch_idx % 10 == 0 and _check_deadline():  # Check every 10 batches
               print(f"[TIMEOUT] Soft deadline at epoch {epoch}, batch {batch_idx}")
               torch.save(model, 'model_emergency.pth')  # Emergency save FULL model
               break
           # ... train batch ...
       if _check_deadline():
           break

   # ALWAYS print final metric
   print(f"Final Validation Performance: {best_val_loss:.6f}")
   ```

11. MODEL CHECKPOINTING - FULL MODEL SAVE (CRITICAL):
    Due to disjointed train/inference environments, ALWAYS save the FULL MODEL object.

    ```python
    # ✅ CORRECT (MANDATORY): Full model save - preserves architecture + weights
    torch.save(model, 'model.pth')

    # Loading - use version-aware helper to handle PyTorch 2.4+ compatibility:
    def load_model(path, device='cpu'):
        """Load full model checkpoint (compatible with PyTorch <2.4 and 2.4+)."""
        import re
        import torch
        # PyTorch 2.4+ requires weights_only=False for full model objects
        # Earlier versions don't have this parameter
        # Parse version safely (handles suffixes like +cu121, a0, .dev, etc.)
        match = re.match(r'^(\d+)\.(\d+)', torch.__version__)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if (major, minor) >= (2, 4):
                return torch.load(path, map_location=device, weights_only=False)
        return torch.load(path, map_location=device)

    model = load_model('model.pth', device=device)
    model.eval()
    ```

    NEVER use state_dict alone:
    ```python
    # ❌ WRONG - causes "size mismatch" or "missing keys" errors:
    torch.save(model.state_dict(), 'model.pth')
    ```

    PYTORCH VERSION NOTES:
    - PyTorch 2.4+ changed `torch.load()` to default to `weights_only=True` for security
    - This BREAKS loading of full model objects (UnpicklingError)
    - The `load_model()` helper above handles both old and new PyTorch versions
    - Error on PyTorch 2.4+ without fix: "WeightsUnpickler error: Unsupported global"
    - Error on PyTorch <2.4 with weights_only: "unexpected keyword argument 'weights_only'"

    WHY FULL MODEL SAVE MATTERS:
    - Training agent defines: `self.net = nn.Sequential(...)`
    - Inference agent defines: `self.layers = nn.Sequential(...)`
    - state_dict keys are "net.0.weight" vs "layers.0.weight" = MISMATCH!
    - Full model save avoids this entirely.

    NOTE: TorchScript (`torch.jit.script`) is an alternative but fails with dynamic control flow.
    Prefer `torch.save(model, ...)` for simplicity and reliability.

12. INCREMENTAL OOF CHECKPOINT (CRITICAL FOR TIMEOUT RESILIENCE):
    Save OOF/test predictions after EACH fold completes, not just at the end.
    This ensures partial results are available if timeout occurs mid-training.

    ```python
    oof_preds = np.zeros((n_train, n_classes))
    test_preds = np.zeros((n_test, n_classes))

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        if _check_deadline():
            print("[TIMEOUT] Soft deadline reached, stopping training")
            break

        # ... train model on fold ...

        # Get predictions
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_test) / len(fold_indices)

        # CHECKPOINT AFTER EACH FOLD (MANDATORY)
        np.save(MODELS_DIR / f'oof_{COMPONENT_NAME}.npy', oof_preds)
        np.save(MODELS_DIR / f'test_{COMPONENT_NAME}.npy', test_preds)
        np.save(MODELS_DIR / f'class_order_{COMPONENT_NAME}.npy', np.array(class_order))
        print(f"[LOG:CHECKPOINT] Saved OOF/test after fold {fold_idx}")

    # Final metric (ALWAYS print, even if stopped early)
    mask = oof_preds.sum(axis=1) > 0  # Only score rows that were predicted
    if mask.any():
        cv_score = metric(y_true[mask], oof_preds[mask])
    else:
        cv_score = 0.0
    print(f"Final Validation Performance: {cv_score:.6f}")
    ```

    WHY THIS MATTERS:
    - If XGBoost times out at fold 3, folds 0-2 are still saved
    - Ensemble Agent can use partial OOF predictions
    - No wasted computation from timed-out components

## MUST NOT:
- sys.exit(), exit(), quit(), raise SystemExit, os._exit()
- try-except blocks that swallow errors silently (let them surface for debugging)
- XGBoost: do not pass early_stopping_rounds/callbacks to fit() without a version check (see API Gotchas)
- Subsample training data unless `KAGGLE_AGENTS_FAST_MODE=1` (FAST_MODE may subsample to meet budget, but keep determinism)
- `pin_memory=True` in DataLoader (causes warnings/crashes). USE `pin_memory=False`.
- `num_workers > 0` in DataLoader (safe default is 0 to avoid fork/spawn issues).
- Overwrite sample_submission.csv (always write to submission.csv)
- `nn.BCELoss()` with `torch.cuda.amp.autocast()` (use `nn.BCEWithLogitsLoss()` - it's AMP-safe)
- Convert predictions to integers for AUC/LogLoss metrics: NEVER `(predictions > 0.5).astype(int)`
- Create dummy/fallback submissions with constant values (0.5, mean, zeros) when errors occur
- Use broad `except Exception` clauses that hide FileNotFoundError, RuntimeError, ValueError
- `generator.next()` (deprecated). ALWAYS use `next(generator)` for iterators (e.g., ImageDataGenerator)

## API Gotchas:
- OneHotEncoder: sparse_output=False (NOT sparse=False) for sklearn 1.2+
- pd.concat() instead of .append() for pandas 2.0+
- Optuna: set_verbosity(WARNING), n_trials <= 5, timeout=60 for validation
- LightGBM callbacks: lgb.early_stopping(100), not early_stopping_rounds param
- XGBoost 2.0+ (CRITICAL - API CHANGED):
  - NEVER use `early_stopping_rounds` or `callbacks` as fit() parameters - they were REMOVED
  - Use version-aware logic:
    ```python
    import xgboost as xgb

    xgb_major = int(xgb.__version__.split(".")[0])
    if xgb_major >= 2:
        model = xgb.XGBClassifier(
            early_stopping_rounds=100,  # In constructor for 2.0+
            n_estimators=1000,
            learning_rate=0.05,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[xgb.callback.EarlyStopping(rounds=100)],
            verbose=False,
        )
    ```
  - DO NOT use: model.fit(..., callbacks=[...]) for 2.0+  # WILL CRASH
- Albumentations: `IAASharpen` is removed, use `Sharpen`. Ensure input is RGB (3 channels) for color transforms.


## PyTorch Gotchas:
- Dataset __getitem__ must return tensors/arrays (never None) so DataLoader can collate

## FAIL FAST PRINCIPLE (CRITICAL):
Do NOT create dummy submissions or fallback predictions when errors occur.
Let errors surface so the Meta-Evaluator can diagnose and fix them.

WRONG - masks the real error (Score will be ~0.5 and error is hidden):
```python
try:
    model = torch.load('model.pth')
    predictions = model(X_test)
except Exception:
    print("Warning: Model not found. Creating dummy submission.")
    predictions = np.full(len(test_df), 0.5)  # BAD! Error is hidden.
```

CORRECT - let error surface (Meta-Evaluator can read and fix):
```python
model = torch.load('model.pth')  # Raises FileNotFoundError if missing
predictions = model(X_test)
# If this fails, the Meta-Evaluator sees the error and can fix the path
```

NEVER catch and hide these errors:
- FileNotFoundError (model/data not found) → Meta-Evaluator will fix the path
- RuntimeError (architecture mismatch) → Meta-Evaluator will fix the model definition
- ValueError (empty arrays, shape mismatch) → Meta-Evaluator will fix the data pipeline

## IMAGE DATA PIPELINE CRITICAL FIXES (MANDATORY FOR IMAGE TASKS):

### 1. VARIABLE IMAGE DIMENSIONS (stack error prevention):
Images in Kaggle datasets often have DIFFERENT sizes (e.g., 258x540 vs 420x540).
The default DataLoader collate_fn uses torch.stack() which FAILS on tensors of different sizes.

SOLUTIONS:
- **TRAINING**: Use `transforms.RandomCrop(256, 256)` or `transforms.Resize((256, 256))` to ensure all tensors have equal size
  ```python
  train_transform = transforms.Compose([
      transforms.RandomCrop(256, 256),  # Guarantees fixed size for batching
      transforms.ToTensor(),
  ])
  ```
- **VALIDATION/TEST**: Use `batch_size=1` to avoid stacking errors. Process one image at a time.
  ```python
  val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  ```
- **ALTERNATIVE**: Implement a custom collate_fn that pads images to the max size in the batch

### 2. NEGATIVE STRIDES (numpy/torch conversion error):
Using `np.flip()`, `np.rot90()`, or array slicing with negative steps creates arrays with negative strides.
PyTorch's `torch.from_numpy()` CANNOT handle negative strides.

ERROR: `ValueError: At least one stride in the given numpy array is negative`

FIX: ALWAYS call `.copy()` or `np.ascontiguousarray()` AFTER any numpy augmentation:
```python
def apply_augmentation(img: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        img = np.flip(img, axis=1)  # Creates negative stride!
    if random.random() > 0.5:
        img = np.rot90(img, k=random.randint(0, 3))  # Creates negative stride!
    return np.ascontiguousarray(img)  # MANDATORY: fixes strides before torch.from_numpy()
```

### 3. NO TRAIN.CSV FOR IMAGE-TO-IMAGE TASKS:
Many image-to-image competitions (denoising, super-resolution) do NOT have a train.csv file.
Data is stored in paired directories: `train/` (noisy) and `train_cleaned/` (clean targets).

DO NOT: `pd.read_csv('train.csv')`  # Will fail with FileNotFoundError

DO THIS INSTEAD:
```python
import os
from pathlib import Path

train_dir = Path('/path/to/train')
clean_dir = Path('/path/to/train_cleaned')

# List files directly from directories
noisy_files = sorted(train_dir.glob('*.png'))
pairs = []
for noisy_path in noisy_files:
    clean_path = clean_dir / noisy_path.name
    if clean_path.exists():
        pairs.append((noisy_path, clean_path))

print(f"Found {len(pairs)} paired training samples.")
```

### 4. DATA PATH VALIDATION (MANDATORY before DataLoader creation):
Image IDs in CSV may not include file extensions. Validate paths before creating Dataset.

```python
from pathlib import Path

def validate_image_paths(
    image_ids: list,
    base_dir: Path,
    extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    allow_missing: bool = False,
) -> list:
    # Validate and resolve image paths, trying multiple extensions.
    valid_paths = []
    missing = []

    if base_dir.exists():
        discovered = {
            p.suffix.lower() for p in base_dir.iterdir() if p.is_file()
        }
        discovered = [ext for ext in discovered if ext in extensions]
        if discovered:
            extensions = tuple(sorted(discovered))

    for img_id in image_ids:
        found = False
        # Try exact path first (in case extension is included in ID)
        for ext in [""] + list(extensions):
            path = base_dir / f"{img_id}{ext}"
            if path.exists():
                valid_paths.append(path)
                found = True
                break
        if not found:
            missing.append(img_id)

    if missing:
        print(f"[LOG:WARNING] {len(missing)} images not found: {missing[:5]}...")
        if allow_missing:
            return valid_paths
        raise FileNotFoundError(f"Missing {len(missing)} images. Check paths and extensions.")

    print(f"[LOG:INFO] Validated {len(valid_paths)} image paths")
    return valid_paths

# USE BEFORE CREATING DATASET:
# train_dir/test_dir should come from data_files or detected paths.
train_dir = Path(data_files["train"]) if "train" in data_files else Path("train")
test_dir = Path(data_files["test"]) if "test" in data_files else Path("test")
allow_missing = os.getenv("KAGGLE_AGENTS_ALLOW_MISSING_IMAGES", "0").lower() in {"1", "true", "yes"}
train_images = validate_image_paths(train_df["id"].tolist(), train_dir, allow_missing=allow_missing)
test_images = validate_image_paths(test_df["id"].tolist(), test_dir, allow_missing=allow_missing)
```

WHY: CSV often has IDs like "abc123" but files are "abc123.tif". Without validation,
Dataset returns empty and training silently fails with "Found array with 0 sample(s)".

## IMAGE-TO-IMAGE / PIXEL-LEVEL TASKS (CRITICAL):
If domain is image_to_image, image_segmentation, or submission format is pixel_level:
1. Output must be a FULL IMAGE (same HxW as input), NOT a single value per image
2. Use encoder-decoder architectures (U-Net, autoencoder), NOT classifiers
3. NEVER use image classifiers (EfficientNet, ResNet with FC head) for these tasks
4. NEVER use global average pooling followed by dense layers
5. Submission must be FLATTENED to pixel-level format:
   - Read sample_submission.csv to get exact ID format and row count
   - ID format is typically: '{image_id}_{row}_{col}' or '{image_id}_{pixel_index}'
   - MUST match EXACT number of rows in sample_submission (often millions of rows)
6. Example flattening code (CRITICAL - use this pattern):
   ```python
   sample_sub = pd.read_csv(sample_submission_path)
   expected_rows = len(sample_sub)

   submission_rows = []
   for img_path in sorted(test_images):  # MUST be sorted for consistent order
       img_id = img_path.stem  # e.g., "1" from "1.png"
       pred = model(preprocess(img))  # Output: HxW image, NOT a single value
       H, W = pred.shape
       for row in range(H):
           for col in range(W):
               pixel_id = f"{img_id}_{row+1}_{col+1}"  # 1-indexed to match sample
               submission_rows.append({"id": pixel_id, "value": float(pred[row, col])})

   assert len(submission_rows) == expected_rows, f"Expected {expected_rows} rows, got {len(submission_rows)}"
   pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
   ```
7. Verify submission shape BEFORE saving: if row count doesn't match sample_submission, your model architecture is WRONG

## SUBMISSION FORMAT CRITICAL FIX (MANDATORY FOR ALL TASKS):

### Problem: Predictions going to WRONG COLUMN (causes score=0.5)
The target column is NOT always `sample_sub.columns[1]`!
Example: "detecting-insults" has columns [Insult, Date, Comment] where Insult (column 0) is the target.
If you put predictions in columns[1] (Date), you get score=0.5 even with perfect CV!

### MANDATORY PATTERN - Identify target column correctly:
```python
# 1. Load sample submission and INSPECT it
sample_sub = pd.read_csv(sample_submission_path)
print("Sample submission columns:", sample_sub.columns.tolist())
print("Sample submission head:\n", sample_sub.head(2))

# 2. CRITICAL: Identify the TARGET column (DO NOT assume columns[1]!)
# The target column has the SAME NAME as the target in train.csv
# Common names: 'target', 'Insult', 'label', 'prediction', 'value', etc.
train_df = pd.read_csv(train_path)
target_col = train_df.columns[0]  # Usually the first column in train is the target

# Verify target_col exists in sample_sub
if target_col not in sample_sub.columns:
    # Fallback: if only 2 columns, use columns[1]; otherwise find numeric column
    if len(sample_sub.columns) == 2:
        target_col = sample_sub.columns[1]
    else:
        # Look for common target names
        for col in ['target', 'label', 'prediction', 'value', 'Insult']:
            if col in sample_sub.columns:
                target_col = col
                break

print(f"Using target column: {target_col}")

# 3. Fill ONLY the target column with predictions (preserve all other columns!)
sample_sub[target_col] = predictions  # predictions must be in same order as sample_sub rows

# 4. VALIDATE before saving
print(f"Predictions sample: {sample_sub[target_col].head()}")
assert len(sample_sub) == len(predictions), f"Row count mismatch: {len(sample_sub)} vs {len(predictions)}"

sample_sub.to_csv('submission.csv', index=False)
```

### CRITICAL CHECKS:
1. Print sample_sub.columns to see ALL column names before assuming anything
2. The target column name MUST match the target from train.csv
3. DO NOT overwrite non-target columns (like Date, Comment, ID, etc.)
4. If sample_sub has 3+ columns, columns[1] is probably NOT the target!

## TABULAR MODELS REQUIRE TABULAR FEATURES (MANDATORY):

LightGBM, XGBoost, CatBoost, and other tree-based models need REAL tabular features.
If train.csv only has [id, label] columns → This is an IMAGE competition!

CRITICAL: DO NOT create dummy/random features - this gives terrible scores!

```python
# CHECK BEFORE USING TABULAR MODELS:
train_df = pd.read_csv('train.csv')
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Number of columns: {len(train_df.columns)}")

if len(train_df.columns) <= 2:
    raise ValueError(
        "No tabular features found! train.csv has only id+label columns. "
        "This appears to be an IMAGE competition. "
        "Use CNN models (EfficientNet, ResNet) with transfer learning, "
        "NOT tree models (LightGBM, XGBoost, CatBoost)."
    )
```

### Example: dog-breed-identification
- train.csv has only [id, breed] columns - NO tabular features!
- WRONG: LightGBM with dummy random features → score 4.78 (terrible)
- RIGHT: EfficientNet-B0 with transfer learning → score < 1.0

### Model Selection Guide:
- IMAGE competition (train/ has images, train.csv has only id+label):
  → Use: EfficientNet, ResNet, VGG with ImageNet pretrained weights
  → DO NOT use: LightGBM, XGBoost, CatBoost (they need tabular features!)

- TABULAR competition (train.csv has many feature columns):
  → Use: LightGBM, XGBoost, CatBoost with hyperparameter tuning

NEVER create dummy/random features to feed into tree models!

## CRITICAL: MULTI-MODAL HYBRID BEST PRACTICE
When a competition has BOTH:
1. Raw image directories (train/, test/, images/)
2. train.csv with many numeric feature columns (hand-crafted / extracted features)
Then prioritize a HYBRID model:
- CNN for images + MLP for tabular features
- Concatenate CNN embedding with normalized tabular features before the classifier head
- Use Keras Functional API (multi-input) or equivalent
- Use light image augmentation (rotation, zoom, flip)
This is a common Kaggle best practice for multi_modal tasks and often beats separate models.

Generic hybrid example (Keras Functional API):
```python
import tensorflow as tf

# X_img: (N, H, W, C), X_tab: (N, n_features)
img_input = tf.keras.Input(shape=(H, W, C), name="image")
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(img_input)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)

tab_input = tf.keras.Input(shape=(n_features,), name="tabular")
t = tf.keras.layers.Dense(128, activation="relu")(tab_input)

combined = tf.keras.layers.Concatenate()([x, t])
combined = tf.keras.layers.Dense(128, activation="relu")(combined)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(combined)

model = tf.keras.Model(inputs=[img_input, tab_input], outputs=out)
```

## FILE ISOLATION (CRITICAL - PREVENTS OVERWRITES):

### WRONG - Components overwriting each other:
```python
submission_df.to_csv("submission.csv")  # LightGBM writes, XGBoost overwrites!
```

### CORRECT - Each component saves to unique path:
```python
# During model training - save component-specific predictions
submission_df.to_csv(f"models/preds_{component_name}.csv", index=False)
np.save(f"models/oof_{component_name}.npy", oof_preds)
np.save(f"models/test_{component_name}.npy", test_preds)

# Only Ensemble Agent creates submission.csv!
```

### Rule:
- Model components: NEVER write to `submission.csv` directly
- Model components: Save to `models/preds_{component_name}.csv`
- Ensemble Agent: Read all `models/preds_*.csv` or `models/test_*.npy`
- Ensemble Agent: Create final `submission.csv`

## ID COLUMN IMMUTABILITY (MANDATORY):

### The 'id' column MUST remain exactly as in the original CSV:
- DO NOT cast id to different type (int -> str or vice versa)
- DO NOT reindex the DataFrame (unless you reassign id from original)
- DO NOT drop the id column
- DO NOT modify id values

### Pattern for Feature Engineering:
```python
# CORRECT: Preserve original id
train_df = pd.read_csv('train.csv')
original_ids = train_df['id'].copy()  # Backup

# ... create features ...

# VERIFY id wasn't corrupted
train_df['id'] = original_ids  # Restore if needed
assert train_df['id'].equals(pd.read_csv('train.csv')['id']), "ID column was modified!"
```

### After ANY merge with target column:
```python
train_eng = train_eng.merge(train_orig[['id', target_col]], on='id', how='left')

# MANDATORY NULL CHECK:
assert not train_eng[target_col].isnull().any(), \
    f"CRITICAL: {train_eng[target_col].isnull().sum()} NaN values in target after merge! " \
    "ID types may have changed (int vs str)."
```

## CLASS ORDER METADATA (MANDATORY FOR MULTICLASS):

### Every model component MUST save its class order:
```python
# After fitting LabelEncoder:
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# MANDATORY: Save class order metadata
np.save(f"models/classes_{component_name}.npy", le.classes_)
print(f"[LOG:INFO] Saved class order: {le.classes_[:5]}... ({len(le.classes_)} classes)")
```

### Why This Matters:
- LabelEncoder order depends on training data order
- Different random_state or data shuffling = different class order
- Averaging predictions with mismatched orders = random guessing

## ENGINEERED DATA VALIDATION (CRITICAL):

### Before using train_engineered.csv/test_engineered.csv:
```python
import pandas as pd
from pathlib import Path

# Load both original and engineered
train_orig = pd.read_csv(train_csv_path)
train_eng_path = Path(train_engineered_path)

if train_eng_path.exists():
    train_eng = pd.read_csv(train_eng_path)

    # CHECK 1: Fail if duplicates found (DO NOT auto-fix!)
    if train_eng['id'].duplicated().any():
        raise ValueError(
            f"DUPLICATE IDS DETECTED! Found {train_eng['id'].duplicated().sum()} duplicates. "
            "Fix the feature engineering code - do NOT use drop_duplicates()."
        )

    # CHECK 2: Row count must match
    if len(train_eng) != len(train_orig):
        print(f"[LOG:ERROR] Engineered data has {len(train_eng)} rows, original has {len(train_orig)}")
        print("[LOG:INFO] Using original data instead")
        train_df = train_orig
    else:
        # CHECK 3: Target column must exist (merge if missing)
        if target_col not in train_eng.columns and target_col in train_orig.columns:
            print(f"[LOG:INFO] Target '{target_col}' not in engineered data, merging from original")
            train_eng = train_eng.merge(train_orig[['id', target_col]], on='id', how='left')
            # Verify merge worked
            assert not train_eng[target_col].isnull().any(), "Target merge failed - check ID types!"
        train_df = train_eng
else:
    train_df = train_orig
```

### Prefer Original Features When Engineered Data is Broken:
```python
# If engineered data has fewer than 10 feature columns, use original
if len(train_eng.columns) < 10:
    print("[LOG:WARNING] Engineered data has too few features, using original train.csv")
    train_df = train_orig  # Fall back to original
```
"""


# ==================== Logging Format ====================

LOGGING_FORMAT = """## Structured Logs (required for feedback loop):
[LOG:FOLD] fold={n} score={score:.6f} time={time:.2f}
[LOG:CV_SUMMARY] mean={mean:.6f} std={std:.6f} scores={list}
[LOG:OPTUNA] trial={n} score={score:.6f} time={time:.2f} params={dict}
[LOG:TIMING] step={name} time={time:.2f} cumulative={cumulative:.2f}
[LOG:FEATURES] top={list[:20]} importances={list[:20]}
[LOG:WARNING] message={str}
[LOG:ERROR] message={str}"""


# ==================== Prompt Composition ====================


def compose_generate_prompt(
    component,
    competition_info,
    paths: dict[str, str],
    context: DynamicContext,
    use_modular_constraints: bool = True,
) -> str:
    """
    Compose a dynamic, context-aware code generation prompt.

    Adaptive injection based on iteration:
    - Iteration 0: SOTA-heavy (learn from winners)
    - Later iterations: Feedback-heavy + truncated SOTA reference

    Now supports modular constraints to reduce token usage (40-60% reduction).

    Args:
        component: AblationComponent to implement
        competition_info: CompetitionInfo with metadata
        paths: Dictionary with train, test, submission, models paths
        context: DynamicContext with SOTA, feedback, rewards
        use_modular_constraints: If True, load domain-specific constraints only

    Returns:
        Composed prompt string
    """
    # Get domain-specific constraints (modular) or full constraints
    if use_modular_constraints:
        try:
            from .constraints import get_constraints_for_domain

            # Handle None domain by defaulting to "tabular"
            domain = getattr(competition_info, "domain", None) or "tabular"
            constraints = get_constraints_for_domain(domain)
            print(f"   📦 Loaded modular constraints for domain: {domain}")
        except Exception:
            constraints = HARD_CONSTRAINTS  # Fallback to full constraints
    else:
        constraints = HARD_CONSTRAINTS

    parts = [
        DEVELOPER_CORE_IDENTITY,
        "",
        constraints,
        "",
        LOGGING_FORMAT,
        "",
        _format_task(component, competition_info, paths),
    ]

    # Non-standard label files instruction (e.g., MLSP 2013 Birds .txt files)
    label_files = paths.get("label_files", [])
    if label_files:
        label_section = """
## NON-STANDARD LABEL FILES (MANDATORY PARSING)

Label files detected: """ + ", ".join(str(lf) for lf in label_files) + """

YOU MUST parse these files - NEVER use dummy labels (np.zeros)!

Steps:
1. Use parse_label_file() helper (injected in code header)
2. Create ID → label mapping
3. For multi-label: pivot to binary matrix
4. Match with training data BEFORE training

Example for MLSP 2013 Birds:
```python
label_df = parse_label_file(LABEL_FILES[0])
label_df.columns = ['rec_id', 'label']
y_train = label_df.pivot_table(index='rec_id', columns='label', aggfunc=len, fill_value=0)
```

WARNING: Using np.zeros for labels causes AUC 0.5 (random predictions)!
"""
        parts.append("")
        parts.append(label_section)

    # Runtime/objective hints (important for timeout-sensitive runs like MLE-bench).
    if context.run_mode or context.objective or context.timeout_per_component is not None:
        parts.append("")
        parts.append("## Objective & Budget")
        if context.run_mode:
            parts.append(f"- run_mode: {context.run_mode}")
        if context.objective:
            parts.append(f"- objective: {context.objective}")
        if context.timeout_per_component is not None:
            parts.append(f"- timeout_per_component_seconds: {context.timeout_per_component}")
        parts.append(
            "- Env knobs: KAGGLE_AGENTS_COMPONENT_TIMEOUT_S, KAGGLE_AGENTS_CV_FOLDS, KAGGLE_AGENTS_FAST_MODE"
        )

    # Memory insights from past runs (best HPs, errors, strategies)
    if context.memory_summary and context.memory_summary != "No memory insights available yet.":
        parts.append("")
        parts.append("## Memory Insights (Use these to avoid repeats and reuse best configs)")
        parts.append(context.memory_summary)

    # Submission validation error (must be fixed immediately).
    if context.submission_validation_error:
        parts.append("")
        parts.append("## CRITICAL: SUBMISSION FORMAT ERROR (MUST FIX)")
        parts.append(
            f"Previous submission failed validation: {context.submission_validation_error}"
        )
        parts.append("")
        parts.append("Fix requirements:")
        parts.append("1. Read sample_submission.csv to match ID values and column order exactly")
        parts.append("2. Match row count exactly (no truncation/padding)")
        parts.append("3. Preserve ID order from sample_submission.csv")
        parts.append(
            "4. For image-to-image: flatten per-pixel predictions to the sample submission ID format"
        )
        parts.append("5. Use assertions before saving")
        parts.append("```python")
        parts.append("sample = pd.read_csv(sample_submission_path)")
        parts.append("assert list(submission.columns) == list(sample.columns)")
        parts.append("assert len(submission) == len(sample)")
        parts.append(
            "assert (submission[sample.columns[0]].values == sample[sample.columns[0]].values).all()"
        )
        parts.append("```")

    # Adaptive training guidance (GPU-accelerated, reduces epochs if timeout)
    if context.run_mode.lower() == "mlebench" or "medal" in context.objective.lower():
        parts.append("")
        parts.append("## NEURAL NETWORK TRAINING (GPU-ACCELERATED)")
        parts.append(
            f"- **EPOCHS**: Train for up to {context.suggested_epochs} epochs with early stopping"
        )
        parts.append(
            "- **GPU**: MUST use CUDA if available: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`"
        )
        parts.append(
            "- **BACKBONE**: Full fine-tuning for maximum performance (do NOT freeze layers)"
        )
        parts.append("- **LEARNING RATE**: Use warmup (5% of epochs) + cosine annealing schedule")
        parts.append("- **AUGMENTATION**: Apply heavy augmentation (Cutmix, Mixup, RandAugment)")
        parts.append(
            f"- **EARLY STOPPING**: Stop if validation loss doesn't improve for {context.early_stopping_patience} epochs (SOTA uses patience=30)"
        )
        parts.append("- **CHECKPOINTING**: Save best model checkpoint by validation metric")
        parts.append("- **MIXED PRECISION**: Use torch.cuda.amp.autocast() for faster training")

        if context.timeout_occurred:
            parts.append("")
            parts.append("⚠️ TIMEOUT DETECTED IN PREVIOUS RUN - ADJUSTMENTS:")
            parts.append(
                f"- REDUCED epochs from {context.epoch_budget} to {context.suggested_epochs}"
            )
            parts.append("- Use smaller batch size if memory issues")
            parts.append("- Consider freezing early backbone layers if still too slow")
            parts.append("- STILL prioritize completing training over speed")

        parts.append("")
        parts.append("## SOFT-DEADLINE PATTERN (MANDATORY)")
        timeout_s = context.timeout_per_component or 3600
        parts.append("```python")
        parts.append("import time")
        parts.append("_START = time.time()")
        parts.append(f"_TIMEOUT = {timeout_s}")
        parts.append("_SOFT_DEADLINE = _TIMEOUT - 120  # Reserve 2min for saving")
        parts.append("")
        parts.append("for epoch in range(MAX_EPOCHS):")
        parts.append("    if time.time() - _START >= _SOFT_DEADLINE:")
        parts.append("        print('[TIMEOUT] Soft deadline reached, saving best model')")
        parts.append("        break")
        parts.append("    # ... train epoch ...")
        parts.append("```")

    # ADAPTIVE: First iteration = SOTA heavy
    if context.iteration_num == 0:
        if context.sota_patterns:
            parts.append("")
            parts.append("## SOTA Patterns (Learn from top solutions):")
            parts.append(context.sota_patterns)

    # ADAPTIVE: Later iterations = Feedback heavy
    else:
        # CRITICAL: Feedback comes first to ensure corrections are applied
        if context.previous_feedback:
            parts.append("")
            parts.append("## Previous Attempt Feedback (MUST FIX):")
            parts.append(context.previous_feedback)

        if context.what_failed:
            parts.append("")
            parts.append("## What Failed (DO NOT REPEAT):")
            parts.append("\n".join(f"- {f}" for f in context.what_failed[:5]))

        if context.reward_guidance:
            parts.append("")
            parts.append("## Meta-Evaluator Guidance:")
            parts.append(context.reward_guidance)

        if context.attempt_feedback:
            parts.append("")
            parts.append("## Prior Attempts (Study + Fix):")
            parts.append(context.attempt_feedback)

        if context.what_worked:
            parts.append("")
            parts.append("## What Worked (Keep these approaches):")
            parts.append("\n".join(f"- {w}" for w in context.what_worked[:5]))

        # DPO: Inject contrastive learning examples (good vs bad code)
        if context.dpo_examples:
            parts.append("")
            parts.append(context.dpo_examples)

        # Still include truncated SOTA as reference
        if context.sota_patterns:
            parts.append("")
            parts.append("## SOTA Reference (condensed):")
            parts.append(context.sota_patterns[:1000])

    # Component-specific minimal guidance
    guidance = _get_component_guidance(component.component_type)
    if guidance:
        parts.append("")
        parts.append(guidance)

    return "\n".join(parts)


def _format_task(component, competition_info, paths: dict[str, str]) -> str:
    """Format the task specification section."""
    component_type = getattr(component, "component_type", "model")
    component_name = getattr(component, "name", "component")
    component_code = getattr(component, "code", "")
    estimated_impact = getattr(component, "estimated_impact", 0.0)

    name = getattr(competition_info, "name", "competition")
    domain = getattr(competition_info, "domain", "tabular")
    problem_type = getattr(competition_info, "problem_type", "classification")
    metric = getattr(competition_info, "evaluation_metric", "accuracy")

    train_path = paths.get("train", "train.csv")
    test_path = paths.get("test", "test.csv")
    models_path = paths.get("models", "models/")
    submission_path = paths.get("submission", "submission.csv")

    return f"""## Task
Component: {component_type} - {component_name}
Goal: {component_code}
Estimated Impact: {estimated_impact:.1%}

## Competition
Name: {name}
Domain: {domain}
Problem Type: {problem_type}
Metric: {metric}

## Paths (CRITICAL - USE EXACTLY AS PROVIDED)
# INPUT_DIR is READ-ONLY in Kaggle Kernels - NEVER write here!
INPUT_DIR: {paths.get("input_dir", ".")}
# OUTPUT_DIR is WRITABLE - use for all outputs (models, submission, etc.)
OUTPUT_DIR: {paths.get("output_dir", ".")}

Train: {train_path}
Test: {test_path}
Models: {models_path}
Submission: {submission_path}

## PATH USAGE (MANDATORY - DO NOT HARDCODE)
**CRITICAL**: Use the EXACT paths provided above. DO NOT hardcode 'train.csv' or 'test.csv'.

```python
# CORRECT: Use the provided paths EXACTLY
from pathlib import Path

TRAIN_PATH = Path("{train_path}")
TEST_PATH = Path("{test_path}")
MODELS_DIR = Path("{models_path}")
SUBMISSION_PATH = Path("{submission_path}")

# Create models directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data based on path type:
if TRAIN_PATH.suffix == '.csv':
    train_df = pd.read_csv(TRAIN_PATH)
elif TRAIN_PATH.is_dir():
    # For directory-based data (images, audio, etc.):
    train_files = sorted(TRAIN_PATH.glob('*'))
    print(f"Found {{len(train_files)}} files in {{TRAIN_PATH}}")
```

**NEVER** do this (WRONG - will cause FileNotFoundError):
```python
train_df = pd.read_csv('train.csv')  # WRONG!
train_df = pd.read_csv(BASE_DIR / 'train.csv')  # WRONG!
```

The paths may point to:
- CSV files: `train.csv`, `test.csv`
- Directories: `supplemental_data/`, `train_images/`, `essential_data/`
- Subdirectories: `essential_data/train.csv`

Always check if the path is a file or directory before loading."""


def _get_component_guidance(component_type: str) -> str:
    """Get minimal, type-specific guidance."""
    guidance = {
        "model": """## Model Component Requirements
- IMPLEMENT soft-deadline pattern (see HARD_CONSTRAINTS #9)
  - For Keras/TensorFlow: MUST use DeadlineCallback in model.fit() callbacks
  - For sklearn/PyTorch manual loops: check _check_deadline() inside fold loop
- Train model with StratifiedKFold CV using n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5"))
- Save OOF predictions to models/oof_{name}.npy for stacking
- Handle class imbalance if ratio > 2:1 (class_weight or scale_pos_weight)
- Print per-fold scores: [LOG:FOLD] fold={n} score={s:.6f}
- Use GPU if available (check torch.cuda.is_available())
- Create submission.csv with probabilities [0,1]
- SUBMISSION FORMAT: Target column is NOT always columns[1]!
  ```python
  sample_sub = pd.read_csv(sample_submission_path)
  print("Columns:", sample_sub.columns.tolist())  # ALWAYS check column names!
  # Use SAME target column name as in train.csv (e.g., 'Insult', 'target', 'label')
  target_col = train_df.columns[0]  # First column of train.csv is usually the target
  sample_sub[target_col] = predictions  # Fill ONLY the target column
  sample_sub.to_csv('submission.csv', index=False)
  ```
- ALWAYS print "Final Validation Performance: {score}" even if stopped early due to deadline
- SAVE PyTorch checkpoints with TorchScript for ensemble compatibility (see HARD_CONSTRAINTS #10):
  ```python
  scripted_model = torch.jit.script(model)
  torch.jit.save(scripted_model, f"models/{component_name}_fold{fold_idx}.pt")
  ```""",
        "feature_engineering": """## Feature Engineering Requirements
- Transform train and test consistently
- NO model training in this component
- Save to train_engineered.csv, test_engineered.csv if creating new files
- Fast execution (<30 seconds)
- Print "Final Validation Performance: 1.0" on completion""",
        "ensemble": """## Ensemble Requirements

### LOADING PREVIOUS MODELS (CRITICAL - READ CAREFULLY):
1. **TorchScript Loading** (PREFERRED - no class definition needed):
   ```python
   model = torch.jit.load(checkpoint_path, map_location=device)
   model.eval()
   ```

2. **State Dict Fallback** (ONLY if TorchScript fails):
   - You MUST define the EXACT same model class as used in training
   - Inspect checkpoint keys to determine architecture:
   ```python
   state_dict = torch.load(path, map_location=device)
   # Look at key names: "net.0.weight" means self.net, NOT self.model
   # Look at number of layers to infer depth
   print([k for k in state_dict.keys()][:10])
   ```

### COMMON PITFALLS (WILL CAUSE state_dict LOADING TO FAIL):
- ❌ Defining model with `self.model` when checkpoint uses `self.net`
- ❌ Using different depth/channels than training component
- ❌ Missing dropout layers that exist in original
- ❌ Different layer ordering or architecture

### OOF-Based Stacking (no checkpoint loading needed):
- Load OOF predictions from models/oof_*.npy files
- Preferred: Stacking with LogisticRegression/Ridge meta-learner
- Fallback: Weighted average if OOF files missing
- Can use correlation analysis to select diverse models
- MUST validate shapes:
  - Load test.csv (or sample_submission) to get n_test
  - Skip any model where oof.shape[0] != n_train or test.shape[0] != n_test

### Final Output:
- Create submission.csv with final ensemble predictions
- Print "Final Validation Performance: {score}" at the end""",
        "preprocessing": """## Preprocessing Requirements
- Clean data, handle missing values, encode categoricals
- NO model training
- Fast execution (<10 seconds)
- Save processed data for subsequent components
- Print "Final Validation Performance: 1.0" on completion""",
        "image_to_image_model": """## Image-to-Image Model Requirements (CRITICAL)
This is a PIXEL-LEVEL prediction task. Your model must output FULL IMAGES, not single values.

### DATA PIPELINE FIXES (MANDATORY - PREVENTS COMMON CRASHES):

1. **VARIABLE IMAGE DIMENSIONS** (torch.stack error):
   Images often have different sizes. Use these solutions:
   ```python
   # TRAINING: Use RandomCrop for consistent tensor sizes
   train_transform = transforms.Compose([
       transforms.RandomCrop(256, 256),  # Fixed size for batching
       transforms.ToTensor(),
   ])
   train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # batch_size > 1 OK

   # VALIDATION/TEST: Use batch_size=1 to handle any size
   val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
   test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
   ```

2. **NEGATIVE STRIDES** (numpy/torch conversion error):
   ```python
   # WRONG - creates negative strides:
   img = np.flip(img, axis=1)
   tensor = torch.from_numpy(img)  # CRASHES!

   # CORRECT - fix strides after augmentation:
   img = np.flip(img, axis=1)
   img = np.ascontiguousarray(img)  # FIX STRIDES
   tensor = torch.from_numpy(img)   # Now works!
   ```

3. **NO TRAIN.CSV** (FileNotFoundError):
   Many image-to-image competitions have NO CSV. Load from directories:
   ```python
   # DO NOT: pd.read_csv('train.csv')

   # DO THIS:
   train_dir = Path('/path/to/train')
   clean_dir = Path('/path/to/train_cleaned')
   noisy_files = sorted(train_dir.glob('*.png'))
   pairs = [(nf, clean_dir / nf.name) for nf in noisy_files if (clean_dir / nf.name).exists()]
   print(f"Found {len(pairs)} paired training samples.")
   ```

### Architecture (MUST USE):
- U-Net: encoder-decoder with skip connections
- Autoencoder: encoder-decoder without skip connections
- DnCNN: deep CNN with residual learning
- Fully Convolutional Network (FCN)

### Architecture (DO NOT USE):
- EfficientNet, ResNet, VGG with classification head
- Any model with global average pooling + dense layers
- Any model that outputs a single value per image

### Model Output:
- Input: Image of shape (H, W, C) or (H, W)
- Output: Image of shape (H, W, C) or (H, W) - SAME spatial dimensions
- Loss: MSE, L1, SSIM, or perceptual loss

### Submission Format (CRITICAL):
Sample submission has MILLIONS of rows (one per pixel), NOT one per image!

```python
# CORRECT pattern for pixel-level submission:
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)  # e.g., 5,789,880 rows

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem
    pred = model(load_image(img_path))  # OUTPUT: (H, W) image
    H, W = pred.shape

    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"  # Match sample format
            submission_rows.append({"id": pixel_id, "value": pred[row, col]})

# VERIFY before saving
assert len(submission_rows) == expected_rows, f"WRONG: {len(submission_rows)} vs {expected_rows}"
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
```

### Common Mistake to Avoid:
If your submission has ~29 rows instead of ~5.8M rows, you used a CLASSIFIER instead of an encoder-decoder.
The number of rows = number of test images means WRONG architecture.""",
    }

    # Handle domain-specific model types
    if component_type == "model":
        # Check if we have context suggesting image-to-image
        return guidance.get(component_type, "")

    return guidance.get(component_type, "")


# ==================== Fix and Debug Prompts ====================

FIX_CODE_PROMPT = """Fix this code error.

## Code
```python
{code}
```

## Error
{error}

## Error Type
{error_type}

## Meta-Feedback (use this to fix root cause)
{meta_feedback}

## Correct Data Paths (USE EXACTLY - DO NOT HARDCODE)
{paths}

## CRITICAL REQUIREMENTS (DO NOT REMOVE):
1. MUST preserve `print(f"Final Validation Performance: {{score:.6f}}")` - Meta-Evaluator depends on this exact string
2. MUST preserve soft-deadline pattern with `_check_deadline()` calls
3. MUST keep all OOF prediction saving (np.save)
4. For FileNotFoundError: Use the EXACT paths from "Correct Data Paths" section above. DO NOT hardcode 'train.csv' or 'test.csv'.

Fix the issue while preserving the component's intent. Return complete fixed code."""


DEBUG_CODE_PROMPT = """Debug this code that failed.

## Code
```python
{code}
```

## Issue
{issue}

## Stdout (last lines)
{stdout}

## Stderr
{stderr}

## Meta-Feedback (if available)
{meta_feedback}

## Correct Data Paths (USE EXACTLY - DO NOT HARDCODE)
{paths}

## CRITICAL REQUIREMENTS (DO NOT REMOVE):
1. MUST preserve `print(f"Final Validation Performance: {{score:.6f}}")` - Meta-Evaluator depends on this exact string
2. MUST preserve soft-deadline pattern with `_check_deadline()` calls
3. MUST keep all OOF prediction saving (np.save)
4. For FileNotFoundError or path issues: Use the EXACT paths from "Correct Data Paths" section above. DO NOT hardcode 'train.csv' or 'test.csv'.

Analyze the output, fix logic errors or missing imports, and return the complete debugged code."""


# ==================== Refinement Prompt ====================

REFINEMENT_WITH_FEEDBACK_PROMPT = """Refine this model based on training feedback.

## Current Score
CV: {current_score}

## Training Feedback
{training_feedback}

## Current Code
```python
{current_code}
```

## Improvement Guidelines
Based on the feedback:
- High variance (std > 0.02): Increase regularization, reduce depth
- Overfitting (train >> val): Add dropout, increase subsample
- Underfitting (low score): Decrease regularization, add features
- Optuna best params: Use as starting point

Keep the same [LOG:*] format for the feedback loop.
Return the complete improved code."""


# ==================== Utility Functions ====================


def format_component_details(component) -> str:
    """Format component details for prompts."""
    name = getattr(component, "name", "Unknown")
    component_type = getattr(component, "component_type", "model")
    estimated_impact = getattr(component, "estimated_impact", 0.0)
    code = getattr(component, "code", "No description")

    return f"""Name: {name}
Type: {component_type}
Estimated Impact: {estimated_impact:.1%}
Description: {code}"""


def format_error_info(error: str) -> dict[str, str]:
    """Categorize and format error information."""
    error_types = {
        "ModuleNotFoundError": "missing_import",
        "FileNotFoundError": "missing_file",
        "KeyError": "missing_key",
        "ValueError": "value_error",
        "TypeError": "type_error",
        "SyntaxError": "syntax_error",
        "MemoryError": "memory_error",
        "Timeout": "timeout",
        "TimeoutError": "timeout",
    }

    error_type = "unknown_error"
    for key, value in error_types.items():
        if key in error:
            error_type = value
            break

    return {
        "error_type": error_type,
        "error": error,
    }


# ==================== Ablation Study Prompts ====================

ABLATION_STUDY_PROMPT = """Analyze the impact of component changes through ablation study.

## Baseline Code
```python
{baseline_code}
```

## Modified Code
```python
{modified_code}
```

## Component Being Tested
{component_name}

Compare baseline vs modified performance. Return analysis in JSON format:
{{"component": "{component_name}", "baseline_score": float, "modified_score": float, "delta": float, "recommendation": "keep|remove|modify"}}"""


ABLATION_STUDY_SEQUENTIAL_PROMPT = """Perform sequential ablation study.

## Current Best Code
```python
{current_code}
```

## Components to Test
{components}

Test each component's impact sequentially. Return results for each."""


SUMMARIZE_ABLATION_PROMPT = """Summarize ablation study results.

## Results
{results}

Provide:
1. Most impactful components (positive delta)
2. Harmful components (negative delta)
3. Recommended final configuration"""


EXTRACT_IMPROVEMENT_PLAN_PROMPT = """Extract improvement plan from ablation results.

## Ablation Results
{results}

## Current Score
{current_score}

Create prioritized list of improvements based on ablation findings."""


EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT = """Extract sequential improvement plan.

## Sequential Results
{results}

## Target Score
{target_score}

Create ordered plan to reach target score."""


PLAN_REFINEMENT_PROMPT = """Refine improvement plan based on actual results.

## Original Plan
{original_plan}

## Actual Results
{actual_results}

## Gap Analysis
{gap_analysis}

Update plan based on what worked and what didn't."""


IMPLEMENT_PLAN_PROMPT = """Implement the improvement plan.

## Current Code
```python
{current_code}
```

## Improvement Plan
{plan}

## Priority
{priority}

Generate improved code implementing the plan."""
