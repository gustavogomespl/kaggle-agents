"""
Base constraints that apply to ALL domains.

These are the core requirements that every generated code must follow.
"""

BASE_CONSTRAINTS = """## CORE REQUIREMENTS (ALL DOMAINS):

### 1. Cross-Validation
- Use StratifiedKFold: `n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5"))`
- Always `shuffle=True, random_state=42`
- Save OOF predictions: `np.save('models/oof_{component_name}.npy', oof_predictions)`

### 2. Output Requirements
- Print "Final Validation Performance: {score:.6f}" at end (CRITICAL for evaluation)
- Clamp predictions: `np.clip(predictions, 0, 1)` before saving
- Match sample_submission.csv exactly: columns, IDs, shape

### 2b. Multi-Modal Hybrid Best Practice
If a competition has BOTH raw image directories (train/, test/, images/) AND a train.csv
with many numeric feature columns, prioritize a HYBRID model:
- CNN for images + MLP for tabular features
- Concatenate CNN embedding with normalized tabular features before the head
- Use Keras Functional API (multi-input) or equivalent
- Use light image augmentation (rotation, zoom, flip)
This is a common Kaggle best practice for multi_modal tasks and often beats separate models.

Generic hybrid example (Keras Functional API):
```python
import tensorflow as tf

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

### 3. Soft-Deadline Pattern (MANDATORY)
CRITICAL: The environment may kill your process at any time. Monitor time actively!

```python
import os, time
_START_TIME = time.time()
_TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "600"))
_SOFT_DEADLINE_S = _TIMEOUT_S - 50  # Reserve 50s for cleanup/save

def _check_deadline() -> bool:
    return (time.time() - _START_TIME) >= _SOFT_DEADLINE_S

# For sklearn/fold-based: Check at start of each fold
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    if _check_deadline():
        print("[LOG:WARNING] Soft deadline reached, stopping early")
        break
    # ... train fold ...

# For PyTorch: Check EVERY BATCH (not just epoch) for long training
for epoch in range(max_epochs):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0 and _check_deadline():  # Check every 10 batches
            print(f"[TIMEOUT] Soft deadline at epoch {epoch}, batch {batch_idx}")
            torch.save(model, 'model_emergency.pth')  # Emergency save FULL model
            break
    if _check_deadline():
        break
```

### 4. Reproducibility
- Set `random_state=42` everywhere
- Use deterministic operations when possible

### 5. MUST NOT:
- `sys.exit()`, `exit()`, `quit()`, `raise SystemExit`, `os._exit()`
- try-except blocks that swallow errors silently (let them surface for debugging)
- Overwrite sample_submission.csv (write to submission.csv)
- `nn.BCELoss()` with `torch.cuda.amp.autocast()` (use `nn.BCEWithLogitsLoss()` - it's AMP-safe)
- Convert predictions to integers for AUC/LogLoss metrics: NEVER `(predictions > 0.5).astype(int)`
- Create dummy/fallback submissions with constant values (0.5, mean, zeros) when errors occur
- Use broad `except Exception` clauses that hide FileNotFoundError, RuntimeError, ValueError

### 6. API Gotchas
- OneHotEncoder: `sparse_output=False` (sklearn 1.2+)
- `pd.concat()` instead of `.append()` (pandas 2.0+)
- LightGBM: `lgb.early_stopping(100)` callback, not parameter
- XGBoost: `xgb.callback.EarlyStopping(rounds=100)`

### 7. PyTorch Gotchas
- DataLoader: `pin_memory=False`, `num_workers=0`
- Dataset `__getitem__` must return tensors (never None)
"""
