"""
Base constraints that apply to ALL domains.

These are the core requirements that every generated code must follow.
"""

BASE_CONSTRAINTS = """## CORE REQUIREMENTS (ALL DOMAINS):

### 0. Canonical Task Contract Precedence
- CANONICAL TASK CONTRACT OVERRIDES generic domain examples and model advice.
- For seq2seq/text-normalization targets, keep targets and predictions as text
  and use the declared sequence metric (exact-match for accuracy). Never encode
  one class per unique target string or coerce target text into a regression
  target.
- Preserve the declared component role. Preprocessing and feature-engineering
  components must not train models or report predictive validation scores.

### 1. Cross-Validation
- Define `RUN_SEED = int(os.getenv("RUN_SEED", "42"))` once and use it everywhere
- Load the injected `CANONICAL_FOLDS_PATH`, `CANONICAL_TRAIN_IDS_PATH`, and
  `CANONICAL_Y_PATH`; fail if any canonical artifact is absent or misaligned
- These names are DEFINED IN THE INJECTED HEADER when a canonical contract
  exists — read the header you were given. If the header has NO
  `CANONICAL DATA CONTRACT` block, the `CANONICAL_*` names do not exist
  anywhere: do NOT invent them or a `load_canonical_data()` loader. In that
  case build a `StratifiedKFold(shuffle=True, random_state=RUN_SEED)` split
  yourself and pass explicit `train_ids=`/`test_ids=` to
  `save_component_artifacts`
- NEVER create a new KFold/StratifiedKFold/GroupKFold inside a model component
  when the canonical contract exists. The canonical assignments already encode
  the task-appropriate split policy
- Iterate the canonical fold labels and write every validation prediction back
  to its original canonical row; use `RUN_SEED` only for model randomness
- Save OOF, test predictions, and row IDs with the injected
  `save_component_artifacts(oof_predictions, test_predictions, ...)` helper;
  do not hand-write the individual files.
- For single-target multiclass classification, reorder every probability
  matrix to `CANONICAL_METADATA["class_order"]` and pass that exact list as
  `class_order=` to `save_component_artifacts`. Never rely on an estimator's
  implicit class-column order.

### 2. Output Requirements
- Only model and ensemble components print
  "Final Validation Performance: {score:.6f}" at the end.
- Preprocessing and feature-engineering components do not print a validation score.
- For probability metrics only, clamp probabilities to `[0, 1]` before saving
- For regression, preserve the prediction scale (RMSLE alone requires non-negative values)
- Match sample_submission.csv exactly: columns, IDs, shape
- Write the final CSV only with the injected `write_submission(test_preds)`
  helper. Never infer ID/target columns by position or call `to_csv` for it.

### 2a. PROBABILITY OUTPUT VALIDATION (CRITICAL - FAIL CLOSED)
For dense model/ensemble classification components, ALWAYS call the injected
`validate_probabilities(...)` helper BEFORE saving OOF and test files. It is
host-owned and already defined in the generated header: call it directly; do
not import, redefine, or assign over it.

The helper rejects row/column mismatches before saving, and its host-owned
implementation will `raise ValueError` for any non-finite prediction with an
error containing `NaN/Inf values; candidate is invalid`.
Never replace NaN/Inf with constants. Only after finite/shape checks does the
helper clip probability outputs to `[0, 1]`; multiclass outputs are normalized
only when their finite row sums are positive.
Temporal CV is the one sanctioned exception: pass the FULL-length OOF with its
warm-up rows still NaN — the helper validates only rows where
`CANONICAL_OOF_ELIGIBLE_MASK` is True and requires the warm-up rows to stay
entirely NaN. Do not fill them and do not validate a masked slice instead.

Packed image-to-image components are explicitly excluded from this helper.
They must use the injected packed evidence contract instead.

```python
# MANDATORY: Call BEFORE saving OOF and test predictions
oof_preds = validate_probabilities(
    oof_preds,
    expected_rows=len(CANONICAL_TRAIN_IDS),
    expected_cols=(N_TARGETS if TARGET_TYPE == "multi_label" else None),
    is_multiclass=(TARGET_TYPE == "single" and n_classes > 2),
    independent_outputs=(TARGET_TYPE == "multi_label"),
    name="OOF",
)
test_preds = validate_probabilities(
    test_preds,
    expected_rows=n_test_entities,
    expected_cols=(N_TARGETS if TARGET_TYPE == "multi_label" else None),
    is_multiclass=(TARGET_TYPE == "single" and n_classes > 2),
    independent_outputs=(TARGET_TYPE == "multi_label"),
    name="Test",
)
```

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

# For fold-based training: use the exact injected split iterator. A temporal
# fold cannot be reconstructed with `CANONICAL_FOLDS != fold_idx`.
for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():
    if _check_deadline():
        print("[LOG:WARNING] Soft deadline reached; checkpointing partial fold state")
        break
    # ... train fold ...

# For PyTorch: Check EVERY BATCH (not just epoch) for long training
for epoch in range(max_epochs):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0 and _check_deadline():  # Check every 10 batches
            print(f"[TIMEOUT] Soft deadline at epoch {epoch}, batch {batch_idx}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                "model_emergency_state.pt",
            )
            break
    if _check_deadline():
        break
```

If the deadline leaves any canonical fold unfinished, save only the documented
partial checkpoint. Do not save a final OOF/test artifact, print a validation
score, or create a submission until every row selected by
`CANONICAL_OOF_ELIGIBLE_MASK` is filled. Temporal warm-up rows outside that
mask must remain NaN.

### 4. Reproducibility
- Set Python, NumPy, framework seeds, and every `random_state` to `RUN_SEED`
- Use deterministic operations when possible

### 5. MUST NOT:
- `sys.exit()`, `exit()`, `quit()`, `raise SystemExit`, `os._exit()`
- try-except blocks that swallow errors silently (let them surface for debugging)
- Overwrite sample_submission.csv (write to submission.csv)
- `nn.BCELoss()` with `torch.cuda.amp.autocast()` (use `nn.BCEWithLogitsLoss()` - it's AMP-safe)
- Convert predictions to integers for AUC/LogLoss metrics: NEVER `(predictions > 0.5).astype(int)`
- Create dummy/fallback submissions with constant values (0.5, mean, zeros) when errors occur
- Use broad `except Exception` clauses that hide FileNotFoundError, RuntimeError, ValueError
- Serialize/pickle a full PyTorch model object; save state_dict plus explicit
  architecture/config metadata instead

### 6. API Gotchas
- OneHotEncoder: `sparse_output=False` (sklearn 1.2+)
- `pd.concat()` instead of `.append()` (pandas 2.0+)
- LightGBM: `lgb.early_stopping(100)` callback, not parameter
- XGBoost: `xgb.callback.EarlyStopping(rounds=100)`

### 7. PyTorch Gotchas
- DataLoader: `pin_memory=False`, `num_workers=0`
- Dataset `__getitem__` must return tensors (never None)
"""
