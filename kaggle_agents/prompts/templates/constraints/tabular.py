"""
Tabular data constraints for tree models and structured data.
"""

TABULAR_CONSTRAINTS = """## TABULAR DATA REQUIREMENTS:

### 0. Task-Contract Guard
When canonical metadata defines seq2seq, text normalization, or text output:
- STOP: ignore every remaining tabular section below.
- DO NOT APPLY TABULAR MODEL GUIDANCE. Keep string outputs and the declared
  exact-match/sequence metric; never create one class per unique target string
  or replace text with a regression target.

### 1. Verify Tabular Features Exist
LightGBM, XGBoost, CatBoost need REAL tabular features.
If train.csv only has [id, label] -> It's an IMAGE competition!

```python
train_df = pd.read_csv('train.csv')
print(f"Columns: {train_df.columns.tolist()}")

if len(train_df.columns) <= 2:
    raise ValueError(
        "No tabular features! train.csv has only id+label. "
        "This is an IMAGE competition - use CNNs, not tree models."
    )
```

### 2. Feature Preprocessing
- Use Pipeline/ColumnTransformer for preprocessing
- Fit feature transformers INSIDE CV folds only (prevent data leakage)
- Handle missing values: `SimpleImputer(strategy='median')`

### 2b. TARGET LABEL ENCODING (CRITICAL - DIFFERENT FROM FEATURE PREPROCESSING!)
**IMPORTANT**: Target variable encoding is DIFFERENT from feature preprocessing:
- **Feature preprocessing** (scalers, encoders for X): Fit INSIDE each CV fold
- **Target LabelEncoder** (for y): Fit ONCE on FULL training data BEFORE CV loop

**WHY THIS MATTERS**:
- If LabelEncoder is fit per-fold, rare classes may be missing from some folds
- This causes `ValueError: y contains previously unseen labels` when validation set has classes not in training fold
- LightGBM/XGBoost internal LabelEncoder will also fail on unseen classes

**CORRECT PATTERN (MANDATORY FOR MULTICLASS)**:
```python
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ==========================================
# STEP 1: Fit LabelEncoder on FULL y BEFORE CV (NOT inside loop!)
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Fit on ALL training data
n_classes = len(le.classes_)
print(f"[LOG:INFO] Classes: {le.classes_} (n={n_classes})")

# For Cover_Type (1-7) or similar 1-indexed targets, simpler approach:
# y_encoded = y - 1  # Direct subtraction, no LabelEncoder needed

# ==========================================
# STEP 2: CV loop uses PRE-ENCODED labels
# ==========================================
for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():

    # y is ALREADY encoded - no transform needed inside loop
    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]  # Safe: all classes known from full fit

    model.fit(X_train, y_train)
    # LightGBM won't fail because y_val classes are subset of y_train's encoder
```

**WRONG PATTERN (CAUSES CRASHES)**:
```python
# ❌ WRONG: Fitting LabelEncoder inside each fold
for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():
    y_train, y_val = y[train_idx], y[val_idx]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # Only sees this fold's classes!
    y_val_enc = le.transform(y_val)  # FAILS if y_val has unseen class
```

**FOR LIGHTGBM SPECIFICALLY**:
LightGBM's `fit()` with `eval_set` uses an internal LabelEncoder. To avoid issues:
```python
# Option 1: Pre-encode y and ensure all classes in every fold's training set
y_encoded = le.fit_transform(y)  # Full fit

# Option 2: Use lgb.Dataset directly with all classes
train_data = lgb.Dataset(X_train, label=y_train, reference=None)
# Set reference to first fold's dataset for consistent class handling
```

**TYPE CONSISTENCY (PREVENTS KeyError: np.str_('2'))**:
Ensure target and class_order have the SAME type:
```python
# Check target type
print(f"Target dtype: {y.dtype}, sample values: {y[:3]}")

# Wide target columns declare class order; a single label column does not:
submission_targets = list(SUBMISSION_TARGET_COLS)
class_order = (
    submission_targets
    if len(submission_targets) > 1
    else le.classes_.tolist()
)

# Option 1: Convert y to string to match class_order
y = y.astype(str)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Option 2: Convert class_order to match y's type (preferred for numeric targets)
class_order_typed = [int(c) for c in class_order]  # or float(c)
le = LabelEncoder()
le.fit(class_order_typed)  # Fit on typed class order
y_encoded = le.transform(y)  # y must be same type
```

### 3. Classification
- Use `predict_proba()` NOT `predict()`
- Handle class imbalance if ratio > 2:1:
  - LightGBM: `scale_pos_weight`
  - XGBoost: `scale_pos_weight`
  - CatBoost: `class_weights`

### 3b. CatBoost Classification (CRITICAL - PREVENTS COMMON ERRORS)
**ALWAYS use CatBoostClassifier for classification tasks, NEVER CatBoostRegressor:**

```python
from catboost import CatBoostClassifier  # NOT CatBoostRegressor!
import torch  # For GPU detection

# CORRECT: Classification with MultiClass loss
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',    # For multiclass: 'MultiClass', NOT 'RMSE'
    eval_metric='Accuracy',        # Or 'MultiClass' (logloss)
    random_seed=42,
    early_stopping_rounds=100,
    verbose=200,
    class_weights='balanced',      # Handle imbalanced classes
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
)

# WRONG: Using Regressor for classification (will produce invalid outputs)
# model = CatBoostRegressor(...)  # NEVER use for classification!

# WRONG: Using RMSE loss for classification
# model = CatBoostClassifier(loss_function='RMSE')  # INVALID for classification!
```

**NEVER drop rows with rare classes** - use class_weights instead:
```python
# ❌ WRONG: Dropping rare classes causes OOF alignment issues
# rare_classes = counts[counts < 10].index.tolist()
# train_df = train_df[~train_df[target].isin(rare_classes)]  # BREAKS CV alignment!
# WHY THIS BREAKS:
# 1. OOF array initialized with len(original_train) but predictions only for filtered rows
# 2. Canonical folds.npy has len(original_train) entries, now misaligned
# 3. Ensemble fails: "shape mismatch: (1200000,6) vs (1200000,7)"

# ✅ CORRECT: Handle ALL classes with class weights
# For LightGBM:
model = lgb.LGBMClassifier(
    class_weight='balanced',       # Automatically handles rare classes
    n_jobs=-1,
    random_state=42,
)

# For CatBoost:
model = CatBoostClassifier(
    class_weights='balanced',      # Automatically handles rare classes
    # OR compute manual weights:
    # class_weights={0: 1.0, 1: 10.0, 2: 5.0, ...}
)

# For XGBoost multiclass: compute sample weights manually
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**CRITICAL: OOF Array Initialization**:
```python
# Initialize OOF with ACTUAL number of classes from LabelEncoder (not assumed)
n_classes = len(le.classes_)  # Get from LabelEncoder, NOT hardcoded
n_train = len(train_df)       # Use ORIGINAL train size
oof_preds = np.zeros((n_train, n_classes))
test_preds = np.zeros((n_test, n_classes))
```

### 4. Callbacks and Early Stopping (CRITICAL FOR STABILITY)
Early stopping triggered too early (< 200 iterations) indicates:
- Learning rate too high → reduce to 0.02-0.05
- Early stopping patience too short → increase to 100-200

```python
# LightGBM - RECOMMENDED SETTINGS
model = lgb.LGBMClassifier(
    n_estimators=3000,           # Allow many iterations
    learning_rate=0.02,          # Conservative LR (NOT 0.1!)
    early_stopping_rounds=150,   # Generous patience
    # ... other params
)

callbacks = [
    lgb.early_stopping(stopping_rounds=150),  # Use 100-200, NOT 50
    lgb.log_evaluation(period=100)
]

# XGBoost
callbacks = [xgb.callback.EarlyStopping(rounds=150)]

# CatBoost
model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    early_stopping_rounds=150,
)
```

**DIAGNOSTIC**: If early stopping triggers at < 100 iterations:
```python
# Check if learning rate is too high
if best_iteration < 100:
    print(f"[LOG:WARN] Early stopping at {best_iteration} iterations - consider reducing learning_rate")
```

### 4b. LightGBM Hyperparameters for Large Datasets (>1M rows)
If you see "[Warning] No further splits with positive gain, best gain: -inf":
```python
params = {
    'objective': 'multiclass',
    'num_class': n_classes,
    'metric': 'multi_logloss',

    # CRITICAL for large datasets - defaults are too restrictive
    'min_data_in_leaf': 100,      # Default 20 too small for millions of rows
    'min_gain_to_split': 0.0,     # Allow any positive gain
    'num_leaves': 127,            # Increase from default 31
    'max_depth': -1,              # Unlimited depth

    'learning_rate': 0.05,
    'n_estimators': 1000,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'class_weight': 'balanced',   # Handle class imbalance
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

model = lgb.LGBMClassifier(**params)
```

### 5. Submission Generation (CRITICAL)
**Use the injected schema-aware helper; it already knows the template roles.**

```python
sample_rows = len(pd.read_csv(SAMPLE_SUBMISSION_PATH, usecols=[SUBMISSION_TARGET_COLS[0]]))
predictions = np.asarray(predictions)
if predictions.ndim == 1:
    predictions = predictions.reshape(-1, 1)
expected_shape = (sample_rows, len(SUBMISSION_TARGET_COLS))
if predictions.shape != expected_shape:
    raise ValueError(
        f"Submission prediction shape {predictions.shape} != {expected_shape}"
    )
write_submission(predictions)  # pass test_ids=... only when reordering is needed
```

**CRITICAL**: Do NOT infer roles from `columns[1:]`, assign template columns by
position, or call `to_csv` for submission.csv.

### 6. Optuna Hyperparameter Tuning
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5, timeout=60)  # Keep short
```

### 7. Large Dataset Handling (CRITICAL - READ THIS)
For datasets with >1M rows:
- **NEVER use nrows parameter** in pd.read_csv() - this causes OOF alignment issues
- **DO NOT use drop_duplicates()** - duplicates are often valid data points
- **DO NOT sample or truncate data** - use ALL rows for training
- **DO NOT use .head() or .sample()** to reduce dataset size
- **DO NOT limit data with nrows** - load the FULL dataset, use memory-efficient dtypes instead

If memory is an issue, use memory-efficient dtypes:
```python
# WRONG: nrows limits data and breaks OOF alignment
train = pd.read_csv('train.csv', nrows=1000000)  # NEVER DO THIS

# CORRECT: Use dtypes for memory efficiency
dtypes = {
    'numeric_col': 'float32',
    'int_col': 'int32',
    'category_col': 'category'
}
train = pd.read_csv('train.csv', dtype=dtypes)  # Loads ALL rows
```

If memory is still an issue, use chunked processing:
```python
# Load in chunks instead of sampling
chunks = pd.read_csv('train.csv', chunksize=1_000_000)
for chunk in chunks:
    process(chunk)
```

ALWAYS validate row count after feature engineering:
```python
if len(train_engineered) != len(train_original):
    raise ValueError(
        f"Feature engineering changed row count: "
        f"{len(train_original)} -> {len(train_engineered)}"
    )
if ID_COL in train_original.columns and ID_COL in train_engineered.columns:
    if not np.array_equal(
        train_engineered[ID_COL].to_numpy(),
        train_original[ID_COL].to_numpy(),
    ):
        raise ValueError("Feature engineering changed canonical row order")
elif not train_engineered.index.equals(train_original.index):
    raise ValueError("Feature engineering changed canonical row alignment")
```

### 8. Regression Model Output
Preserve the learned prediction scale by default. Apply bounds only when the
injected metric or data contract explicitly requires them (for example, a
non-negative lower bound for RMSLE). Never derive clipping bounds from a target
column name, competition identity, or assumed real-world semantics.

```python
def validate_regression_predictions(predictions, explicit_bounds=None):
    predictions = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Regression predictions contain NaN or Inf")
    if explicit_bounds is not None:
        lower, upper = explicit_bounds
        if lower is not None:
            predictions = np.maximum(predictions, lower)
        if upper is not None:
            predictions = np.minimum(predictions, upper)
    return predictions
```

### 9. TabFM: Zero-Shot Tabular Foundation Model (STRONG stagnation breaker)
TabFM is an OPTIONAL tabular foundation-model arm. Its score and artifacts are
valid only when they were produced by the real TabFM implementation. A tree
model or any other estimator trained under the TabFM component name is a false
attribution and is forbidden.

```python
# Dependency setup may fail this optional component; it must never change model
# identity. Installation is allowed only when the active runtime policy permits.
try:
    from tabfm import tabfm_v1_0_0
except ImportError:
    raise RuntimeError("TabFM unavailable; prune tabfm_zero_shot") from None

model = tabfm_v1_0_0.load()  # sklearn-compatible; uses GPU if available
model.fit(X_ctx, y_ctx)      # 'fit' = context ingestion (no weight updates)
proba = model.predict_proba(X_val)  # or model.predict(X_val) for regression
```

MANDATORY MODEL-IDENTITY CONTRACT:
- On import, load, compatibility, fit, or predict failure: log the cause, raise
  `RuntimeError`, and let the planner mark/prune `tabfm_zero_shot`.
- NEVER train LightGBM, XGBoost, CatBoost, sklearn, or another estimator in this
  component. A substitute belongs in its own separately named component.
- Write `oof_tabfm_zero_shot.npy` and `test_tabfm_zero_shot.npy` only after
  genuine TabFM inference succeeds and all alignment/finite checks pass.
- Do not reuse stale TabFM-named artifacts after a failed attempt.

HARD LIMITS (check BEFORE using; on violation fail/prune this component):
- Classification: at most 10 classes (multi-label NOT supported).
- Context size: ICL context should be <= ~50,000 rows. For larger training
  sets, subsample ONLY the fit() context (stratified for classification):
  `ctx = train_fold.sample(n=50_000, random_state=42)` — this does NOT violate
  the no-sampling rule in section 7, because predictions are still produced
  for ALL validation/test rows (only the in-context examples are subsampled).
- OOF contract still applies: iterate the canonical folds, fit() on the
  (subsampled) train-fold context, predict the FULL validation fold, save
  models/oof_{name}.npy and models/test_{name}.npy exactly like other models.
"""

MULTI_LABEL_CONSTRAINTS = """## MULTI-LABEL CLASSIFICATION (target_type="multi_label")

### CRITICAL: Use Sigmoid PER CLASS, NOT Softmax
- **Softmax**: Classes are mutually exclusive (single-label multiclass)
- **Sigmoid**: Each class is INDEPENDENT (multi-label)

```python
# CORRECT for multi-label
predictions = torch.sigmoid(logits)  # Independent per class
# or with numpy:
predictions = 1 / (1 + np.exp(-logits))

# WRONG for multi-label (DO NOT use)
predictions = torch.softmax(logits, dim=1)  # Sum = 1, classes exclusive
predictions = predictions / predictions.sum(axis=1, keepdims=True)  # Also wrong!
```

### Metric Calculation
Log-loss PER COLUMN, then AVERAGE (not overall log_loss):
```python
from sklearn.metrics import log_loss
import numpy as np

# CORRECT: Per-column log-loss
scores = [log_loss(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]
final_score = np.mean(scores)
print(f"Final Validation Performance: {final_score:.6f}")

# WRONG: Overall log-loss (treats as single multi-class)
# score = log_loss(y_true, y_pred)  # DO NOT USE for multi-label
```

### Submission Format
- Each row should have INDEPENDENT probabilities [0, 1]
- Rows should NOT sum to 1 (that would be softmax)
- If binary submission is required:
```python
binary_preds = (predictions > 0.5).astype(int)
```

### Loss Function in Training
Use BCEWithLogitsLoss (binary cross-entropy), NOT CrossEntropyLoss:
```python
import torch.nn as nn

# CORRECT for multi-label
criterion = nn.BCEWithLogitsLoss()  # Applies sigmoid internally
loss = criterion(logits, targets.float())

# WRONG for multi-label
# criterion = nn.CrossEntropyLoss()  # This is for single-label
```
"""
