"""
Tabular data constraints for tree models and structured data.
"""

TABULAR_CONSTRAINTS = """## TABULAR DATA REQUIREMENTS:

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
for fold_idx in range(n_folds):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    # y is ALREADY encoded - no transform needed inside loop
    y_train = y_encoded[train_mask]
    y_val = y_encoded[val_mask]  # Safe: all classes known from full fit

    model.fit(X_train, y_train)
    # LightGBM won't fail because y_val classes are subset of y_train's encoder
```

**WRONG PATTERN (CAUSES CRASHES)**:
```python
# ❌ WRONG: Fitting LabelEncoder inside each fold
for fold_idx in range(n_folds):
    y_train, y_val = y[train_mask], y[val_mask]

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

# If sample_submission columns are strings but y is int:
class_order = sample_sub.columns[1:].tolist()

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
# WRONG: Dropping rare classes causes OOF alignment issues
# rare_classes = counts[counts < 10].index.tolist()
# train_df = train_df[~train_df[target].isin(rare_classes)]  # BREAKS CV alignment!

# CORRECT: Handle rare classes with class weights
model = CatBoostClassifier(
    class_weights='balanced',      # Automatically handles rare classes
    # OR compute manual weights:
    # class_weights={0: 1.0, 1: 10.0, 2: 5.0, ...}
)
```

### 4. Callbacks and Early Stopping
```python
# LightGBM
callbacks = [
    lgb.early_stopping(stopping_rounds=100),  # Use 100, not 50
    lgb.log_evaluation(period=50)
]

# XGBoost
callbacks = [xgb.callback.EarlyStopping(rounds=100)]
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

### 5. Target Column Identification
```python
sample_sub = pd.read_csv(sample_submission_path)
print("Columns:", sample_sub.columns.tolist())

# Target is NOT always columns[1]!
# Match target column name from train.csv
train_df = pd.read_csv(train_path)
target_col = [c for c in sample_sub.columns if c in train_df.columns and c != 'id'][0]
print(f"Target column: {target_col}")

sample_sub[target_col] = predictions
sample_sub.to_csv('submission.csv', index=False)
```

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
assert len(train_engineered) >= len(train_original) * 0.95, \\
    f"CRITICAL: Data loss detected {len(train_original)} → {len(train_engineered)}"
```

### 8. Regression Model Output
For regression tasks, ensure predictions are in valid range:
```python
# Clip predictions to valid range (example for positive targets)
predictions = np.clip(predictions, 0, None)

# For specific domains (e.g., taxi fares, prices):
predictions = np.clip(predictions, min_valid, max_valid)
```
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
