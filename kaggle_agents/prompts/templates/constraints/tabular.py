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
- Fit INSIDE CV folds only (prevent data leakage)
- Handle missing values: `SimpleImputer(strategy='median')`

### 3. Classification
- Use `predict_proba()` NOT `predict()`
- Handle class imbalance if ratio > 2:1:
  - LightGBM: `scale_pos_weight`
  - XGBoost: `scale_pos_weight`
  - CatBoost: `class_weights`

### 4. Callbacks
```python
# LightGBM
callbacks = [lgb.early_stopping(stopping_rounds=100)]

# XGBoost
callbacks = [xgb.callback.EarlyStopping(rounds=100)]
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
