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
"""
