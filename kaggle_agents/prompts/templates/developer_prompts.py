"""
Prompt templates for the Developer Agent.

These templates guide code generation for implementing
ablation components with retry and debug capabilities.
"""

from typing import Dict

# Base system prompt for the developer
DEVELOPER_SYSTEM_PROMPT = """You are a Kaggle Grandmaster and expert Python developer specializing in winning Machine Learning competitions.

Your role is to write PRODUCTION-READY, HIGH-PERFORMANCE code that implements machine learning components that achieve top leaderboard scores.

You are known for:
1. **Kaggle Best Practices**: Class imbalance handling, proper CV, probability predictions, no data leakage
2. **Clean Code**: Well-structured, readable, minimal comments (code should be self-explanatory)
3. **Reproducibility**: Always set random_state=42, use StratifiedKFold for CV
4. **Feature Engineering Excellence**: Create impactful features (polynomial, interactions, target encoding)
5. **Model Selection**: Use proven winners (LightGBM, XGBoost, CatBoost) with proper hyperparameters
6. **Efficiency**: Vectorized operations, fast execution (<60s for models, <10s for preprocessing)
7. **Submission Safety**: Match sample_submission exactly (columns/order/id) and clamp probabilities to [0,1]
8. **Schema Awareness**: Detect target column reliably (prefer sample_submission second column; fallback to known names or last non-id col) and log schema (shapes, columns, dtypes)
9. **Leak-Free Pipelines**: Always use Pipeline/ColumnTransformer with SimpleImputer+OneHotEncoder for categoricals and SimpleImputer for numerics; fit inside CV only

CRITICAL RULES (Never Break):
- ALWAYS use predict_proba() for probability predictions (NOT predict())
- ALWAYS check class distribution and apply class weights if imbalanced (ratio > 2:1)
- ALWAYS use StratifiedKFold(n_splits=5) for cross-validation
- ALWAYS print CV scores, class distribution, prediction distribution
- ALWAYS fit all preprocessing (imputer/encoder/scaler) INSIDE a Pipeline/ColumnTransformer that is fit per CV fold (no global median/mean before split)
- ALWAYS validate submission against sample_submission (shape, columns, id order) before saving; clamp probs to [0,1] if needed
- ALWAYS print informative logs: shapes, column list, dtypes, class distribution, per-fold scores, prediction stats
- NEVER use try-except to hide errors (let them surface for debugging)
- NEVER subsample training data (use all available data)
- NEVER use sys.exit(), exit(), quit(), raise SystemExit, os._exit(), or ANY termination commands
- ALWAYS save submission.csv with probabilities (0.0-1.0), NOT binary predictions (0/1)

CRITICAL API USAGE (Common Mistakes to Avoid):
LightGBM/XGBoost Early Stopping:
- ❌ WRONG: model.fit(X, y, early_stopping_rounds=100)  # This will fail!
- ✅ CORRECT: model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
- ✅ ALTERNATIVE: Don't use early stopping and just set n_estimators appropriately
- For LightGBM: Use lgb.early_stopping(stopping_rounds=100) in callbacks parameter
- For XGBoost: Use xgb.callback.EarlyStopping(rounds=100) in callbacks parameter
- NEVER pass early_stopping_rounds as a direct parameter to fit()

Categorical Features (MANDATORY - WILL FAIL WITHOUT THIS):
- ❌ CRITICAL ERROR: Training LightGBM/XGBoost/sklearn models on unencoded categorical features WILL FAIL
- ✅ REQUIRED: ALWAYS encode categorical columns before training (except CatBoost which handles them natively)
- ✅ MUST use ColumnTransformer with OneHotEncoder for categorical features (object/category dtypes)
- ✅ Pipeline structure: ColumnTransformer([('num', SimpleImputer(), numeric_cols), ('cat', Pipeline([SimpleImputer(), OneHotEncoder(sparse_output=False)]), categorical_cols)])
- Check for 'object' or 'category' dtypes - if they exist, they MUST be encoded
- Never pass raw categorical strings to LightGBM/XGBoost/sklearn models

sklearn/pandas Version Compatibility (CRITICAL):
- ❌ WRONG: OneHotEncoder(sparse=False)  # Deprecated in sklearn 1.2+
- ✅ CORRECT: OneHotEncoder(sparse_output=False)  # Use this for sklearn 1.2+
- ❌ WRONG: df.append(other_df)  # Removed in pandas 2.0+
- ✅ CORRECT: pd.concat([df, other_df], ignore_index=True)  # Use this for pandas 2.0+
- ❌ WRONG: series.append(other_series)  # Removed in pandas 2.0+
- ✅ CORRECT: pd.concat([series, other_series], ignore_index=True)  # Use this for pandas 2.0+
- ALWAYS use sparse_output parameter (not sparse) in OneHotEncoder
- ALWAYS use pd.concat() instead of .append() for combining DataFrames/Series

Optuna Dependencies (CRITICAL):
- ❌ WRONG: from optuna.integration import OptunaSearchCV  # May not be installed
- ✅ CORRECT: Use try/except to check if optuna-integration is available
- ✅ SILENCE OPTUNA: ALWAYS set optuna.logging.set_verbosity(optuna.logging.WARNING) to prevent "False Positive Error Detection"
- ✅ LIMIT TUNING: For validation, use n_trials=2 or timeout=60 to prevent timeouts.
- ✅ Example:
  ```python
  import optuna
  optuna.logging.set_verbosity(optuna.logging.WARNING)  # SILENCE OPTUNA
  
  try:
      from optuna.integration import OptunaSearchCV
      USE_OPTUNA_INTEGRATION = True
  except ImportError:
      USE_OPTUNA_INTEGRATION = False
      print("optuna-integration not available, using manual Optuna tuning")
  ```
- If optuna-integration is missing, use manual Optuna with study.optimize()
- NEVER fail because a dependency is missing - always have a fallback

🚀 GPU ACCELERATION (CRITICAL - ALWAYS ENABLE):
- **ALWAYS check for GPU availability at the start of code**
- **ALWAYS enable GPU for LightGBM, XGBoost, and CatBoost models**
- GPU acceleration is **MANDATORY** for competitive training speed
- Detection pattern (REQUIRED in ALL model code):
  ```python
  import torch
  use_gpu = torch.cuda.is_available()
  print(f"GPU Available: {use_gpu}")
  if use_gpu:
      print("✅ GPU acceleration ENABLED")
  else:
      print("⚠️  Running on CPU (slower)")
  ```
- **LightGBM GPU params (REQUIRED if GPU available)**:
  ```python
  if use_gpu:
      lgb_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
  else:
      lgb_params = {'device': 'cpu'}
  model = lgb.LGBMClassifier(**lgb_params, n_estimators=..., ...)
  ```
- **XGBoost GPU params (REQUIRED if GPU available)**:
  ```python
  if use_gpu:
      xgb_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
  else:
      xgb_params = {'tree_method': 'hist'}
  
  # ROBUST FALLBACK: Wrap fit in try-except to handle GPU failures
  try:
      model = xgb.XGBClassifier(**xgb_params, n_estimators=..., ...)
      model.fit(X, y)
  except Exception as e:
      print(f"⚠️  GPU Training failed: {e}. Falling back to CPU...")
      xgb_params['tree_method'] = 'hist'
      xgb_params.pop('predictor', None)
      model = xgb.XGBClassifier(**xgb_params, n_estimators=..., ...)
      model.fit(X, y)
  ```
- **CatBoost GPU params (REQUIRED if GPU available)**:
  ```python
  task_type = "GPU" if use_gpu else "CPU"
  model = CatBoostClassifier(task_type=task_type, iterations=..., ...)
  ```
- **NEVER skip GPU configuration** - it's the difference between 30s and 5min training time
- **ALWAYS print GPU status** so users know if GPU is being used

MANDATORY OUTPUT FORMAT (MLE-STAR Pattern):
- Your response must contain ONLY a single Python code block
- No additional text, explanations, or markdown outside the code block
- The code MUST print 'Final Validation Performance: {score}' at the end
- The score must be the cross-validation performance metric
- Example: print(f"Final Validation Performance: {cv_accuracy:.6f}")

📊 MANDATORY PERFORMANCE LOGGING (CRITICAL FOR DEBUGGING):
- Print timing for EACH major step using consistent format
- Format: "⏱️ [STEP_NAME] completed in {time:.2f}s"
- Track cumulative time from start
- Example implementation:
  ```python
  import time
  
  start_time = time.time()
  step_times = {}
  
  def log_step(step_name, start):
      elapsed = time.time() - start
      step_times[step_name] = elapsed
      cumulative = time.time() - start_time
      print(f"⏱️ [{step_name}] completed in {elapsed:.2f}s (cumulative: {cumulative:.2f}s)")
      return time.time()
  
  # Usage:
  step_start = time.time()
  # ... load data ...
  step_start = log_step("DATA_LOADING", step_start)
  
  # ... preprocess ...
  step_start = log_step("PREPROCESSING", step_start)
  
  # ... train model ...
  step_start = log_step("MODEL_TRAINING", step_start)
  
  # At the end, print summary:
  print("\\n📊 Performance Summary:")
  for step, duration in step_times.items():
      print(f"  {step}: {duration:.2f}s")
  print(f"  TOTAL: {time.time() - start_time:.2f}s")
  ```
- REQUIRED steps to log:
  1. DATA_LOADING - Time to load train/test CSVs
  2. PREPROCESSING - Time for encoding, imputation, feature engineering
  3. CV_FOLD_N - Time for each cross-validation fold (N=1,2,3,4,5)
  4. MODEL_TRAINING - Total training time across all folds
  5. PREDICTION - Time to make test predictions
  6. SUBMISSION_SAVE - Time to save submission file
- For Optuna tuning, log each trial:
  "⏱️ [OPTUNA_TRIAL_N] score={score:.4f} in {time:.2f}s"
- Print memory usage after large operations (optional but recommended):
  ```python
  import psutil
  process = psutil.Process()
  mem_mb = process.memory_info().rss / 1024 / 1024
  print(f"📈 Memory usage: {mem_mb:.1f} MB")
  ```
- Print intermediate scores during CV:
  "📊 Fold {n}/5: score={fold_score:.4f} (running mean: {mean_score:.4f})"

Your code should:
- Import all necessary libraries
- Load data from correct paths
- Check for class imbalance and handle appropriately
- Implement the specified component with best practices
- Use cross-validation to estimate performance
- Make probability predictions (not hard predictions)
- Save outputs/models to correct locations
- Print execution time and key metrics with structured logging
- Be a complete, executable single-file Python program
"""

# Template for generating code from ablation component
GENERATE_CODE_PROMPT = """Generate complete, production-ready Python code for the following component.

## Component Details
{component_details}

## Competition Context
Competition: {competition_name}
Domain: {domain}
Problem Type: {problem_type}
Metric: {metric}

## Data Paths
Train Data: {train_data_path}
Test Data: {test_data_path}
Models Directory: {models_dir}
Submission Path: {submission_path}

## Dataset Information
{dataset_info}

## CRITICAL TYPE-SPECIFIC REQUIREMENTS

### If component_type == "preprocessing" or "feature_engineering":
- **DO NOT train any models** (no fit(), no GridSearch, no model training)
- ONLY clean data, handle missing values, scale features, or create new features
- Must execute in **under 10 seconds** (keep it simple and fast)
- Can save processed data to models directory for later use
- **MUST print "Final Validation Performance: 1.0" at the end** (placeholder for successful completion)
- Example: print("Final Validation Performance: 1.0  # Feature engineering complete")

### If component_type == "model":
- **MUST train a model** (LightGBM, XGBoost, or CatBoost recommended for best performance)
- **For classification**: use **predict_proba()** to get probabilities (NOT predict() for hard predictions)
- **For regression**: use **predict()** to get continuous values
- **MUST create submission.csv** at {submission_path} with predictions
- **MUST load sample_submission.csv** (if available) and use it as template: same shape/columns/id ordering
- **MUST detect target_col** reliably: prefer `target_col = sample_sub.columns[1]` if present; else use common names (`target`, `label`, `loan_paid_back`) or last non-id column; assert existence in train df
- **CRITICAL: MUST save Out-of-Fold (OOF) predictions** for stacking ensemble:
  - During CV, collect predictions on validation folds (oof_predictions array, same length as train)
  - Save to: `{models_dir}/oof_{{component_name}}.npy` using `np.save()`
  - This enables proper stacking ensemble later (train meta-model on OOF predictions)
  - Print: "OOF predictions saved to {{oof_path}}"
- Use **dataset-adaptive hyperparameters** (automatically calculated):
  - Variables `n_estimators`, `max_depth`, `learning_rate` are set based on dataset size
  - Small datasets (<5k): More trees, deeper (n_estimators=1000, max_depth=8)
  - Large datasets (>100k): Fewer trees, shallower (n_estimators=400, max_depth=5)
  - **num_leaves** (LightGBM): Use `2^max_depth - 1` for consistency

🚀 **GPU ACCELERATION (CRITICAL - MANDATORY)**:
  - **STEP 1**: ALWAYS detect GPU at the start of your code:
    ```python
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"GPU Available: {use_gpu}")
    ```
  - **STEP 2**: Configure model params based on GPU availability:
    - **LightGBM**: Add `'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0` if GPU available
    - **XGBoost**: Add `'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'` if GPU available
    - **CatBoost**: Set `task_type="GPU"` if GPU available (otherwise "CPU")
  - **STEP 3**: Print GPU status so users can verify:
    ```python
    if use_gpu:
        print("✅ GPU ENABLED for training (fast)")
    else:
        print("⚠️  CPU mode (slower)")
    ```
  - **Example for LightGBM**:
    ```python
    lgb_gpu_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0} if use_gpu else {'device': 'cpu'}
    model = lgb.LGBMClassifier(**lgb_gpu_params, n_estimators=n_estimators, max_depth=max_depth, ...)
    ```
  - **Example for XGBoost (ROBUST FALLBACK REQUIRED)**:
    ```python
    try:
        model = xgb.XGBClassifier(**xgb_params, ...)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ GPU failed: {e}. Retrying with CPU...")
        xgb_params['tree_method'] = 'hist' # Fallback
        model = xgb.XGBClassifier(**xgb_params, ...)
        model.fit(X_train, y_train)
    ```
  - **This is NOT optional** - GPU reduces training time from minutes to seconds

🛑 **OPTUNA CONFIGURATION (CRITICAL)**:
  - **SILENCE LOGGING**: `optuna.logging.set_verbosity(optuna.logging.WARNING)` (Prevent "False Positive Error Detection")
  - **LIMIT TUNING**: Use `n_trials=2` or `timeout=60` for validation runs. Do NOT run for 10 minutes.

- Target execution time: 60-90 seconds per model (use early stopping)
- Print CV score or validation metrics

#### CRITICAL: Class Imbalance Handling
- **ALWAYS check class distribution** in training data
- **Calculate class weights** if imbalanced (ratio > 2:1)
- For XGBoost: use `scale_pos_weight = negative_count / positive_count`
- For LightGBM: use `is_unbalance=True` or `class_weight='balanced'`
- For sklearn models: use `class_weight='balanced'`
- **ALWAYS use StratifiedKFold** for cross-validation (5 folds)
- **Print class distribution** before and after predictions

#### CRITICAL: Probability Predictions
- Use `model.predict_proba(X_test)[:, 1]` for binary classification
- For multiclass: use `model.predict_proba(X_test)` (all class probabilities)
- Submission must contain probabilities (0.0-1.0), NOT hard predictions (0/1)
- Example: `submission['prediction'] = model.predict_proba(X_test)[:, 1]`
- Validate submission with:
  - `assert submission.shape == sample_sub.shape`
  - `assert submission.columns.tolist() == sample_sub.columns.tolist()`
  - `assert submission['id'].equals(sample_sub['id'])`
  - print head/dtypes/range checks
- Clamp probabilities to [0, 1] with `np.clip` before saving
- Log: min/max/mean of predictions and per-fold scores
- For CatBoost: specify `cat_features` by indices; prefer `auto_class_weights='Balanced'` (avoid unsupported `class_weight` in older versions)

### If component_type == "ensemble":
- **PREFERRED: Stacking Ensemble** (best performance)
  - Load OOF predictions from all base models: `np.load('{models_dir}/oof_{{model_name}}.npy')`
  - Stack OOF predictions horizontally: `oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat, ...])`
  - Train meta-model (LogisticRegression/Ridge) on stacked OOF: `meta_model.fit(oof_stack, y_train)`
  - Load test predictions from each model and stack them
  - Use meta-model to predict on stacked test: `final_predictions = meta_model.predict_proba(test_stack)[:, 1]`
- **FALLBACK: Weighted Average** (if OOF files missing)
  - Load submission files from each model
  - Calculate weights based on CV scores or use equal weights
  - Combine: `final = w1*pred1 + w2*pred2 + w3*pred3`
- **MUST create submission.csv** with ensemble predictions
- **MUST use sample_submission.csv** as the template and validate shape/columns/id
- Print which models were used and their contribution/weights

## General Requirements
1. Load data from the provided paths
2. Implement the component exactly as specified above
3. Print progress and key metrics
4. Handle errors gracefully
5. Use sklearn Pipeline/ColumnTransformer so that imputers/encoders are fit inside CV splits (no leakage)
6. Default preprocessing: numeric -> SimpleImputer(strategy='median'); categorical (object/category) -> SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore', sparse_output=False); wrap model in Pipeline
7. CRITICAL: Use sparse_output=False (NOT sparse=False) for OneHotEncoder (sklearn 1.2+ compatibility)
8. CRITICAL: Use pd.concat() instead of .append() for DataFrames/Series (pandas 2.0+ compatibility)

## CRITICAL GUARDRAILS (You are a Kaggle Grandmaster - follow best practices)
- **NO try-except blocks that hide errors** - let errors surface for debugging
- **NO subsampling of training data** - use all available data
- **NO sys.exit()** or similar termination commands
- **ALWAYS print intermediate validation scores** during training
- **ALWAYS use all provided features** - don't drop columns arbitrarily
- **ALWAYS set random_state/random_seed** for reproducibility (use 42)
- **Print execution time** at the end

## Code Structure
```python
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time
# ... other imports

# Silence Optuna logs (CRITICAL - prevents false error detection)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    pass

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========== PERFORMANCE LOGGING SETUP ==========
start_time = time.time()
step_times = {{}}

def log_step(step_name, step_start):
    # Log timing for a processing step
    elapsed = time.time() - step_start
    step_times[step_name] = elapsed
    cumulative = time.time() - start_time
    print(f"⏱️ [{{step_name}}] completed in {{elapsed:.2f}}s (cumulative: {{cumulative:.2f}}s)")
    return time.time()

def print_performance_summary():
    # Print final performance summary
    print("\\n📊 Performance Summary:")
    for step, duration in step_times.items():
        print(f"  {{step}}: {{duration:.2f}}s")
    print(f"  TOTAL: {{time.time() - start_time:.2f}}s")
# ================================================

step_start = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')
sample_sub = pd.read_csv('{submission_path}'.replace('submission.csv', 'sample_submission.csv'))
step_start = log_step("DATA_LOADING", step_start)

print(f"Train shape: {{train_df.shape}}, Test shape: {{test_df.shape}}")
print(f"Train columns: {{train_df.columns.tolist()}}")
print("Train dtypes:")
print(train_df.dtypes)

# CRITICAL: Detect target column (MUST match sample_submission)
# Priority 1: Use sample_submission.columns[1] (most reliable)
# Priority 2: Use common target names if they exist
# Priority 3: Last non-id column
candidate_target = sample_sub.columns[1] if len(sample_sub.columns) > 1 else None
fallback_targets = ['target', 'label', 'loan_paid_back', 'survived', 'price', 'sales']
target_col = candidate_target if candidate_target and candidate_target in train_df.columns else None
if target_col is None:
    for t in fallback_targets:
        if t in train_df.columns:
            target_col = t
            break
if target_col is None:
    non_id_cols = [c for c in train_df.columns if c.lower() != 'id']
    target_col = non_id_cols[-1] if non_id_cols else 'target'

# Validate target column exists
assert target_col in train_df.columns, f"Target column '{{target_col}}' not found in train data. Available: {{train_df.columns.tolist()}}"
print(f"✓ Target column detected: '{{target_col}}'")
print(f"  NOTE: This MUST match the submission column name from sample_submission")

# Separate X/y
y_train = train_df[target_col]
X_train = train_df.drop(columns=[target_col, 'id'] if 'id' in train_df.columns else [target_col])
X_test = test_df.drop(columns=['id'], errors='ignore')

# CRITICAL: Identify categorical columns and prepare preprocessing
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Numeric features: {{len(numeric_features)}}")
print(f"Categorical features: {{len(categorical_features)}}")
if categorical_features:
    print(f"  Categorical columns: {{categorical_features}}")
    print("  ⚠️  MUST encode categorical features before training")

# Validate target column
print(f"  Unique values: {{y_train.nunique()}}")
print(f"  Null count: {{y_train.isnull().sum()}}")
print(f"  Data type: {{y_train.dtype}}")

# Check class distribution (for model components)
if y_train.nunique() <= 30:
    print(f"  Class distribution: {{y_train.value_counts().to_dict()}}")

# Implement component
print("Implementing {component_name}...")

# Dynamic hyperparameter adjustment based on dataset size
n_rows = train_df.shape[0]
n_features = X_train.shape[1]

# Adjust n_estimators based on dataset size
if n_rows < 5_000:
    n_estimators = 1000
    max_depth = 8
    learning_rate = 0.05
elif n_rows < 20_000:
    n_estimators = 800
    max_depth = 7
    learning_rate = 0.04
elif n_rows < 100_000:
    n_estimators = 600
    max_depth = 6
    learning_rate = 0.03
else:
    n_estimators = 400
    max_depth = 5
    learning_rate = 0.03

print(f"📊 Dataset-adaptive hyperparameters:")
print(f"  n_estimators: {{n_estimators}} (based on {{n_rows:,}} rows)")
print(f"  max_depth: {{max_depth}}")
print(f"  learning_rate: {{learning_rate}}")

# ... your code here

# For model components: Calculate class weights if needed
positive_count = (y_train == 1).sum()
negative_count = (y_train == 0).sum()
imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
print(f"Class imbalance ratio: {{imbalance_ratio:.2f}}")

if imbalance_ratio > 2.0:
    print("⚠️  Class imbalance detected - applying weights")
    scale_pos_weight = negative_count / positive_count

# CRITICAL: Build preprocessing pipeline (MANDATORY for categorical features)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ] if categorical_features else [
        ('num', SimpleImputer(strategy='median'), numeric_features)
    ]
)

# Example: Wrap model in pipeline with preprocessing
# from xgboost import XGBClassifier
# model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         learning_rate=learning_rate,
#         scale_pos_weight=scale_pos_weight if imbalance_ratio > 2.0 else 1.0,
#         random_state=42
#     ))
# ])
#
# Alternative: Apply preprocessing manually
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_test_preprocessed = preprocessor.transform(X_test)

# Cross-validation with StratifiedKFold (CRITICAL: Save OOF predictions for stacking)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Initialize OOF predictions array
oof_predictions = np.zeros(len(X_train))
cv_scores = []

step_start = log_step("PREPROCESSING", step_start)

print("\\nTraining with 5-fold cross-validation...")
cv_start = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    fold_start = time.time()
    print(f"  Fold {{fold}}/5...")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train model on fold
    model.fit(X_tr, y_tr)

    # Predict on validation fold (save for OOF)
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_predictions[val_idx] = val_preds

    # Calculate fold score
    from sklearn.metrics import roc_auc_score
    fold_score = roc_auc_score(y_val, val_preds)
    cv_scores.append(fold_score)
    fold_time = time.time() - fold_start
    running_mean = np.mean(cv_scores)
    print(f"    📊 Fold {{fold}}/5: score={{fold_score:.4f}} (running mean: {{running_mean:.4f}}) in {{fold_time:.2f}}s")
    step_times[f"CV_FOLD_{{fold}}"] = fold_time

step_times["MODEL_TRAINING"] = time.time() - cv_start
print(f"⏱️ [MODEL_TRAINING] completed in {{step_times['MODEL_TRAINING']:.2f}}s")
print(f"\\nCV Score (ROC-AUC): {{np.mean(cv_scores):.4f}} (+/- {{np.std(cv_scores):.4f}})")

# CRITICAL: Save OOF predictions for stacking ensemble
oof_path = Path('{models_dir}') / 'oof_{{component_name}}.npy'
np.save(oof_path, oof_predictions)
print(f"✓ OOF predictions saved to: {{oof_path}}")

# Make predictions (MUST use predict_proba for probabilities)
step_start = time.time()
predictions = model.predict_proba(X_test)[:, 1]  # Binary classification
predictions = np.clip(predictions, 0, 1)
step_start = log_step("PREDICTION", step_start)
print(f"Prediction distribution: min={{predictions.min():.4f}}, max={{predictions.max():.4f}}, mean={{predictions.mean():.4f}}")

# Save outputs
print("Saving outputs...")
submission = sample_sub.copy()
# CRITICAL: Use the EXACT column name from sample_submission (DO NOT hardcode 'target')
target_submission_col = sample_sub.columns[1]
print(f"✓ Using submission column: '{{target_submission_col}}' (from sample_submission)")
submission[target_submission_col] = predictions

# CRITICAL VALIDATION: Ensure submission matches sample_submission exactly
assert submission.shape == sample_sub.shape, f"Submission shape mismatch: {{submission.shape}} vs {{sample_sub.shape}}"
assert submission.columns.tolist() == sample_sub.columns.tolist(), f"Column mismatch: {{submission.columns.tolist()}} vs {{sample_sub.columns.tolist()}}"
assert submission['id'].equals(sample_sub['id']), "Submission id column does not match sample_submission"
print(f"✓ Validation passed: columns={{submission.columns.tolist()}}, shape={{submission.shape}}")
print("Submission head:")
print(submission.head())
print(f"Prediction stats: min={{predictions.min():.4f}}, max={{predictions.max():.4f}}, mean={{predictions.mean():.4f}}")
step_start = time.time()
submission.to_csv('{submission_path}', index=False)
log_step("SUBMISSION_SAVE", step_start)
print(f"✅ Submission saved: {{len(submission)}} rows to {{'{submission_path}'}}")

# Print performance summary
print_performance_summary()

# MANDATORY: Final validation performance output
print(f"Final Validation Performance: {{np.mean(cv_scores):.6f}}")
print("✅ Complete!")
```

Generate the complete code below:
"""

# Template for fixing code errors
FIX_CODE_PROMPT = """The following code failed with an error. Fix it.

## Original Code
```python
{code}
```

## Error
{error}

## Error Type
{error_type}

## Fix Instructions
1. Identify the root cause of the error
2. Fix the issue while preserving the component's intent
3. Add error handling if appropriate
4. Return the complete fixed code

Generate the fixed code below:
"""

# Template for debugging code
DEBUG_CODE_PROMPT = """Debug and improve the following code that is not working as expected.

## Code
```python
{code}
```

## Issue
{issue}

## Stdout (last 50 lines)
{stdout}

## Stderr
{stderr}

## Debug Instructions
1. Analyze the output to identify the problem
2. Fix any logic errors, missing imports, or incorrect paths
3. Add debugging print statements if needed
4. Ensure the code achieves the intended goal

Generate the debugged code below:
"""

# Template for code refactoring
REFACTOR_CODE_PROMPT = """Refactor the following working code to improve quality.

## Code
```python
{code}
```

## Refactoring Goals
- Improve readability and structure
- Add proper error handling
- Optimize performance where possible
- Add docstrings and comments
- Follow PEP 8 style guide

Generate the refactored code below:
"""

# Template for component integration
INTEGRATE_COMPONENT_PROMPT = """Integrate the following component into the existing pipeline.

## New Component
{new_component_code}

## Existing Pipeline
{existing_pipeline_code}

## Integration Instructions
1. Add the new component at the appropriate stage
2. Ensure data flows correctly between components
3. Handle any compatibility issues
4. Maintain existing functionality

Generate the integrated code below:
"""

# Domain-specific code templates
DOMAIN_CODE_TEMPLATES = {
    "tabular": """
# Tabular Data Template
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import {metric_function}
import xgboost as xgb
import lightgbm as lgb

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

# Separate features and target
X = train_df.drop('{target_col}', axis=1)
y = train_df['{target_col}']
X_test = test_df

# Feature engineering
# TODO: Implement feature engineering

# Model training
# GPU Detection (CRITICAL - MANDATORY)
import torch
use_gpu = torch.cuda.is_available()
print(f"GPU Available: {use_gpu}")

if use_gpu:
    print("✅ GPU ENABLED for training")
    xgb_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'random_state': 42, 'n_estimators': 100}
else:
    print("⚠️  Running on CPU (slower)")
    xgb_params = {'tree_method': 'hist', 'random_state': 42, 'n_estimators': 100}

model = xgb.XGBClassifier(**xgb_params)
model.fit(X, y)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='{metric}')
print(f"CV Score: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std():.4f}})")

# Predictions
predictions = model.predict(X_test)

# Save model
import joblib
joblib.dump(model, '{model_path}')
""",

    "computer_vision": """
# Computer Vision Template
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Data augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model setup (using timm for pretrained models)
model = timm.create_model('resnet50', pretrained=True, num_classes={num_classes})
model = model.to(device)

# Training
# TODO: Implement training loop

# Save model
torch.save(model.state_dict(), '{model_path}')
""",

    "nlp": """
# NLP Template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels={num_labels}
)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# TODO: Load and prepare data
# train_dataset = Dataset.from_pandas(train_df)
# train_dataset = train_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  # TODO: Add dataset
)

# Train
# trainer.train()

# Save
model.save_pretrained('{model_path}')
""",

    "time_series": """
# Time Series Template
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Load data
df = pd.read_csv('{data_path}', parse_dates=['{date_col}'])

# Prepare for Prophet
df_prophet = df.rename(columns={{'{date_col}': 'ds', '{target_col}': 'y'}})

# Train model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df_prophet)

# Make predictions
future = model.make_future_dataframe(periods={forecast_periods})
forecast = model.predict(future)

# Save
import joblib
joblib.dump(model, '{model_path}')
""",
}

# Advanced ML code templates
ADVANCED_ML_TEMPLATES = {
    "stacking_ensemble_classification": """
# Stacking Ensemble Template for Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import time

RANDOM_SEED = 42
start_time = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')

# Prepare features and target
# TODO: Identify target column and separate features
X = train_df.drop('{target_col}', axis=1)
y = train_df['{target_col}']
X_test = test_df.drop('{target_col}', axis=1, errors='ignore')

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))
X_test_imputed = imputer.transform(X_test.select_dtypes(include=[np.number]))

# Check class imbalance
positive_count = (y == 1).sum()
negative_count = (y == 0).sum()
imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
print(f"Class imbalance ratio: {{imbalance_ratio:.2f}}")

scale_pos_weight = negative_count / positive_count if imbalance_ratio > 2.0 else 1.0

# GPU Detection (CRITICAL - MANDATORY)
import torch
use_gpu = torch.cuda.is_available()
print(f"GPU Available: {use_gpu}")

if use_gpu:
    print("✅ GPU ENABLED for training")
    xgb_gpu_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    lgb_gpu_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
else:
    print("⚠️  Running on CPU (slower)")
    xgb_gpu_params = {'tree_method': 'hist'}
    lgb_gpu_params = {'device': 'cpu'}

# Define base learners (diverse models)
base_learners = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **xgb_gpu_params
    )),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.03,
        is_unbalance=(imbalance_ratio > 2.0),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        **lgb_gpu_params
    )),
    ('catboost', CatBoostClassifier(
        iterations=2000,
        depth=7,
        learning_rate=0.03,
        random_state=RANDOM_SEED,
        verbose=False,
        task_type="GPU" if use_gpu else "CPU"
    ))
]

# Meta-learner
meta_learner = LogisticRegression(
    random_state=RANDOM_SEED,
    max_iter=1000,
    class_weight='balanced' if imbalance_ratio > 2.0 else None
)

print("\\nBuilding stacking ensemble...")
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# Train stacking model
print("Training stacking ensemble...")
stacking_clf.fit(X_imputed, y)

# Cross-validation
print("\\nEvaluating with cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(stacking_clf, X_imputed, y, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"CV Score (ROC-AUC): {{cv_scores.mean():.4f}} (+/- {{cv_scores.std():.4f}})")

# Make predictions
print("\\nMaking predictions...")
predictions = stacking_clf.predict_proba(X_test_imputed)[:, 1]

# Save submission
# Load sample_submission to get correct column names
sample_sub = pd.read_csv('{submission_path}'.replace('submission.csv', 'sample_submission.csv'))
submission = sample_sub.copy()
target_submission_col = sample_sub.columns[1]
print(f"✓ Using submission column: '{{target_submission_col}}' (from sample_submission)")
submission[target_submission_col] = predictions
assert submission.shape == sample_sub.shape, f"Shape mismatch: {{submission.shape}} vs {{sample_sub.shape}}"
assert submission.columns.tolist() == sample_sub.columns.tolist(), f"Column mismatch: {{submission.columns.tolist()}} vs {{sample_sub.columns.tolist()}}"
print(f"✓ Validation passed: columns={{submission.columns.tolist()}}")
submission.to_csv('{submission_path}', index=False)

elapsed_time = time.time() - start_time
print(f"\\n⏱️  Execution time: {{elapsed_time:.2f}}s")
print(f"✅ Stacking ensemble complete! Submission saved with {{len(submission)}} rows")
""",

    "stacking_ensemble_regression": """
# Stacking Ensemble Template for Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import time

RANDOM_SEED = 42
start_time = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')

# Detect target column
target_col = None
for cand in ['target', 'y', 'loss', 'label']:
    if cand in train_df.columns:
        target_col = cand
        break
if not target_col:
    # Use last numeric column
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = numeric_cols[-1] if numeric_cols else 'target'

print(f"Target column: {{target_col}}")

# Prepare features and target
X = train_df.drop(target_col, axis=1, errors='ignore')
y = train_df[target_col]
X_test = test_df.copy()

# Remove ID columns
id_cols = [c for c in X.columns if 'id' in c.lower()]
if id_cols:
    test_ids = X_test[id_cols[0]] if id_cols[0] in X_test.columns else np.arange(len(X_test))
    X = X.drop(columns=id_cols)
    X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns])
else:
    test_ids = np.arange(len(X_test))

# Handle missing values
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Separate numeric and categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Impute numerics
num_imputer = SimpleImputer(strategy='median')
X_num = num_imputer.fit_transform(X[numeric_cols]) if numeric_cols else np.array([]).reshape(len(X), 0)
X_test_num = num_imputer.transform(X_test[numeric_cols]) if numeric_cols else np.array([]).reshape(len(X_test), 0)

# Encode categoricals
if cat_cols:
    X_cat = X[cat_cols].copy()
    X_test_cat = X_test[cat_cols].copy()

    for col in cat_cols:
        le = LabelEncoder()
        # Combine to fit on all categories
        combined = pd.concat([X_cat[col].astype(str), X_test_cat[col].astype(str)])
        le.fit(combined.fillna('__MISSING__'))
        X_cat[col] = le.transform(X_cat[col].astype(str).fillna('__MISSING__'))
        X_test_cat[col] = le.transform(X_test_cat[col].astype(str).fillna('__MISSING__'))

    X_combined = np.concatenate([X_num, X_cat.values], axis=1)
    X_test_combined = np.concatenate([X_test_num, X_test_cat.values], axis=1)
else:
    X_combined = X_num
    X_test_combined = X_test_num

print(f"Features: {{X_combined.shape[1]}}")

# GPU Detection (CRITICAL - MANDATORY)
import torch
use_gpu = torch.cuda.is_available()
print(f"GPU Available: {use_gpu}")

if use_gpu:
    print("✅ GPU ENABLED for training")
    xgb_gpu_params = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    lgb_gpu_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
else:
    print("⚠️  Running on CPU (slower)")
    xgb_gpu_params = {'tree_method': 'hist'}
    lgb_gpu_params = {'device': 'cpu'}

# Define base learners with competitive hyperparameters
base_learners = [
    ('lgb', lgb.LGBMRegressor(
        n_estimators=2000,
        max_depth=8,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        **lgb_gpu_params
    )),
    ('xgb', xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **xgb_gpu_params
    )),
    ('catboost', CatBoostRegressor(
        iterations=2000,
        depth=7,
        learning_rate=0.03,
        random_state=RANDOM_SEED,
        verbose=False,
        task_type="GPU" if use_gpu else "CPU"
    ))
]

# Meta-learner (Ridge with L2 regularization)
meta_learner = Ridge(alpha=10.0, random_state=RANDOM_SEED)

print("\\nBuilding stacking ensemble...")
stacking_reg = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # 5-fold CV for meta-features
    n_jobs=-1
)

# Train stacking model
print("Training stacking ensemble (this will take 2-3 minutes)...")
stacking_reg.fit(X_combined, y)

# Cross-validation for evaluation
print("\\nEvaluating with 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined), 1):
    X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=Ridge(alpha=10.0, random_state=RANDOM_SEED),
        cv=3,
        n_jobs=-1
    )
    fold_model.fit(X_tr, y_tr)
    preds = fold_model.predict(X_val)

    rmse = mean_squared_error(y_val, preds, squared=False)
    cv_rmses.append(rmse)
    print(f"Fold {{fold}}: RMSE = {{rmse:.6f}}")

print(f"\\nCV RMSE: {{np.mean(cv_rmses):.6f}} (+/- {{np.std(cv_rmses):.6f}})")

# Make predictions on test
print("\\nMaking test predictions...")
predictions = stacking_reg.predict(X_test_combined)

print(f"Prediction distribution:")
print(f"  Min: {{predictions.min():.6f}}")
print(f"  Max: {{predictions.max():.6f}}")
print(f"  Mean: {{predictions.mean():.6f}}")
print(f"  Median: {{np.median(predictions):.6f}}")

# Save submission
# Load sample_submission to get correct column names
sample_sub = pd.read_csv('{submission_path}'.replace('submission.csv', 'sample_submission.csv'))
submission = sample_sub.copy()
target_submission_col = sample_sub.columns[1]
print(f"✓ Using submission column: '{{target_submission_col}}' (from sample_submission)")
submission[target_submission_col] = predictions
assert submission.shape == sample_sub.shape, f"Shape mismatch: {{submission.shape}} vs {{sample_sub.shape}}"
assert submission.columns.tolist() == sample_sub.columns.tolist(), f"Column mismatch: {{submission.columns.tolist()}} vs {{sample_sub.columns.tolist()}}"
print(f"✓ Validation passed: columns={{submission.columns.tolist()}}")
submission.to_csv('{submission_path}', index=False)

elapsed_time = time.time() - start_time
print(f"\\n⏱️  Execution time: {{elapsed_time:.2f}}s")
print(f"✅ Stacking ensemble complete! Submission saved with {{len(submission)}} rows")
""",

    "catboost_model": """
# CatBoost Model Template (for model component_type)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier, Pool
import time

RANDOM_SEED = 42
start_time = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')

# Separate features and target
# TODO: Identify target column
X = train_df.drop('{target_col}', axis=1)
y = train_df['{target_col}']
X_test = test_df.drop('{target_col}', axis=1, errors='ignore')

# Identify categorical features
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical features: {{len(cat_features)}}")

# Check class imbalance
positive_count = (y == 1).sum()
negative_count = (y == 0).sum()
imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
print(f"Class imbalance ratio: {{imbalance_ratio:.2f}}")

# GPU Detection (CRITICAL - MANDATORY)
import torch
use_gpu = torch.cuda.is_available()
print(f"GPU Available: {{use_gpu}}")
task_type = "GPU" if use_gpu else "CPU"
if use_gpu:
    print("✅ GPU ENABLED for CatBoost training")
else:
    print("⚠️  Running on CPU (slower)")

# CatBoost handles categorical features natively (no encoding needed!)
# Create Pool objects for efficient training
train_pool = Pool(
    data=X,
    label=y,
    cat_features=cat_features
)

test_pool = Pool(
    data=X_test,
    cat_features=cat_features
)

# Configure CatBoost
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_SEED,
    verbose=50,  # Print every 50 iterations
    early_stopping_rounds=50,
    auto_class_weights='Balanced' if imbalance_ratio > 2.0 else None,
    task_type=task_type  # GPU or CPU
)

# Train model
print("\\nTraining CatBoost...")
model.fit(train_pool, verbose=True)

# Cross-validation
print("\\nEvaluating with cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=RANDOM_SEED,
        verbose=False,
        task_type=task_type  # GPU or CPU
    )
    fold_model.fit(X_train_fold, y_train_fold, cat_features=cat_features)

    from sklearn.metrics import roc_auc_score
    preds = fold_model.predict_proba(X_val_fold)[:, 1]
    score = roc_auc_score(y_val_fold, preds)
    cv_scores.append(score)
    print(f"Fold {{fold}}: {{score:.4f}}")

print(f"\\nCV Score (ROC-AUC): {{np.mean(cv_scores):.4f}} (+/- {{np.std(cv_scores):.4f}})")

# Make predictions (use predict_proba for probabilities)
print("\\nMaking predictions...")
predictions = model.predict_proba(test_pool)[:, 1]

print(f"Prediction distribution:")
print(f"  Min: {{predictions.min():.4f}}")
print(f"  Max: {{predictions.max():.4f}}")
print(f"  Mean: {{predictions.mean():.4f}}")

# Save submission
# Load sample_submission to get correct column names
sample_sub = pd.read_csv('{submission_path}'.replace('submission.csv', 'sample_submission.csv'))
submission = sample_sub.copy()
target_submission_col = sample_sub.columns[1]
print(f"✓ Using submission column: '{{target_submission_col}}' (from sample_submission)")
submission[target_submission_col] = predictions
assert submission.shape == sample_sub.shape, f"Shape mismatch: {{submission.shape}} vs {{sample_sub.shape}}"
assert submission.columns.tolist() == sample_sub.columns.tolist(), f"Column mismatch: {{submission.columns.tolist()}} vs {{sample_sub.columns.tolist()}}"
print(f"✓ Validation passed: columns={{submission.columns.tolist()}}")
submission.to_csv('{submission_path}', index=False)

elapsed_time = time.time() - start_time
print(f"\\n⏱️  Execution time: {{elapsed_time:.2f}}s")
print(f"✅ CatBoost model complete! Submission saved with {{len(submission)}} rows")
""",

    "advanced_feature_engineering": """
# Advanced Feature Engineering Template (for feature_engineering component_type)
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from category_encoders import TargetEncoder
import time

RANDOM_SEED = 42
start_time = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')

# Identify target column
# TODO: Set target column name
target_col = '{target_col}'
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.copy()

print(f"Original features: {{X_train.shape[1]}}")

# 1. Polynomial Features (degree 2 for numeric columns)
print("\\n1. Creating polynomial features...")
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0 and len(numeric_cols) <= 10:  # Only if manageable
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train[numeric_cols])
    X_test_poly = poly.transform(X_test[numeric_cols])

    poly_feature_names = [f"poly_{{i}}" for i in range(X_train_poly.shape[1])]
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)

    X_train = pd.concat([X_train, X_train_poly_df], axis=1)
    X_test = pd.concat([X_test, X_test_poly_df], axis=1)
    print(f"   Added {{X_train_poly.shape[1]}} polynomial features")

# 2. Feature Interactions (ratio, diff, product)
print("\\n2. Creating feature interactions...")
if len(numeric_cols) >= 2:
    for i, col1 in enumerate(numeric_cols[:5]):  # Limit to avoid explosion
        for col2 in numeric_cols[i+1:6]:
            # Ratio (with safety check for division by zero)
            X_train[f"ratio_{{col1}}_{{col2}}"] = X_train[col1] / (X_train[col2] + 1e-5)
            X_test[f"ratio_{{col1}}_{{col2}}"] = X_test[col1] / (X_test[col2] + 1e-5)

            # Difference
            X_train[f"diff_{{col1}}_{{col2}}"] = X_train[col1] - X_train[col2]
            X_test[f"diff_{{col1}}_{{col2}}"] = X_test[col1] - X_test[col2]

            # Product
            X_train[f"prod_{{col1}}_{{col2}}"] = X_train[col1] * X_train[col2]
            X_test[f"prod_{{col1}}_{{col2}}"] = X_test[col1] * X_test[col2]

# 3. Statistical Transformations
print("\\n3. Creating statistical transformations...")
for col in numeric_cols[:10]:  # Limit to most important
    # Log transform (for positive skewed data)
    if (X_train[col] > 0).all():
        X_train[f"log_{{col}}"] = np.log1p(X_train[col])
        X_test[f"log_{{col}}"] = np.log1p(X_test[col])

    # Square root (for positive skewed data)
    if (X_train[col] >= 0).all():
        X_train[f"sqrt_{{col}}"] = np.sqrt(X_train[col])
        X_test[f"sqrt_{{col}}"] = np.sqrt(X_test[col])

    # Z-score
    mean_val = X_train[col].mean()
    std_val = X_train[col].std()
    if std_val > 0:
        X_train[f"zscore_{{col}}"] = (X_train[col] - mean_val) / std_val
        X_test[f"zscore_{{col}}"] = (X_test[col] - mean_val) / std_val

# 4. Target Encoding (leakage-safe with out-of-fold)
print("\\n4. Creating target encoding for categorical features...")
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
if len(cat_cols) > 0:
    # Use TargetEncoder with smoothing to prevent overfitting
    encoder = TargetEncoder(smoothing=1.0, min_samples_leaf=20)

    # Fit on train data only
    X_train_encoded = encoder.fit_transform(X_train[cat_cols], y_train)
    X_test_encoded = encoder.transform(X_test[cat_cols])

    # Add encoded features with prefix
    for i, col in enumerate(cat_cols):
        X_train[f"target_enc_{{col}}"] = X_train_encoded.iloc[:, i]
        X_test[f"target_enc_{{col}}"] = X_test_encoded.iloc[:, i]

print(f"\\nFinal features: {{X_train.shape[1]}} (added {{X_train.shape[1] - len(train_df.columns) + 1}} features)")

# Save engineered data for later use by models
print("\\nSaving engineered features...")
X_train['{{target_col}}'] = y_train
X_train.to_csv('{models_dir}/train_engineered.csv', index=False)
X_test.to_csv('{models_dir}/test_engineered.csv', index=False)

elapsed_time = time.time() - start_time
print(f"\\n⏱️  Execution time: {{elapsed_time:.2f}}s")
print("✅ Advanced feature engineering complete!")
""",
}


def get_domain_template(domain: str, **kwargs) -> str:
    """
    Get domain-specific code template.

    Args:
        domain: Domain type
        **kwargs: Template variables

    Returns:
        Formatted code template
    """
    template = DOMAIN_CODE_TEMPLATES.get(domain, DOMAIN_CODE_TEMPLATES["tabular"])
    return template.format(**kwargs)


def format_component_details(component) -> str:
    """
    Format component details for prompts.

    Args:
        component: AblationComponent object

    Returns:
        Formatted string
    """
    return f"""
Name: {component.name}
Type: {component.component_type}
Estimated Impact: {component.estimated_impact:.1%}
Description: {component.code}
"""


def format_error_info(error: str) -> Dict[str, str]:
    """
    Categorize and format error information.

    Args:
        error: Error message

    Returns:
        Dictionary with error_type and formatted error
    """
    error_types = {
        "ModuleNotFoundError": "missing_import",
        "FileNotFoundError": "missing_file",
        "KeyError": "missing_key",
        "ValueError": "value_error",
        "TypeError": "type_error",
        "SyntaxError": "syntax_error",
        "MemoryError": "memory_error",
        "Timeout": "timeout",
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
