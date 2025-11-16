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

CRITICAL RULES (Never Break):
- ALWAYS use predict_proba() for probability predictions (NOT predict())
- ALWAYS check class distribution and apply class weights if imbalanced (ratio > 2:1)
- ALWAYS use StratifiedKFold(n_splits=5) for cross-validation
- ALWAYS print CV scores, class distribution, prediction distribution
- NEVER use try-except to hide errors (let them surface for debugging)
- NEVER subsample training data (use all available data)
- NEVER use sys.exit() or similar termination commands
- ALWAYS save submission.csv with probabilities (0.0-1.0), NOT binary predictions (0/1)

Your code should:
- Import all necessary libraries
- Load data from correct paths
- Check for class imbalance and handle appropriately
- Implement the specified component with best practices
- Use cross-validation to estimate performance
- Make probability predictions (not hard predictions)
- Save outputs/models to correct locations
- Print execution time and key metrics
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

## CRITICAL TYPE-SPECIFIC REQUIREMENTS

### If component_type == "preprocessing" or "feature_engineering":
- **DO NOT train any models** (no fit(), no GridSearch, no model training)
- ONLY clean data, handle missing values, scale features, or create new features
- Must execute in **under 10 seconds** (keep it simple and fast)
- Can save processed data to models directory for later use

### If component_type == "model":
- **MUST train a model** (LightGBM, XGBoost, or CatBoost recommended for best performance)
- **For classification**: use **predict_proba()** to get probabilities (NOT predict() for hard predictions)
- **For regression**: use **predict()** to get continuous values
- **MUST create submission.csv** at {submission_path} with predictions
- Use competitive hyperparameters:
  - **n_estimators**: 1500-2500 (with early_stopping for efficiency)
  - **max_depth**: 6-9 (deeper for complex patterns, shallower for overfitting prevention)
  - **learning_rate**: 0.02-0.05 (lower = more trees, better generalization)
  - **num_leaves** (LightGBM): 31-127
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

### If component_type == "ensemble":
- Combine predictions from multiple models
- Must create submission.csv with ensemble predictions

## General Requirements
1. Load data from the provided paths
2. Implement the component exactly as specified above
3. Print progress and key metrics
4. Handle errors gracefully

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

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

start_time = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')

# Check class distribution (for model components)
print(f"Class distribution: {{train_df[target_col].value_counts().to_dict()}}")

# Implement component
print("Implementing {component_name}...")
# ... your code here

# For model components: Calculate class weights if needed
positive_count = (y_train == 1).sum()
negative_count = (y_train == 0).sum()
imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
print(f"Class imbalance ratio: {{imbalance_ratio:.2f}}")

if imbalance_ratio > 2.0:
    print("⚠️  Class imbalance detected - applying weights")
    # For XGBoost:
    scale_pos_weight = negative_count / positive_count
    # model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

    # For LightGBM:
    # model = LGBMClassifier(is_unbalance=True, random_state=42)

    # For sklearn:
    # model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Cross-validation with StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
print(f"CV Score: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std():.4f}})")

# Make predictions (MUST use predict_proba for probabilities)
predictions = model.predict_proba(X_test)[:, 1]  # Binary classification
print(f"Prediction distribution: min={{predictions.min():.4f}}, max={{predictions.max():.4f}}, mean={{predictions.mean():.4f}}")

# Save outputs
print("Saving outputs...")
submission = pd.DataFrame({{'id': test_df['id'], 'prediction': predictions}})
submission.to_csv('{submission_path}', index=False)
print(f"Submission saved: {{len(submission)}} rows")

elapsed_time = time.time() - start_time
print(f"⏱️  Execution time: {{elapsed_time:.2f}}s")
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
model = xgb.XGBClassifier(random_state=42, n_estimators=100)
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

# Define base learners (diverse models)
base_learners = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.03,
        is_unbalance=(imbalance_ratio > 2.0),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )),
    ('catboost', CatBoostClassifier(
        iterations=2000,
        depth=7,
        learning_rate=0.03,
        random_state=RANDOM_SEED,
        verbose=False
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
submission = pd.DataFrame({{'id': test_df['id'], 'prediction': predictions}})
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
        verbose=-1
    )),
    ('xgb', xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )),
    ('catboost', CatBoostRegressor(
        iterations=2000,
        depth=7,
        learning_rate=0.03,
        random_state=RANDOM_SEED,
        verbose=False
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
submission = pd.DataFrame({{'id': test_ids, target_col: predictions}})
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
    auto_class_weights='Balanced' if imbalance_ratio > 2.0 else None
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
        verbose=False
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
submission = pd.DataFrame({{'id': test_df['id'], 'prediction': predictions}})
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
