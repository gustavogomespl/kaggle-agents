# train_model

## Name
train_model

## Description
Train a machine learning model with cross-validation and return trained model with performance metrics. Supports multiple model types including tree-based models, linear models, and ensemble methods.

## Applicable Situations
- Model Building, Validation, and Prediction phase
- When you have preprocessed features ready for training
- Need to train and evaluate model performance
- Want to compare different model types

## Parameters

### X_train
- **Type**: `pandas.DataFrame` | `numpy.ndarray`
- **Description**: Training features
- **Required**: Yes

### y_train
- **Type**: `pandas.Series` | `numpy.ndarray`
- **Description**: Training target variable
- **Required**: Yes

### model_type
- **Type**: `str`
- **Description**: Type of model to train
- **Enum**: `xgboost` | `lightgbm` | `catboost` | `random_forest` | `logistic_regression` | `linear_regression`
- **Required**: Yes

### params
- **Type**: `dict` | `None`
- **Description**: Model-specific hyperparameters
- **Default**: `None` (uses sensible defaults)
- **Required**: No

### cv_folds
- **Type**: `int`
- **Description**: Number of cross-validation folds
- **Default**: `5`
- **Required**: No

### random_state
- **Type**: `int`
- **Description**: Random seed for reproducibility
- **Default**: `42`
- **Required**: No

## Result
Returns a dictionary containing:
- `model`: Trained model object
- `cv_scores`: List of cross-validation scores
- `mean_cv_score`: Average cross-validation score
- `std_cv_score`: Standard deviation of CV scores
- `feature_importance`: Dictionary of feature importances (if available)

## Notes
- Always use cross-validation to avoid overfitting
- XGBoost, LightGBM, and CatBoost typically perform well on tabular data
- Random Forest is a good baseline model
- For classification, ensure target variable is properly encoded
- For regression, consider scaling features for linear models
- CatBoost can handle categorical features natively
- Monitor CV scores to detect overfitting (high train score, low CV score)

## Example

### Input
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load processed data
X = pd.read_csv('processed_train.csv')
y = X.pop('target')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
result = train_model(
    X_train=X_train,
    y_train=y_train,
    model_type='xgboost',
    params={
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
    },
    cv_folds=5
)

print(f"Mean CV Score: {result['mean_cv_score']:.4f}")
print(f"Std CV Score: {result['std_cv_score']:.4f}")
```

### Output
```python
# Result:
{
    'model': <xgboost.XGBClassifier object>,
    'cv_scores': [0.85, 0.87, 0.86, 0.84, 0.88],
    'mean_cv_score': 0.8600,
    'std_cv_score': 0.0141,
    'feature_importance': {
        'feature1': 0.25,
        'feature2': 0.18,
        'feature3': 0.15,
        ...
    }
}
```

## Implementation Reference
```python
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_model(X_train, y_train, model_type, params=None, cv_folds=5, random_state=42):
    # Default parameters
    default_params = {
        'xgboost': {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
        'lightgbm': {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100},
        'catboost': {'depth': 6, 'learning_rate': 0.1, 'iterations': 100},
        'random_forest': {'max_depth': 10, 'n_estimators': 100}
    }

    if params is None:
        params = default_params.get(model_type, {})

    params['random_state'] = random_state

    # Select model
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**params)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**params)
    # ... other model types

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    # Train final model
    model.fit(X_train, y_train)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    else:
        feature_importance = {}

    return {
        'model': model,
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'feature_importance': feature_importance
    }
```
