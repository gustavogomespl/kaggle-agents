"""
Prompt templates for the Developer Agent.

These templates guide code generation for implementing
ablation components with retry and debug capabilities.
"""

from typing import Dict

# Base system prompt for the developer
DEVELOPER_SYSTEM_PROMPT = """You are an expert Python developer specializing in Machine Learning and Kaggle competitions.

Your role is to write PRODUCTION-READY code that implements machine learning components.

You follow these principles:
1. **Clean Code**: Well-structured, readable, documented
2. **Error Handling**: Proper try-except blocks, informative errors
3. **Reproducibility**: Set random seeds, save artifacts
4. **Best Practices**: Follow scikit-learn/pandas conventions
5. **Efficiency**: Vectorized operations, memory-conscious

Your code should:
- Import all necessary libraries
- Load data from correct paths
- Implement the specified component
- Save outputs/models to correct locations
- Handle edge cases and errors gracefully
- Print progress and results
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
- **MUST train a simple, fast model** (LightGBM, XGBoost, or RandomForest recommended)
- **MUST make predictions** on test data
- **MUST create submission.csv** at {submission_path} with predictions
- Keep model simple (max_depth=5, n_estimators=100-200) to avoid timeout
- Target execution time: 30-60 seconds maximum
- Print CV score or validation metrics

### If component_type == "ensemble":
- Combine predictions from multiple models
- Must create submission.csv with ensemble predictions

## General Requirements
1. Load data from the provided paths
2. Implement the component exactly as specified above
3. Print progress and key metrics
4. Handle errors gracefully

## Code Structure
```python
# Imports
import pandas as pd
import numpy as np
# ... other imports

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load data
print("Loading data...")
train_df = pd.read_csv('{train_data_path}')
test_df = pd.read_csv('{test_data_path}')

# Implement component
print("Implementing {component_name}...")
# ... your code here

# Save outputs
print("Saving outputs...")
# ... save models, predictions, etc.

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
