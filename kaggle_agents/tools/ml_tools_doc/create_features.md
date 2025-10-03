# create_features

## Name
create_features

## Description
Create engineered features from existing columns using various techniques including polynomial features, interactions, aggregations, and domain-specific transformations.

## Applicable Situations
- Feature Engineering phase
- When you want to create new predictive features
- Combining existing features to capture relationships
- Extracting information from datetime, text, or categorical columns

## Parameters

### df
- **Type**: `pandas.DataFrame`
- **Description**: Input DataFrame with base features
- **Required**: Yes

### feature_type
- **Type**: `str`
- **Description**: Type of features to create
- **Enum**: `polynomial` | `interaction` | `aggregation` | `datetime` | `text` | `binning`
- **Required**: Yes

### columns
- **Type**: `list`
- **Description**: Columns to use for feature creation
- **Required**: Yes

### degree
- **Type**: `int`
- **Description**: Degree for polynomial features
- **Default**: `2`
- **Required**: No (only for polynomial type)

### operations
- **Type**: `list`
- **Description**: Operations for aggregation (`sum`, `mean`, `max`, `min`, `std`)
- **Default**: `['mean', 'sum']`
- **Required**: No (only for aggregation type)

### bins
- **Type**: `int` | `list`
- **Description**: Number of bins or bin edges for binning
- **Default**: `10`
- **Required**: No (only for binning type)

## Result
Returns a pandas DataFrame with original columns plus newly created features.

## Notes
- **Polynomial features**: Create x², x³, etc. Use carefully to avoid overfitting
- **Interaction features**: Create x₁×x₂, x₁×x₃, etc. Captures relationships between features
- **Aggregation**: Group by categorical columns and aggregate numeric columns
- **Datetime features**: Extract year, month, day, hour, dayofweek, is_weekend, etc.
- **Text features**: Extract length, word count, special character count
- **Binning**: Convert continuous variables into categorical bins
- Always check feature correlation to avoid redundancy
- Consider computational cost for high-degree polynomials

## Example

### Input
```python
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000],
    'date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05'],
    'category': ['A', 'B', 'A', 'B']
})

# Create polynomial features
df_poly = create_features(
    df,
    feature_type='polynomial',
    columns=['age', 'income'],
    degree=2
)

# Create interaction features
df_inter = create_features(
    df,
    feature_type='interaction',
    columns=['age', 'income']
)

# Extract datetime features
df['date'] = pd.to_datetime(df['date'])
df_datetime = create_features(
    df,
    feature_type='datetime',
    columns=['date']
)
```

### Output
```python
# Polynomial features added:
# age_squared, age_cubed, income_squared, income_cubed

# Interaction features added:
# age_income_interaction

# Datetime features added:
# date_year, date_month, date_day, date_dayofweek, date_is_weekend
```

## Implementation Reference
```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def create_features(df, feature_type, columns, degree=2, operations=None, bins=10):
    df_new = df.copy()

    if feature_type == 'polynomial':
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)

        for i, name in enumerate(feature_names):
            if name not in columns:  # Skip original features
                df_new[name] = poly_features[:, i]

    elif feature_type == 'interaction':
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                df_new[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]

    elif feature_type == 'datetime':
        for col in columns:
            df_new[f'{col}_year'] = df[col].dt.year
            df_new[f'{col}_month'] = df[col].dt.month
            df_new[f'{col}_day'] = df[col].dt.day
            df_new[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df_new[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            df_new[f'{col}_quarter'] = df[col].dt.quarter

    elif feature_type == 'binning':
        for col in columns:
            df_new[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=False)

    elif feature_type == 'aggregation':
        if operations is None:
            operations = ['mean', 'sum']
        # Requires groupby column (assume first column is categorical)
        # Implementation depends on specific use case

    return df_new
```
