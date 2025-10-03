# fill_missing_values

## Name
fill_missing_values

## Description
Fill missing values in a DataFrame using various strategies including statistical methods (mean, median, mode) and propagation methods (forward fill, backward fill).

## Applicable Situations
- When dealing with datasets that have missing values (NaN, None)
- Before training machine learning models that don't handle missing data
- During data cleaning phase
- When missing data is less than 30% of total data

## Parameters

### df
- **Type**: `pandas.DataFrame`
- **Description**: Input DataFrame with missing values
- **Required**: Yes

### strategy
- **Type**: `str`
- **Description**: Strategy to fill missing values
- **Enum**: `mean` | `median` | `mode` | `ffill` | `bfill` | `constant`
- **Default**: `mean`
- **Required**: No

### columns
- **Type**: `list` | `None`
- **Description**: Specific columns to fill. If None, applies to all columns with missing values
- **Default**: `None`
- **Required**: No

### fill_value
- **Type**: `Any`
- **Description**: Value to use when strategy is 'constant'
- **Default**: `0`
- **Required**: No (only when strategy='constant')

## Result
Returns a pandas DataFrame with missing values filled according to the specified strategy.

## Notes
- `mean` and `median` strategies only work with numeric columns
- `mode` strategy works with both numeric and categorical columns
- `ffill` (forward fill) uses previous valid value to fill gaps
- `bfill` (backward fill) uses next valid value to fill gaps
- For time series data, prefer `ffill` or `bfill`
- Always check the percentage of missing values before deciding strategy
- Consider creating a missing value indicator column for important features

## Example

### Input
```python
df = pd.DataFrame({
    'age': [25, np.nan, 30, np.nan, 35],
    'salary': [50000, 60000, np.nan, 70000, 80000],
    'category': ['A', 'B', np.nan, 'A', 'B']
})

# Fill numeric columns with mean
df_filled = fill_missing_values(df, strategy='mean', columns=['age', 'salary'])

# Fill categorical columns with mode
df_filled = fill_missing_values(df_filled, strategy='mode', columns=['category'])
```

### Output
```python
# Result DataFrame:
   age   salary category
0  25.0  50000.0        A
1  30.0  60000.0        B
2  30.0  65000.0        A
3  30.0  70000.0        A
4  35.0  80000.0        B
```

## Implementation Reference
```python
def fill_missing_values(df, strategy='mean', columns=None, fill_value=0):
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.columns[df_copy.isnull().any()].tolist()

    for col in columns:
        if strategy == 'mean':
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'ffill':
            df_copy[col].fillna(method='ffill', inplace=True)
        elif strategy == 'bfill':
            df_copy[col].fillna(method='bfill', inplace=True)
        elif strategy == 'constant':
            df_copy[col].fillna(fill_value, inplace=True)

    return df_copy
```
