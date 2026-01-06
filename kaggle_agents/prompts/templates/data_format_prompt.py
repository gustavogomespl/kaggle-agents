"""
Data Format Discovery Prompt.

LLM prompt for analyzing non-standard Kaggle competition data formats
and generating parsing instructions.
"""

DATA_FORMAT_DISCOVERY_PROMPT = """You are an expert data scientist analyzing a Kaggle competition's data format.
Your task is to understand how the data is structured and generate parsing instructions.

## Competition: {competition}

## Competition Description:
{description}

## Content from Data Page:
{data_page_content}

## Files Found in Data Directory:
{file_listing}

## How SOTA Notebooks Load This Data:
{sota_loading_code}

## Your Task:

Analyze all the information above and generate a JSON object with parsing instructions.
The JSON must include:

1. **format_type**: The primary data format - one of: "csv", "txt", "json", "parquet", "custom"

2. **id_column**: The name of the ID/identifier column (e.g., "id", "rec_id", "image_id")

3. **target_column**: The name of the target/label column (e.g., "target", "label", "species")

4. **train_file**: Relative path to the file containing training labels/data

5. **test_file**: Relative path to the file containing test data (or directory)

6. **train_test_split_method**: How train/test split is defined:
   - "file_based": Separate files for train and test
   - "column_based": A column indicates train (0) vs test (1)
   - "directory_based": Separate directories for train and test data

7. **loading_code**: Python code that:
   - Reads all necessary files
   - Creates `train_df` DataFrame with columns: [id_column, target_column, ...]
   - Creates `test_df` DataFrame with columns: [id_column, ...]
   - Handles any special parsing (multi-label, custom delimiters, etc.)

   The code should use these variables:
   - `working_dir`: Path object pointing to the data directory
   - `pd`: pandas module
   - `Path`: pathlib.Path class

8. **column_mapping**: Dictionary mapping standard names to actual column names:
   {{
     "id": "<actual_id_column_name>",
     "target": "<actual_target_column_name>",
     "filename": "<column_with_filenames_if_exists>"
   }}

9. **can_generate_csv**: Boolean - True if we can generate standard train.csv/test.csv files,
   False if the format is too complex and loading_code should be passed to the developer agent.

10. **multi_label**: Boolean - True if this is a multi-label classification problem
    (one sample can have multiple labels)

11. **notes**: Any important notes about the data format that the developer should know

## Important Guidelines:

- If files use space or tab delimiters instead of commas, specify this in loading_code
- If labels are in format "id,label1,label2,..." (multi-label), handle this properly
- If there's a separate file mapping IDs to filenames, include that in loading_code
- Look at SOTA notebooks for hints on how to parse the data correctly
- Be specific about file paths - use exact filenames from the file listing

## Response Format:

Return ONLY a valid JSON object (no markdown, no explanation):

```json
{{
  "format_type": "...",
  "id_column": "...",
  "target_column": "...",
  "train_file": "...",
  "test_file": "...",
  "train_test_split_method": "...",
  "loading_code": "...",
  "column_mapping": {{}},
  "can_generate_csv": true/false,
  "multi_label": true/false,
  "notes": "..."
}}
```
"""


DATA_FORMAT_REFINEMENT_PROMPT = """The previous parsing attempt failed with error:
{error}

Please fix the loading_code to handle this error.

Previous loading_code:
```python
{previous_code}
```

File listing for reference:
{file_listing}

Return the corrected JSON with fixed loading_code.
"""
