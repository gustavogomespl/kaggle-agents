## Running Tests

This directory contains unit and integration tests for the Kaggle Agents framework.

### Installation

Install development dependencies:

```bash
uv sync --extra dev
```

Or install manually:

```bash
uv pip install pytest pytest-cov pytest-asyncio
```

### Running Tests

Run all tests:

```bash
uv run pytest
```

Run with coverage report:

```bash
uv run pytest --cov=kaggle_agents --cov-report=html --cov-report=term
```

Run specific test file:

```bash
uv run pytest tests/test_cross_validation.py
```

Run specific test:

```bash
uv run pytest tests/test_cross_validation.py::TestAdaptiveCrossValidator::test_get_stratified_kfold
```

Run with verbose output:

```bash
uv run pytest -v
```

### Test Structure

- `conftest.py`: Pytest configuration and shared fixtures
- `test_cross_validation.py`: Tests for adaptive CV strategies
- `test_feature_engineering.py`: Tests for feature engineering utilities
- `test_state.py`: Tests for state management and reducers

### Coverage

After running tests with coverage, open the HTML report:

```bash
open htmlcov/index.html
```

### Writing New Tests

Follow these guidelines:

1. **Test naming**: Use descriptive names starting with `test_`
2. **Fixtures**: Use pytest fixtures for common test data
3. **Assertions**: Use specific assertions (e.g., `assert x == y` instead of `assert x`)
4. **Mocking**: Mock external services (Kaggle API, LLMs) in tests
5. **Isolation**: Each test should be independent

Example test:

```python
def test_example_feature(engineer, sample_data):
    \"\"\"Test example feature creation.\"\"\"
    train, test = sample_data
    result = engineer.create_feature(train, test)

    assert "new_feature" in result.columns
    assert result["new_feature"].dtype == np.float64
```

### Integration Tests

Integration tests validate the complete workflow. They require:

- Mock LLM responses
- Mock Kaggle API calls
- Sample datasets

Run integration tests separately:

```bash
uv run pytest tests/integration/ -v
```

### CI/CD

Tests are automatically run in CI on:

- Pull requests
- Pushes to main branch
- Release tags

Ensure all tests pass before merging.
