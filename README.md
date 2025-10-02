# Kaggle Agents

Multi-agent framework for autonomous Kaggle competition participation using LangGraph. Inspired by the AutoKaggle paper, this system employs specialized agents to handle the complete competition pipeline from data acquisition through submission.

## Architecture

The system implements a directed graph workflow with intelligent decision-making at each phase:

```
Data Collection → EDA → Strategy → Feature Engineering → Model Training → Ensemble → Submission → Leaderboard
                                      ↑                                                              ↓
                                      └──────────────────────────────────────────────────────────────┘
```

### Core Components

**State Management**: Dataclass-based state with custom reducers
- List fields use `operator.add` for appending
- Dict fields use custom merge reducer for updates
- Extends `MessagesState` for LLM conversation tracking

**Agent System**: Eight specialized agents with advanced ML capabilities
- **Data Collector**: Kaggle API integration for dataset acquisition
- **EDA Agent**: Statistical analysis and insight generation
- **Strategy Agent**: High-level decision making based on data characteristics
  - Selects optimal models for the problem
  - Determines feature engineering priorities
  - Chooses encoding and validation strategies
- **Feature Engineer**: Advanced feature creation
  - Adaptive categorical encoding (Target, CatBoost, OneHot, Label)
  - Polynomial feature generation
  - Date feature extraction
  - Missing value indicators
  - Feature scaling when required
- **Model Trainer**: Multi-model training with hyperparameter optimization
  - Optuna-based tuning for XGBoost, LightGBM, CatBoost, Random Forest
  - Adaptive cross-validation (StratifiedKFold, TimeSeriesSplit, etc.)
- **Ensemble Agent**: Model combination for improved performance
  - Stacking with meta-learner
  - Blending with weighted averaging
- **Submission Agent**: Prediction generation and Kaggle submission
- **Leaderboard Monitor**: Performance tracking and intelligent iteration

**Workflow Control**: LangGraph orchestration with adaptive routing
- Strategy-driven pipeline execution
- Conditional iteration based on performance analysis
- Terminates at top 20% placement or max iterations

## Installation

Requires Python 3.10+ and uv package manager.

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Required Configuration

```bash
OPENAI_API_KEY=sk-...  # Required for LLM operations
```

### Optional Configuration

```bash
# Kaggle API (enables automatic submission)
KAGGLE_USERNAME=username
KAGGLE_KEY=api_key

# LangSmith (enables tracing)
LANGSMITH_API_KEY=ls_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=kaggle-agents

# Model parameters
LLM_MODEL=gpt-4-turbo-preview
TEMPERATURE=0.0
MAX_ITERATIONS=5
```

## Usage

### Basic Execution

```bash
uv run python -m kaggle_agents.main <competition-name>
```

Example:
```bash
uv run python -m kaggle_agents.main titanic
```

### Advanced Options

```bash
# Custom iteration limit
uv run python -m kaggle_agents.main titanic --max-iterations 10

# Workflow visualization
uv run python -m kaggle_agents.main titanic --visualize
```

## Implementation Details

### State Schema

```python
@dataclass
class KaggleState(MessagesState):
    competition_name: str
    train_data_path: str
    test_data_path: str

    # Uses custom reducers for efficient updates
    eda_summary: Annotated[Dict[str, Any], merge_dict]
    data_insights: Annotated[List[str], add]
    features_engineered: Annotated[List[str], add]
    models_trained: Annotated[List[Dict[str, Any]], add]

    iteration: int
    max_iterations: int
```

### Workflow Graph

The workflow uses LangGraph's `StateGraph` with:
- **Nodes**: Python functions implementing agent logic
- **Edges**: Define execution flow between agents
- **Conditional edges**: Route based on performance metrics
- **Checkpointer**: Optional persistence for state recovery

### Advanced Model Training

State-of-the-art model optimization pipeline:
- **Hyperparameter Optimization**: Optuna-based tuning with TPE sampler
  - XGBoost: learning rate, max depth, subsample, regularization
  - LightGBM: num_leaves, learning rate, feature fraction
  - CatBoost: depth, learning rate, l2 regularization
  - Random Forest: n_estimators, max depth, min samples
- **Adaptive Cross-Validation**: Strategy selection based on data type
  - StratifiedKFold for imbalanced classification
  - TimeSeriesSplit for temporal data
  - GroupKFold for grouped data
- **Ensemble Methods**:
  - Stacking: Out-of-fold predictions with meta-learner
  - Blending: Weighted averaging of top models
- Automatic problem type detection
- Feature importance extraction
- Model persistence and versioning

### Intelligent Feature Engineering

Strategy-driven feature creation:
- **Missing Value Handling**:
  - Missing value indicators
  - Context-aware imputation
- **Adaptive Categorical Encoding**:
  - Low cardinality: OneHot or Label encoding
  - High cardinality: Target or CatBoost encoding
- **Advanced Feature Creation**:
  - Polynomial features for non-linear relationships
  - Date decomposition (year, month, day, dayofweek, quarter)
  - Interaction features between numeric columns
  - Aggregation features (sum, mean, std, min, max)
- **Feature Scaling**: StandardScaler or MinMaxScaler when required
- LLM-guided strategy formulation

## Project Structure

```
kaggle_agents/
├── agents/
│   ├── data_collector.py       # Kaggle API integration
│   ├── eda_agent.py             # Statistical analysis
│   ├── strategy_agent.py        # High-level decision making
│   ├── feature_engineer.py      # Advanced feature engineering
│   ├── model_trainer.py         # Model training with optimization
│   ├── ensemble_agent.py        # Model ensembling
│   ├── submission_agent.py      # Prediction and submission
│   └── leaderboard_monitor.py   # Performance tracking
├── workflows/
│   └── kaggle_workflow.py       # LangGraph orchestration
├── tools/
│   └── kaggle_api.py            # Kaggle API client
├── utils/
│   ├── config.py                # Configuration management
│   ├── state.py                 # State schema and reducers
│   ├── feature_engineering.py   # Advanced FE utilities
│   └── hyperparameter_tuning.py # Optuna optimization
└── main.py                      # CLI entrypoint
```

## Monitoring and Debugging

### LangSmith Integration

When configured, all workflow executions are traced in LangSmith:
- Agent decision points
- LLM calls and responses
- State transitions
- Error tracking

Access traces at: https://smith.langchain.com

### Checkpointing

Enable persistence for workflow recovery:

```python
from langgraph.checkpoint.memory import MemorySaver

workflow = create_kaggle_workflow(checkpointer=MemorySaver())
```

## Limitations

- Optimized for tabular datasets (CSV format)
- Requires OpenAI API access
- Basic feature engineering (no deep learning)
- Limited to standard Kaggle competition formats

## Technical Stack

- **langgraph** (0.6.8): Agent orchestration framework
- **langchain**: LLM interface and abstractions
- **optuna**: Hyperparameter optimization with TPE sampler
- **kaggle**: Official Kaggle API client
- **scikit-learn**: Classical ML models and preprocessing
- **xgboost**: Gradient boosting implementation
- **lightgbm**: Efficient gradient boosting
- **catboost**: Gradient boosting with categorical support
- **category-encoders**: Advanced categorical encoding
- **pandas/numpy**: Data manipulation

## Key Differentiators

This system achieves competitive performance through:

1. **Strategy-First Approach**: LLM-powered analysis determines optimal approach before execution
2. **Adaptive Feature Engineering**: Encoding and transformation strategies adapt to data characteristics
3. **Hyperparameter Optimization**: Automated tuning with Optuna for all major models
4. **Ensemble Methods**: Stacking and blending for robust predictions
5. **Intelligent Iteration**: Performance-based decisions with overfitting detection
   - Compares CV scores with public leaderboard scores
   - Detects overfitting severity (moderate/severe)
   - Automatically adjusts strategy based on diagnostics
6. **Adaptive Cross-Validation**: CV strategy selection based on data type
   - StratifiedKFold for imbalanced classification
   - TimeSeriesSplit for temporal data
   - GroupKFold for grouped data
7. **Production-Ready**: Full error handling, logging, checkpointing, and tracing
8. **Test Coverage**: Comprehensive unit and integration tests

## Testing

Run the test suite:

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=kaggle_agents --cov-report=html

# Run specific tests
uv run pytest tests/test_cross_validation.py -v
```

See `tests/README.md` for detailed testing documentation.

## References

Based on the AutoKaggle paper: "AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions"
