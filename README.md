# Kaggle Agents

Multi-agent framework for autonomous Kaggle competition participation using LangGraph. Inspired by the AutoKaggle paper, this system employs specialized agents to handle the complete competition pipeline from data acquisition through submission.

## Architecture

The system implements a directed graph workflow where specialized agents handle distinct phases of the competition lifecycle:

```
Data Collection → EDA → Feature Engineering → Model Training → Submission → Leaderboard Monitor
                           ↑                                                       ↓
                           └───────────────────────────────────────────────────────┘
```

### Core Components

**State Management**: Uses dataclass-based state with custom reducers for efficient updates
- List fields use `operator.add` for appending
- Dict fields use custom merge reducer for updates
- Extends `MessagesState` for LLM conversation tracking

**Agent System**: Six specialized agents implement the competition pipeline
- **Data Collector**: Kaggle API integration for dataset acquisition
- **EDA Agent**: Statistical analysis and insight generation
- **Feature Engineer**: Automated feature creation and preprocessing
- **Model Trainer**: Multi-model training with cross-validation (RF, XGBoost, LightGBM)
- **Submission Agent**: Prediction generation and Kaggle submission
- **Leaderboard Monitor**: Performance tracking and iteration control

**Workflow Control**: LangGraph orchestration with conditional routing
- Linear progression through primary pipeline
- Conditional iteration based on leaderboard percentile
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

### Model Training

Automated training pipeline with cross-validation:
- Automatic problem type detection (classification/regression)
- Parallel model training: Random Forest, XGBoost, LightGBM, Linear models
- 5-fold cross-validation for robust evaluation
- Feature importance extraction
- Best model selection based on CV scores

### Feature Engineering

Automated feature generation pipeline:
- Missing value imputation (median for numeric, mode for categorical)
- Categorical encoding using LabelEncoder
- Interaction feature creation
- Aggregation features (sum, mean, std)
- LLM-guided feature strategy generation

## Project Structure

```
kaggle_agents/
├── agents/
│   ├── data_collector.py      # Kaggle API integration
│   ├── eda_agent.py            # Statistical analysis
│   ├── feature_engineer.py    # Feature generation
│   ├── model_trainer.py        # Multi-model training
│   ├── submission_agent.py     # Prediction and submission
│   └── leaderboard_monitor.py # Performance tracking
├── workflows/
│   └── kaggle_workflow.py      # LangGraph orchestration
├── tools/
│   └── kaggle_api.py           # Kaggle API client
├── utils/
│   ├── config.py               # Configuration management
│   └── state.py                # State schema and reducers
└── main.py                     # CLI entrypoint
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
- **kaggle**: Official Kaggle API client
- **scikit-learn**: Classical ML models and preprocessing
- **xgboost**: Gradient boosting implementation
- **lightgbm**: Efficient gradient boosting
- **pandas/numpy**: Data manipulation

## References

Based on the AutoKaggle paper: "AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions"
