<img width="2676" height="854" alt="image" src="https://github.com/user-attachments/assets/398dcf92-804c-441d-9399-8957d31ca445" />

# Kaggle Agents

Autonomous agent system for Kaggle competitions using LangGraph, DSPy, and concepts from Google's MLE Star framework (https://www.arxiv.org/abs/2506.15692).

## Overview

Kaggle Agents is a multi-agent system that autonomously participates in Kaggle competitions. The system implements a search-first strategy combined with ablation-driven optimization to achieve competitive results with minimal human intervention.

### Core Capabilities

- Autonomous competition solving from problem analysis to submission
- Search-first strategy leveraging SOTA solutions from Kaggle notebooks
- Ablation-driven planning and systematic component testing
- Iterative improvement with automatic leaderboard monitoring
- Multi-domain support: tabular, computer vision, NLP, time series
- Self-improving prompts via DSPy optimization

### Basic Usage

```python
from kaggle_agents.core import solve_competition

results = solve_competition(
    competition_name="titanic",
    competition_description="Predict survival on the Titanic",
    problem_type="binary_classification",
    evaluation_metric="accuracy",
    max_iterations=5,
)
```

The system automatically:
1. Detects the competition domain
2. Searches for SOTA solutions
3. Creates an ablation plan
4. Implements and tests components
5. Validates code quality
6. Submits to Kaggle
7. Monitors leaderboard
8. Iterates until target percentile is achieved

## Architecture

### Pipeline

```
Domain Detection -> Search Agent -> Planner Agent -> Developer Agent ->
Robustness Agent -> Submission Agent -> Iteration Control
                         ^                                      |
                         |______________________________________|
```

The workflow uses LangGraph's StateGraph with conditional routing:
- Components are implemented sequentially
- After all components, code is validated
- Submission is uploaded to Kaggle
- If target percentile not achieved, the cycle repeats with new SOTA search

### Agents

**1. Domain Detection**

Identifies the competition type based on description and available data. Supports tabular, computer vision, NLP, and time series domains.

**2. Search Agent**

Implements search-first strategy:
- Generates diverse search queries
- Retrieves top Kaggle notebooks via API
- Analyzes SOTA solutions
- Extracts common patterns and techniques

**3. Planner Agent**

Creates systematic ablation plan:
- Analyzes SOTA patterns
- Identifies reusable components
- Estimates component impact
- Prioritizes implementation order

**4. Developer Agent**

Generates and executes code:
- LLM-based code generation
- Sandbox execution with subprocess isolation
- Automatic debugging (10 iterations)
- Retry mechanism (5 attempts)
- Timeout protection (10 minutes)

**5. Robustness Agent**

Validates code quality using 4 modules:

1. Debugging: Checks for exceptions and warnings
2. Data Leakage: Detects target leakage and improper splits
3. Data Usage: Ensures proper data utilization
4. Format Compliance: Validates submission format

Threshold: 70% overall score to pass

**6. Submission Agent**

Handles Kaggle integration:
- Uploads submission via Kaggle API
- Fetches public score
- Calculates leaderboard percentile
- Detects goal achievement (default: top 20%)

**7. Iteration Control**

Manages the workflow loop:
- Checks termination conditions
- Tracks iteration count
- Maintains iteration memory
- Decides whether to continue or end

## Installation

### Local Installation

Requirements:
- Python 3.11+
- OpenAI API key
- Kaggle API credentials (optional, for submissions)

Setup:

```bash
git clone https://github.com/yourusername/kaggle-agents.git
cd kaggle-agents
pip install -e .
```

### Dependencies

- langgraph >= 0.2.0 (workflow orchestration)
- dspy-ai >= 2.5.0 (prompt optimization)
- openai >= 1.0.0 (LLM provider)
- kaggle >= 1.6.0 (Kaggle API)
- pandas >= 2.0.0 (data manipulation)
- rich >= 13.0.0 (terminal output)

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (Kaggle integration)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
export KAGGLE_AUTO_SUBMIT="true"  # default: false

# Optional (agent configuration)
export LLM_MODEL="gpt-4o-minio-mini"           # default: gpt-4o-minio-mini
export LLM_TEMPERATURE="0.1"       # default: 0.1
export MAX_ITERATIONS="5"          # default: 10
export TARGET_PERCENTILE="20.0"    # default: 20.0
export MIN_VALIDATION_SCORE="0.7"  # default: 0.7
```

### Python API

```python
from kaggle_agents.core import get_config, set_config

config = get_config()
config.llm.model = "gpt-4o-mini"
config.kaggle.auto_submit = True
config.iteration.max_iterations = 10
set_config(config)
```

## Logging

The system includes a centralized logging system for tracking execution:

```python
from kaggle_agents.core import get_logger, setup_logging

# Setup logging
setup_logging(
    log_dir="./logs",
    log_level="INFO",
    enable_console=True,
    enable_file=True,
)

# Use in your code
logger = get_logger("my_module")
logger.info("Processing started")
```

Logs are saved to `./logs/` with automatic rotation. See [LOGGING.md](LOGGING.md) for complete documentation.

## Usage

### Level 1: Simple API (Recommended)

```python
from kaggle_agents.core import solve_competition

results = solve_competition(
    competition_name="house-prices-advanced-regression-techniques",
    competition_description="Predict house prices",
    problem_type="regression",
    evaluation_metric="rmse",
    max_iterations=5,
)
```

### Level 2: Orchestrator API

```python
from kaggle_agents.core import KaggleOrchestrator

orchestrator = KaggleOrchestrator()
results = orchestrator.solve_competition(
    competition_name="digit-recognizer",
    competition_description="MNIST digit recognition",
    problem_type="multiclass_classification",
    evaluation_metric="accuracy",
    max_iterations=3,
    simple_mode=False,  # enables full workflow with iterations
)
```

### Level 3: Workflow API

```python
from kaggle_agents.workflow import run_workflow

final_state = run_workflow(
    competition_name="nlp-getting-started",
    working_dir="./work",
    competition_info={
        "name": "nlp-getting-started",
        "description": "NLP with Disaster Tweets",
        "problem_type": "binary_classification",
        "evaluation_metric": "f1_score",
        "domain": "nlp",
    },
    max_iterations=5,
)
```

### Level 4: LangGraph Raw

```python
from kaggle_agents.workflow import create_workflow
from kaggle_agents.core import create_initial_state

workflow = create_workflow()
compiled = workflow.compile()

state = create_initial_state("competition-name", "./work")
final_state = compiled.invoke(state)
```

## Results and Tracking

### Workflow Results

```python
from kaggle_agents.core import WorkflowResults

results: WorkflowResults = solve_competition(...)

print(f"Competition: {results.competition_name}")
print(f"Success: {results.success}")
print(f"Iterations: {results.iterations}")
print(f"SOTA Solutions: {results.sota_solutions_found}")
print(f"Components Planned: {results.components_planned}")
print(f"Components Implemented: {results.components_implemented}")
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Total Time: {results.total_time:.1f}s")
print(f"Termination: {results.termination_reason}")
```

### Detailed State Access

```python
state = results.final_state

# SOTA solutions
for sota in state["sota_solutions"]:
    print(f"{sota.title}: {sota.score}")

# Ablation plan
for component in state["ablation_plan"]:
    print(f"{component.name}: {component.estimated_impact}%")

# Development results
for result in state["development_results"]:
    print(f"Success: {result.success}, Time: {result.execution_time:.2f}s")

# Validation results
for validation in state["validation_results"]:
    print(f"{validation.module}: {validation.score:.1%}")

# Submissions
for submission in state["submissions"]:
    print(f"Score: {submission.public_score}, Percentile: {submission.percentile}%")
```

## DSPy Optimization

The system can self-improve by optimizing prompts based on execution feedback.

### How It Works

1. Each agent execution creates training examples
2. Specialized reward models evaluate performance
3. MIPROv2 optimizer improves prompts
4. Updated prompts lead to better performance

### Running Optimization

```python
from kaggle_agents.optimization import optimize_all_agents

# Run competitions to collect training data
results = solve_competition(...)

# Optimize prompts
optimize_all_agents(
    num_candidates=10,
    max_bootstrapped_demos=5,
    max_labeled_demos=10,
)
```

### Reward Models

- Planner Reward: Evaluates ablation plan quality
- Developer Reward: Measures code quality and success rate
- Validation Reward: Assesses validation scores
- Kaggle Reward: Tracks leaderboard performance
- Combined Reward: Weighted combination of all metrics

## Project Structure

```
kaggle-agents/
├── kaggle_agents/
│   ├── core/
│   │   ├── state.py           # State management
│   │   ├── config.py          # Configuration
│   │   └── orchestrator.py    # High-level API
│   ├── agents/
│   │   ├── search_agent.py
│   │   ├── planner_agent.py
│   │   ├── developer_agent.py
│   │   ├── robustness_agent.py
│   │   └── submission_agent.py
│   ├── tools/
│   │   ├── kaggle_search.py
│   │   ├── code_executor.py
│   │   └── llm_client.py
│   ├── optimization/
│   │   ├── prompt_optimizer.py
│   │   └── reward_model.py
│   ├── domain.py              # Domain detection
│   └── workflow.py            # LangGraph workflow
├── examples/
│   └── run_titanic_competition.py
├── tests/
│   └── test_*.py
└── README.md
```

## Examples

### Tabular Competition

```python
results = solve_competition(
    competition_name="house-prices-advanced-regression-techniques",
    competition_description="Predict house prices",
    problem_type="regression",
    evaluation_metric="rmse",
)
```

### Computer Vision

```python
results = solve_competition(
    competition_name="digit-recognizer",
    competition_description="MNIST digit recognition",
    problem_type="multiclass_classification",
    evaluation_metric="accuracy",
)
```

### Natural Language Processing

```python
results = solve_competition(
    competition_name="nlp-getting-started",
    competition_description="Disaster tweet classification",
    problem_type="binary_classification",
    evaluation_metric="f1_score",
)
```

## Testing

```bash
# Run example
python examples/run_titanic_competition.py

# Run with custom configuration
MAX_ITERATIONS=3 python examples/run_titanic_competition.py

# Run tests
pytest tests/

# Run with coverage
pytest --cov=kaggle_agents tests/
```

## Research Foundation

This project implements concepts from:

**Google's MLE STAR framework**
- Search-first strategy
- Ablation-driven optimization
- Robustness validation modules

**DSPy (Declarative Self-improving Language Programs)**
- Prompt optimization
- Reward-based learning
- MIPROv2 optimizer

**LangGraph**
- StateGraph workflow orchestration
- Conditional routing
- Checkpointing support

## Contributing

Contributions are welcome. Please submit pull requests or open issues for bugs and feature requests.

### Development Setup

```bash
git clone https://github.com/yourusername/kaggle-agents.git
cd kaggle-agents
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT License - use as you want.

## Citation

```bibtex
@software{kaggle_agents_2025,
  title={Kaggle Agents: Autonomous Competition Solving},
  author={Gustavo Paulino Gomes},
  year={2025},
  url={https://github.com/gustavogomespl/kaggle-agents}
}
```

## Status

Version: 1.0.0

Pipeline Status: Production Ready

- Domain Detection: Complete
- Search Agent: Complete
- Planner Agent: Complete
- Developer Agent: Complete
- Robustness Agent: Complete
- Submission Agent: Complete
- LangGraph Workflow: Complete
- DSPy Optimization: Complete
