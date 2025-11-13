# 🏆 Kaggle Agents - Autonomous Competition Solving

**State-of-the-art autonomous agent system for Kaggle competitions**

Built with LangGraph, DSPy, and inspired by Google's Automated Data science and Knowledge (ADK) framework.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![DSPy](https://img.shields.io/badge/DSPy-Optimization-orange.svg)](https://github.com/stanfordnlp/dspy)

---

## 🎯 Overview

Kaggle Agents is a **fully autonomous system** that can compete in Kaggle competitions with minimal human intervention. It implements cutting-edge concepts from Google's ADK research and uses advanced prompt optimization techniques to achieve competitive results.

### Key Features

- 🤖 **Fully Autonomous**: From problem analysis to submission upload
- 🔍 **Search-First Strategy**: Leverages SOTA solutions from Kaggle notebooks
- 📊 **Ablation-Driven Planning**: Systematic component testing and iteration
- 🛡️ **Robustness Validation**: 4-module validation system (Google ADK)
- 🔄 **Iterative Improvement**: Automatically iterates until goal achieved
- 🎓 **Multi-Domain Support**: Tabular, Computer Vision, NLP, Time Series
- ⚡ **DSPy Optimization**: Self-improving prompts via reinforcement learning
- 📈 **Leaderboard Monitoring**: Automatic score tracking and percentile calculation

### What It Does

```python
from kaggle_agents.core import solve_competition

# That's it! One function call to solve a competition
results = solve_competition(
    competition_name="titanic",
    competition_description="Predict survival on the Titanic",
    problem_type="binary_classification",
    evaluation_metric="accuracy",
    max_iterations=5,
)

# Automatically:
# ✅ Detects domain (tabular, CV, NLP, etc.)
# ✅ Searches for SOTA solutions
# ✅ Creates ablation plan
# ✅ Implements and tests components
# ✅ Validates code quality
# ✅ Submits to Kaggle
# ✅ Monitors leaderboard
# ✅ Iterates until top 20% achieved
```

---

## 🏗️ Architecture

### Agent Pipeline

```
╔════════════════════════════════════════════════════════╗
║                                                        ║
║        🚀 AUTONOMOUS PIPELINE WORKFLOW 🚀              ║
║                                                        ║
║  1. Domain Detection     → Identify problem type      ║
║         ↓                                              ║
║  2. Search Agent         → Find SOTA solutions        ║
║         ↓                                              ║
║  3. Planner Agent        → Create ablation plan       ║
║         ↓                                              ║
║  4. Developer Agent      → Implement components       ║
║         ↓                                              ║
║  5. Robustness Agent     → Validate code quality      ║
║         ↓                                              ║
║  6. Submission Agent     → Upload & monitor score     ║
║         ↓                                              ║
║  7. Iteration Control    → Check goal & repeat        ║
║         ↓                                              ║
║  [Top 20% achieved?] ─No→ Repeat from step 2          ║
║         ↓ Yes                                          ║
║  🎉 SUCCESS!                                           ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

### Core Components

- **LangGraph StateGraph**: Orchestrates workflow with conditional routing
- **7 Specialized Agents**: Each with specific responsibilities
- **Type-Safe State Management**: Using TypedDict and dataclasses
- **DSPy Optimization**: Self-improving prompts with MIPROv2
- **Sandbox Execution**: Safe code execution with subprocess isolation

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (or other LLM provider)
- Kaggle API credentials (optional, for submissions)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/kaggle-agents.git
cd kaggle-agents

# Install dependencies
pip install -e .

# Or with pip install requirements
pip install -r requirements.txt
```

### Dependencies

Core dependencies:
- `langgraph>=0.2.0` - Workflow orchestration
- `dspy-ai>=2.5.0` - Prompt optimization
- `openai>=1.0.0` - LLM provider
- `kaggle>=1.6.0` - Kaggle API integration
- `pandas>=2.0.0` - Data manipulation
- `rich>=13.0.0` - Beautiful terminal output

---

## 🚀 Quick Start

### 1. Set Up Credentials

```bash
# Required: OpenAI API key
export OPENAI_API_KEY="sk-..."

# Optional: Kaggle credentials (for submissions)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Optional: Enable auto-submit (disabled by default for safety)
export KAGGLE_AUTO_SUBMIT="true"
```

### 2. Run Your First Competition

```python
from kaggle_agents.core import solve_competition

# Solve the Titanic competition
results = solve_competition(
    competition_name="titanic",
    competition_description="Predict survival on the Titanic using passenger data",
    problem_type="binary_classification",
    evaluation_metric="accuracy",
    max_iterations=3,
)

# Check results
print(f"Success: {results.success}")
print(f"Iterations: {results.iterations}")
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Validation Score: {results.final_state.get('overall_validation_score', 0):.1%}")
```

### 3. See What It Created

```bash
# Check the working directory
ls -la ./kaggle_competitions/titanic/

# You'll find:
# - solution.py        (generated code)
# - submission.csv     (predictions)
# - models/           (trained models)
# - notebooks/        (downloaded SOTA solutions)
```

---

## 📖 Detailed Usage

### API Levels

Kaggle Agents provides 4 levels of API for different use cases:

#### Level 1: Super Simple (Recommended)

```python
from kaggle_agents.core import solve_competition

results = solve_competition(
    competition_name="house-prices-advanced-regression-techniques",
    competition_description="Predict house prices using advanced regression",
    problem_type="regression",
    evaluation_metric="rmse",
    max_iterations=5,
)
```

#### Level 2: Orchestrator API

```python
from kaggle_agents.core import KaggleOrchestrator

orchestrator = KaggleOrchestrator()

results = orchestrator.solve_competition(
    competition_name="digit-recognizer",
    competition_description="MNIST digit recognition",
    problem_type="multiclass_classification",
    evaluation_metric="accuracy",
    max_iterations=3,
    simple_mode=False,  # Full workflow with iterations
)
```

#### Level 3: Workflow API

```python
from kaggle_agents.workflow import run_workflow

final_state = run_workflow(
    competition_name="nlp-getting-started",
    working_dir="./work",
    competition_info={
        "name": "nlp-getting-started",
        "description": "Natural Language Processing with Disaster Tweets",
        "problem_type": "binary_classification",
        "evaluation_metric": "f1_score",
        "domain": "nlp",
    },
    max_iterations=5,
)
```

#### Level 4: LangGraph Raw (Advanced)

```python
from kaggle_agents.workflow import create_workflow
from kaggle_agents.core import create_initial_state

# Create custom workflow
workflow = create_workflow()
compiled = workflow.compile()

# Create initial state
state = create_initial_state("competition-name", "./work")

# Run workflow
final_state = compiled.invoke(state)
```

---

## 🤖 Agent Details

### 1. Domain Detection Agent

**Purpose**: Automatically identify the competition type

**Domains Supported**:
- Tabular (regression, classification)
- Computer Vision (image classification, object detection)
- Natural Language Processing (text classification, NER)
- Time Series (forecasting)

**How it works**:
```python
# Analyzes competition description and available data
domain, confidence = detect_competition_domain(competition_info, working_dir)

# Returns: ("tabular", 0.95)
```

---

### 2. Search Agent

**Purpose**: Find SOTA solutions from Kaggle notebooks

**Strategy**: Search-first approach (Google ADK)

**Process**:
1. Generate diverse search queries
2. Search Kaggle notebooks via API
3. Download and analyze top solutions
4. Extract patterns and techniques

**Example Output**:
```python
{
    "title": "Titanic - XGBoost Baseline [0.78]",
    "score": 0.78,
    "votes": 1234,
    "techniques": ["XGBoost", "Feature Engineering", "Ensemble"],
    "code_snippets": [...],
}
```

---

### 3. Planner Agent

**Purpose**: Create systematic ablation plan

**Strategy**: Ablation-driven optimization (Google ADK)

**Process**:
1. Analyze SOTA patterns
2. Identify common components
3. Estimate component impact
4. Create prioritized plan

**Example Plan**:
```python
[
    {
        "name": "xgboost_baseline",
        "type": "model",
        "description": "XGBoost baseline model",
        "estimated_impact": 15.0,
        "priority": 1,
    },
    {
        "name": "feature_engineering",
        "type": "preprocessing",
        "description": "Advanced feature engineering",
        "estimated_impact": 10.0,
        "priority": 2,
    },
    ...
]
```

---

### 4. Developer Agent

**Purpose**: Implement components with code generation

**Features**:
- LLM-based code generation
- Sandbox execution
- Automatic debugging (10 iterations)
- Retry mechanism (5 attempts)

**Process**:
```
Generate Code → Execute → Check Errors → Debug → Retry → Success
```

**Safety**:
- Subprocess isolation
- Timeout protection (10 minutes)
- Resource limits

---

### 5. Robustness Agent

**Purpose**: Validate code quality (Google ADK)

**4 Validation Modules**:

1. **Debugging** (30% weight)
   - No uncaught exceptions
   - Proper error handling
   - No warnings

2. **Data Leakage** (30% weight)
   - No target leakage
   - Proper train/test split
   - No test data in training

3. **Data Usage** (20% weight)
   - All data used
   - No unnecessary sampling
   - Proper missing value handling

4. **Format Compliance** (20% weight)
   - Submission file exists
   - Correct CSV format
   - No missing values
   - Correct number of rows

**Threshold**: 70% overall score to pass

---

### 6. Submission Agent

**Purpose**: Upload to Kaggle and monitor leaderboard

**Features**:
- Automatic submission upload
- Score fetching (waits 30s)
- Percentile calculation
- Goal achievement detection

**Goal Detection**:
```python
if percentile <= target_percentile:  # Default: top 20%
    print("🎉 GOAL ACHIEVED!")
    state["should_continue"] = False
    state["termination_reason"] = "goal_achieved"
```

**Safety**:
- Disabled by default (set `KAGGLE_AUTO_SUBMIT=true`)
- Authentication validation
- Submission file validation

---

### 7. Iteration Control

**Purpose**: Manage iteration loop and termination

**Termination Conditions**:
1. Goal achieved (top 20%)
2. Max iterations reached
3. Validation failure
4. Manual stop

**Iteration Strategy**:
- Each iteration: New SOTA search → Re-plan → Implement → Validate → Submit
- Incremental improvement
- Memory of previous iterations

---

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Configuration
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4"              # Default: gpt-4o-mini
export LLM_TEMPERATURE="0.1"          # Default: 0.1

# Kaggle Configuration
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
export KAGGLE_AUTO_SUBMIT="false"     # Default: false (safety)

# Agent Configuration
export MAX_ITERATIONS="5"             # Default: 10
export TARGET_PERCENTILE="20.0"       # Default: 20.0 (top 20%)
export MIN_VALIDATION_SCORE="0.7"     # Default: 0.7 (70%)

# Path Configuration
export KAGGLE_WORK_DIR="./kaggle_competitions"  # Default
```

### Python Configuration

```python
from kaggle_agents.core import get_config, set_config

# Get current config
config = get_config()

# Modify config
config.llm.model = "gpt-4"
config.kaggle.auto_submit = True
config.iteration.max_iterations = 10

# Set config
set_config(config)
```

---

## 🎓 DSPy Optimization

### What is DSPy Optimization?

DSPy allows the system to **self-improve** by optimizing prompts based on feedback.

### How It Works

1. **Collect Training Data**: Each agent execution creates training examples
2. **Define Reward Models**: 5 specialized models for different agents
3. **Optimize Prompts**: MIPROv2 optimizer improves prompts
4. **Apply Updates**: Better prompts → better performance

### Running Optimization

```python
from kaggle_agents.optimization import optimize_all_agents

# Collect training data first (run competitions)
results = solve_competition(...)

# Then optimize
optimize_all_agents(
    num_candidates=10,
    max_bootstrapped_demos=5,
    max_labeled_demos=10,
)
```

### Reward Models

1. **Planner Reward**: Quality of ablation plans
2. **Developer Reward**: Code quality and success rate
3. **Validation Reward**: Validation scores
4. **Kaggle Reward**: Leaderboard performance
5. **Combined Reward**: Weighted combination

---

## 📊 Results and Tracking

### Workflow Results

```python
from kaggle_agents.core import WorkflowResults

results: WorkflowResults = solve_competition(...)

# Access results
print(f"Competition: {results.competition_name}")
print(f"Success: {results.success}")
print(f"Iterations: {results.iterations}")
print(f"SOTA Solutions Found: {results.sota_solutions_found}")
print(f"Components Planned: {results.components_planned}")
print(f"Components Implemented: {results.components_implemented}")
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Total Time: {results.total_time:.1f}s")
print(f"Termination: {results.termination_reason}")
```

### Detailed State

```python
# Access final state for detailed info
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

---

## 🧪 Testing

### Run Example

```bash
# Run the Titanic example
python examples/run_titanic_competition.py

# Run with custom config
MAX_ITERATIONS=3 python examples/run_titanic_competition.py
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_agents.py -v

# Run with coverage
pytest --cov=kaggle_agents tests/
```

---

## 🛠️ Development

### Project Structure

```
kaggle-agents/
├── kaggle_agents/
│   ├── core/                    # Core infrastructure
│   │   ├── state.py            # State management
│   │   ├── config.py           # Configuration
│   │   └── orchestrator.py     # High-level API
│   ├── agents/                  # Specialized agents
│   │   ├── search_agent.py
│   │   ├── planner_agent.py
│   │   ├── developer_agent.py
│   │   ├── robustness_agent.py
│   │   └── submission_agent.py
│   ├── tools/                   # Utility tools
│   │   ├── kaggle_search.py
│   │   ├── code_executor.py
│   │   └── llm_client.py
│   ├── optimization/            # DSPy optimization
│   │   ├── prompt_optimizer.py
│   │   └── reward_model.py
│   ├── domain.py               # Domain detection
│   └── workflow.py             # LangGraph workflow
├── examples/
│   └── run_titanic_competition.py
├── tests/
│   └── test_*.py
├── README.md
└── requirements.txt
```

### Adding a New Agent

```python
from kaggle_agents.core import KaggleState
from typing import Dict, Any

class MyCustomAgent:
    def __call__(self, state: KaggleState) -> Dict[str, Any]:
        # Your agent logic here
        return {
            "my_field": "value",
            "last_updated": datetime.now(),
        }

# Create node function for LangGraph
def my_custom_agent_node(state: KaggleState) -> Dict[str, Any]:
    agent = MyCustomAgent()
    return agent(state)

# Add to workflow
workflow.add_node("my_custom_agent", my_custom_agent_node)
```

---

## 📚 Examples

### Example 1: Tabular Competition

```python
results = solve_competition(
    competition_name="house-prices-advanced-regression-techniques",
    competition_description="Predict house prices",
    problem_type="regression",
    evaluation_metric="rmse",
)
```

### Example 2: Computer Vision

```python
results = solve_competition(
    competition_name="digit-recognizer",
    competition_description="MNIST digit recognition",
    problem_type="multiclass_classification",
    evaluation_metric="accuracy",
)
```

### Example 3: NLP

```python
results = solve_competition(
    competition_name="nlp-getting-started",
    competition_description="Disaster tweet classification",
    problem_type="binary_classification",
    evaluation_metric="f1_score",
)
```

### Example 4: Custom Workflow

```python
from kaggle_agents.core import KaggleOrchestrator

orchestrator = KaggleOrchestrator()

# Customize settings
results = orchestrator.solve_competition(
    competition_name="my-competition",
    competition_description="...",
    problem_type="regression",
    evaluation_metric="mae",
    max_iterations=10,
    simple_mode=False,
    target_percentile=10.0,  # Top 10%!
)
```

---

## 🔬 Research and Inspiration

This project implements concepts from:

1. **Google's ADK (Automated Data science and Knowledge)**
   - Search-first strategy
   - Ablation-driven optimization
   - Robustness validation modules

2. **DSPy (Declarative Self-improving Language Programs)**
   - Prompt optimization
   - Reward-based learning
   - MIPROv2 optimizer

3. **LangGraph (LangChain)**
   - StateGraph for workflow orchestration
   - Conditional routing
   - Checkpointing

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/kaggle-agents.git
cd kaggle-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Areas for Contribution

- 🔧 New agent implementations
- 📊 Better evaluation metrics
- 🎯 Domain-specific optimizations
- 📝 Documentation improvements
- 🧪 Test coverage
- 🚀 Performance optimizations

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **Google Research**: ADK framework and concepts
- **Stanford NLP**: DSPy library
- **LangChain**: LangGraph orchestration
- **Kaggle Community**: SOTA solutions and inspiration

---

## 📞 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🎓 Citation

```bibtex
@software{kaggle_agents_2025,
  title={Kaggle Agents: Autonomous Competition Solving},
  author={Gustavo Paulino Gomes},
  year={2025},
  url={https://github.com/yourusername/kaggle-agents}
}
```

---

## 🎉 Status

**Current Version**: 1.0.0

**Pipeline Status**:
```
✅ Domain Detection     100%
✅ Search Agent         100%
✅ Planner Agent        100%
✅ Developer Agent      100%
✅ Robustness Agent     100%
✅ Submission Agent     100%
✅ LangGraph Workflow   100%
✅ DSPy Optimization    100%

🎉 COMPLETE AUTONOMOUS SYSTEM!
```

**Production Ready**: YES 🚀

---

Made with ❤️ for the Kaggle community
