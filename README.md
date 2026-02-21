<img width="2676" height="854" alt="image" src="https://github.com/user-attachments/assets/398dcf92-804c-441d-9399-8957d31ca445" />

# Kaggle Agents

Autonomous agent system for Kaggle competitions using LangGraph, DSPy, and concepts from Google's MLE Star framework (https://www.arxiv.org/abs/2506.15692).

## Colab notebooks example

Example for MLE Bench evaluation use (https://colab.research.google.com/drive/1AluH6I7vniCIo-ULCBRJJlco7K7tdFk_?usp=sharing#scrollTo=081AGrB8PYJD)

## Quickstart

1) Install in dev mode:
```bash
git clone https://github.com/yourusername/kaggle-agents.git
cd kaggle-agents
pip install -e .
```

2) Configure credentials (Kaggle optional for submission):
```bash
export OPENAI_API_KEY="..."
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
# Optional: choose models per role
# PLANNER_MODEL=claude-4-5-sonnet EVALUATOR_MODEL=claude-4-5-sonnet DEVELOPER_MODEL=gpt-5-mini
```

3) Run automation via CLI:
```bash
python -m kaggle_agents.main playground-series-s5e11 --max-iterations 2
```
This downloads data, plans, generates code, validates, and creates `submission.csv` in the competition directory.

4) Notebook example: check `kaggle_agents_colab.ipynb` for Colab usage.

## Configuration Notes
- Python 3.11+.
- Adjust `max_iterations` if you want fewer iterations to reduce cost.
- To use auto-submit set `KAGGLE_AUTO_SUBMIT=true` and ensure you have `kaggle.json` or environment variables configured.

## Minimal Structure
- `kaggle_agents/main.py`: CLI entry point.
- `kaggle_agents/workflow.py`: LangGraph orchestration.
- Agents: planner, developer, robustness, submission, reporting.

## Best Practices
- Use the generated `submission.csv` in the competition directory.
- Keep `sample_submission.csv` as a template to avoid column/id errors.
- Avoid callbacks/early_stopping if the XGBoost/LightGBM version is unknown; prefer moderate hyperparameters and pipelines with OneHotEncoder.

## License

MIT License - use as you want.
