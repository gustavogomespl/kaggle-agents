<img width="2676" height="854" alt="image" src="https://github.com/user-attachments/assets/398dcf92-804c-441d-9399-8957d31ca445" />

# Kaggle Agents

Autonomous agent system for Kaggle competitions using LangGraph, DSPy, and concepts from Google's MLE Star framework (https://www.arxiv.org/abs/2506.15692).

## Colab notebooks example

Example for MLE Bench evaluation use (https://colab.research.google.com/drive/1AluH6I7vniCIo-ULCBRJJlco7K7tdFk_?usp=sharing#scrollTo=081AGrB8PYJD)

Example for Kaggle competition use (https://colab.research.google.com/drive/14INytAtGtAQ5935yEj27cJi4zwsFTbkJ#scrollTo=run_workflow)

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
python -m notebooks/kaggle_eval.py --competition playground-series-s5e11 --max-iterations 2
```
This downloads data, plans, generates code, validates, and creates `submission.csv` in the competition directory.

4) Notebook example: check the links for Colab usage.

## Configuration Notes
- Python 3.11+.
- Adjust `max_iterations` if you want fewer iterations to reduce cost.
- To use auto-submit set `KAGGLE_AUTO_SUBMIT=true` and ensure you have `kaggle.json` or environment variables configured.

## License

MIT License - use as you want.
