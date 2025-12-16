"""
Prompt templates for the Developer Agent.

Refactored to be agentic, feedback-driven, and RL-friendly.
Inspired by Claude Code's concise style.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

# ==================== Core Identity ====================

DEVELOPER_CORE_IDENTITY = """You are a Kaggle Grandmaster implementing ML components.

Style:
- Write minimal, working code - no unnecessary abstractions
- No comments unless logic is non-obvious
- Use proven patterns from SOTA solutions when provided
- Print structured logs for the feedback loop

Output: A single Python code block. No explanations outside the code."""


# ==================== Hard Constraints ====================

HARD_CONSTRAINTS = """## MUST (violations cause failures):
1. predict_proba() for classification (NOT predict())
2. CV folds must respect `KAGGLE_AGENTS_CV_FOLDS` (default 5): StratifiedKFold(n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5")), shuffle=True, random_state=42)
3. Pipeline/ColumnTransformer for preprocessing - fit INSIDE CV folds only
4. Save OOF predictions: np.save('models/oof_{component_name}.npy', oof_predictions)
5. Clamp predictions: np.clip(predictions, 0, 1) before saving
6. Match sample_submission.csv exactly: columns, IDs, shape
7. Print "Final Validation Performance: {score:.6f}" at the end
8. Set random_state=42 everywhere for reproducibility
9. MANDATORY SOFT-DEADLINE PATTERN (prevents hard timeout kills):
   ```python
   import os, time
   _START_TIME = time.time()
   _TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "600"))
   _SOFT_DEADLINE_S = _TIMEOUT_S - 45  # Reserve 45s for cleanup/save

   def _check_deadline() -> bool:
       '''Return True if deadline exceeded.'''
       return (time.time() - _START_TIME) >= _SOFT_DEADLINE_S

   # Call inside training loops:
   for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
       if _check_deadline():
           print("[LOG:WARNING] Soft deadline reached, stopping early")
           break
       # ... train fold ...

   # ALWAYS print final metric even if stopped early
   print(f"Final Validation Performance: {cv_score:.6f}")
   ```

## MUST NOT:
- sys.exit(), exit(), quit(), raise SystemExit, os._exit()
- try-except blocks that swallow errors (let them surface)
- early_stopping_rounds as direct fit() parameter (use callbacks)
- Subsample training data unless `KAGGLE_AGENTS_FAST_MODE=1` (FAST_MODE may subsample to meet budget, but keep determinism)

## API Gotchas:
- OneHotEncoder: sparse_output=False (NOT sparse=False) for sklearn 1.2+
- pd.concat() instead of .append() for pandas 2.0+
- Optuna: set_verbosity(WARNING), n_trials <= 5, timeout=60 for validation
- LightGBM callbacks: lgb.early_stopping(100), not early_stopping_rounds param
- XGBoost callbacks: xgb.callback.EarlyStopping(rounds=100)

## PyTorch Gotchas:
- Dataset __getitem__ must return tensors/arrays (never None) so DataLoader can collate"""


# ==================== Logging Format ====================

LOGGING_FORMAT = """## Structured Logs (required for feedback loop):
[LOG:FOLD] fold={n} score={score:.6f} time={time:.2f}
[LOG:CV_SUMMARY] mean={mean:.6f} std={std:.6f} scores={list}
[LOG:OPTUNA] trial={n} score={score:.6f} time={time:.2f} params={dict}
[LOG:TIMING] step={name} time={time:.2f} cumulative={cumulative:.2f}
[LOG:FEATURES] top={list[:20]} importances={list[:20]}
[LOG:WARNING] message={str}
[LOG:ERROR] message={str}"""


# ==================== Dynamic Context ====================

@dataclass
class DynamicContext:
    """Context injected into prompts based on current workflow state."""

    sota_patterns: str = ""
    previous_feedback: str = ""
    attempt_feedback: str = ""
    reward_guidance: str = ""
    iteration_num: int = 0
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    best_score: Optional[float] = None
    target_score: Optional[float] = None
    run_mode: str = ""
    objective: str = ""
    timeout_per_component: Optional[int] = None
    fast_mode: bool = False


def build_context(state: dict[str, Any], component: Any | None = None) -> DynamicContext:
    """
    Build dynamic context from KaggleState for prompt injection.

    Extracts:
    - SOTA solutions → sota_patterns
    - Previous execution results → previous_feedback
    - Meta-evaluator guidance → reward_guidance
    - Iteration memory → what_worked, what_failed

    Args:
        state: KaggleState dictionary
        component: Optional component being implemented (for filtering attempt history)

    Returns:
        DynamicContext with extracted information
    """
    from ...utils.log_parser import parse_training_logs, format_feedback_for_llm

    context = DynamicContext()
    context.iteration_num = state.get("current_iteration", 0)
    context.best_score = state.get("best_score")
    context.target_score = state.get("target_score")
    context.run_mode = str(state.get("run_mode", ""))
    context.objective = str(state.get("objective", ""))
    context.fast_mode = (
        str(state.get("run_mode", "")).lower() == "mlebench"
        or str(os.getenv("KAGGLE_AGENTS_FAST_MODE", "")).lower() in {"1", "true", "yes"}
        or str(os.getenv("FAST_MODE", "")).lower() in {"1", "true", "yes"}
    )
    timeout_val = state.get("timeout_per_component")
    if isinstance(timeout_val, str):
        try:
            timeout_val = int(timeout_val)
        except ValueError:
            timeout_val = None
    context.timeout_per_component = timeout_val if isinstance(timeout_val, int) else None

    # Extract SOTA patterns from search results
    sota_solutions = state.get("sota_solutions", [])
    if sota_solutions:
        context.sota_patterns = _format_sota_for_prompt(sota_solutions)

    # Extract feedback from previous development results
    dev_results = state.get("development_results", [])
    if dev_results:
        last_result = dev_results[-1]
        if hasattr(last_result, "stdout") and last_result.stdout:
            training_feedback = parse_training_logs(last_result.stdout)
            if training_feedback:
                context.previous_feedback = format_feedback_for_llm(training_feedback)

    # Extract meta-evaluator guidance
    refinement_guidance = state.get("refinement_guidance", {})
    reward_signals = state.get("reward_signals", {})

    guidance_parts = []
    if refinement_guidance.get("developer_guidance"):
        guidance_parts.append(refinement_guidance["developer_guidance"])

    if refinement_guidance.get("priority_fixes"):
        fixes = refinement_guidance["priority_fixes"]
        if fixes:
            guidance_parts.append(f"Priority fixes: {', '.join(fixes[:3])}")

    if reward_signals:
        r_combined = reward_signals.get("r_combined", 0)
        r_performance = reward_signals.get("r_performance", 0)
        r_medal = reward_signals.get("r_medal")
        if isinstance(r_medal, (int, float)):
            guidance_parts.append(
                f"Reward: r_combined={r_combined:.3f}, r_performance={r_performance:.3f}, r_medal={float(r_medal):.3f}"
            )
        else:
            guidance_parts.append(
                f"Reward: r_combined={r_combined:.3f}, r_performance={r_performance:.3f}"
            )

    if guidance_parts:
        context.reward_guidance = "\n".join(guidance_parts)

    # Extract what worked/failed from iteration memory
    iteration_memory = state.get("iteration_memory", [])
    if iteration_memory:
        latest = iteration_memory[-1]
        if hasattr(latest, "what_worked"):
            context.what_worked = latest.what_worked or []
        if hasattr(latest, "what_failed"):
            context.what_failed = latest.what_failed or []

    # Poetiq-style feedback injection: include selected prior attempts + feedback
    attempts = state.get("code_attempts", [])
    if attempts:
        component_name = getattr(component, "name", None) if component is not None else None

        def _get_field(a: Any, key: str) -> Any:
            if isinstance(a, dict):
                return a.get(key)
            return getattr(a, key, None)

        relevant = (
            [a for a in attempts if _get_field(a, "component_name") == component_name]
            if component_name
            else list(attempts)
        )

        # Selection controls (defaults tuned for token safety)
        try:
            selection_probability = float(os.getenv("ATTEMPT_SELECTION_PROB", "1.0"))
        except ValueError:
            selection_probability = 1.0

        try:
            max_attempts = int(os.getenv("ATTEMPT_CONTEXT_MAX", "3"))
        except ValueError:
            max_attempts = 3

        selection_probability = max(0.0, min(selection_probability, 1.0))
        max_attempts = max(0, min(max_attempts, 5))

        rng = random.Random(42)
        selected = [a for a in relevant if rng.random() < selection_probability]

        def _attempt_score(a: Any) -> float:
            cv = _get_field(a, "cv_score")
            if isinstance(cv, (int, float)):
                return float(cv)
            return 1.0 if bool(_get_field(a, "success")) else 0.0

        selected.sort(key=_attempt_score, reverse=True)
        selected = selected[:max_attempts]

        if selected:
            context.attempt_feedback = _format_attempts_for_prompt(selected)

    return context


def _format_sota_for_prompt(solutions: list, max_solutions: int = 3) -> str:
    """Format SOTA solutions into prompt-friendly text."""
    lines = []
    for i, sol in enumerate(solutions[:max_solutions], 1):
        title = getattr(sol, "title", "Unknown")
        score = getattr(sol, "score", 0)
        lines.append(f"### Solution {i}: {title} (Score: {score})")

        models = getattr(sol, "models_used", [])
        if models:
            lines.append(f"Models: {', '.join(models[:5])}")

        strategies = getattr(sol, "strategies", [])
        if strategies:
            lines.append(f"Strategies: {'; '.join(strategies[:3])}")

        snippets = getattr(sol, "code_snippets", [])
        if snippets:
            snippet = snippets[0][:800] if len(snippets[0]) > 800 else snippets[0]
            lines.append(f"```python\n{snippet}\n```")

        lines.append("")

    return "\n".join(lines)

def _format_attempts_for_prompt(attempts: list[Any]) -> str:
    """Format prior attempts (code + feedback) into prompt-friendly text."""

    def _get_field(a: Any, key: str) -> Any:
        if isinstance(a, dict):
            return a.get(key)
        return getattr(a, key, None)

    blocks: list[str] = []
    for idx, attempt in enumerate(attempts, start=1):
        stage = _get_field(attempt, "stage") or "unknown"
        attempt_num = _get_field(attempt, "attempt")
        success = bool(_get_field(attempt, "success"))
        cv_score = _get_field(attempt, "cv_score")
        error = _get_field(attempt, "error")
        meta_feedback = _get_field(attempt, "meta_feedback")
        code_excerpt = (_get_field(attempt, "code_excerpt") or "").strip()
        stdout_tail = (_get_field(attempt, "stdout_tail") or "").strip()

        header = f"<attempt_{idx}> stage={stage} attempt={attempt_num} success={success}"
        if isinstance(cv_score, (int, float)):
            header += f" cv_score={float(cv_score):.6f}"

        parts = [header]
        if error:
            parts.append(f"error: {str(error)[:400]}")
        if meta_feedback:
            parts.append("meta_feedback:")
            parts.append(str(meta_feedback)[:700])
        if stdout_tail:
            parts.append("stdout_tail:")
            parts.append(str(stdout_tail)[:700])
        if code_excerpt:
            parts.append("code_excerpt:")
            parts.append(f"```python\n{code_excerpt[:1600]}\n```")
        parts.append(f"</attempt_{idx}>")
        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


# ==================== Prompt Composition ====================

def compose_generate_prompt(
    component,
    competition_info,
    paths: dict[str, str],
    context: DynamicContext,
) -> str:
    """
    Compose a dynamic, context-aware code generation prompt.

    Adaptive injection based on iteration:
    - Iteration 0: SOTA-heavy (learn from winners)
    - Later iterations: Feedback-heavy + truncated SOTA reference

    Args:
        component: AblationComponent to implement
        competition_info: CompetitionInfo with metadata
        paths: Dictionary with train, test, submission, models paths
        context: DynamicContext with SOTA, feedback, rewards

    Returns:
        Composed prompt string
    """
    parts = [
        DEVELOPER_CORE_IDENTITY,
        "",
        HARD_CONSTRAINTS,
        "",
        LOGGING_FORMAT,
        "",
        _format_task(component, competition_info, paths),
    ]

    # Runtime/objective hints (important for timeout-sensitive runs like MLE-bench).
    if context.run_mode or context.objective or context.timeout_per_component is not None:
        parts.append("")
        parts.append("## Objective & Budget")
        if context.run_mode:
            parts.append(f"- run_mode: {context.run_mode}")
        if context.objective:
            parts.append(f"- objective: {context.objective}")
        if context.timeout_per_component is not None:
            parts.append(f"- timeout_per_component_seconds: {context.timeout_per_component}")
        parts.append("- Env knobs: KAGGLE_AGENTS_COMPONENT_TIMEOUT_S, KAGGLE_AGENTS_CV_FOLDS, KAGGLE_AGENTS_FAST_MODE")

    # MLE-bench specific guidance for medal optimization
    if context.run_mode.lower() == "mlebench" or "medal" in context.objective.lower():
        parts.append("")
        parts.append("## MLE-BENCH MEDAL OPTIMIZATION (CRITICAL)")
        parts.append("- GOAL: Achieve at least a Bronze medal")
        parts.append("- TIME BUDGET: Complete within timeout - IMPLEMENT soft-deadline pattern")
        parts.append("- CV FOLDS: Use 2-3 folds max (speed over precision)")
        parts.append("- EARLY STOPPING: Stop training early if deadline approaches")
        parts.append("- SUBMISSION: MUST generate valid submission even if training incomplete")
        parts.append("- FALLBACK: If time runs out, use best checkpoint so far")
        parts.append("- For image models: freeze backbone, train head only, 2-3 epochs max")

    # ADAPTIVE: First iteration = SOTA heavy
    if context.iteration_num == 0:
        if context.sota_patterns:
            parts.append("")
            parts.append("## SOTA Patterns (Learn from top solutions):")
            parts.append(context.sota_patterns)

    # ADAPTIVE: Later iterations = Feedback heavy
    else:
        if context.previous_feedback:
            parts.append("")
            parts.append("## Previous Attempt Feedback:")
            parts.append(context.previous_feedback)

        if context.attempt_feedback:
            parts.append("")
            parts.append("## Prior Attempts (Study + Fix):")
            parts.append(context.attempt_feedback)

        if context.what_worked:
            parts.append("")
            parts.append("## What Worked (Keep these approaches):")
            parts.append("\n".join(f"- {w}" for w in context.what_worked[:5]))

        if context.what_failed:
            parts.append("")
            parts.append("## What Failed (Avoid these):")
            parts.append("\n".join(f"- {f}" for f in context.what_failed[:5]))

        if context.reward_guidance:
            parts.append("")
            parts.append("## Meta-Evaluator Guidance:")
            parts.append(context.reward_guidance)

        # Still include truncated SOTA as reference
        if context.sota_patterns:
            parts.append("")
            parts.append("## SOTA Reference (condensed):")
            parts.append(context.sota_patterns[:1000])

    # Component-specific minimal guidance
    guidance = _get_component_guidance(component.component_type)
    if guidance:
        parts.append("")
        parts.append(guidance)

    return "\n".join(parts)


def _format_task(component, competition_info, paths: dict[str, str]) -> str:
    """Format the task specification section."""
    component_type = getattr(component, "component_type", "model")
    component_name = getattr(component, "name", "component")
    component_code = getattr(component, "code", "")
    estimated_impact = getattr(component, "estimated_impact", 0.0)

    name = getattr(competition_info, "name", "competition")
    domain = getattr(competition_info, "domain", "tabular")
    problem_type = getattr(competition_info, "problem_type", "classification")
    metric = getattr(competition_info, "evaluation_metric", "accuracy")

    return f"""## Task
Component: {component_type} - {component_name}
Goal: {component_code}
Estimated Impact: {estimated_impact:.1%}

## Competition
Name: {name}
Domain: {domain}
Problem Type: {problem_type}
Metric: {metric}

## Paths
Train: {paths.get('train', 'train.csv')}
Test: {paths.get('test', 'test.csv')}
Models: {paths.get('models', 'models/')}
Submission: {paths.get('submission', 'submission.csv')}"""


def _get_component_guidance(component_type: str) -> str:
    """Get minimal, type-specific guidance."""
    guidance = {
        "model": """## Model Component Requirements
- IMPLEMENT soft-deadline pattern (see HARD_CONSTRAINTS #9) - check _check_deadline() INSIDE fold loop
- Train model with StratifiedKFold CV using n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5"))
- Save OOF predictions to models/oof_{name}.npy for stacking
- Handle class imbalance if ratio > 2:1 (class_weight or scale_pos_weight)
- Print per-fold scores: [LOG:FOLD] fold={n} score={s:.6f}
- Use GPU if available (check torch.cuda.is_available())
- Create submission.csv with probabilities [0,1]
- ALWAYS print "Final Validation Performance: {score}" even if stopped early due to deadline""",

        "feature_engineering": """## Feature Engineering Requirements
- Transform train and test consistently
- NO model training in this component
- Save to train_engineered.csv, test_engineered.csv if creating new files
- Fast execution (<30 seconds)
- Print "Final Validation Performance: 1.0" on completion""",

        "ensemble": """## Ensemble Requirements
- Load OOF predictions from models/oof_*.npy files
- Preferred: Stacking with LogisticRegression/Ridge meta-learner
- Fallback: Weighted average if OOF files missing
- Can use correlation analysis to select diverse models
- Create submission.csv with final ensemble predictions""",

        "preprocessing": """## Preprocessing Requirements
- Clean data, handle missing values, encode categoricals
- NO model training
- Fast execution (<10 seconds)
- Save processed data for subsequent components
- Print "Final Validation Performance: 1.0" on completion""",
    }

    return guidance.get(component_type, "")


# ==================== Fix and Debug Prompts ====================

FIX_CODE_PROMPT = """Fix this code error.

## Code
```python
{code}
```

## Error
{error}

## Error Type
{error_type}

## Meta-Feedback (use this to fix root cause)
{meta_feedback}

Fix the issue while preserving the component's intent. Return complete fixed code."""


DEBUG_CODE_PROMPT = """Debug this code that failed.

## Code
```python
{code}
```

## Issue
{issue}

## Stdout (last lines)
{stdout}

## Stderr
{stderr}

## Meta-Feedback (if available)
{meta_feedback}

Analyze the output, fix logic errors or missing imports, and return the complete debugged code."""


# ==================== Refinement Prompt ====================

REFINEMENT_WITH_FEEDBACK_PROMPT = """Refine this model based on training feedback.

## Current Score
CV: {current_score}

## Training Feedback
{training_feedback}

## Current Code
```python
{current_code}
```

## Improvement Guidelines
Based on the feedback:
- High variance (std > 0.02): Increase regularization, reduce depth
- Overfitting (train >> val): Add dropout, increase subsample
- Underfitting (low score): Decrease regularization, add features
- Optuna best params: Use as starting point

Keep the same [LOG:*] format for the feedback loop.
Return the complete improved code."""


# ==================== Utility Functions ====================

def format_component_details(component) -> str:
    """Format component details for prompts."""
    name = getattr(component, "name", "Unknown")
    component_type = getattr(component, "component_type", "model")
    estimated_impact = getattr(component, "estimated_impact", 0.0)
    code = getattr(component, "code", "No description")

    return f"""Name: {name}
Type: {component_type}
Estimated Impact: {estimated_impact:.1%}
Description: {code}"""


def format_error_info(error: str) -> dict[str, str]:
    """Categorize and format error information."""
    error_types = {
        "ModuleNotFoundError": "missing_import",
        "FileNotFoundError": "missing_file",
        "KeyError": "missing_key",
        "ValueError": "value_error",
        "TypeError": "type_error",
        "SyntaxError": "syntax_error",
        "MemoryError": "memory_error",
        "Timeout": "timeout",
        "TimeoutError": "timeout",
    }

    error_type = "unknown_error"
    for key, value in error_types.items():
        if key in error:
            error_type = value
            break

    return {
        "error_type": error_type,
        "error": error,
    }


# ==================== Ablation Study Prompts ====================

ABLATION_STUDY_PROMPT = """Analyze the impact of component changes through ablation study.

## Baseline Code
```python
{baseline_code}
```

## Modified Code
```python
{modified_code}
```

## Component Being Tested
{component_name}

Compare baseline vs modified performance. Return analysis in JSON format:
{{"component": "{component_name}", "baseline_score": float, "modified_score": float, "delta": float, "recommendation": "keep|remove|modify"}}"""


ABLATION_STUDY_SEQUENTIAL_PROMPT = """Perform sequential ablation study.

## Current Best Code
```python
{current_code}
```

## Components to Test
{components}

Test each component's impact sequentially. Return results for each."""


SUMMARIZE_ABLATION_PROMPT = """Summarize ablation study results.

## Results
{results}

Provide:
1. Most impactful components (positive delta)
2. Harmful components (negative delta)
3. Recommended final configuration"""


EXTRACT_IMPROVEMENT_PLAN_PROMPT = """Extract improvement plan from ablation results.

## Ablation Results
{results}

## Current Score
{current_score}

Create prioritized list of improvements based on ablation findings."""


EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT = """Extract sequential improvement plan.

## Sequential Results
{results}

## Target Score
{target_score}

Create ordered plan to reach target score."""


PLAN_REFINEMENT_PROMPT = """Refine improvement plan based on actual results.

## Original Plan
{original_plan}

## Actual Results
{actual_results}

## Gap Analysis
{gap_analysis}

Update plan based on what worked and what didn't."""


IMPLEMENT_PLAN_PROMPT = """Implement the improvement plan.

## Current Code
```python
{current_code}
```

## Improvement Plan
{plan}

## Priority
{priority}

Generate improved code implementing the plan."""
