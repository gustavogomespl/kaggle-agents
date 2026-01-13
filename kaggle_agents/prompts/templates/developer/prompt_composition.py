"""
Prompt composition functions for the Developer Agent.

Contains the main prompt building logic that assembles context-aware prompts.
"""

from ..builders import DynamicContext
from .core import DEVELOPER_CORE_IDENTITY, HARD_CONSTRAINTS, LOGGING_FORMAT


def compose_generate_prompt(
    component,
    competition_info,
    paths: dict[str, str],
    context: DynamicContext,
    use_modular_constraints: bool = True,
) -> str:
    """
    Compose a dynamic, context-aware code generation prompt.

    Adaptive injection based on iteration:
    - Iteration 0: SOTA-heavy (learn from winners)
    - Later iterations: Feedback-heavy + truncated SOTA reference

    Now supports modular constraints to reduce token usage (40-60% reduction).

    Args:
        component: AblationComponent to implement
        competition_info: CompetitionInfo with metadata
        paths: Dictionary with train, test, submission, models paths
        context: DynamicContext with SOTA, feedback, rewards
        use_modular_constraints: If True, load domain-specific constraints only

    Returns:
        Composed prompt string
    """
    # Get domain-specific constraints (modular) or full constraints
    if use_modular_constraints:
        try:
            from ..constraints import get_constraints_for_domain

            # Handle None domain by defaulting to "tabular"
            domain = getattr(competition_info, "domain", None) or "tabular"
            constraints = get_constraints_for_domain(domain)
            print(f"   Loaded modular constraints for domain: {domain}")
        except Exception:
            constraints = str(HARD_CONSTRAINTS)  # Fallback to full constraints
    else:
        constraints = str(HARD_CONSTRAINTS)

    parts = [
        DEVELOPER_CORE_IDENTITY,
        "",
        constraints,
        "",
        LOGGING_FORMAT,
        "",
        _format_task(component, competition_info, paths),
    ]

    # Inject dynamic canonical data instructions for model components
    comp_type = getattr(component, "component_type", "model")
    if comp_type in ("model", "ensemble"):
        try:
            from ....utils.data_contract import get_canonical_data_instructions

            output_dir = paths.get("output_dir", ".")
            canonical_instructions = get_canonical_data_instructions(output_dir)
            if canonical_instructions:
                parts.append("")
                parts.append(canonical_instructions)
        except Exception:
            pass  # Canonical data not available yet

    # Non-standard label files instruction (e.g., MLSP 2013 Birds .txt files)
    label_files = paths.get("label_files", [])
    if label_files:
        label_section = """
## NON-STANDARD LABEL FILES (MANDATORY PARSING)

Label files detected: """ + ", ".join(str(lf) for lf in label_files) + """

YOU MUST parse these files - NEVER use dummy labels (np.zeros)!

Steps:
1. Use parse_label_file() helper (injected in code header)
2. Create ID -> label mapping
3. For multi-label: pivot to binary matrix
4. Match with training data BEFORE training

Example for MLSP 2013 Birds:
```python
label_df = parse_label_file(LABEL_FILES[0])
label_df.columns = ['rec_id', 'label']
y_train = label_df.pivot_table(index='rec_id', columns='label', aggfunc=len, fill_value=0)
```

WARNING: Using np.zeros for labels causes AUC 0.5 (random predictions)!
"""
        parts.append("")
        parts.append(label_section)

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
        parts.append(
            "- Env knobs: KAGGLE_AGENTS_COMPONENT_TIMEOUT_S, KAGGLE_AGENTS_CV_FOLDS, KAGGLE_AGENTS_FAST_MODE"
        )

    # Memory insights from past runs (best HPs, errors, strategies)
    if context.memory_summary and context.memory_summary != "No memory insights available yet.":
        parts.append("")
        parts.append("## Memory Insights (Use these to avoid repeats and reuse best configs)")
        parts.append(context.memory_summary)

    # Submission validation error (must be fixed immediately).
    if context.submission_validation_error:
        parts.append("")
        parts.append("## CRITICAL: SUBMISSION FORMAT ERROR (MUST FIX)")
        parts.append(
            f"Previous submission failed validation: {context.submission_validation_error}"
        )
        parts.append("")
        parts.append("Fix requirements:")
        parts.append("1. Read sample_submission.csv to match ID values and column order exactly")
        parts.append("2. Match row count exactly (no truncation/padding)")
        parts.append("3. Preserve ID order from sample_submission.csv")
        parts.append(
            "4. For image-to-image: flatten per-pixel predictions to the sample submission ID format"
        )
        parts.append("5. Use assertions before saving")
        parts.append("```python")
        parts.append("sample = pd.read_csv(sample_submission_path)")
        parts.append("assert list(submission.columns) == list(sample.columns)")
        parts.append("assert len(submission) == len(sample)")
        parts.append(
            "assert (submission[sample.columns[0]].values == sample[sample.columns[0]].values).all()"
        )
        parts.append("```")

    # Adaptive training guidance (GPU-accelerated, reduces epochs if timeout)
    if context.run_mode.lower() == "mlebench" or "medal" in context.objective.lower():
        parts.append("")
        parts.append("## NEURAL NETWORK TRAINING (GPU-ACCELERATED)")
        parts.append(
            f"- **EPOCHS**: Train for up to {context.suggested_epochs} epochs with early stopping"
        )
        parts.append(
            "- **GPU**: MUST use CUDA if available: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`"
        )
        parts.append(
            "- **BACKBONE**: Full fine-tuning for maximum performance (do NOT freeze layers)"
        )
        parts.append("- **LEARNING RATE**: Use warmup (5% of epochs) + cosine annealing schedule")
        parts.append("- **AUGMENTATION**: Apply heavy augmentation (Cutmix, Mixup, RandAugment)")
        parts.append(
            f"- **EARLY STOPPING**: Stop if validation loss doesn't improve for {context.early_stopping_patience} epochs (SOTA uses patience=30)"
        )
        parts.append("- **CHECKPOINTING**: Save best model checkpoint by validation metric")
        parts.append("- **MIXED PRECISION**: Use torch.cuda.amp.autocast() for faster training")

        if context.timeout_occurred:
            parts.append("")
            parts.append("WARNING: TIMEOUT DETECTED IN PREVIOUS RUN - ADJUSTMENTS:")
            parts.append(
                f"- REDUCED epochs from {context.epoch_budget} to {context.suggested_epochs}"
            )
            parts.append("- Use smaller batch size if memory issues")
            parts.append("- Consider freezing early backbone layers if still too slow")
            parts.append("- STILL prioritize completing training over speed")

        parts.append("")
        parts.append("## SOFT-DEADLINE PATTERN (MANDATORY)")
        timeout_s = context.timeout_per_component or 3600
        parts.append("```python")
        parts.append("import time")
        parts.append("_START = time.time()")
        parts.append(f"_TIMEOUT = {timeout_s}")
        parts.append("_SOFT_DEADLINE = _TIMEOUT - 120  # Reserve 2min for saving")
        parts.append("")
        parts.append("for epoch in range(MAX_EPOCHS):")
        parts.append("    if time.time() - _START >= _SOFT_DEADLINE:")
        parts.append("        print('[TIMEOUT] Soft deadline reached, saving best model')")
        parts.append("        break")
        parts.append("    # ... train epoch ...")
        parts.append("```")

    # ADAPTIVE: First iteration = SOTA heavy
    if context.iteration_num == 0:
        if context.sota_patterns:
            parts.append("")
            parts.append("## SOTA Patterns (Learn from top solutions):")
            parts.append(context.sota_patterns)

    # ADAPTIVE: Later iterations = Feedback heavy
    else:
        # CRITICAL: Feedback comes first to ensure corrections are applied
        if context.previous_feedback:
            parts.append("")
            parts.append("## Previous Attempt Feedback (MUST FIX):")
            parts.append(context.previous_feedback)

        if context.what_failed:
            parts.append("")
            parts.append("## What Failed (DO NOT REPEAT):")
            parts.append("\n".join(f"- {f}" for f in context.what_failed[:5]))

        if context.reward_guidance:
            parts.append("")
            parts.append("## Meta-Evaluator Guidance:")
            parts.append(context.reward_guidance)

        if context.attempt_feedback:
            parts.append("")
            parts.append("## Prior Attempts (Study + Fix):")
            parts.append(context.attempt_feedback)

        if context.what_worked:
            parts.append("")
            parts.append("## What Worked (Keep these approaches):")
            parts.append("\n".join(f"- {w}" for w in context.what_worked[:5]))

        # DPO: Inject contrastive learning examples (good vs bad code)
        if context.dpo_examples:
            parts.append("")
            parts.append(context.dpo_examples)

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

    train_path = paths.get("train", "train.csv")
    test_path = paths.get("test", "test.csv")
    models_path = paths.get("models", "models/")
    submission_path = paths.get("submission", "submission.csv")

    return f"""## Task
Component: {component_type} - {component_name}
Goal: {component_code}
Estimated Impact: {estimated_impact:.1%}

## Competition
Name: {name}
Domain: {domain}
Problem Type: {problem_type}
Metric: {metric}

## Paths (CRITICAL - USE EXACTLY AS PROVIDED)
# INPUT_DIR is READ-ONLY in Kaggle Kernels - NEVER write here!
INPUT_DIR: {paths.get("input_dir", ".")}
# OUTPUT_DIR is WRITABLE - use for all outputs (models, submission, etc.)
OUTPUT_DIR: {paths.get("output_dir", ".")}

Train: {train_path}
Test: {test_path}
Models: {models_path}
Submission: {submission_path}

## PATH USAGE (MANDATORY - DO NOT HARDCODE)
**CRITICAL**: Use the EXACT paths provided above. DO NOT hardcode 'train.csv' or 'test.csv'.

```python
# CORRECT: Use the provided paths EXACTLY
from pathlib import Path

TRAIN_PATH = Path("{train_path}")
TEST_PATH = Path("{test_path}")
MODELS_DIR = Path("{models_path}")
SUBMISSION_PATH = Path("{submission_path}")

# Create models directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data based on path type:
if TRAIN_PATH.suffix == '.csv':
    train_df = pd.read_csv(TRAIN_PATH)
elif TRAIN_PATH.is_dir():
    # For directory-based data (images, audio, etc.):
    train_files = sorted(TRAIN_PATH.glob('*'))
    print(f"Found {{len(train_files)}} files in {{TRAIN_PATH}}")
```

**NEVER** do this (WRONG - will cause FileNotFoundError or NameError):
```python
train_df = pd.read_csv('train.csv')  # WRONG! Relative path fails
train_df = pd.read_csv(BASE_DIR / 'train.csv')  # WRONG! BASE_DIR is NOT defined
test_df = pd.read_csv(BASE_DIR / 'test.csv')  # WRONG! Use TEST_PATH instead
sample = pd.read_csv(BASE_DIR / 'sample_submission.csv')  # WRONG! Use SAMPLE_SUBMISSION_PATH
```

## PATH CONSTANTS (CRITICAL - DO NOT USE BASE_DIR)
The following path constants ARE pre-defined in the execution environment:
- **TRAIN_PATH**: Path to training data (use directly, NOT `OUTPUT_DIR / "train.csv"`)
- **TEST_PATH**: Path to test data
- **SAMPLE_SUBMISSION_PATH**: Path to sample_submission.csv
- **OUTPUT_DIR**: Directory for all outputs (models, predictions, submission.csv)
- **SUBMISSION_PATH**: `OUTPUT_DIR / "submission.csv"`

**BASE_DIR IS NOT DEFINED** - using it will cause `NameError: name 'BASE_DIR' is not defined`!
- For train data: use `TRAIN_PATH` directly
- For test data: use `TEST_PATH` directly
- For sample submission: use `SAMPLE_SUBMISSION_PATH` directly
- For intermediate files: use `OUTPUT_DIR / "filename"`

The paths may point to:
- CSV files: `train.csv`, `test.csv`
- Directories: `supplemental_data/`, `train_images/`, `essential_data/`
- Subdirectories: `essential_data/train.csv`

Always check if the path is a file or directory before loading."""


def _get_component_guidance(component_type: str) -> str:
    """Get minimal, type-specific guidance."""
    from .component_guidance import COMPONENT_GUIDANCE

    # Handle domain-specific model types
    if component_type == "model":
        return COMPONENT_GUIDANCE.get(component_type, "")

    return COMPONENT_GUIDANCE.get(component_type, "")
