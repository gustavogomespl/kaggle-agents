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
    requirements: str = "",
) -> str:
    """
    Compose a dynamic, context-aware code generation prompt.

    Adaptive injection based on iteration:
    - Iteration 0: retrieval-heavy (test external technique hypotheses)
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
    if requirements:
        parts.extend(("", "## Dynamic Component Contract", requirements))

    # The target section comes from the ONE resolved decision. Probing
    # `output_dir/canonical` here was a second, independent authority check:
    # it could announce a canonical contract the executable header never
    # injected (or stay silent about one it did), so the model was told to
    # parse a stale annotation file the preamble had deliberately hidden.
    comp_type = getattr(component, "component_type", "model")
    target_source = paths.get("target_source")
    for section in _target_source_sections(target_source, paths, comp_type):
        parts.append("")
        parts.append(section)

    # Runtime/objective hints (important for timeout-sensitive runs like MLE-bench).
    if context.run_mode or context.objective or context.timeout_per_component is not None:
        parts.append("")
        parts.append("## Objective & Budget")
        if context.run_mode:
            displayed_run_mode = (
                "fixed_budget_evaluation"
                if context.run_mode.lower() == "mlebench"
                else context.run_mode
            )
            parts.append(f"- run_mode: {displayed_run_mode}")
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
        parts.append("1. Keep the injected header and use write_submission(test_preds)")
        parts.append("2. Match row count exactly (no truncation/padding)")
        parts.append("3. Preserve the template's echoed columns and row order")
        parts.append(
            "4. For image-to-image: flatten per-pixel predictions to the sample submission ID format"
        )
        parts.append("5. Do not assign template columns by position or call to_csv directly")

    # Adaptive training guidance (GPU-accelerated, bounded by measured runtime)
    if context.run_mode.lower() == "mlebench" or "medal" in context.objective.lower():
        parts.append("")
        parts.append("## BUDGET-MATCHED NEURAL TRAINING")
        parts.append(
            f"- **UPPER BOUND**: At most {context.suggested_epochs} epochs; "
            "select the feasible step budget from a throughput pilot"
        )
        parts.append(
            "- **DEVICE**: Use CUDA when available, while preserving a valid CPU path"
        )
        parts.append(
            "- **BACKBONE**: Compare frozen, partial, or full fine-tuning only "
            "when each fits the deadline; choose by identical OOF folds"
        )
        parts.append(
            "- **SCHEDULE**: Derive warmup and decay from the measured number of "
            "optimizer steps; do not assume a fixed epoch percentage"
        )
        parts.append(
            "- **AUGMENTATION**: Use only label-preserving transforms supported "
            "by the observed task; retain each candidate through OOF evidence"
        )
        parts.append(
            f"- **EARLY STOPPING UPPER BOUND**: {context.early_stopping_patience} "
            "checks, shortened when the remaining deadline requires it"
        )
        parts.append("- **CHECKPOINTING**: Save best model checkpoint by validation metric")
        parts.append(
            "- **MIXED PRECISION**: Use it only when supported and numerically "
            "validated for the chosen device/model"
        )

        if context.timeout_occurred:
            parts.append("")
            parts.append("WARNING: TIMEOUT DETECTED IN PREVIOUS RUN - ADJUSTMENTS:")
            parts.append(
                f"- REDUCED epochs from {context.epoch_budget} to {context.suggested_epochs}"
            )
            parts.append("- Use smaller batch size if memory issues")
            parts.append("- Reduce trainable capacity if the throughput pilot still misses the deadline")
            parts.append("- Prioritize a complete, validated candidate within the deadline")

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

    # ADAPTIVE: First iteration = external-hypothesis heavy
    if context.iteration_num == 0:
        if context.sota_patterns:
            parts.append("")
            parts.append(
                "## Retrieved External Technique Hypotheses "
                "(unverified; validate locally):"
            )
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

        # Still include bounded retrieved hypotheses as reference
        if context.sota_patterns:
            parts.append("")
            parts.append("## Retrieved External Hypotheses (condensed):")
            parts.append(context.sota_patterns[:1000])

    # Component-specific minimal guidance
    guidance = _get_component_guidance(component.component_type)
    if guidance:
        parts.append("")
        parts.append(guidance)

    return "\n".join(parts)


_DENSE_CANONICAL_INSTRUCTIONS = """
## MANDATORY: Canonical Data Contract

The canonical contract is the ONLY authoritative source of rows, folds and
targets for this run. It is already loaded by the injected header.

```python
# Already available - do NOT reload or redefine:
#   CANONICAL_TRAIN_IDS, CANONICAL_Y, CANONICAL_FOLDS, CANONICAL_TEST_IDS
#   CANONICAL_FEATURE_COLS, ID_COL, TARGET_COL, TARGET_COLS, N_FOLDS

# Use the injected audited splitter for CV. It preserves temporal
# forward-chaining and warm-up eligibility when the canonical strategy is
# temporal; a simple fold-complement split does not.
for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = CANONICAL_Y[train_idx], CANONICAL_Y[val_idx]
    model.fit(X_train, y_train)
    oof[val_idx] = model.predict_proba(X_val)
```

### CRITICAL RULES:
1. NEVER use train_test_split() - use the canonical folds
2. NEVER create your own KFold/StratifiedKFold - folds are pre-defined
3. NEVER sample or shuffle the data independently
4. ALWAYS save OOF predictions in canonical order (aligned with CANONICAL_TRAIN_IDS)
5. Align a loaded training table with align_train_to_canonical(df)

### Saving Predictions:
```python
assert len(oof) == len(CANONICAL_TRAIN_IDS), "OOF must match canonical row count"
save_component_artifacts(
    oof,
    test_predictions,
    train_ids=CANONICAL_TRAIN_IDS,
    test_ids=CANONICAL_TEST_IDS,
)
```
"""

_PACKED_CANONICAL_INSTRUCTIONS = """
## MANDATORY: Canonical Data Contract (packed image targets)

Targets are packed pixel arrays, not a dense `y` vector. The injected header
already loaded and validated them; do NOT open the archive yourself.

```python
# Already available - do NOT reload or redefine:
#   CANONICAL_TRAIN_IDS, CANONICAL_TEST_IDS, CANONICAL_FOLDS
#   CANONICAL_TARGET_VALUES, CANONICAL_TARGET_OFFSETS,
#   CANONICAL_TARGET_SHAPES, CANONICAL_TARGET_IMAGE_IDS
#   CANONICAL_IMAGE_INPUT_PATHS, CANONICAL_IMAGE_TEST_INPUT_PATHS

for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():
    ...  # index the packed target accessor by row, never by filename
```

### CRITICAL RULES:
1. Row i of every canonical array describes CANONICAL_TRAIN_IDS[i]
2. Read model inputs from CANONICAL_IMAGE_INPUT_PATHS (train) and
   CANONICAL_IMAGE_TEST_INPUT_PATHS (test) - never re-scan directories
3. Predictions are packed in the same order and saved with the injected helper
"""


def _target_source_sections(
    target_source,
    paths: dict,
    component_type: str,
) -> list[str]:
    """Build the target-related prompt sections from the resolved decision."""
    if target_source is None:
        return []

    sections: list[str] = []
    if getattr(target_source, "canonical_authoritative", False):
        if component_type in ("model", "ensemble"):
            sections.append(
                _PACKED_CANONICAL_INSTRUCTIONS
                if getattr(target_source, "packed_image_contract", False)
                else _DENSE_CANONICAL_INSTRUCTIONS
            )
        auxiliary = _auxiliary_records(paths, target_source)
        if auxiliary:
            described = "\n".join(
                f"- `{record['path']}` (layout: {record['layout'] or 'unclassified'})"
                for record in auxiliary
            )
            sections.append(
                "## AUXILIARY PUBLIC ARTIFACTS (NOT TARGETS)\n\n"
                "These public files may carry useful features or metadata. The\n"
                "canonical contract already owns the targets and folds, so do NOT\n"
                "read a target from them and do NOT rebuild the row set from them.\n\n"
                f"{described}\n"
            )
        return sections

    if getattr(target_source, "mode", "none") == "sparse_preload":
        label_files = list(getattr(target_source, "label_files", ()))
        sections.append(
            """
## NON-STANDARD LABEL FILES (MANDATORY PARSING)

Verified sparse-label artifacts: """
            + ", ".join(str(lf) for lf in label_files)
            + """

Their sparse layout was verified by inspection before injection. Parse them
with the injected helper and fail clearly if the target role cannot be
validated. Never manufacture dummy targets.

Steps:
1. Use parse_label_file() helper (injected in code header)
2. Preserve the semantic record ID -> target mapping
3. Pivot to a binary matrix only if the observed artifact is truly multi-label
4. Match with training data BEFORE training

Generic variable-width example:
```python
targets_df = parse_label_file(LABEL_FILES[0])
assert {'record_id', 'target'}.issubset(targets_df.columns)
# Only for a verified multi-label target:
y_train = targets_df.pivot_table(
    index='record_id',
    columns='target',
    aggfunc='size',
    fill_value=0,
)
```
"""
        )
    return sections


def _auxiliary_records(paths: dict, target_source) -> list[dict]:
    """Neutral auxiliary artifact records for the current decision."""
    try:
        from ....agents.developer.target_source import auxiliary_public_artifacts
    except ImportError:  # pragma: no cover - defensive
        return []
    records = paths.get("public_artifacts")
    if not records:
        return []
    return list(auxiliary_public_artifacts({"public_artifacts": records}, target_source))


def _format_task(component, competition_info, paths: dict[str, str]) -> str:
    """Format the task specification section."""
    component_type = getattr(component, "component_type", "model")
    component_name = getattr(component, "name", "component")
    component_code = getattr(component, "code", "")

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
- Directories: `media/`, `train_images/`, `dataset_bundle/`
- Nested files: `dataset_bundle/train.csv`

Always check if the path is a file or directory before loading."""


def _get_component_guidance(component_type: str) -> str:
    """Get minimal, type-specific guidance."""
    from .component_guidance import COMPONENT_GUIDANCE

    # Handle domain-specific model types
    if component_type == "model":
        return COMPONENT_GUIDANCE.get(component_type, "")

    return COMPONENT_GUIDANCE.get(component_type, "")
