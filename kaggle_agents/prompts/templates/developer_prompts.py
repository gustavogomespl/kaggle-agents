"""
Prompt templates for the Developer Agent.

Refactored to be agentic, feedback-driven, and RL-friendly.
Inspired by Claude Code's concise style.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.config import is_metric_minimization

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
7. Print "Final Validation Performance: {score:.6f}" at the end (CRITICAL: Meta-Evaluator depends on this exact string)
8. Set random_state=42 everywhere for reproducibility
9. MANDATORY SOFT-DEADLINE PATTERN (prevents hard timeout kills):

   For sklearn/manual training loops:
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

   For Keras/TensorFlow model.fit() - MUST use callback:
   ```python
   import tensorflow as tf
   import time

   class DeadlineCallback(tf.keras.callbacks.Callback):
       '''Stops training when soft deadline is reached.'''
       def __init__(self, start_time, soft_deadline_s):
           super().__init__()
           self.start_time = start_time
           self.soft_deadline_s = soft_deadline_s

       def on_epoch_end(self, epoch, logs=None):
           if time.time() - self.start_time >= self.soft_deadline_s:
               print(f"[TIMEOUT] Soft deadline reached at epoch {epoch+1}, stopping training")
               self.model.stop_training = True

   # Setup deadline
   _START_TIME = time.time()
   _TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "3300"))
   _SOFT_DEADLINE_S = _TIMEOUT_S - 120  # Reserve 2min for saving

   # Use with model.fit():
   callbacks = [
       DeadlineCallback(_START_TIME, _SOFT_DEADLINE_S),
       EarlyStopping(patience=30, restore_best_weights=True),
       ModelCheckpoint("best_model.h5", save_best_only=True),
   ]
   model.fit(..., callbacks=callbacks)

   # ALWAYS print final metric even if stopped early
   print(f"Final Validation Performance: {val_loss:.6f}")
   ```

   For PyTorch manual training loops:
   ```python
   import time
   _START_TIME = time.time()
   _TIMEOUT_S = int(os.getenv("KAGGLE_AGENTS_COMPONENT_TIMEOUT_S", "3300"))
   _SOFT_DEADLINE_S = _TIMEOUT_S - 120

   for epoch in range(max_epochs):
       if time.time() - _START_TIME >= _SOFT_DEADLINE_S:
           print(f"[TIMEOUT] Soft deadline reached at epoch {epoch+1}")
           break
       # ... train epoch ...

   # ALWAYS print final metric
   print(f"Final Validation Performance: {best_val_loss:.6f}")
   ```

10. MODEL CHECKPOINTING FOR ENSEMBLE COMPATIBILITY:
    When saving model checkpoints that may be loaded by other components (e.g., ensemble):

    PREFER TorchScript (preserves architecture - no class definition needed to reload):
    ```python
    # ❌ WRONG - state_dict requires EXACT class definition to reload:
    torch.save(model.state_dict(), "model.pth")

    # ✅ CORRECT - TorchScript preserves full architecture:
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, "model.pt")

    # Loading in ensemble (no class needed):
    model = torch.jit.load("model.pt", map_location=device)
    model.eval()
    ```

    If state_dict MUST be used, ALSO save model config:
    ```python
    import json
    config = {"class_name": "DnCNN", "depth": 15, "channels": 64, "container_attr": "net"}
    with open("model_config.json", "w") as f:
        json.dump(config, f)
    ```

    WHY: Ensemble components cannot load state_dicts if the model class definition
    differs (e.g., self.net vs self.model, depth=15 vs depth=17). TorchScript avoids this.

## MUST NOT:
- sys.exit(), exit(), quit(), raise SystemExit, os._exit()
- try-except blocks that swallow errors (let them surface)
- early_stopping_rounds as direct fit() parameter (use callbacks)
- Subsample training data unless `KAGGLE_AGENTS_FAST_MODE=1` (FAST_MODE may subsample to meet budget, but keep determinism)
- `pin_memory=True` in DataLoader (causes warnings/crashes). USE `pin_memory=False`.
- `num_workers > 0` in DataLoader (safe default is 0 to avoid fork/spawn issues).
- Overwrite sample_submission.csv (always write to submission.csv)

## API Gotchas:
- OneHotEncoder: sparse_output=False (NOT sparse=False) for sklearn 1.2+
- pd.concat() instead of .append() for pandas 2.0+
- Optuna: set_verbosity(WARNING), n_trials <= 5, timeout=60 for validation
- LightGBM callbacks: lgb.early_stopping(100), not early_stopping_rounds param
- XGBoost callbacks: xgb.callback.EarlyStopping(rounds=100)
- Albumentations: `IAASharpen` is removed, use `Sharpen`. Ensure input is RGB (3 channels) for color transforms.


## PyTorch Gotchas:
- Dataset __getitem__ must return tensors/arrays (never None) so DataLoader can collate

## IMAGE DATA PIPELINE CRITICAL FIXES (MANDATORY FOR IMAGE TASKS):

### 1. VARIABLE IMAGE DIMENSIONS (stack error prevention):
Images in Kaggle datasets often have DIFFERENT sizes (e.g., 258x540 vs 420x540).
The default DataLoader collate_fn uses torch.stack() which FAILS on tensors of different sizes.

SOLUTIONS:
- **TRAINING**: Use `transforms.RandomCrop(256, 256)` or `transforms.Resize((256, 256))` to ensure all tensors have equal size
  ```python
  train_transform = transforms.Compose([
      transforms.RandomCrop(256, 256),  # Guarantees fixed size for batching
      transforms.ToTensor(),
  ])
  ```
- **VALIDATION/TEST**: Use `batch_size=1` to avoid stacking errors. Process one image at a time.
  ```python
  val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  ```
- **ALTERNATIVE**: Implement a custom collate_fn that pads images to the max size in the batch

### 2. NEGATIVE STRIDES (numpy/torch conversion error):
Using `np.flip()`, `np.rot90()`, or array slicing with negative steps creates arrays with negative strides.
PyTorch's `torch.from_numpy()` CANNOT handle negative strides.

ERROR: `ValueError: At least one stride in the given numpy array is negative`

FIX: ALWAYS call `.copy()` or `np.ascontiguousarray()` AFTER any numpy augmentation:
```python
def apply_augmentation(img: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        img = np.flip(img, axis=1)  # Creates negative stride!
    if random.random() > 0.5:
        img = np.rot90(img, k=random.randint(0, 3))  # Creates negative stride!
    return np.ascontiguousarray(img)  # MANDATORY: fixes strides before torch.from_numpy()
```

### 3. NO TRAIN.CSV FOR IMAGE-TO-IMAGE TASKS:
Many image-to-image competitions (denoising, super-resolution) do NOT have a train.csv file.
Data is stored in paired directories: `train/` (noisy) and `train_cleaned/` (clean targets).

DO NOT: `pd.read_csv('train.csv')`  # Will fail with FileNotFoundError

DO THIS INSTEAD:
```python
from pathlib import Path

train_dir = Path('/path/to/train')
clean_dir = Path('/path/to/train_cleaned')

# List files directly from directories
noisy_files = sorted(train_dir.glob('*.png'))
pairs = []
for noisy_path in noisy_files:
    clean_path = clean_dir / noisy_path.name
    if clean_path.exists():
        pairs.append((noisy_path, clean_path))

print(f"Found {len(pairs)} paired training samples.")
```

## IMAGE-TO-IMAGE / PIXEL-LEVEL TASKS (CRITICAL):
If domain is image_to_image, image_segmentation, or submission format is pixel_level:
1. Output must be a FULL IMAGE (same HxW as input), NOT a single value per image
2. Use encoder-decoder architectures (U-Net, autoencoder), NOT classifiers
3. NEVER use image classifiers (EfficientNet, ResNet with FC head) for these tasks
4. NEVER use global average pooling followed by dense layers
5. Submission must be FLATTENED to pixel-level format:
   - Read sample_submission.csv to get exact ID format and row count
   - ID format is typically: '{image_id}_{row}_{col}' or '{image_id}_{pixel_index}'
   - MUST match EXACT number of rows in sample_submission (often millions of rows)
6. Example flattening code (CRITICAL - use this pattern):
   ```python
   sample_sub = pd.read_csv(sample_submission_path)
   expected_rows = len(sample_sub)

   submission_rows = []
   for img_path in sorted(test_images):  # MUST be sorted for consistent order
       img_id = img_path.stem  # e.g., "1" from "1.png"
       pred = model(preprocess(img))  # Output: HxW image, NOT a single value
       H, W = pred.shape
       for row in range(H):
           for col in range(W):
               pixel_id = f"{img_id}_{row+1}_{col+1}"  # 1-indexed to match sample
               submission_rows.append({"id": pixel_id, "value": float(pred[row, col])})

   assert len(submission_rows) == expected_rows, f"Expected {expected_rows} rows, got {len(submission_rows)}"
   pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
   ```
7. Verify submission shape BEFORE saving: if row count doesn't match sample_submission, your model architecture is WRONG

## SUBMISSION FORMAT CRITICAL FIX (MANDATORY FOR ALL TASKS):

### Problem: Predictions going to WRONG COLUMN (causes score=0.5)
The target column is NOT always `sample_sub.columns[1]`!
Example: "detecting-insults" has columns [Insult, Date, Comment] where Insult (column 0) is the target.
If you put predictions in columns[1] (Date), you get score=0.5 even with perfect CV!

### MANDATORY PATTERN - Identify target column correctly:
```python
# 1. Load sample submission and INSPECT it
sample_sub = pd.read_csv(sample_submission_path)
print("Sample submission columns:", sample_sub.columns.tolist())
print("Sample submission head:\n", sample_sub.head(2))

# 2. CRITICAL: Identify the TARGET column (DO NOT assume columns[1]!)
# The target column has the SAME NAME as the target in train.csv
# Common names: 'target', 'Insult', 'label', 'prediction', 'value', etc.
train_df = pd.read_csv(train_path)
target_col = train_df.columns[0]  # Usually the first column in train is the target

# Verify target_col exists in sample_sub
if target_col not in sample_sub.columns:
    # Fallback: if only 2 columns, use columns[1]; otherwise find numeric column
    if len(sample_sub.columns) == 2:
        target_col = sample_sub.columns[1]
    else:
        # Look for common target names
        for col in ['target', 'label', 'prediction', 'value', 'Insult']:
            if col in sample_sub.columns:
                target_col = col
                break

print(f"Using target column: {target_col}")

# 3. Fill ONLY the target column with predictions (preserve all other columns!)
sample_sub[target_col] = predictions  # predictions must be in same order as sample_sub rows

# 4. VALIDATE before saving
print(f"Predictions sample: {sample_sub[target_col].head()}")
assert len(sample_sub) == len(predictions), f"Row count mismatch: {len(sample_sub)} vs {len(predictions)}"

sample_sub.to_csv('submission.csv', index=False)
```

### CRITICAL CHECKS:
1. Print sample_sub.columns to see ALL column names before assuming anything
2. The target column name MUST match the target from train.csv
3. DO NOT overwrite non-target columns (like Date, Comment, ID, etc.)
4. If sample_sub has 3+ columns, columns[1] is probably NOT the target!

## TABULAR MODELS REQUIRE TABULAR FEATURES (MANDATORY):

LightGBM, XGBoost, CatBoost, and other tree-based models need REAL tabular features.
If train.csv only has [id, label] columns → This is an IMAGE competition!

CRITICAL: DO NOT create dummy/random features - this gives terrible scores!

```python
# CHECK BEFORE USING TABULAR MODELS:
train_df = pd.read_csv('train.csv')
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Number of columns: {len(train_df.columns)}")

if len(train_df.columns) <= 2:
    raise ValueError(
        "No tabular features found! train.csv has only id+label columns. "
        "This appears to be an IMAGE competition. "
        "Use CNN models (EfficientNet, ResNet) with transfer learning, "
        "NOT tree models (LightGBM, XGBoost, CatBoost)."
    )
```

### Example: dog-breed-identification
- train.csv has only [id, breed] columns - NO tabular features!
- WRONG: LightGBM with dummy random features → score 4.78 (terrible)
- RIGHT: EfficientNet-B0 with transfer learning → score < 1.0

### Model Selection Guide:
- IMAGE competition (train/ has images, train.csv has only id+label):
  → Use: EfficientNet, ResNet, VGG with ImageNet pretrained weights
  → DO NOT use: LightGBM, XGBoost, CatBoost (they need tabular features!)

- TABULAR competition (train.csv has many feature columns):
  → Use: LightGBM, XGBoost, CatBoost with hyperparameter tuning

NEVER create dummy/random features to feed into tree models!"""


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
    # Adaptive training fields
    epoch_budget: int = 300  # Maximum epochs for current iteration (SOTA uses 600)
    timeout_occurred: bool = False  # Whether timeout occurred in last attempt
    suggested_epochs: int = 300  # Suggested epochs based on timeout history
    early_stopping_patience: int = 30  # SOTA uses patience=30
    # Submission validation retry
    submission_validation_error: Optional[str] = None  # Error from last invalid submission
    # DPO: Preference pairs for contrastive learning
    dpo_examples: str = ""  # Formatted DPO pairs (good vs bad code examples)


def build_context(state: dict[str, Any], component: Optional[Any] = None) -> DynamicContext:
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
    # Get timeout configuration
    timeout_val = state.get("timeout_per_component")
    if isinstance(timeout_val, str):
        try:
            timeout_val = int(timeout_val)
        except ValueError:
            timeout_val = None
    context.timeout_per_component = timeout_val if isinstance(timeout_val, int) else None

    # Adaptive training: detect epoch budget, patience, and timeout history
    context.epoch_budget = int(state.get("epoch_budget", 600))  # SOTA uses 600
    context.early_stopping_patience = int(state.get("early_stopping_patience", 30))  # SOTA uses 30
    min_epochs = int(os.getenv("KAGGLE_AGENTS_MIN_EPOCHS", "5"))

    # Check if timeout occurred in last execution
    dev_results = state.get("development_results", [])
    if dev_results:
        last_result = dev_results[-1]
        last_stdout = str(getattr(last_result, "stdout", "") or "").lower()
        last_stderr = str(getattr(last_result, "stderr", "") or "").lower()
        last_exec_time = getattr(last_result, "execution_time", 0) or 0

        # Detect timeout via multiple signals
        timeout_component = context.timeout_per_component or 3600
        context.timeout_occurred = (
            "timeout" in last_stderr
            or "deadline" in last_stdout
            or "[timeout]" in last_stdout
            or last_exec_time >= timeout_component * 0.95
        )

    # Calculate suggested epochs (reduce 50% if timeout occurred)
    if context.timeout_occurred:
        reduction_factor = float(os.getenv("KAGGLE_AGENTS_EPOCH_REDUCTION", "0.5"))
        context.suggested_epochs = max(min_epochs, int(context.epoch_budget * reduction_factor))
    else:
        context.suggested_epochs = context.epoch_budget

    # fast_mode only activates when epochs are very low
    context.fast_mode = (
        context.suggested_epochs <= min_epochs
        or str(os.getenv("KAGGLE_AGENTS_FAST_MODE", "")).lower() in {"1", "true", "yes"}
        or str(os.getenv("FAST_MODE", "")).lower() in {"1", "true", "yes"}
    )

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

    # Extract submission validation error for retry context
    context.submission_validation_error = state.get("submission_validation_error")

    # DPO: Extract preference pairs for contrastive learning
    preference_pairs = state.get("preference_pairs", [])
    if preference_pairs:
        context.dpo_examples = _format_dpo_for_prompt(preference_pairs, component)

    return context


def _format_dpo_for_prompt(pairs: list, component: Optional[Any] = None) -> str:
    """
    Format DPO preference pairs as contrastive examples for prompts.

    Shows what code patterns succeeded vs failed, helping the model
    learn from past mistakes and successes.

    Args:
        pairs: List of PreferencePair objects
        component: Optional component to filter relevant pairs

    Returns:
        Formatted string with contrastive examples
    """
    if not pairs:
        return ""

    # Filter pairs by component type if available
    component_type = getattr(component, "component_type", None) if component else None
    if component_type:
        relevant_pairs = [p for p in pairs if getattr(p, "component_type", "") == component_type]
        if not relevant_pairs:
            relevant_pairs = pairs  # Fall back to all pairs
    else:
        relevant_pairs = pairs

    # Sort by margin (most informative examples first)
    def get_margin(p):
        return getattr(p, "margin", 0.0)

    sorted_pairs = sorted(relevant_pairs, key=get_margin, reverse=True)

    # Take top 3 most informative pairs
    selected = sorted_pairs[:3]

    lines = ["## DPO: Learned Code Preferences (from past fixes)\n"]
    lines.append("Learn from these successful fixes - avoid the rejected patterns, follow the chosen patterns:\n")

    for i, pair in enumerate(selected, 1):
        context_desc = getattr(pair, "context", "Code fix")[:50]
        margin = getattr(pair, "margin", 0.0)

        # Get code snippets (truncated for prompt efficiency)
        rejected = getattr(pair, "rejected", "")
        chosen = getattr(pair, "chosen", "")

        # Extract key differences (first 150 chars of each)
        rejected_snippet = rejected[:150].strip()
        chosen_snippet = chosen[:150].strip()

        if rejected_snippet and chosen_snippet:
            lines.append(f"### Example {i}: {context_desc}")
            lines.append(f"**Improvement margin:** {margin:.2f}")
            lines.append("")
            lines.append("**❌ AVOID (this pattern failed):**")
            lines.append(f"```python")
            lines.append(rejected_snippet + "...")
            lines.append("```")
            lines.append("")
            lines.append("**✅ PREFER (this pattern succeeded):**")
            lines.append(f"```python")
            lines.append(chosen_snippet + "...")
            lines.append("```")
            lines.append("")

    if len(lines) > 2:  # More than just header
        lines.append("**INSTRUCTION**: When implementing similar code, follow the preferred patterns above.")
        return "\n".join(lines)

    return ""


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
            from .constraints import get_constraints_for_domain
            # Handle None domain by defaulting to "tabular"
            domain = getattr(competition_info, "domain", None) or "tabular"
            constraints = get_constraints_for_domain(domain)
            print(f"   📦 Loaded modular constraints for domain: {domain}")
        except Exception:
            constraints = HARD_CONSTRAINTS  # Fallback to full constraints
    else:
        constraints = HARD_CONSTRAINTS

    parts = [
        DEVELOPER_CORE_IDENTITY,
        "",
        constraints,
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

    # Submission validation error (must be fixed immediately).
    if context.submission_validation_error:
        parts.append("")
        parts.append("## CRITICAL: SUBMISSION FORMAT ERROR (MUST FIX)")
        parts.append(f"Previous submission failed validation: {context.submission_validation_error}")
        parts.append("")
        parts.append("Fix requirements:")
        parts.append("1. Read sample_submission.csv to match ID values and column order exactly")
        parts.append("2. Match row count exactly (no truncation/padding)")
        parts.append("3. Preserve ID order from sample_submission.csv")
        parts.append("4. For image-to-image: flatten per-pixel predictions to the sample submission ID format")
        parts.append("5. Use assertions before saving")
        parts.append("```python")
        parts.append("sample = pd.read_csv(sample_submission_path)")
        parts.append("assert list(submission.columns) == list(sample.columns)")
        parts.append("assert len(submission) == len(sample)")
        parts.append("assert (submission[sample.columns[0]].values == sample[sample.columns[0]].values).all()")
        parts.append("```")

    # Adaptive training guidance (GPU-accelerated, reduces epochs if timeout)
    if context.run_mode.lower() == "mlebench" or "medal" in context.objective.lower():
        parts.append("")
        parts.append("## NEURAL NETWORK TRAINING (GPU-ACCELERATED)")
        parts.append(f"- **EPOCHS**: Train for up to {context.suggested_epochs} epochs with early stopping")
        parts.append("- **GPU**: MUST use CUDA if available: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`")
        parts.append("- **BACKBONE**: Full fine-tuning for maximum performance (do NOT freeze layers)")
        parts.append("- **LEARNING RATE**: Use warmup (5% of epochs) + cosine annealing schedule")
        parts.append("- **AUGMENTATION**: Apply heavy augmentation (Cutmix, Mixup, RandAugment)")
        parts.append(f"- **EARLY STOPPING**: Stop if validation loss doesn't improve for {context.early_stopping_patience} epochs (SOTA uses patience=30)")
        parts.append("- **CHECKPOINTING**: Save best model checkpoint by validation metric")
        parts.append("- **MIXED PRECISION**: Use torch.cuda.amp.autocast() for faster training")

        if context.timeout_occurred:
            parts.append("")
            parts.append("⚠️ TIMEOUT DETECTED IN PREVIOUS RUN - ADJUSTMENTS:")
            parts.append(f"- REDUCED epochs from {context.epoch_budget} to {context.suggested_epochs}")
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
- IMPLEMENT soft-deadline pattern (see HARD_CONSTRAINTS #9)
  - For Keras/TensorFlow: MUST use DeadlineCallback in model.fit() callbacks
  - For sklearn/PyTorch manual loops: check _check_deadline() inside fold loop
- Train model with StratifiedKFold CV using n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5"))
- Save OOF predictions to models/oof_{name}.npy for stacking
- Handle class imbalance if ratio > 2:1 (class_weight or scale_pos_weight)
- Print per-fold scores: [LOG:FOLD] fold={n} score={s:.6f}
- Use GPU if available (check torch.cuda.is_available())
- Create submission.csv with probabilities [0,1]
- SUBMISSION FORMAT: Target column is NOT always columns[1]!
  ```python
  sample_sub = pd.read_csv(sample_submission_path)
  print("Columns:", sample_sub.columns.tolist())  # ALWAYS check column names!
  # Use SAME target column name as in train.csv (e.g., 'Insult', 'target', 'label')
  target_col = train_df.columns[0]  # First column of train.csv is usually the target
  sample_sub[target_col] = predictions  # Fill ONLY the target column
  sample_sub.to_csv('submission.csv', index=False)
  ```
- ALWAYS print "Final Validation Performance: {score}" even if stopped early due to deadline
- SAVE PyTorch checkpoints with TorchScript for ensemble compatibility (see HARD_CONSTRAINTS #10):
  ```python
  scripted_model = torch.jit.script(model)
  torch.jit.save(scripted_model, f"models/{component_name}_fold{fold_idx}.pt")
  ```""",

        "feature_engineering": """## Feature Engineering Requirements
- Transform train and test consistently
- NO model training in this component
- Save to train_engineered.csv, test_engineered.csv if creating new files
- Fast execution (<30 seconds)
- Print "Final Validation Performance: 1.0" on completion""",

        "ensemble": """## Ensemble Requirements

### LOADING PREVIOUS MODELS (CRITICAL - READ CAREFULLY):
1. **TorchScript Loading** (PREFERRED - no class definition needed):
   ```python
   model = torch.jit.load(checkpoint_path, map_location=device)
   model.eval()
   ```

2. **State Dict Fallback** (ONLY if TorchScript fails):
   - You MUST define the EXACT same model class as used in training
   - Inspect checkpoint keys to determine architecture:
   ```python
   state_dict = torch.load(path, map_location=device)
   # Look at key names: "net.0.weight" means self.net, NOT self.model
   # Look at number of layers to infer depth
   print([k for k in state_dict.keys()][:10])
   ```

### COMMON PITFALLS (WILL CAUSE state_dict LOADING TO FAIL):
- ❌ Defining model with `self.model` when checkpoint uses `self.net`
- ❌ Using different depth/channels than training component
- ❌ Missing dropout layers that exist in original
- ❌ Different layer ordering or architecture

### OOF-Based Stacking (no checkpoint loading needed):
- Load OOF predictions from models/oof_*.npy files
- Preferred: Stacking with LogisticRegression/Ridge meta-learner
- Fallback: Weighted average if OOF files missing
- Can use correlation analysis to select diverse models

### Final Output:
- Create submission.csv with final ensemble predictions
- Print "Final Validation Performance: {score}" at the end""",

        "preprocessing": """## Preprocessing Requirements
- Clean data, handle missing values, encode categoricals
- NO model training
- Fast execution (<10 seconds)
- Save processed data for subsequent components
- Print "Final Validation Performance: 1.0" on completion""",

        "image_to_image_model": """## Image-to-Image Model Requirements (CRITICAL)
This is a PIXEL-LEVEL prediction task. Your model must output FULL IMAGES, not single values.

### DATA PIPELINE FIXES (MANDATORY - PREVENTS COMMON CRASHES):

1. **VARIABLE IMAGE DIMENSIONS** (torch.stack error):
   Images often have different sizes. Use these solutions:
   ```python
   # TRAINING: Use RandomCrop for consistent tensor sizes
   train_transform = transforms.Compose([
       transforms.RandomCrop(256, 256),  # Fixed size for batching
       transforms.ToTensor(),
   ])
   train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # batch_size > 1 OK

   # VALIDATION/TEST: Use batch_size=1 to handle any size
   val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
   test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
   ```

2. **NEGATIVE STRIDES** (numpy/torch conversion error):
   ```python
   # WRONG - creates negative strides:
   img = np.flip(img, axis=1)
   tensor = torch.from_numpy(img)  # CRASHES!

   # CORRECT - fix strides after augmentation:
   img = np.flip(img, axis=1)
   img = np.ascontiguousarray(img)  # FIX STRIDES
   tensor = torch.from_numpy(img)   # Now works!
   ```

3. **NO TRAIN.CSV** (FileNotFoundError):
   Many image-to-image competitions have NO CSV. Load from directories:
   ```python
   # DO NOT: pd.read_csv('train.csv')

   # DO THIS:
   train_dir = Path('/path/to/train')
   clean_dir = Path('/path/to/train_cleaned')
   noisy_files = sorted(train_dir.glob('*.png'))
   pairs = [(nf, clean_dir / nf.name) for nf in noisy_files if (clean_dir / nf.name).exists()]
   print(f"Found {len(pairs)} paired training samples.")
   ```

### Architecture (MUST USE):
- U-Net: encoder-decoder with skip connections
- Autoencoder: encoder-decoder without skip connections
- DnCNN: deep CNN with residual learning
- Fully Convolutional Network (FCN)

### Architecture (DO NOT USE):
- EfficientNet, ResNet, VGG with classification head
- Any model with global average pooling + dense layers
- Any model that outputs a single value per image

### Model Output:
- Input: Image of shape (H, W, C) or (H, W)
- Output: Image of shape (H, W, C) or (H, W) - SAME spatial dimensions
- Loss: MSE, L1, SSIM, or perceptual loss

### Submission Format (CRITICAL):
Sample submission has MILLIONS of rows (one per pixel), NOT one per image!

```python
# CORRECT pattern for pixel-level submission:
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)  # e.g., 5,789,880 rows

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem
    pred = model(load_image(img_path))  # OUTPUT: (H, W) image
    H, W = pred.shape

    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"  # Match sample format
            submission_rows.append({"id": pixel_id, "value": pred[row, col]})

# VERIFY before saving
assert len(submission_rows) == expected_rows, f"WRONG: {len(submission_rows)} vs {expected_rows}"
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
```

### Common Mistake to Avoid:
If your submission has ~29 rows instead of ~5.8M rows, you used a CLASSIFIER instead of an encoder-decoder.
The number of rows = number of test images means WRONG architecture.""",
    }

    # Handle domain-specific model types
    if component_type == "model":
        # Check if we have context suggesting image-to-image
        return guidance.get(component_type, "")

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

## CRITICAL REQUIREMENTS (DO NOT REMOVE):
1. MUST preserve `print(f"Final Validation Performance: {{score:.6f}}")` - Meta-Evaluator depends on this exact string
2. MUST preserve soft-deadline pattern with `_check_deadline()` calls
3. MUST keep all OOF prediction saving (np.save)

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

## CRITICAL REQUIREMENTS (DO NOT REMOVE):
1. MUST preserve `print(f"Final Validation Performance: {{score:.6f}}")` - Meta-Evaluator depends on this exact string
2. MUST preserve soft-deadline pattern with `_check_deadline()` calls
3. MUST keep all OOF prediction saving (np.save)

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


# ==================== Dynamic Instructions Builder ====================


def build_budget_instructions(timeout_hint: int | None) -> list[str]:
    """Build time budget instructions."""
    if not isinstance(timeout_hint, (int, float)):
        return []

    return [
        "\n⏱️ TIME BUDGET (CRITICAL):",
        f"  - Component must complete within ~{int(timeout_hint)}s (env: KAGGLE_AGENTS_COMPONENT_TIMEOUT_S).",
        "  - Implement a soft-deadline (e.g., budget-45s): if exceeded, stop training, save best artifacts, and still print the final metric line.",
        "  - Read env vars: KAGGLE_AGENTS_FAST_MODE and KAGGLE_AGENTS_CV_FOLDS to reduce compute when needed.",
    ]


def build_mlebench_objective_instructions() -> list[str]:
    """Build MLE-bench objective instructions."""
    return [
        "\n🏁 MLE-BENCH OBJECTIVE:",
        "  - Optimize for MLE-bench medal: prioritize fast end-to-end runtime + robust valid submission.",
        "  - Prefer cheaper training (fewer folds/epochs) and inference-time tricks (TTA) over expensive CV sweeps.",
    ]


def build_iteration_context(current_iteration: int, refinement_guidance: dict) -> list[str]:
    """Build iteration context instructions."""
    instructions = []

    if current_iteration > 0:
        instructions.append(f"\n⚡ REFINEMENT ITERATION {current_iteration}")
        instructions.append("Focus on improvements that address previous shortcomings.")

    if refinement_guidance and refinement_guidance.get("developer_guidance"):
        instructions.append("\nMETA-EVALUATOR GUIDANCE:")
        instructions.append(f"  {refinement_guidance['developer_guidance']}")

    if refinement_guidance and refinement_guidance.get("priority_fixes"):
        instructions.append("\nAVOID THESE ERROR PATTERNS:")
        for error in refinement_guidance["priority_fixes"][:3]:
            instructions.append(f"  - {error}")

    return instructions


def build_previous_results_context(dev_results: list) -> list[str]:
    """Build context from previous development results."""
    if not dev_results:
        return []

    instructions = []
    successful_components = [r for r in dev_results if r.success]
    failed_components = [r for r in dev_results if not r.success]

    if successful_components:
        instructions.append("\n✅ SUCCESSFUL PATTERNS FROM PREVIOUS COMPONENTS:")
        for result in successful_components[-2:]:
            if "LightGBM" in result.code:
                instructions.append("  - LightGBM implementation worked well")
            if "StratifiedKFold" in result.code:
                instructions.append("  - StratifiedKFold cross-validation successful")
            if "predict_proba" in result.code:
                instructions.append("  - predict_proba() for probabilities confirmed working")

    if failed_components:
        instructions.append("\nAVOID THESE ERRORS FROM PREVIOUS ATTEMPTS:")
        for result in failed_components[-2:]:
            if result.errors:
                error_msg = result.errors[0][:300]
                instructions.append(f"  - {error_msg}")

    return instructions


def build_performance_gap_instructions(
    current_score: float,
    target_score: float | None,
    metric_name: str,
) -> list[str]:
    """Build performance gap instructions."""
    if current_score <= 0 or target_score is None:
        return []

    minimize = is_metric_minimization(metric_name)
    gap = (float(current_score) - float(target_score)) if minimize else (float(target_score) - float(current_score))
    if gap <= 0:
        return []
    instructions = [
        f"\nPERFORMANCE GAP: {gap:.4f} to reach target ({float(target_score):.4f}, {'minimize' if minimize else 'maximize'})"
    ]

    if gap < 0.01:
        instructions.append("  - Small gap: Focus on fine-tuning hyperparameters")
    elif gap < 0.05:
        instructions.append("  - Medium gap: Consider feature engineering or ensemble methods")
    else:
        instructions.append("  - Large gap: May need different model architecture or approach")

    return instructions


def build_cv_instructions(working_dir: str, component_name: str) -> list[str]:
    """Build cross-validation instructions."""
    return [
        "\n🔄 CONSISTENT CROSS-VALIDATION (CRITICAL):",
        f"  - Check if '{working_dir}/folds.csv' exists.",
        "  - IF EXISTS: Load it and use the 'fold' column for splitting.",
        "    ```python",
        "    folds = pd.read_csv('folds.csv')",
        "    # Assuming X is aligned with folds (reset_index if needed)",
        "    for fold in sorted(folds['fold'].unique()):",
        "        val_idx = folds[folds['fold'] == fold].index",
        "        train_idx = folds[folds['fold'] != fold].index",
        "        # ... train/val split ...",
        "    ```",
        "  - IF NOT EXISTS: Use StratifiedKFold(n_splits=int(os.getenv('KAGGLE_AGENTS_CV_FOLDS','5')), shuffle=True, random_state=42)",
        f"  - CRITICAL: MUST save Out-of-Fold (OOF) predictions during CV to models/oof_{component_name}.npy",
        "  - OOF predictions enable proper stacking ensemble (meta-model trained on OOF)",
        "  - MUST print 'Final Validation Performance: {score}'",
        "  - If metric is NaN/inf, replace with 0.0 before printing Final Validation Performance",
        "  - MUST handle class imbalance with class_weight='balanced'",
    ]


def build_stacking_oof_instructions(working_dir: str, component_name: str) -> list[str]:
    """Build stacking/OOF instructions."""
    return [
        "\nSTACKING & OOF REQUIREMENTS (CRITICAL):",
        "  1. Initialize `oof_preds` array of zeros with length of train set.",
        "  2. Initialize `test_preds` array of zeros with length of test set.",
        "  3. During CV loop:",
        "     - Fill `oof_preds[val_idx]` with predictions for validation fold.",
        "     - Predict on test set and accumulate: `test_preds += model.predict_proba(X_test)[:, 1] / n_folds`",
        f"  4. Save OOF predictions: `np.save(str(Path('{working_dir}') / 'models' / 'oof_{component_name}.npy'), oof_preds)`",
        f"  5. Save Test predictions: `np.save(str(Path('{working_dir}') / 'models' / 'test_{component_name}.npy'), test_preds)`",
        "  6. Ensemble will ONLY run if BOTH oof_{name}.npy AND test_{name}.npy exist for at least 2 models.",
        "  7. This enables the Ensemble Agent to use Stacking later.",
    ]


def build_optuna_tuning_instructions(n_trials: int = 5, timeout: int = 540) -> list[str]:
    """Build Optuna hyperparameter tuning instructions."""
    return [
        "\nHYPERPARAMETER OPTIMIZATION (OPTUNA) REQUIRED:",
        "  - MUST use 'optuna' library for hyperparameter search",
        f"  - Run AT MOST {n_trials} trials (n_trials={n_trials}) and timeout={timeout}s to prevent timeouts",
        "  - CRITICAL: Check if 'optuna-integration' is available with try/except:",
        "    try:",
        "        from optuna.integration import OptunaSearchCV",
        "    except ImportError:",
        "        # Use manual Optuna with study.optimize() instead",
        "  - If optuna-integration is missing, use manual Optuna tuning with study.optimize()",
        "  - Use 'TPESampler' for efficient sampling",
        "  - CRITICAL: Do NOT pass 'callbacks' or 'early_stopping_rounds' to .fit() for XGBoost/LightGBM/CatBoost sklearn API; use fixed n_estimators",
        "  - Optimize for the competition metric (minimize RMSE/LogLoss or maximize AUC/Accuracy)",
        "  - Print the best parameters found",
        "  - Train final model with best parameters",
        "\n⚡ SPEED OPTIMIZATION (CRITICAL TO AVOID TIMEOUT):",
        "  - **SUBSAMPLE FOR TUNING**: If train dataset > 10,000 rows:",
        "    1. Create tuning subset with train_test_split",
        "    2. For CLASSIFICATION only: pass stratify=y when sampling (y discrete: y.nunique() < 20 or dtype category/object)",
        "    3. For REGRESSION (continuous y): DO NOT use stratify parameter",
        "    4. Run Optuna study on 25% sample (reduce to 15% if memory errors occur)",
        "    5. After finding best_params, retrain on FULL dataset",
        "  - **REDUCE ESTIMATORS DURING TUNING**:",
        "    - Inside objective(): Use n_estimators=150-200 (fast convergence)",
        "    - Final model: Use n_estimators=1000 with early_stopping_rounds=50 (if supported)",
        "  - **TIMEOUT BUDGET**: Set study.optimize(n_trials=5, timeout=600) for max 10 min tuning",
        "  - **MEMORY SAFETY (PREVENT OOM CRASHES)**:",
        "    - ALWAYS set n_jobs=1 in model __init__ (LGBMClassifier, XGBClassifier, etc.)",
        "    - ALWAYS set n_jobs=1 in cross_val_score (avoid nested parallelism → memory explosion)",
        "    - Add 'import gc; gc.collect()' inside objective() after computing score",
        "    - Delete model object explicitly: 'del model' before gc.collect()",
        "    - If memory errors persist, reduce train_size from 0.25 → 0.15 (15% of data)",
        "  - **ROBUST TRIALS**: Wrap objective logic in try/except; on exception log and return 0.0 so trials finish",
        "  - **NO-COMPLETION GUARD**: After study.optimize, if NO trials completed, fall back to safe default params instead of study.best_params",
    ]


def build_feature_engineering_instructions() -> list[str]:
    """Build feature engineering instructions."""
    return [
        "\n🔧 FEATURE ENGINEERING REQUIREMENTS:",
        "  - Create NEW features from existing ones",
        "  - IMPLEMENT SOTA TECHNIQUES:",
        "    - Target Encoding: MUST be done inside Cross-Validation (fit on train folds, transform val fold) to prevent leakage.",
        "    - Frequency Encoding: Map categorical features to their frequency/count.",
        "    - Aggregations: Mean/Count of numeric features grouped by categorical features.",
        "  - Save engineered features to file for model components",
        "  - NO model training in this component",
        "  - Print feature importance or correlation metrics",
        "\nFEATURE SELECTION (CRITICAL):",
        "  - After creating new features, perform selection to remove noise:",
        "  1. Train a quick LightGBM/XGBoost on the new feature set.",
        "  2. Calculate feature importance (gain/split).",
        "  3. Drop features with 0 importance or very low importance (< 1e-4).",
        "  4. Save ONLY the selected features to 'train_engineered.csv' and 'test_engineered.csv'.",
        "  5. Print list of dropped features.",
    ]


def build_ensemble_instructions(target_col: str = "target") -> list[str]:
    """Build ensemble instructions."""
    return [
        "\nENSEMBLE REQUIREMENTS:",
        "  - Combine predictions from multiple models",
        "  - PREFERRED STRATEGY: Stacking Ensemble (best performance)",
        "    - Load OOF predictions from models/oof_*.npy files",
        "    - Stack OOF predictions: oof_stack = np.column_stack([oof1, oof2, ...])",
        "    - Train meta-model (LogisticRegression/Ridge) on stacked OOF",
        "    - Load test predictions from each model and stack them",
        "    - Use meta-model to predict on stacked test predictions",
        "  - FALLBACK: Weighted average if OOF files missing",
        "    - Load submission files from each model",
        "    - Combine with weights: final = w1*pred1 + w2*pred2 + ...",
        "  - Generate final submission.csv",
        f"  - CRITICAL: Use target_col from dataset info (target_col='{target_col}' if available)",
        "  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target' or 'prediction')",
        "  - Print which models were used and their contribution/weights",
    ]


def build_standard_requirements() -> list[str]:
    """Build standard requirements."""
    return [
        "\nSTANDARD REQUIREMENTS:",
        "  - Save models to models/ directory",
        "  - Print progress and metrics throughout execution",
        "  - NO sys.exit() or exit() calls",
        "  - CRITICAL: Do NOT use deprecated 'pandas.append()'. Use 'pd.concat()' instead.",
        "  - Complete, executable single-file Python program",
    ]


def build_image_model_instructions(is_image_to_image: bool, data_files: dict | None, suggested_epochs: int = 600, early_stopping_patience: int = 60) -> list[str]:
    """Build image model instructions with adaptive epoch budget and patience."""
    instructions = [
        "\n🖼️ IMAGE MODELLING (DEEP TRAINING - SOTA Pattern):",
        f"  - Train for up to {suggested_epochs} epochs with early stopping (patience={early_stopping_patience})",
        "  - MUST use GPU: device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "  - Use mixed precision: torch.cuda.amp.autocast() for 2x faster training",
        "  - Full backbone fine-tuning for maximum performance (do NOT freeze layers)",
        "  - Learning rate schedule: warmup for 5% of epochs, then cosine decay to 1e-6",
        "  - Use pretrained backbone (torchvision/timm) - efficientnet_b0 or resnet50 recommended",
        "  - For Keras: use DeadlineCallback in model.fit() callbacks (see HARD_CONSTRAINTS)",
        "  - Do NOT run full test inference inside each fold; run once after training best checkpoint",
        "  - Save checkpoint every epoch, keep best by validation metric",
    ]

    if suggested_epochs < 20:
        instructions.append(f"  - ⚠️ Reduced epochs ({suggested_epochs}) due to previous timeout")
        instructions.append("  - Consider using smaller model (efficientnet_b0) or lower resolution if still slow")

    if isinstance(data_files, dict) and data_files.get("train_csv"):
        instructions.append(f"  - Labels are in Train CSV at: {data_files['train_csv']} (not inside train/)")

    if is_image_to_image:
        clean_path = ""
        if isinstance(data_files, dict):
            clean_path = data_files.get("clean_train", "") or ""

        instructions.extend([
            "\n🧽 IMAGE-TO-IMAGE (PIXEL-LEVEL) REQUIREMENTS:",
            "  - MUST learn noisy->clean mapping. Use train/ as noisy inputs and clean targets from clean_train.",
            "  - If using a pretrained backbone, use it ONLY as an encoder; discard classification heads.",
        ])

        if clean_path:
            instructions.append(f"  - Clean target dir (paired with train/): {clean_path}")

        instructions.append("  - Output full-resolution (or resized) images, then flatten to pixel-level CSV using sample_submission IDs.")

        # Add critical data pipeline fixes for image-to-image
        instructions.extend([
            "\n⚠️ DATA PIPELINE REQUIREMENTS (CRITICAL - SEE HARD_CONSTRAINTS):",
            "  - **VARIABLE DIMENSIONS**: Images may have different sizes (e.g., 258x540 vs 420x540)",
            "    - TRAINING: Use `transforms.RandomCrop(256, 256)` or `transforms.Resize((256, 256))` for consistent tensor sizes",
            "    - VALIDATION/TEST: Use `batch_size=1` in DataLoader to avoid torch.stack() errors",
            "  - **NEGATIVE STRIDES**: Call `np.ascontiguousarray()` or `.copy()` after `np.flip()`/`np.rot90()` augmentations",
            "  - **NO TRAIN.CSV**: Load image pairs directly from directories with `glob`/`pathlib`, NOT from CSV files:",
            "    ```python",
            "    noisy_files = sorted(train_dir.glob('*.png'))",
            "    pairs = [(nf, clean_dir / nf.name) for nf in noisy_files if (clean_dir / nf.name).exists()]",
            "    ```",
            "  - **LARGER BATCH FOR TRAINING**: Once dimensions are fixed with RandomCrop, use batch_size=16 or 32 for faster training",
        ])

    return instructions


def build_model_component_instructions(
    component,
    state: dict,
    working_dir: str,
    is_image: bool,
    is_image_to_image: bool,
    is_classification: bool,
    sample_integer_labels: bool,
    target_col: str = "target",
    suggested_epochs: int = 600,
    early_stopping_patience: int = 30,
) -> list[str]:
    """Build model component instructions with adaptive epoch budget and patience (SOTA pattern)."""
    instructions = [
        "\nMODEL COMPONENT REQUIREMENTS:",
        "  - MUST train a model and generate predictions",
    ]

    if is_image_to_image:
        instructions.extend([
            "  - MUST train on (noisy -> clean) image pairs and output FULL images (H x W), NOT a single scalar.",
            "  - MUST write pixel-level submission.csv matching sample_submission (id format: image_row_col).",
            "  - Use an encoder-decoder (U-Net/autoencoder). DO NOT use a classifier head or global pooling.",
        ])
    elif is_classification:
        if sample_integer_labels:
            instructions.append("  - MUST create submission.csv with integer class labels (0..K-1) matching sample_submission")
        else:
            instructions.append("  - MUST create submission.csv with probability predictions (0.0-1.0)")
    else:
        instructions.append("  - MUST create submission.csv with numeric predictions (regression)")

    instructions.append("  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target')")

    if not is_image:
        instructions.extend([
            f"  - CRITICAL: Use target_col from dataset info (target_col='{target_col}' if available)",
            "  - CRITICAL: MUST encode categorical features (object/category dtypes) using ColumnTransformer + OneHotEncoder",
            "  - CRITICAL: Never pass raw categorical strings to LightGBM/XGBoost/sklearn (will fail with 'could not convert string to float')",
            "  - CatBoost is the ONLY exception that handles categorical features natively",
            "  - Use OneHotEncoder(handle_unknown='ignore', sparse_output=False) (NOT sparse=...)",
        ])
    else:
        data_files = state.get("data_files", {}) if state else {}
        instructions.extend(build_image_model_instructions(is_image_to_image, data_files, suggested_epochs, early_stopping_patience))
        if not is_image_to_image:
            train_csv_path = data_files.get("train_csv", "") if isinstance(data_files, dict) else ""
            instructions.extend([
                "  - CRITICAL: This is an image competition. Do NOT use tabular models unless you have real numeric features.",
                "    - If train.csv only has id+label (<=2 cols), you MUST train an image model (CNN/transformer) or add an embedding extractor first.",
            ])
            if train_csv_path:
                instructions.append(f"  - Train CSV path (check columns): {train_csv_path}")

    # Add CV and OOF instructions
    component_name = getattr(component, "name", "component")
    instructions.extend(build_cv_instructions(working_dir, component_name))
    instructions.extend(build_stacking_oof_instructions(working_dir, component_name))

    # Add submission format instructions (CRITICAL for CV vs public score match)
    instructions.extend([
        "\n⚠️ SUBMISSION FORMAT (CRITICAL - SEE HARD_CONSTRAINTS):",
        "  - Target column is NOT always sample_sub.columns[1]!",
        "  - First: print(sample_sub.columns) to see ALL column names",
        "  - The target column has SAME NAME as target in train.csv (e.g., 'Insult', 'target', 'label')",
        "  - Example: 'detecting-insults' has target='Insult' at columns[0], NOT columns[1]",
        "  - ONLY fill the target column: sample_sub[target_col] = predictions",
        "  - DO NOT overwrite non-target columns (Date, Comment, ID, etc.)",
    ])

    return instructions


def build_dynamic_instructions(
    component,
    state: dict,
    config,
    working_dir: str,
) -> str:
    """
    Build dynamic instructions based on current state (MLE-STAR pattern).

    Creates context-aware guidance by analyzing:
    - Previous component results (what worked/failed)
    - Current iteration number (more specific in later iterations)
    - Performance trends
    - Common error patterns

    Args:
        component: Component being implemented
        state: Current workflow state
        config: Agent configuration
        working_dir: Working directory path

    Returns:
        Dynamic instructions string
    """
    instructions = []

    instructions.append(f"Implement {component.component_type}: {component.name}")

    run_mode = str(state.get("run_mode", "")).lower()
    objective = str(state.get("objective", "")).lower()
    domain = str(state.get("domain_detected", state.get("domain", "tabular"))).lower()
    submission_format_type = str(state.get("submission_format_type") or "").lower()
    is_image = domain.startswith("image") or domain in {"computer_vision", "vision"}
    is_image_to_image = domain == "image_to_image" or submission_format_type == "pixel_level"

    # Detect problem type
    problem_type = ""
    try:
        comp_info = state.get("competition_info")
        problem_type = comp_info.problem_type if comp_info else ""
    except Exception:
        problem_type = ""
    is_classification = "class" in str(problem_type).lower()

    # Check sample submission for integer labels
    sample_integer_labels = False
    sample_submission_path = state.get("sample_submission_path")
    if sample_submission_path:
        try:
            import pandas as pd
            import numpy as np

            sample_sub = pd.read_csv(sample_submission_path)
            if sample_sub.shape[1] >= 2:
                sample_vals = sample_sub.iloc[:, 1]
                if pd.api.types.is_numeric_dtype(sample_vals):
                    vals = sample_vals.to_numpy()
                    if vals.size:
                        sample_integer_labels = np.allclose(vals, np.round(vals))
        except Exception:
            sample_integer_labels = False

    # Get timeout hint
    timeout_hint = state.get("timeout_per_component")
    if not isinstance(timeout_hint, (int, float)):
        try:
            timeout_hint = int(timeout_hint) if timeout_hint is not None else None
        except Exception:
            timeout_hint = None

    target_col = state.get("target_col", "target")
    current_iteration = state.get("current_iteration", 0)
    refinement_guidance = state.get("refinement_guidance", {})
    dev_results = state.get("development_results", [])
    current_score = state.get("current_performance_score", 0.0)
    target_score = state.get("target_score")

    if isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = None

    competition_info = state.get("competition_info")
    metric_name = competition_info.evaluation_metric if competition_info else ""

    # Build budget instructions
    instructions.extend(build_budget_instructions(timeout_hint))

    # Build MLE-bench instructions if applicable
    if run_mode == "mlebench" or "medal" in objective:
        instructions.extend(build_mlebench_objective_instructions())

    # Build iteration context
    instructions.extend(build_iteration_context(current_iteration, refinement_guidance))

    # Build refinement guidance
    if refinement_guidance and "component_type_guidance" in refinement_guidance:
        comp_guidance = refinement_guidance["component_type_guidance"].get(component.component_type)
        if comp_guidance:
            instructions.append(f"\n🎯 {component.component_type.upper()} SPECIFIC GUIDANCE:")
            instructions.append(f"  {comp_guidance}")

    # Build previous results context
    instructions.extend(build_previous_results_context(dev_results))

    # Build performance gap instructions
    instructions.extend(build_performance_gap_instructions(current_score, target_score, metric_name))

    # Get adaptive epoch budget and patience from state (SOTA pattern)
    epoch_budget = int(state.get("epoch_budget", 600))  # SOTA uses 600
    early_stopping_patience = int(state.get("early_stopping_patience", 30))  # SOTA uses 30
    min_epochs = int(os.getenv("KAGGLE_AGENTS_MIN_EPOCHS", "5"))

    # Check if last run timed out and reduce epochs
    suggested_epochs = epoch_budget
    if dev_results:
        last_result = dev_results[-1]
        last_stdout = str(getattr(last_result, "stdout", "") or "").lower()
        last_stderr = str(getattr(last_result, "stderr", "") or "").lower()
        last_exec_time = getattr(last_result, "execution_time", 0) or 0
        timeout_component = timeout_hint or 3600

        timed_out = (
            "timeout" in last_stderr
            or "deadline" in last_stdout
            or "[timeout]" in last_stdout
            or last_exec_time >= timeout_component * 0.95
        )
        if timed_out:
            reduction_factor = float(os.getenv("KAGGLE_AGENTS_EPOCH_REDUCTION", "0.5"))
            suggested_epochs = max(min_epochs, int(epoch_budget * reduction_factor))

    # Component-type specific instructions
    if component.component_type == "model":
        instructions.extend(build_model_component_instructions(
            component=component,
            state=state,
            working_dir=working_dir,
            is_image=is_image,
            is_image_to_image=is_image_to_image,
            is_classification=is_classification,
            sample_integer_labels=sample_integer_labels,
            target_col=target_col,
            suggested_epochs=suggested_epochs,
            early_stopping_patience=early_stopping_patience,
        ))

        # Optuna instructions if component name suggests tuning
        name_lower = component.name.lower()
        if "optuna" in name_lower or "tuned" in name_lower or "optimized" in name_lower:
            n_trials = getattr(getattr(config, "ablation", None), "optuna_trials", 5) if config else 5
            timeout = (getattr(getattr(config, "ablation", None), "testing_timeout", 600) if config else 600) - 60
            instructions.extend(build_optuna_tuning_instructions(n_trials, timeout))

    elif component.component_type == "feature_engineering":
        instructions.extend(build_feature_engineering_instructions())

    elif component.component_type == "ensemble":
        instructions.extend(build_ensemble_instructions(target_col))

    # Standard requirements
    instructions.extend(build_standard_requirements())

    return "\n".join(instructions)
