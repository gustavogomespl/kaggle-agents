"""
Dynamic context builder for Developer Agent prompts.

Extracts context from KaggleState for prompt injection:
- SOTA solutions
- Previous execution results
- Meta-evaluator guidance
- Iteration memory
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any


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
    best_score: float | None = None
    target_score: float | None = None
    run_mode: str = ""
    objective: str = ""
    timeout_per_component: int | None = None
    fast_mode: bool = False
    # Adaptive training fields
    epoch_budget: int = 300  # Maximum epochs for current iteration (SOTA uses 600)
    timeout_occurred: bool = False  # Whether timeout occurred in last attempt
    suggested_epochs: int = 300  # Suggested epochs based on timeout history
    early_stopping_patience: int = 30  # SOTA uses patience=30
    # Submission validation retry
    submission_validation_error: str | None = None  # Error from last invalid submission
    # Memory summary (best models/HPs/errors/strategies)
    memory_summary: str | None = None
    # DPO: Preference pairs for contrastive learning
    dpo_examples: str = ""  # Formatted DPO pairs (good vs bad code examples)
    # Audio-specific context (submission format, precomputed features)
    audio_context: str = ""  # Formatted audio-specific context for Developer Agent


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
    from ....core.state import get_memory_summary_for_planning
    from ....utils.log_parser import format_feedback_for_llm, parse_training_logs

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

    # Memory summary from structured state (best HPs, errors, strategies)
    try:
        context.memory_summary = get_memory_summary_for_planning(state)
    except Exception:
        context.memory_summary = None

    # DPO: Extract preference pairs for contrastive learning
    preference_pairs = state.get("preference_pairs", [])
    if preference_pairs:
        context.dpo_examples = _format_dpo_for_prompt(preference_pairs, component)

    # Audio-specific context: submission format and precomputed features
    audio_context = _build_audio_context(state)
    if audio_context:
        context.audio_context = audio_context

    return context


def _format_dpo_for_prompt(pairs: list, component: Any | None = None) -> str:
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
    lines.append(
        "Learn from these successful fixes - avoid the rejected patterns, follow the chosen patterns:\n"
    )

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
            lines.append("```python")
            lines.append(rejected_snippet + "...")
            lines.append("```")
            lines.append("")
            lines.append("**✅ PREFER (this pattern succeeded):**")
            lines.append("```python")
            lines.append(chosen_snippet + "...")
            lines.append("```")
            lines.append("")

    if len(lines) > 2:  # More than just header
        lines.append(
            "**INSTRUCTION**: When implementing similar code, follow the preferred patterns above."
        )
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


def _build_audio_context(state: dict[str, Any]) -> str:
    """
    Build audio-specific context for Developer Agent prompts.

    Extracts submission format info and precomputed features from state
    to ensure the Developer Agent generates code with the correct:
    - Submission format (Wide vs Long with correct ID pattern)
    - Precomputed feature loading code
    - Train/test split based on CVfolds

    Args:
        state: KaggleState dictionary

    Returns:
        Formatted audio context string, or empty string if not audio domain
    """
    # Check if this is an audio domain
    domain_type = str(state.get("domain_type", "")).lower()
    if "audio" not in domain_type:
        return ""

    lines = ["## Audio Competition Context (CRITICAL)\n"]

    # Extract submission format info
    submission_format = state.get("submission_format_info")
    if submission_format and isinstance(submission_format, dict):
        format_type = submission_format.get("format_type", "unknown")
        id_column = submission_format.get("id_column", "Id")
        target_columns = submission_format.get("target_columns", [])
        id_pattern = submission_format.get("id_pattern")
        id_multiplier = submission_format.get("id_multiplier")
        num_classes = submission_format.get("num_classes")

        lines.append("### Submission Format (MUST FOLLOW EXACTLY)")
        lines.append(f"- **Format Type:** {format_type.upper()}")
        lines.append(f"- **ID Column:** `{id_column}`")
        lines.append(f"- **Target Columns:** {target_columns}")
        if num_classes:
            lines.append(f"- **Number of Classes:** {num_classes}")

        if format_type == "long" and id_multiplier:
            lines.append(f"- **ID Pattern:** `{id_pattern}`")
            lines.append(f"- **ID Multiplier:** {id_multiplier}")
            lines.append("")
            lines.append("**LONG FORMAT: Submission code pattern:**")
            lines.append("```python")
            lines.append("# For LONG format: Id encodes (rec_id, class_id)")
            lines.append("submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)")
            lines.append("pred_map = {}")
            lines.append("for i, rec_id in enumerate(test_rec_ids):")
            lines.append(f"    for class_id in range({num_classes or 'num_classes'}):")
            lines.append(f"        submission_id = rec_id * {id_multiplier} + class_id")
            lines.append("        pred_map[submission_id] = predictions[i, class_id]")
            lines.append(f"submission['{target_columns[0] if target_columns else 'Probability'}'] = submission['{id_column}'].map(pred_map)")
            lines.append("submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)")
            lines.append("```")

        elif format_type == "wide":
            lines.append("")
            lines.append("**WIDE FORMAT: Submission code pattern:**")
            lines.append("```python")
            lines.append("# For WIDE format: One column per class")
            lines.append("submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)")
            lines.append(f"for i, col in enumerate({target_columns}):")
            lines.append("    submission[col] = predictions[:, i]")
            lines.append("submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)")
            lines.append("```")

        lines.append("")

    # Extract train/test split info (labels or CVfolds)
    train_rec_ids = state.get("train_rec_ids", [])
    test_rec_ids = state.get("test_rec_ids", [])
    train_test_source = state.get("train_test_ids_source") or ("cvfolds" if state.get("cv_folds_used") else "")
    if train_rec_ids and test_rec_ids:
        source_label = train_test_source or "state"
        lines.append(f"### Train/Test Split ({source_label})")
        lines.append(f"- **Train samples:** {len(train_rec_ids)} rec_ids")
        lines.append(f"- **Test samples:** {len(test_rec_ids)} rec_ids")
        lines.append("- **Use these rec_ids for filtering, do NOT infer from sample_submission.csv**")
        lines.append("")

    # Extract precomputed features info
    precomputed_features = state.get("precomputed_features_info")
    if precomputed_features and isinstance(precomputed_features, dict):
        features_found = precomputed_features.get("features_found", {})
        if features_found:
            lines.append("### Precomputed Features Available")
            lines.append("Use these instead of re-extracting features:")
            lines.append("")

            for feature_type, file_path in features_found.items():
                if feature_type in ("cv_folds", "id_mapping"):
                    continue  # Skip metadata files
                shape = precomputed_features.get("feature_shapes", {}).get(feature_type)
                shape_str = f" (shape: {shape})" if shape else ""
                lines.append(f"- **{feature_type}:** `{file_path}`{shape_str}")

            lines.append("")
            lines.append("**Loading code:**")
            lines.append("```python")
            for feature_type, file_path in features_found.items():
                if feature_type in ("cv_folds", "id_mapping"):
                    continue
                path_str = str(file_path)
                if path_str.endswith((".npy", ".npz")):
                    lines.append(f"{feature_type}_features = np.load(Path('{path_str}'))")
                elif path_str.endswith(".parquet"):
                    lines.append(f"{feature_type}_df = pd.read_parquet(Path('{path_str}'))")
                else:
                    lines.append(f"{feature_type}_df = pd.read_csv(Path('{path_str}'))")
            lines.append("```")
            lines.append("")

    # Only return if we have meaningful content beyond the header
    if len(lines) > 1:
        return "\n".join(lines)

    return ""
