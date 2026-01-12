"""
HuggingFace compatibility utilities.

Handles API changes across different transformers versions, specifically:
- evaluation_strategy -> eval_strategy rename in transformers >= 4.38.0
"""

from __future__ import annotations

import importlib.metadata
from typing import Any


def get_transformers_version() -> tuple[int, int, int]:
    """
    Get transformers version as tuple.

    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    try:
        version_str = importlib.metadata.version("transformers")
        parts = version_str.split(".")
        # Handle versions like "4.38.0a0" or "4.38.0+cu118"
        major = int(parts[0])
        minor = int(parts[1])
        patch_str = parts[2].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        patch = int(patch_str) if patch_str else 0
        return (major, minor, patch)
    except Exception:
        # Default fallback to a safe version
        return (4, 30, 0)


def get_training_args_kwargs(
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build TrainingArguments kwargs with version-appropriate evaluation_strategy.

    transformers < 4.38.0: evaluation_strategy
    transformers >= 4.38.0: eval_strategy (evaluation_strategy deprecated)

    Args:
        eval_strategy: One of "no", "steps", "epoch"
        eval_steps: Evaluation interval when strategy is "steps"
        **kwargs: Additional TrainingArguments parameters

    Returns:
        Dict of kwargs compatible with installed transformers version
    """
    version = get_transformers_version()

    result = dict(kwargs)
    result["eval_steps"] = eval_steps

    # Version 4.38.0 deprecated evaluation_strategy in favor of eval_strategy
    if version >= (4, 38, 0):
        result["eval_strategy"] = eval_strategy
    else:
        result["evaluation_strategy"] = eval_strategy

    return result


def create_seq2seq_training_args(
    output_dir: str,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-4,
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 100,
    max_steps: int = -1,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    predict_with_generate: bool = True,
    generation_max_length: int = 128,
    **kwargs: Any,
):
    """
    Create Seq2SeqTrainingArguments with version-appropriate parameters.

    Handles the evaluation_strategy -> eval_strategy rename in transformers >= 4.38.0.
    Also enforces max_steps guard for bounded training time.

    Args:
        output_dir: Directory for checkpoints
        max_steps: Maximum training steps (-1 for epoch-based, recommended: 2000 for fast mode)
        ... other standard Seq2SeqTrainingArguments parameters

    Returns:
        Seq2SeqTrainingArguments instance
    """
    from transformers import Seq2SeqTrainingArguments

    # Build version-compatible kwargs
    args_kwargs = get_training_args_kwargs(
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=save_steps,
        logging_steps=logging_steps,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        **kwargs,
    )

    return Seq2SeqTrainingArguments(**args_kwargs)


def create_training_args(
    output_dir: str,
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    **kwargs: Any,
):
    """
    Create TrainingArguments with version-appropriate parameters.

    Args:
        output_dir: Directory for checkpoints
        eval_strategy: Evaluation strategy
        eval_steps: Evaluation interval
        **kwargs: Additional parameters

    Returns:
        TrainingArguments instance
    """
    from transformers import TrainingArguments

    args_kwargs = get_training_args_kwargs(
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        output_dir=output_dir,
        **kwargs,
    )

    return TrainingArguments(**args_kwargs)


# Code snippet for injection into generated code
# This allows generated code to be self-contained without importing from kaggle_agents
HF_COMPAT_CODE_SNIPPET = '''
# === HuggingFace Version Compatibility ===
import importlib.metadata

def _get_hf_version():
    """Get transformers version as tuple."""
    try:
        v = importlib.metadata.version("transformers").split(".")
        major = int(v[0])
        minor = int(v[1])
        patch = int(v[2].split("+")[0].split("a")[0].split("b")[0].split("rc")[0])
        return (major, minor, patch)
    except Exception:
        return (4, 30, 0)

_HF_VERSION = _get_hf_version()

def _hf_eval_strategy_kwarg(strategy="steps", steps=500):
    """Return version-appropriate eval strategy kwargs."""
    if _HF_VERSION >= (4, 38, 0):
        return {"eval_strategy": strategy, "eval_steps": steps}
    else:
        return {"evaluation_strategy": strategy, "eval_steps": steps}

# Usage: Seq2SeqTrainingArguments(output_dir="./", **_hf_eval_strategy_kwarg("steps", 500))
# === END HF Compat ===
'''


def get_hf_compat_code_snippet() -> str:
    """
    Get code snippet for injection into generated code.

    Returns the HF_COMPAT_CODE_SNIPPET that can be inserted at the top
    of generated code to make it self-contained.
    """
    return HF_COMPAT_CODE_SNIPPET
