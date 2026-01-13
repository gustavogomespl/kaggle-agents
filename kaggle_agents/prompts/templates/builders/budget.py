"""
Time budget and MLE-bench objective instruction builders.
"""

from __future__ import annotations


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


def build_timeout_safe_training_instructions() -> list[str]:
    """
    Generate instructions for timeout-safe training with epoch prediction.

    Critical for image/audio competitions where epochs take 20+ minutes.
    The key insight: check BEFORE starting an epoch if there's enough time,
    not after (when it's too late).
    """
    return [
        "\n### TIMEOUT-SAFE TRAINING (CRITICAL FOR IMAGE/AUDIO):",
        "  Long epochs (20+ min) can cause hard timeout kills mid-training.",
        "  Use this pattern to exit gracefully and save predictions:",
        "",
        "  ```python",
        "  import time",
        "  import numpy as np",
        "  ",
        "  # Configuration",
        "  HARD_TIMEOUT = int(os.environ.get('KAGGLE_AGENTS_COMPONENT_TIMEOUT_S', 3600))",
        "  MIN_SAVE_BUFFER = 120  # Always reserve 2 min for final save",
        "  START_TIME = time.time()",
        "  epoch_times = []",
        "  ",
        "  def get_remaining_time():",
        "      return HARD_TIMEOUT - (time.time() - START_TIME)",
        "  ",
        "  def can_start_next_epoch():",
        "      '''Check BEFORE starting epoch if we have time to complete it.'''",
        "      remaining = get_remaining_time()",
        "      if not epoch_times:",
        "          # First epoch: only start if >50% time remaining",
        "          return remaining > HARD_TIMEOUT * 0.5",
        "      # Need: estimated_epoch_time * 1.1 (10% safety) + save buffer",
        "      est_epoch = np.mean(epoch_times) * 1.1",
        "      required = est_epoch + MIN_SAVE_BUFFER",
        "      print(f'[TIMEOUT] remaining={remaining:.0f}s, need={required:.0f}s for next epoch')",
        "      return remaining > required",
        "  ",
        "  def save_checkpoint(oof, test, epoch):",
        "      '''Save predictions after EVERY epoch - crucial for graceful timeout.'''",
        "      np.save(MODELS_DIR / f'oof_checkpoint_e{epoch}.npy', oof)",
        "      np.save(MODELS_DIR / f'test_checkpoint_e{epoch}.npy', test)",
        "      print(f'[CHECKPOINT] Saved predictions at epoch {epoch}')",
        "  ",
        "  # Training loop with PRE-EPOCH timeout check",
        "  for epoch in range(MAX_EPOCHS):",
        "      # CHECK BEFORE STARTING (not after!)",
        "      if not can_start_next_epoch():",
        "          print(f'[TIMEOUT] Graceful stop before epoch {epoch}')",
        "          break  # oof_preds/test_preds have last checkpoint",
        "      ",
        "      epoch_start = time.time()",
        "      # ... train one epoch ...",
        "      epoch_times.append(time.time() - epoch_start)",
        "      ",
        "      # ALWAYS save checkpoint after epoch completes",
        "      save_checkpoint(oof_preds, test_preds, epoch)",
        "  ```",
        "",
        "  WRONG: Check timeout at end of epoch (too late - process already killed!)",
        "  WRONG: Use fixed 50-600s buffer (epochs can take 1400s+)",
        "  RIGHT: Check BEFORE starting each epoch if time is sufficient",
        "  RIGHT: Save checkpoint AFTER each epoch completes",
        "  RIGHT: Estimate epoch time from first epoch, predict if next is feasible",
    ]
