"""
Image model instruction builder.
"""

from __future__ import annotations

from typing import Any


def build_image_model_instructions(
    is_image_to_image: bool,
    data_files: dict[str, Any] | None,
    suggested_epochs: int = 600,
    early_stopping_patience: int = 60,
) -> list[str]:
    """Build image model instructions with adaptive epoch budget and patience."""
    instructions = [
        "\n🖼️ IMAGE MODELLING (DEEP TRAINING - SOTA Pattern):",
        f"  - Train for up to {suggested_epochs} epochs with early stopping (patience={early_stopping_patience})",
        "  - MUST use GPU: device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "  - Use mixed precision: torch.cuda.amp.autocast() for 2x faster training",
        "  - Full backbone fine-tuning for maximum performance (do NOT freeze layers)",
        "  - Learning rate schedule: warmup for 5% of epochs, then cosine decay to 1e-6",
        "  - Use pretrained backbone (torchvision/timm) - efficientnet_b0 or resnet50 recommended",
        "  - For multi-class: use softmax outputs and ensure probabilities sum to 1 per row",
        "  - For log_loss: use label_smoothing (0.05-0.1) + MixUp/CutMix; clip probs to [1e-15, 1-1e-15]",
        "  - Map class indices to sample_submission columns order (do NOT sort labels independently)",
        "  - For Keras: use DeadlineCallback in model.fit() callbacks (see HARD_CONSTRAINTS)",
        "  - Do NOT run full test inference inside each fold; run once after training best checkpoint",
        "  - Save checkpoint every epoch, keep best by validation metric",
        "  - If dataset is huge (many images or >5GB), prefer 1 holdout split + smaller resolution (224/256) to avoid timeouts",
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
