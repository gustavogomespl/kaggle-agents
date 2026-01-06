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
        "  - Start with frozen backbone for 1-2 epochs; unfreeze last block only if time allows",
        "  - Learning rate schedule: warmup for 5% of epochs, then cosine decay to 1e-6",
        "  - Use pretrained backbone (torchvision/timm) - efficientnet_b0 or resnet50 recommended",
        "  - Avoid heavy backbones (e.g., resnet152) unless you have ample time budget",
        "\n  🔴 CRITICAL PREPROCESSING (model-specific normalization - DO NOT SKIP):",
        "    - EfficientNet (TF/Keras): tf.keras.applications.efficientnet.preprocess_input() → scales to [-1, 1]",
        "    - ResNet/VGG (TF/Keras): tf.keras.applications.resnet.preprocess_input() → ImageNet BGR mean subtraction",
        "    - MobileNet (TF/Keras): tf.keras.applications.mobilenet_v2.preprocess_input() → scales to [-1, 1]",
        "    - PyTorch timm: use timm.data.create_transform(is_training=False) or model.default_cfg['mean']/['std']",
        "    - PyTorch torchvision: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "  - ⚠️ NEVER use simple /255.0 normalization for pretrained ImageNet models (breaks feature extraction!)",
        "  - ⚠️ Using wrong preprocessing = near-random predictions (LogLoss ~4.0 for 120 classes)",
        "  - For custom models trained from scratch: /255.0 or [0,1] normalization is acceptable",
        "  - Use ONE framework per run (PyTorch OR Keras) and keep inference consistent",
        "  - If using TensorFlow: tf.io.read_file + tf.image.decode_jpeg/png(channels=3), tf.ensure_shape, Dataset.ignore_errors(), Dataset.repeat()",
        "  - Avoid tf.image.decode_image unless you also call tf.ensure_shape and set static shape",
        "  - If using TensorFlow: avoid .numpy() inside tf.data map; use tf.print/tf.debugging for logs",
        "  - Save best full model: PyTorch -> models/best_model.pth, Keras -> models/best_model.keras",
        "  - PyTorch must save full model object (torch.save(model, ...)), NOT state_dict",
        "  - For multi-class: use softmax outputs and ensure probabilities sum to 1 per row",
        "  - For log_loss: use label_smoothing (0.05-0.1) + MixUp/CutMix; clip probs to [1e-15, 1-1e-15]",
        "  - Map class indices to sample_submission columns order (do NOT sort labels independently)",
        "  - For Keras: use DeadlineCallback in model.fit() callbacks (see HARD_CONSTRAINTS)",
        "  - Do NOT run full test inference inside each fold; run once after training best checkpoint",
        "  - Save checkpoint every epoch, keep best by validation metric",
        "  - If dataset is huge (many images or >5GB), prefer 1 holdout split + smaller resolution (224/256) to avoid timeouts",
    ]

    if suggested_epochs < 50:
        instructions.extend([
            f"\n  ⚠️ TIMEOUT ADAPTATION (epochs reduced to {suggested_epochs}):",
            "    - Use EfficientNet-B0 (fastest/smallest) instead of B3/B4/B7",
            "    - Reduce image resolution: 224 → 160 → 128 (smaller = faster)",
            "    - Reduce batch size: 32 → 16 → 8 (if memory is tight)",
            "    - Use 1-2 CV folds instead of 5 (faster validation)",
            "    - Skip heavy augmentation (MixUp/CutMix) on validation set",
            "    - Keep backbone frozen (don't unfreeze layers)",
            "    - Reduce early_stopping_patience proportionally (e.g., patience=5 for 50 epochs)",
        ])

    # Always add input size consistency and binary output instructions
    instructions.extend([
        "\n  🔴 INPUT SIZE CONSISTENCY (CRITICAL FOR ENSEMBLE):",
        "    - SAVE the image size used during training as metadata:",
        "      ```python",
        "      IMG_HEIGHT, IMG_WIDTH = 160, 160  # or whatever you use",
        "      np.save(f'models/input_size_{component_name}.npy', np.array([IMG_HEIGHT, IMG_WIDTH]))",
        "      ```",
        "    - AT INFERENCE: Load the saved size and resize test images to match:",
        "      ```python",
        "      input_size = np.load(f'models/input_size_{component_name}.npy')",
        "      test_images = tf.image.resize(test_images, input_size)  # or transforms.Resize(tuple(input_size))",
        "      ```",
        "    - NEVER assume 224x224 - always check the saved size metadata",
        "    - If input_size file doesn't exist, default to model.input_shape[1:3] or 224x224",
        "\n  🔴 BINARY CLASSIFICATION OUTPUT FORMAT (CRITICAL FOR ENSEMBLE):",
        "    - Use SINGLE sigmoid output: `Dense(1, activation='sigmoid')` or `nn.Sigmoid()`",
        "    - DO NOT use softmax with 2 classes (causes shape mismatch in ensemble)",
        "    - Save predictions as shape (N,) or (N, 1), NEVER (N, 2)",
        "    - For test predictions: `preds = model.predict(X_test).flatten()`",
        "    - This ensures all models produce compatible prediction shapes for ensemble averaging",
    ])

    if isinstance(data_files, dict) and data_files.get("train_csv"):
        instructions.append(
            f"  - Labels are in Train CSV at: {data_files['train_csv']} (not inside train/)"
        )
    if isinstance(data_files, dict):
        train_dir = data_files.get("train") or ""
        test_dir = data_files.get("test") or ""
        if train_dir:
            instructions.append(f"  - Use training images from: {train_dir} (do not hardcode paths)")
            instructions.append(
                "  - Detect image extension by scanning train dir (tif is common); do not assume jpg/png"
            )
        if test_dir:
            instructions.append(f"  - Use test images from: {test_dir} (do not hardcode paths)")

    if is_image_to_image:
        clean_path = ""
        if isinstance(data_files, dict):
            clean_path = data_files.get("clean_train", "") or ""

        instructions.extend(
            [
                "\n🧽 IMAGE-TO-IMAGE (PIXEL-LEVEL) REQUIREMENTS:",
                "  - MUST learn noisy->clean mapping. Use train/ as noisy inputs and clean targets from clean_train.",
                "  - If using a pretrained backbone, use it ONLY as an encoder; discard classification heads.",
            ]
        )

        if clean_path:
            instructions.append(f"  - Clean target dir (paired with train/): {clean_path}")

        instructions.append(
            "  - Output full-resolution (or resized) images, then flatten to pixel-level CSV using sample_submission IDs."
        )

        # Add critical data pipeline fixes for image-to-image
        instructions.extend(
            [
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
            ]
        )

    return instructions
