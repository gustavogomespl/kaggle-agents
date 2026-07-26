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
        "\n🖼️ IMAGE MODELLING (VALIDATION-GUIDED TRAINING):",
        f"  - Train for up to {suggested_epochs} epochs with early stopping (patience={early_stopping_patience})",
        "  - MUST use GPU: device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "  - Use mixed precision when supported and verify numerical stability",
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
        "  - ⚠️ Incompatible preprocessing can invalidate pretrained features",
        "  - For custom models trained from scratch: /255.0 or [0,1] normalization is acceptable",
        "  - Use ONE framework per run (PyTorch OR Keras) and keep inference consistent",
        "  - If using TensorFlow: tf.io.read_file + a format-appropriate decoder + tf.ensure_shape; let missing/corrupt paths raise with their record ID",
        "  - NEVER suppress dataset element errors, drop failed decodes, or filter unresolved records; any lost row invalidates canonical OOF alignment",
        "  - Avoid tf.image.decode_image unless you also call tf.ensure_shape and set static shape",
        "  - If using TensorFlow: avoid .numpy() inside tf.data map; use tf.print/tf.debugging for logs",
        "  - Save best checkpoint: PyTorch state_dict + architecture/config metadata -> models/best_model_state.pt; Keras -> models/best_model.keras",
        "  - PyTorch must NEVER save a full model object; save/load model.state_dict() into an explicitly reconstructed architecture",
        "  - For multi-class: use softmax outputs and ensure probabilities sum to 1 per row",
        "  - For log_loss: use label_smoothing (0.05-0.1) + MixUp/CutMix; clip probs to [1e-15, 1-1e-15]",
        "  - Map class indices to sample_submission columns order (do NOT sort labels independently)",
        "  - For Keras: use DeadlineCallback in model.fit() callbacks (see HARD_CONSTRAINTS)",
        "  - Iterate every injected CANONICAL_FOLDS label, fill its exact OOF rows, and average test predictions across completed canonical folds",
        "  - NEVER create a holdout or private image split; canonical folds encode grouping and leakage constraints",
        "  - Save a state_dict checkpoint every epoch and keep the best validation state for each canonical fold",
        "  - If the dataset is huge, reduce observed-data-derived resolution, batch size, or epochs; never replace canonical CV with a holdout",
    ]

    if suggested_epochs < 50:
        instructions.extend([
            f"\n  ⚠️ TIMEOUT ADAPTATION (epochs reduced to {suggested_epochs}):",
            "    - Use EfficientNet-B0 (fastest/smallest) instead of B3/B4/B7",
            "    - Reduce image resolution: 224 → 160 → 128 (smaller = faster)",
            "    - Reduce batch size: 32 → 16 → 8 (if memory is tight)",
            "    - Keep every canonical fold; reduce per-fold epochs/resolution instead of dropping validation rows",
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

    # Initialize defaults for data_files paths
    train_dir = ""
    test_dir = ""

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

    # Add robust path building guidance for all image competitions
    # Use train_dir if available from data_files, otherwise use 'train' as default
    train_dir_for_guidance = train_dir if train_dir else "train"
    instructions.extend([
        "\n  🔴 ROBUST FILE PATH MAPPING (CRITICAL - IDs may not match filenames):",
        "    - DO NOT assume `path = dir / f'{id}.jpg'` - this pattern frequently fails",
        "    - INSTEAD: Scan directory first, build id_to_path mapping:",
        "      ```python",
        "      from pathlib import Path",
        f"      img_dir = Path('{train_dir_for_guidance}')",
        "      all_images = list(img_dir.rglob('*.*'))  # Get all files",
        "      image_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}",
        "      all_images = [f for f in all_images if f.suffix.lower() in image_exts]",
        "      paths_by_id = {}",
        "      for path in all_images:",
        "          paths_by_id.setdefault(path.stem, []).append(path)",
        "      duplicate_ids = {k: v for k, v in paths_by_id.items() if len(v) != 1}",
        "      if duplicate_ids:",
        "          raise ValueError(f'Ambiguous image IDs: {list(duplicate_ids)[:5]}')",
        "      id_to_path = {k: v[0] for k, v in paths_by_id.items()}",
        "      df['image_path'] = df[ID_COL].astype(str).map(id_to_path)",
        "      missing = df.loc[df['image_path'].isna(), ID_COL].astype(str).tolist()",
        "      if missing:",
        "          raise FileNotFoundError(f'Unresolved canonical image IDs: {missing[:10]}')",
        "      ```",
        "    - Assert every canonical ID resolves exactly once; never filter the dataframe to matched files",
    ])

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
                "  - **VARIABLE DIMENSIONS**: Images may have different native sizes",
                "    - TRAINING: Use `transforms.RandomCrop(256, 256)` or `transforms.Resize((256, 256))` for consistent tensor sizes",
                "    - VALIDATION/TEST: Use `batch_size=1` in DataLoader to avoid torch.stack() errors",
                "  - **NEGATIVE STRIDES**: Call `np.ascontiguousarray()` or `.copy()` after `np.flip()`/`np.rot90()` augmentations",
                "  - **NO TRAIN.CSV**: Load image pairs directly from directories with `glob`/`pathlib`, NOT from CSV files:",
                "    ```python",
                "    noisy_files = sorted(train_dir.glob('*.png'))",
                "    pairs = [(nf, clean_dir / nf.name) for nf in noisy_files]",
                "    missing_targets = [clean for _, clean in pairs if not clean.is_file()]",
                "    if missing_targets:",
                "        raise FileNotFoundError(f'Missing paired targets: {missing_targets[:10]}')",
                "    ```",
                "  - **LARGER BATCH FOR TRAINING**: Once dimensions are fixed with RandomCrop, use batch_size=16 or 32 for faster training",
            ]
        )

    return instructions
