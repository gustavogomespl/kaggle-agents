"""
Image competition fallback plans.

Includes image classification/regression and image-to-image tasks.

CRITICAL NOTES FOR MEDICAL IMAGING (e.g., RANZCR, chest X-rays):
- Use higher resolution (384-512) for thin structures like catheters
- Train more epochs (10-20) with early stopping
- Use GroupKFold on PatientID to prevent leakage
- Unfreeze backbone for domain adaptation from ImageNet to X-rays
"""

from typing import Any


def create_image_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    *,
    fast_mode: bool = False,
    competition_name: str = "",
) -> list[dict[str, Any]]:
    """
    Create fallback plan for image competitions (PyTorch/TensorFlow DataLoaders).

    Uses transfer learning with pre-trained CNNs (EfficientNet, ResNet).

    Args:
        domain: Competition domain (image_classification, image_regression, etc.)
        sota_analysis: SOTA analysis results
        fast_mode: If True, return minimal 2-component plan for speed
        competition_name: Name of competition for domain-specific settings

    Returns:
        List of component dictionaries (2 in fast mode, 3 normally)
    """
    is_regression = "regression" in domain
    task = "regression" if is_regression else "classification"

    # Detect medical imaging competitions that need special handling
    comp_lower = competition_name.lower() if competition_name else ""
    is_medical = any(kw in comp_lower for kw in [
        "ranzcr", "rsna", "siim", "chest", "xray", "x-ray", "medical", "radiology",
        "ct", "mri", "dicom", "catheter", "pneumonia", "covid", "lung"
    ])

    # FAST MODE: Only 2 components for maximum speed (MLE-bench optimization)
    if fast_mode:
        # Medical imaging needs different settings even in fast mode
        if is_medical:
            return [
                {
                    "name": f"efficientnet_b0_medical_{task}",
                    "component_type": "model",
                    "description": """EfficientNet-B0 for MEDICAL IMAGING with domain-appropriate settings.

CRITICAL MEDICAL IMAGING REQUIREMENTS:
1. **TARGET COLUMNS**: Read sample_submission.csv FIRST to get N_CLASSES (e.g., RANZCR has 11 columns)
2. **RESOLUTION**: Use IMG_SIZE=384 (NOT 224) - thin catheters/lines need higher resolution
3. **EPOCHS**: Train for 10-15 epochs with early stopping (patience=3), NOT just 2 epochs
4. **BACKBONE**: Unfreeze last 2 blocks of backbone for X-ray domain adaptation
5. **CV**: Use GroupKFold on PatientID column to prevent patient-level leakage
6. **BATCH SIZE**: Use 32-64 (NOT 128) to fit higher resolution images in memory

Mixed precision training. Save full model to models/best_model.pth.""",
                    "estimated_impact": 0.35,
                    "rationale": "Medical imaging requires higher resolution and more training than natural images. Domain shift from ImageNet to X-rays requires backbone fine-tuning.",
                    "code_outline": "efficientnet_b0(pretrained=True), unfreeze last 2 blocks, IMG_SIZE=384, BATCH_SIZE=32, 10-15 epochs with early stopping, GroupKFold on PatientID, save to models/best_model.pth",
                },
                {
                    "name": "tta_inference_only",
                    "component_type": "ensemble",
                    "description": "Test-Time Augmentation ONLY (no additional training). Load the single trained full model from models/best_model.* (auto-detect extension) and apply 5 simple transforms (original, hflip, vflip, rotate90, rotate180), average predictions. Write submission.csv with ALL target columns.",
                    "estimated_impact": 0.05,
                    "rationale": "Free accuracy boost without additional training time. Just inference with multiple transforms.",
                    "code_outline": "Load full model from models/best_model.* (auto-detect extension), for each test image: apply transforms, average predictions, clip to [0,1], write submission.csv with N_CLASSES columns",
                },
            ]
        # Standard (non-medical) fast mode
        return [
            {
                "name": f"efficientnet_b0_fast_{task}",
                "component_type": "model",
                "description": "EfficientNet-B0 with FROZEN backbone. Only train classifier head for 2-3 epochs. Use 2-fold CV (KAGGLE_AGENTS_CV_FOLDS=2). Mixed precision training. Lightweight augmentations only (flip, normalize). IMPLEMENT soft-deadline pattern. Save full model to models/best_model.pth (PyTorch) or models/best_model.keras (Keras).",
                "estimated_impact": 0.30,
                "rationale": "Frozen backbone = fastest training. 2 epochs is enough for head fine-tuning. This prioritizes getting a valid submission quickly.",
                "code_outline": "efficientnet_b0(pretrained=True), freeze all backbone layers, train head only, 2 epochs, 2-fold CV, save full model to models/best_model.pth (PyTorch) or models/best_model.keras (Keras), implement _check_deadline() pattern",
            },
            {
                "name": "tta_inference_only",
                "component_type": "ensemble",
                "description": "Test-Time Augmentation ONLY (no additional training). Load the single trained full model from models/best_model.* (auto-detect extension) and apply 5 simple transforms (original, hflip, vflip, rotate90, rotate180), average predictions. Write submission.csv.",
                "estimated_impact": 0.05,
                "rationale": "Free accuracy boost without additional training time. Just inference with multiple transforms.",
                "code_outline": "Load full model from models/best_model.* (auto-detect extension), for each test image: apply transforms, average predictions, clip to [0,1], write submission.csv",
            },
        ]

    # NORMAL MODE: 3 components (2 models + TTA ensemble)
    return [
        {
            "name": f"efficientnet_b0_{task}",
            "component_type": "model",
            "description": f"EfficientNet-B0 pre-trained fine-tuned for {task}. PyTorch DataLoader with ImageFolder or custom Dataset. Data augmentation (rotation, flip, color jitter). Use transfer learning from ImageNet weights.",
            "estimated_impact": 0.28,
            "rationale": "EfficientNet achieves SOTA on ImageNet with excellent efficiency. Transfer learning transfers learned features. Data augmentation prevents overfitting on small datasets.",
            "code_outline": "torchvision.models.efficientnet_b0(pretrained=True), replace classifier head, train with CrossEntropyLoss/MSELoss, CV folds via KAGGLE_AGENTS_CV_FOLDS, save full model to models/best_model.pth (PyTorch) or models/best_model.keras (Keras) and OOF predictions for ensemble",
        },
        {
            "name": f"resnet50_{task}",
            "component_type": "model",
            "description": "ResNet50 fine-tuned with different augmentation strategy (Cutout, Mixup) for architectural diversity in ensemble.",
            "estimated_impact": 0.24,
            "rationale": "ResNet provides complementary features vs EfficientNet. Ensemble benefits from architectural diversity.",
            "code_outline": "torchvision.models.resnet50(pretrained=True), replace head, Cutout + Mixup augmentations, AdamW, CV folds via KAGGLE_AGENTS_CV_FOLDS",
        },
        {
            "name": "tta_ensemble",
            "component_type": "ensemble",
            "description": "Test-time augmentation (TTA) + weighted ensemble of EfficientNet and ResNet predictions. Apply multiple transforms to test images and average predictions.",
            "estimated_impact": 0.15,
            "rationale": "TTA averages predictions over multiple augmented views of each test image, reducing variance. Weighted ensemble (by CV score) combines different architectures. Typical +2-5% improvement.",
            "code_outline": "For each test image: apply 5 transforms (original, hflip, vflip, rotate90, rotate270), get predictions from each model, average TTA predictions per model, then weighted average models by CV score",
        },
    ]


def create_image_to_image_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    *,
    fast_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Create fallback plan for image-to-image tasks (denoising, super-resolution, style transfer).

    CRITICAL: These tasks require PIXEL-LEVEL predictions, not per-image predictions.
    The submission format is typically one row per pixel: id=image_row_col, value=pixel_intensity.

    Args:
        domain: Competition domain (image_to_image)
        sota_analysis: SOTA analysis results
        fast_mode: If True, return minimal plan for speed

    Returns:
        List of component dictionaries with encoder-decoder architectures
    """
    if fast_mode:
        return [
            {
                "name": "simple_autoencoder_denoiser",
                "component_type": "model",
                "description": """Simple convolutional autoencoder for image-to-image transformation.

CRITICAL - THIS IS A PIXEL-LEVEL PREDICTION TASK:
- Model must output FULL IMAGE (same H x W as input), NOT a single value
- Use encoder-decoder architecture (Conv2d -> ConvTranspose2d)
- DO NOT use classifiers (EfficientNet, ResNet with FC head)

Architecture:
- Encoder: 3-4 Conv2d layers with ReLU, max pooling
- Decoder: 3-4 ConvTranspose2d layers with ReLU
- Output: Same size as input (H x W) for grayscale

Training:
- Input: noisy/degraded images
- Target: clean images
- Loss: MSE or L1 loss between output and target image

SUBMISSION FORMAT (CRITICAL - MUST FOLLOW):
```python
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)  # Typically MILLIONS of rows

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem  # e.g., "1" from "1.png"
    pred = model(preprocess(img))  # OUTPUT: (H, W) image
    H, W = pred.shape
    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"
            submission_rows.append({"id": pixel_id, "value": float(pred[row, col])})

assert len(submission_rows) == expected_rows
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
```""",
                "estimated_impact": 0.35,
                "rationale": "Simple autoencoder is fast to train and provides baseline for denoising. Pixel-level output is critical for correct submission format.",
                "code_outline": "Conv2d encoder, ConvTranspose2d decoder, MSE loss, output same size as input, flatten to pixel-level CSV",
            },
            {
                "name": "submission_format_validator",
                "component_type": "ensemble",
                "description": "Validate pixel-level submission format matches sample_submission.csv exactly.",
                "estimated_impact": 0.05,
                "rationale": "Critical validation to catch format errors before submission.",
                "code_outline": "Load sample_sub, verify row count matches, verify ID format matches exactly",
            },
        ]

    # Full mode: U-Net and ensemble
    return [
        {
            "name": "unet_encoder_decoder",
            "component_type": "model",
            "description": """U-Net architecture for image-to-image transformation with skip connections.

CRITICAL - THIS IS A PIXEL-LEVEL PREDICTION TASK:
- Model must output FULL IMAGE (same H x W as input)
- U-Net preserves fine details through skip connections
- DO NOT use classifiers (EfficientNet, ResNet with FC head)

U-Net Architecture:
- Encoder: 4 blocks of (Conv2d, BatchNorm, ReLU, MaxPool)
- Bottleneck: Conv2d block
- Decoder: 4 blocks of (ConvTranspose2d, concat skip, Conv2d, BatchNorm, ReLU)
- Output: Conv2d(1, 1, 1) for single-channel grayscale output

SUBMISSION FORMAT (CRITICAL):
Read sample_submission.csv to get expected row count (millions of rows).
Flatten each output image to pixel format: {img_id}_{row}_{col} -> value""",
            "estimated_impact": 0.40,
            "rationale": "U-Net is SOTA for image-to-image tasks. Skip connections preserve fine details crucial for denoising/super-resolution.",
            "code_outline": "PyTorch U-Net with skip connections, MSE loss, 3-5 epochs, output full image, flatten to pixel CSV",
        },
        {
            "name": "residual_autoencoder",
            "component_type": "model",
            "description": """Residual autoencoder that predicts the NOISE (residual) rather than clean image.

Architecture:
- Similar to U-Net but predicts: clean = noisy - predicted_noise
- Residual learning makes training more stable
- Output: Same size as input

This provides model diversity for ensemble.""",
            "estimated_impact": 0.35,
            "rationale": "Residual learning (predicting noise) often works better than direct denoising. Provides ensemble diversity.",
            "code_outline": "Conv encoder-decoder, predict residual, output = input - residual, same pixel-level submission format",
        },
        {
            "name": "pixel_ensemble_average",
            "component_type": "ensemble",
            "description": """Average predictions from U-Net and Residual autoencoder at pixel level.

1. Load predictions from both models
2. Average pixel values: final[i,j] = (unet[i,j] + residual[i,j]) / 2
3. Flatten to submission format
4. Validate row count matches sample_submission.csv""",
            "estimated_impact": 0.10,
            "rationale": "Ensembling reduces prediction variance. Simple average works well for image tasks.",
            "code_outline": "Load both model outputs, pixel-wise average, flatten to CSV, validate format",
        },
    ]
