"""
Image competition fallback plans.

Includes image classification/regression and image-to-image tasks.
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
        competition_name: Retained for API compatibility; never used to select
            hyperparameters or dataset-specific recipes

    Returns:
        List of component dictionaries (2 in fast mode, 3 normally)
    """
    is_regression = "regression" in domain
    task = "regression" if is_regression else "classification"

    adaptive_contract = """
DATA- AND CONTRACT-DERIVED TRAINING POLICY:
1. Read sample_submission.csv and the canonical metadata before model setup.
   Derive the target columns, task output shape, fold count, and any validated
   group column; never infer these from a competition name.
2. Inspect a representative sample of the supplied images. Record codecs,
   channel counts, bit depth, height/width quantiles, and aspect ratios. Choose
   resize/pad/crop behavior from those observations and preserve the geometry
   needed by the declared target.
3. Run a short training/validation throughput probe. Select input geometry,
   batch size, and the maximum number of steps from measured memory,
   examples/second, dataset size, and the component deadline.
4. Start with the least expensive pretrained configuration that fits the
   contract. Select fine-tuning depth and stopping point only through
   comparable OOF folds and validation curves.
5. Use grouped CV only when canonical_metadata supplies a validated group_col;
   otherwise use the declared canonical folds without inventing group IDs.
6. Derive augmentations from label-preserving symmetries of the observed task.
   Keep each augmentation, alternate backbone, TTA view, or blend only when it
   improves the declared metric on identical OOF folds.
7. If decoding dominates the throughput probe, cache each source once at the
   selected geometry using collision-free full-path hashes and a manifest.
   Preserve high-bit-depth inputs losslessly; make the cache resumable.
""".strip()

    # FAST MODE: Only 2 components for maximum speed (MLE-bench optimization)
    if fast_mode:
        return [
            {
                "name": f"adaptive_pretrained_image_{task}",
                "component_type": "model",
                "description": (
                    f"Build a budget-aware pretrained image {task} baseline.\n\n"
                    f"{adaptive_contract}\n\n"
                    "Use mixed precision when supported, enforce the soft "
                    "deadline, and persist the best validated full model plus "
                    "aligned OOF/test predictions."
                ),
                "estimated_impact": 0.30,
                "rationale": (
                    "A measured pilot bounds compute while letting the public "
                    "data and metric decide geometry, capacity, and training depth."
                ),
                "code_outline": (
                    "Inspect image/schema statistics; run a throughput and memory "
                    "probe; choose a compact available pretrained backbone and "
                    "training schedule from the deadline; use canonical folds; "
                    "early-stop on the declared metric; save the best full model "
                    "and aligned OOF/test predictions."
                ),
            },
            {
                "name": "validated_inference_refinement",
                "component_type": "ensemble",
                "description": (
                    "Evaluate only label-preserving inference views derived from "
                    "the observed task. Simulate them on OOF predictions, retain "
                    "the refinement only if it improves the declared metric and "
                    "fits the measured inference budget, then align all target "
                    "columns to sample_submission.csv."
                ),
                "estimated_impact": 0.05,
                "rationale": (
                    "Validation-gated inference avoids assuming that a geometric "
                    "transform is label preserving or worth its runtime cost."
                ),
                "code_outline": (
                    "Load the validated model; derive candidate symmetries from "
                    "task metadata; compare each candidate on held-out folds; "
                    "apply only accepted views to test data; align predictions "
                    "to the submission contract."
                ),
            },
        ]

    # NORMAL MODE: 3 components (2 models + TTA ensemble)
    return [
        {
            "name": f"adaptive_pretrained_primary_{task}",
            "component_type": "model",
            "description": (
                f"Primary pretrained image {task} model.\n\n{adaptive_contract}\n\n"
                "Choose the backbone from locally available candidates using "
                "pilot throughput and identical-fold OOF performance."
            ),
            "estimated_impact": 0.28,
            "rationale": (
                "Selecting capacity and preprocessing from measured data avoids "
                "binding the fallback to one image scale or acquisition domain."
            ),
            "code_outline": (
                "Profile supplied images and runtime; evaluate compact pretrained "
                "candidates on canonical folds; derive loss/output shape from the "
                "submission and metric contracts; select training depth by OOF; "
                "save full model and aligned predictions."
            ),
        },
        {
            "name": f"validated_diverse_image_{task}",
            "component_type": "model",
            "description": (
                "Train a second image model only after the primary pipeline is "
                "valid. Select a candidate with a different inductive bias or "
                "preprocessing policy using residual/error complementarity on "
                "the same OOF rows; reuse all schema, group, and budget contracts."
            ),
            "estimated_impact": 0.24,
            "rationale": (
                "Measured error diversity is stronger evidence for a second "
                "model than a fixed architecture pairing."
            ),
            "code_outline": (
                "Compare candidate residuals to the primary OOF predictions; "
                "train the best feasible complementary candidate with identical "
                "folds; abandon it if it does not improve validated coverage or score."
            ),
        },
        {
            "name": "validated_image_ensemble",
            "component_type": "ensemble",
            "description": (
                "Fit a blend and optional inference views using only aligned OOF "
                "predictions. Candidate transforms must be justified by observed "
                "task symmetries; weights, inclusion, and runtime are selected on "
                "the declared metric, then frozen before test inference."
            ),
            "estimated_impact": 0.15,
            "rationale": (
                "OOF-gated blending preserves useful diversity without assuming "
                "that extra models or views improve every image task."
            ),
            "code_outline": (
                "Validate artifact alignment; evaluate candidate views and blend "
                "weights on identical OOF rows with the metric contract; keep "
                "only repeatable gains; generate the exact submission schema."
            ),
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

    These tasks require dense outputs rather than one scalar per image. The
    exact submission granularity and ID encoding come from sample_submission.

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
- Derive encoder/decoder depth and receptive field from observed image geometry
- Derive output channels and spatial shape from paired targets and submission contract
- Reject any configuration whose output cannot align exactly to the target tensor

Training:
- Input: noisy/degraded images
- Target: clean images
- Select the loss on held-out images using the declared evaluation metric
- Select maximum steps and stopping point from pilot throughput and validation curves

SUBMISSION FORMAT (CRITICAL - MUST FOLLOW):
```python
save_component_artifacts(
    oof_images,
    test_images,
    train_ids=CANONICAL_TRAIN_IDS,
    test_ids=CANONICAL_TEST_IDS,
)
write_submission(None)
```
The injected writer maps the saved packed test artifact to observed template
IDs in bounded chunks and rejects ambiguous coordinate conventions.

Train on paired crops/padding produced from one shared spatial transform per
noisy/clean pair. Use inference batch_size=1, retain a valid-pixel mask for any
stride padding, and remove padding before metrics/artifacts. Save variable-size
OOF/test images with the injected save_component_artifacts helper; it writes
component-specific .npz files with embedded IDs. Never import/redefine that
helper, use object arrays, or write the CSV manually.""",
                "estimated_impact": 0.35,
                "rationale": "Simple autoencoder is fast to train and provides baseline for denoising. Pixel-level output is critical for correct submission format.",
                "code_outline": "Conv2d encoder, ConvTranspose2d decoder, MSE loss, paired crop/pad, packed OOF/test artifacts, write_submission helper",
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

U-Net candidate:
- Derive the number of downsampling blocks from observed input dimensions
- Preserve skip connections only at compatible measured tensor shapes
- Derive output channels, activation, and spatial alignment from paired targets
- Select capacity and stopping point under the measured component budget

SUBMISSION FORMAT (CRITICAL):
Read sample_submission.csv and align dense predictions to its exact IDs and
order. Do not assume separators, image dimensions, or coordinate indexing.
Use paired crop/pad transforms in training, batch_size=1 plus a valid-pixel mask
for validation/test, save OOF/test with the injected save_component_artifacts
helper, and finish with the injected write_submission helper.""",
            "estimated_impact": 0.40,
            "rationale": "U-Net is a candidate because skip connections can preserve spatial detail; retain it only when held-out image metrics improve.",
            "code_outline": "Profile exact relative-path pairs, apply paired crop/pad, derive a valid U-Net depth/output contract, save packed aligned OOF/test, call write_submission",
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
            "code_outline": "Conv encoder-decoder, predict residual, output = input - residual, paired transforms, packed image artifacts, write_submission helper",
        },
    ]
