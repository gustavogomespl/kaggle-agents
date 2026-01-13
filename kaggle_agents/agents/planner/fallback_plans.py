"""Fallback plan creation for different competition domains."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import KaggleState


def create_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    curriculum_insights: str = "",
    *,
    state: KaggleState | None = None,
    is_image_competition_without_features_fn=None,
) -> list[dict[str, Any]]:
    """
    Create domain-specific fallback plan when LLM parsing fails.

    Routes to appropriate domain-specific fallback method based on domain type.

    Args:
        domain: Competition domain (e.g., 'image_classification', 'text_classification', 'tabular')
        sota_analysis: SOTA analysis results
        curriculum_insights: Insights from previous iterations (optional)
        state: Current workflow state (optional)
        is_image_competition_without_features_fn: Function to detect image competition

    Returns:
        List of component dictionaries (3-5 components depending on domain)
    """
    print(f"  [DEBUG] Creating fallback plan for domain: '{domain}'")

    # SAFETY CHECK: Prevent tabular models for image competitions
    if is_image_competition_without_features_fn and is_image_competition_without_features_fn(state):
        print(
            "  [WARNING] Forcing IMAGE fallback plan (detected image competition without features)"
        )
        print("            Tree models (LightGBM/XGBoost) require tabular features!")
        return create_image_fallback_plan("image_classification", sota_analysis, fast_mode=False)

    run_mode = str((state or {}).get("run_mode", "")).lower()
    objective = str((state or {}).get("objective", "")).lower()
    timeout_cap = (state or {}).get("timeout_per_component")
    if isinstance(timeout_cap, str):
        try:
            timeout_cap = int(timeout_cap)
        except ValueError:
            timeout_cap = None

    # Speed-first when optimizing for MLE-bench medals or tight component caps.
    fast_mode = (
        run_mode == "mlebench"
        or "medal" in objective
        or (isinstance(timeout_cap, int) and timeout_cap <= 1200)
    )

    # Define domain sets for routing
    IMAGE_DOMAINS = {
        "image_classification",
        "image_regression",
        "object_detection",
        "image_to_image",
        "image_segmentation",
    }
    TEXT_DOMAINS = {"text_classification", "text_regression", "seq_to_seq"}
    AUDIO_DOMAINS = {"audio_classification", "audio_regression"}

    # Route to domain-specific fallback method
    if domain in ("image_to_image", "image_segmentation"):
        return create_image_to_image_fallback_plan(domain, sota_analysis, fast_mode=fast_mode)
    if domain in IMAGE_DOMAINS or domain.startswith("image_"):
        return create_image_fallback_plan(domain, sota_analysis, fast_mode=fast_mode)
    if domain in TEXT_DOMAINS or domain.startswith("text_"):
        return create_text_fallback_plan(domain, sota_analysis)
    if domain in AUDIO_DOMAINS or domain.startswith("audio_"):
        return create_audio_fallback_plan(domain, sota_analysis)
    # Tabular (existing logic)
    return create_tabular_fallback_plan(
        domain,
        sota_analysis,
        curriculum_insights,
        fast_mode=fast_mode,
        state=state,
    )


def create_tabular_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    curriculum_insights: str = "",
    *,
    fast_mode: bool = False,
    state: KaggleState | None = None,
) -> list[dict[str, Any]]:
    """
    Create fallback plan for tabular competitions (classification/regression).

    Uses tree-based models (LightGBM, XGBoost, CatBoost) with ensemble.

    Args:
        domain: Competition domain
        sota_analysis: SOTA analysis results
        curriculum_insights: Insights from previous iterations (optional)
        fast_mode: Whether to use speed-optimized plan
        state: Current workflow state (used to filter failed components)

    Returns:
        List of component dictionaries (5 components: 1 FE + 4 models + 1 ensemble)
    """
    # Get failed components to avoid repeating them
    failed_names = set()
    if state:
        failed_names = set(state.get("failed_component_names", []))
        if failed_names:
            print(f"   📋 Filtering out previously failed components: {failed_names}")

    if fast_mode:
        # Full candidate pool with alternatives for failed components
        all_fast_candidates = [
            {
                "name": "lightgbm_fast_cv",
                "component_type": "model",
                "description": "LightGBM baseline tuned for speed (no Optuna). Use fewer estimators + early stopping/callbacks. Respect KAGGLE_AGENTS_CV_FOLDS for faster iteration.",
                "estimated_impact": 0.18,
                "rationale": "High ROI baseline for tabular tasks; fast enough to iterate under tight time budgets (MLE-bench).",
                "code_outline": "LGBMClassifier/Regressor with sane defaults, 3-fold CV when FAST_MODE, save OOF/test preds",
            },
            {
                "name": "xgboost_fast_cv",
                "component_type": "model",
                "description": "XGBoost baseline tuned for speed (no Optuna). Use hist/gpu_hist where available. Respect time budget and fold count env vars.",
                "estimated_impact": 0.16,
                "rationale": "Provides diversity vs LightGBM with similar compute budget; useful for a quick ensemble.",
                "code_outline": "XGBClassifier/Regressor with fixed params, 3-fold CV when FAST_MODE, save OOF/test preds",
            },
            {
                "name": "catboost_fast_cv",
                "component_type": "model",
                "description": "CatBoost baseline tuned for speed (no Optuna). Handles categoricals natively. Respect time budget and fold count env vars.",
                "estimated_impact": 0.15,
                "rationale": "Alternative to XGBoost/LightGBM with different regularization; handles categoricals well.",
                "code_outline": "CatBoostClassifier/Regressor with sane defaults, 3-fold CV when FAST_MODE, save OOF/test preds",
            },
            {
                "name": "logistic_tfidf",
                "component_type": "model",
                "description": "Logistic Regression with TF-IDF features. Very fast fallback for text-heavy tabular data.",
                "estimated_impact": 0.12,
                "rationale": "Extremely fast linear model; useful when tree models timeout.",
                "code_outline": "TfidfVectorizer + LogisticRegression/Ridge, save OOF/test preds",
            },
            {
                "name": "random_forest_fast",
                "component_type": "model",
                "description": "Random Forest baseline with limited trees (n_estimators=200) for speed.",
                "estimated_impact": 0.13,
                "rationale": "Robust tree ensemble that rarely fails; good fallback option.",
                "code_outline": "RandomForestClassifier/Regressor with n_estimators=200, 3-fold CV, save OOF/test preds",
            },
            {
                "name": "stacking_ensemble",
                "component_type": "ensemble",
                "description": "Stack OOF predictions from available models with LogisticRegression/Ridge meta-learner. Fallback to weighted average if needed.",
                "estimated_impact": 0.10,
                "rationale": "Cheap ensemble step that often improves generalization without additional heavy training.",
                "code_outline": "Load models/oof_*.npy + models/test_*.npy, fit meta-model on OOF, predict test, write submission",
            },
        ]

        # Filter out failed components
        filtered_plan = [c for c in all_fast_candidates if c["name"] not in failed_names]

        # Ensure we have at least 2 models (excluding ensemble) for meaningful stacking
        model_count = sum(1 for c in filtered_plan if c["component_type"] == "model")
        if model_count < 2:
            print("   ⚠️ Less than 2 models available after filtering. Adding simple baseline.")
            filtered_plan.insert(0, {
                "name": "simple_ridge_baseline",
                "component_type": "model",
                "description": "Simple Ridge regression baseline. Cannot fail, always produces predictions.",
                "estimated_impact": 0.08,
                "rationale": "Failsafe baseline that always works.",
                "code_outline": "StandardScaler + Ridge, 3-fold CV, save OOF/test preds",
            })

        # Keep top 2 models + ensemble (avoid bloated plans)
        models = [c for c in filtered_plan if c["component_type"] == "model"][:2]
        ensemble = [c for c in filtered_plan if c["component_type"] == "ensemble"][:1]
        return models + ensemble

    plan = []

    # ALWAYS add feature engineering first (high impact)
    plan.append(
        {
            "name": "advanced_feature_engineering",
            "component_type": "feature_engineering",
            "description": "Create polynomial features (degree 2), feature interactions (ratio, diff, product), statistical transformations (log, sqrt), and target encoding for categorical features",
            "estimated_impact": 0.15,
            "rationale": "Comprehensive feature engineering improves scores by 10-20% in tabular competitions",
            "code_outline": "Use PolynomialFeatures(degree=2), create ratio/diff/product features, apply log/sqrt transforms, use TargetEncoder",
        }
    )

    # ALWAYS add 3 diverse models for ensemble diversity
    plan.extend(
        [
            {
                "name": "lightgbm_optuna_tuned",
                "component_type": "model",
                "description": "LightGBM with Optuna hyperparameter optimization: 15 trials, tuning learning_rate, num_leaves, max_depth, min_child_samples",
                "estimated_impact": 0.22,
                "rationale": "LightGBM consistently wins tabular competitions. Optuna finds better parameters than manual tuning.",
                "code_outline": "LGBMRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=100",
            },
            {
                "name": "xgboost_optuna_tuned",
                "component_type": "model",
                "description": "XGBoost with Optuna hyperparameter optimization: 15 trials, tuning max_depth, learning_rate, subsample, colsample_bytree",
                "estimated_impact": 0.20,
                "rationale": "XGBoost provides different regularization than LightGBM. Optuna ensures optimal capacity.",
                "code_outline": "XGBRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=50",
            },
            {
                "name": "catboost_optuna_tuned",
                "component_type": "model",
                "description": "CatBoost with Optuna hyperparameter optimization: 15 trials, tuning depth, learning_rate, l2_leaf_reg",
                "estimated_impact": 0.19,
                "rationale": "CatBoost handles categorical features natively. Tuning depth is critical for performance.",
                "code_outline": "CatBoostRegressor/Classifier with OptunaSearchCV, cat_features parameter, 5-fold CV",
            },
            {
                "name": "neural_network_mlp",
                "component_type": "model",
                "description": "Simple MLP Neural Network using Scikit-Learn or PyTorch (if available). Standard scaling is CRITICAL.",
                "estimated_impact": 0.15,
                "rationale": "Neural Networks capture different patterns than tree-based models, adding valuable diversity to the ensemble.",
                "code_outline": "MLPClassifier/Regressor or PyTorch simple net. Must use StandardScaler/MinMaxScaler on inputs. Early stopping.",
            },
        ]
    )

    # ALWAYS add stacking ensemble (combines the 4 models above)
    plan.append(
        {
            "name": "stacking_ensemble",
            "component_type": "ensemble",
            "description": "Stack LightGBM, XGBoost, CatBoost, and NN predictions using Ridge/Logistic regression as meta-learner",
            "estimated_impact": 0.25,
            "rationale": "Stacking combines diverse models (Trees + NN) and typically improves scores by 5-10%",
            "code_outline": "StackingRegressor/Classifier with base_estimators=[lgb, xgb, cat, nn], final_estimator=Ridge/LogisticRegression, cv=5",
        }
    )

    # Filter out failed components (if any)
    if failed_names:
        plan = [c for c in plan if c["name"] not in failed_names]
        # Ensure we still have at least 1 model
        model_count = sum(1 for c in plan if c["component_type"] == "model")
        if model_count == 0:
            print("   ⚠️ All models filtered out! Adding simple baseline.")
            plan.insert(0, {
                "name": "simple_ridge_baseline",
                "component_type": "model",
                "description": "Simple Ridge regression baseline. Cannot fail, always produces predictions.",
                "estimated_impact": 0.08,
                "rationale": "Failsafe baseline that always works.",
                "code_outline": "StandardScaler + Ridge, 5-fold CV, save OOF/test preds",
            })

    return plan


def create_image_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    *,
    fast_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Create fallback plan for image competitions (PyTorch/TensorFlow DataLoaders).

    Uses transfer learning with pre-trained CNNs (EfficientNet, ResNet).

    Args:
        domain: Competition domain (image_classification, image_regression, etc.)
        sota_analysis: SOTA analysis results
        fast_mode: If True, return minimal 2-component plan for speed

    Returns:
        List of component dictionaries (2 in fast mode, 3 normally)
    """
    is_regression = "regression" in domain
    task = "regression" if is_regression else "classification"

    # FAST MODE: Only 2 components for maximum speed (MLE-bench optimization)
    if fast_mode:
        return [
            {
                "name": f"efficientnet_b0_fast_{task}",
                "component_type": "model",
                "description": "EfficientNet-B0 with FROZEN backbone. Only train classifier head for 2-3 epochs. Use 2-fold CV (KAGGLE_AGENTS_CV_FOLDS=2). Mixed precision training. Lightweight augmentations only (flip, normalize). IMPLEMENT soft-deadline pattern.",
                "estimated_impact": 0.30,
                "rationale": "Frozen backbone = fastest training. 2 epochs is enough for head fine-tuning. This prioritizes getting a valid submission quickly.",
                "code_outline": "efficientnet_b0(pretrained=True), freeze all backbone layers, train head only, 2 epochs, 2-fold CV, save best checkpoint, implement _check_deadline() pattern",
            },
            {
                "name": "tta_inference_only",
                "component_type": "ensemble",
                "description": "Test-Time Augmentation ONLY (no additional training). Load the single trained model and apply 5 simple transforms (original, hflip, vflip, rotate90, rotate180), average predictions. Write submission.csv.",
                "estimated_impact": 0.05,
                "rationale": "Free accuracy boost without additional training time. Just inference with multiple transforms.",
                "code_outline": "Load models/best_model.* (auto-detect extension), for each test image: apply transforms, average predictions, clip to [0,1], write submission.csv",
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
            "code_outline": "torchvision.models.efficientnet_b0(pretrained=True), replace classifier head, train with CrossEntropyLoss/MSELoss, CV folds via KAGGLE_AGENTS_CV_FOLDS, save OOF predictions for ensemble",
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


def create_text_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create fallback plan for text/NLP competitions (HuggingFace transformers).

    Uses pre-trained language models (RoBERTa, DistilBERT, or T5 for seq2seq).

    Args:
        domain: Competition domain (text_classification, seq_to_seq, etc.)
        sota_analysis: SOTA analysis results

    Returns:
        List of component dictionaries (3 components for classification, 1 for seq2seq)
    """
    if domain == "seq_to_seq":
        # Sequence-to-sequence tasks (translation, text normalization, summarization)
        return [
            {
                "name": "lookup_baseline_deterministic",
                "component_type": "model",
                "description": """LookupBaseline for deterministic semiotic classes.
Uses frequency-based mapping from training data for PLAIN, PUNCT, LETTERS, VERBATIM.
Expected coverage: 80%+ of tokens with zero inference cost.""",
                "estimated_impact": 0.35,
                "rationale": "SOTA pattern: lookup handles majority of tokens deterministically. Proven in text normalization competitions to achieve 99%+ accuracy on deterministic classes.",
                "code_outline": """from kaggle_agents.utils.text_normalization import LookupBaseline, DETERMINISTIC_CLASSES

lookup = LookupBaseline()
lookup.fit(train_df, class_col='class', before_col='before', after_col='after')
predictions = lookup.predict_batch(test_df)
# Use predictions['prediction'] for deterministic classes""",
            },
            {
                "name": "t5_seq2seq_ambiguous",
                "component_type": "model",
                "description": """T5-small fine-tuned ONLY on ambiguous tokens (CARDINAL, DATE, TIME, MONEY, MEASURE).
Uses create_hybrid_pipeline() to identify which tokens need neural processing.
Input format: "class: DATE before: 1/2/2023"
Output: "january second twenty twenty three" """,
                "estimated_impact": 0.25,
                "rationale": "Neural model for context-dependent transformations that lookup cannot handle. Training only on ambiguous tokens (20% of data) makes training 5x faster.",
                "code_outline": """from kaggle_agents.utils.text_normalization import create_hybrid_pipeline, get_neural_training_config

pipeline = create_hybrid_pipeline(train_df, fast_mode=True, timeout_s=1800)
ambiguous_df = pipeline['ambiguous_df']
neural_config = pipeline['neural_config']

# Train T5 only on ambiguous samples with max_steps guard
from transformers import T5ForConditionalGeneration, Trainer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
# Use neural_config['max_steps'] to prevent timeout""",
            },
            {
                "name": "hybrid_ensemble",
                "component_type": "ensemble",
                "description": """Combines lookup baseline with T5 neural predictions.
Uses lookup for confident predictions, T5 for ambiguous cases.""",
                "estimated_impact": 0.10,
                "rationale": "Hybrid approach proven in SOTA to achieve 99%+ accuracy. Lookup provides fast, deterministic results for common cases; neural handles edge cases.",
                "code_outline": """from kaggle_agents.utils.text_normalization import apply_hybrid_predictions

final_predictions = apply_hybrid_predictions(
    test_df, lookup, neural_predictions, neural_indices
)""",
            },
        ]
    # Classification or regression tasks
    return [
        {
            "name": "roberta_classifier",
            "component_type": "model",
            "description": "RoBERTa-base fine-tuned for text classification with learning rate warmup and linear decay schedule.",
            "estimated_impact": 0.28,
            "rationale": "RoBERTa improves on BERT with dynamic masking and larger training corpus. Achieves SOTA on GLUE, SuperGLUE, and many NLP benchmarks. Warmup stabilizes training.",
            "code_outline": "transformers.RobertaForSequenceClassification.from_pretrained('roberta-base'), AutoTokenizer, Trainer API with TrainingArguments, AdamW optimizer with warmup_steps=500, 5-fold StratifiedKFold CV, save OOF predictions",
        },
        {
            "name": "distilbert_classifier",
            "component_type": "model",
            "description": "DistilBERT fine-tuned (60% faster than BERT, lighter for ensemble diversity).",
            "estimated_impact": 0.22,
            "rationale": "DistilBERT is 60% faster and 40% smaller than BERT while retaining 97% of performance through knowledge distillation. Provides architectural diversity for ensemble while being computationally efficient.",
            "code_outline": "transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'), similar training setup to RoBERTa, 5-fold CV",
        },
        {
            "name": "transformer_ensemble",
            "component_type": "ensemble",
            "description": "Weighted average of RoBERTa and DistilBERT predictions using CV scores as weights.",
            "estimated_impact": 0.12,
            "rationale": "Different architectures (RoBERTa vs DistilBERT) capture different linguistic patterns. Ensemble reduces variance and overfitting to specific model biases.",
            "code_outline": "Load OOF predictions from both models, compute optimal weights via Ridge regression on validation fold, apply weighted average to test predictions",
        },
    ]


def create_audio_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create fallback plan for audio competitions (mel-spectrograms + CNNs).

    Converts audio to spectrograms, then uses image models.

    Args:
        domain: Competition domain (audio_classification, audio_regression)
        sota_analysis: SOTA analysis results

    Returns:
        List of component dictionaries (4 components: 1 preprocessing + 2 models + 1 ensemble)
    """
    return [
        {
            "name": "mel_spectrogram_preprocessing",
            "component_type": "preprocessing",
            "description": "Convert audio files to mel-spectrograms using librosa. Save as PNG images for CNN input.",
            "estimated_impact": 0.20,
            "rationale": "Mel-spectrograms are the standard time-frequency representation for audio. Convert audio problem to computer vision problem, enabling use of powerful pre-trained image models.",
            "code_outline": "Use librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128), convert to dB scale with librosa.power_to_db(), normalize to [0, 255], save as 3-channel PNG to spectrograms/ directory",
        },
        {
            "name": "efficientnet_audio",
            "component_type": "model",
            "description": "EfficientNet-B0 trained on mel-spectrogram images. Transfer learning from ImageNet.",
            "estimated_impact": 0.25,
            "rationale": "CNNs excel at recognizing patterns in spectrograms (frequency bands, temporal patterns). EfficientNet provides excellent accuracy with computational efficiency.",
            "code_outline": "Load mel-spectrogram images with PyTorch DataLoader, torchvision.models.efficientnet_b0(pretrained=True), replace classifier, train with data augmentation on spectrograms",
        },
        {
            "name": "resnet_audio",
            "component_type": "model",
            "description": "ResNet50 for architectural diversity in ensemble.",
            "estimated_impact": 0.20,
            "rationale": "ResNet learns different features than EfficientNet due to different architecture (residual connections). Ensemble benefits from this diversity.",
            "code_outline": "Similar pipeline to EfficientNet but with torchvision.models.resnet50(pretrained=True)",
        },
        {
            "name": "audio_ensemble",
            "component_type": "ensemble",
            "description": "Weighted average of EfficientNet and ResNet predictions.",
            "estimated_impact": 0.12,
            "rationale": "Ensemble reduces overfitting to specific architecture biases and improves generalization.",
            "code_outline": "Load OOF predictions, compute weights by CV score, weighted average for test predictions",
        },
    ]


def create_diversified_fallback_plan(
    state: KaggleState,
    sota_analysis: dict[str, Any],
    focus: str,
) -> list[dict[str, Any]]:
    """
    Create a diversified fallback plan with a specific focus.

    Args:
        state: Current state
        sota_analysis: SOTA analysis
        focus: Focus area ('deep_learning', 'feature_engineering', 'ensemble')

    Returns:
        Diversified plan as list of dicts
    """
    if focus == "deep_learning":
        return [
            {
                "name": "nn_tabular",
                "component_type": "model",
                "description": "Neural network for tabular data (TabNet or MLP)",
                "estimated_impact": 0.18,
                "rationale": "Deep learning alternative to tree models",
                "code_outline": "TabNet/MLP with entity embeddings, batch norm, dropout",
            },
            {
                "name": "gradient_blend",
                "component_type": "ensemble",
                "description": "Gradient-based blending of NN and tree models",
                "estimated_impact": 0.12,
                "rationale": "Combine NN and tree strengths",
                "code_outline": "Weighted average with learned weights via gradient descent",
            },
        ]
    if focus == "feature_engineering":
        return [
            {
                "name": "target_encoding_cv",
                "component_type": "feature_engineering",
                "description": "Target encoding with proper CV to avoid leakage",
                "estimated_impact": 0.15,
                "rationale": "Powerful encoding for categorical features",
                "code_outline": "category_encoders.TargetEncoder with cv=5 folds",
            },
            {
                "name": "feature_selection",
                "component_type": "feature_engineering",
                "description": "Feature selection using importance + RFE",
                "estimated_impact": 0.10,
                "rationale": "Remove noise features",
                "code_outline": "RFECV or SelectFromModel with LightGBM importances",
            },
            {
                "name": "lightgbm_tuned",
                "component_type": "model",
                "description": "LightGBM with Optuna hyperparameter tuning",
                "estimated_impact": 0.20,
                "rationale": "Better hyperparameters",
                "code_outline": "Optuna study with n_trials=50 for LGBM params",
            },
        ]
    # ensemble focus
    return [
        {
            "name": "stacking_meta",
            "component_type": "ensemble",
            "description": "Stacking ensemble with ridge meta-learner",
            "estimated_impact": 0.15,
            "rationale": "Combine diverse model predictions",
            "code_outline": "StackingClassifier/Regressor with Ridge meta",
        },
        {
            "name": "voting_diverse",
            "component_type": "ensemble",
            "description": "Voting ensemble with diverse base models",
            "estimated_impact": 0.10,
            "rationale": "Simple but effective ensemble",
            "code_outline": "VotingClassifier with LGBM, XGB, CatBoost",
        },
    ]
