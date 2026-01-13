"""Extended strategies for computer vision competitions."""

EXTENDED_STRATEGIES_CV = {
    "feature_engineering_heavy": {
        "name": "feature_engineering_heavy",
        "prompt_modifier": """
Focus on advanced image augmentation and preprocessing:
- Heavy augmentation: Cutmix, Mixup, GridMask, RandomErasing
- Test Time Augmentation (TTA) with multiple crops
- External data integration if allowed
- Multi-scale feature extraction
""",
        "model_preference": ["efficientnet_b3", "resnet50"],
        "component_emphasis": ["preprocessing", "augmentation"],
    },
    "neural_exploration": {
        "name": "neural_exploration",
        "prompt_modifier": """
Explore SOTA vision architectures:
- Vision Transformers (ViT, DeiT, Swin)
- ConvNeXt for modern CNN approach
- Hybrid CNN-Transformer models
- Knowledge distillation from larger models
""",
        "model_preference": ["vit", "swin", "convnext", "deit"],
        "component_emphasis": ["model"],
    },
    "hyperparameter_variant": {
        "name": "hyperparameter_variant",
        "prompt_modifier": """
Explore training variations:
- Different learning rate schedules (cosine, warmup)
- Vary image sizes (224, 384, 512)
- Different optimizers (AdamW, SAM, LAMB)
- Label smoothing and loss variants
""",
        "model_preference": ["efficientnet", "resnet"],
        "component_emphasis": ["model"],
        "inherit_from_best": True,
    },
}
