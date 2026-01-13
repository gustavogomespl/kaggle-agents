"""Extended strategies for NLP competitions."""

EXTENDED_STRATEGIES_NLP = {
    "feature_engineering_heavy": {
        "name": "feature_engineering_heavy",
        "prompt_modifier": """
Focus on text preprocessing and feature extraction:
- Advanced tokenization strategies
- Domain-specific vocabulary expansion
- Text augmentation (back-translation, synonym replacement)
- Sentence-level and document-level features
""",
        "model_preference": ["roberta_base", "deberta"],
        "component_emphasis": ["preprocessing"],
    },
    "neural_exploration": {
        "name": "neural_exploration",
        "prompt_modifier": """
Explore advanced NLP architectures:
- DeBERTa-v3 (large and xlarge variants)
- Longformer for long documents
- Multi-task learning approaches
- Ensemble of different model sizes
""",
        "model_preference": ["deberta_v3_large", "longformer", "roberta_large"],
        "component_emphasis": ["model"],
    },
    "hyperparameter_variant": {
        "name": "hyperparameter_variant",
        "prompt_modifier": """
Explore training variations:
- Different learning rates (1e-5, 2e-5, 3e-5)
- Layer-wise learning rate decay
- Different pooling strategies (CLS, mean, max)
- Gradient accumulation for larger batch sizes
""",
        "model_preference": ["roberta", "deberta"],
        "component_emphasis": ["model"],
        "inherit_from_best": True,
    },
}
