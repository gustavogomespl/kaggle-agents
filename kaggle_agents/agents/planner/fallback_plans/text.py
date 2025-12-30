"""
Text/NLP competition fallback plan.

Uses pre-trained language models (RoBERTa, DistilBERT, T5).
"""

from typing import Any


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
        return [{
            "name": "t5_base_seq2seq",
            "component_type": "model",
            "description": "T5-base fine-tuned for seq2seq task using HuggingFace Trainer API. T5 is designed for text-to-text tasks.",
            "estimated_impact": 0.30,
            "rationale": "T5 (Text-to-Text Transfer Transformer) is specifically designed for seq2seq tasks. Achieves SOTA on translation, summarization, and text normalization benchmarks.",
            "code_outline": "transformers.T5ForConditionalGeneration.from_pretrained('t5-base'), T5Tokenizer, Seq2SeqTrainer with DataCollatorForSeq2Seq, train with learning_rate=1e-4, evaluate with BLEU/ROUGE metrics"
        }]
    else:
        # Classification or regression tasks
        return [
            {
                "name": "roberta_classifier",
                "component_type": "model",
                "description": "RoBERTa-base fine-tuned for text classification with learning rate warmup and linear decay schedule.",
                "estimated_impact": 0.28,
                "rationale": "RoBERTa improves on BERT with dynamic masking and larger training corpus. Achieves SOTA on GLUE, SuperGLUE, and many NLP benchmarks. Warmup stabilizes training.",
                "code_outline": "transformers.RobertaForSequenceClassification.from_pretrained('roberta-base'), AutoTokenizer, Trainer API with TrainingArguments, AdamW optimizer with warmup_steps=500, 5-fold StratifiedKFold CV, save OOF predictions"
            },
            {
                "name": "distilbert_classifier",
                "component_type": "model",
                "description": "DistilBERT fine-tuned (60% faster than BERT, lighter for ensemble diversity).",
                "estimated_impact": 0.22,
                "rationale": "DistilBERT is 60% faster and 40% smaller than BERT while retaining 97% of performance through knowledge distillation. Provides architectural diversity for ensemble while being computationally efficient.",
                "code_outline": "transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'), similar training setup to RoBERTa, 5-fold CV"
            },
            {
                "name": "transformer_ensemble",
                "component_type": "ensemble",
                "description": "Weighted average of RoBERTa and DistilBERT predictions using CV scores as weights.",
                "estimated_impact": 0.12,
                "rationale": "Different architectures (RoBERTa vs DistilBERT) capture different linguistic patterns. Ensemble reduces variance and overfitting to specific model biases.",
                "code_outline": "Load OOF predictions from both models, compute optimal weights via Ridge regression on validation fold, apply weighted average to test predictions"
            }
        ]
