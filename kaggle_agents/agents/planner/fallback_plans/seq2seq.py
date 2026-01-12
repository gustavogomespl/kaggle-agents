"""
Seq2seq / text normalization competition fallback plan.

Specialized for tasks like text normalization, translation, and summarization.
Uses a hybrid approach: rule-based for deterministic patterns + neural for ambiguous cases.
"""

from typing import Any


def create_seq2seq_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
    competition_name: str = "",
) -> list[dict[str, Any]]:
    """
    Create fallback plan for seq2seq/text normalization competitions.

    Text normalization competitions (like Kaggle's text-normalization-challenge-english-language)
    are best solved with a HYBRID approach:
    1. Rule-based system for deterministic classes (PUNCT, LETTERS, etc.)
    2. Neural seq2seq for ambiguous classes (CARDINAL, DATE with multiple formats)
    3. Rule-priority ensemble (apply rules first, model for fallback)

    Args:
        domain: Competition domain (seq_to_seq, text_normalization)
        sota_analysis: SOTA analysis results
        competition_name: Name of the competition

    Returns:
        List of component dictionaries (3 components for hybrid approach)
    """
    is_text_norm = domain == "text_normalization" or any(
        kw in competition_name.lower()
        for kw in ["normalization", "normalize", "text-norm", "tts", "speech"]
    )

    if is_text_norm:
        # Specialized plan for text normalization (ITN/TN tasks)
        return [
            {
                "name": "rule_based_normalizer",
                "component_type": "preprocessing",
                "description": (
                    "Class-specific regex rules for deterministic text normalization. "
                    "Handles PUNCT (keep as-is), LETTERS (spell out), PLAIN (keep), "
                    "VERBATIM (keep), TRANS (transliteration), ELECTRONIC (URLs/emails). "
                    "These classes have deterministic transformations without ambiguity."
                ),
                "estimated_impact": 0.70,
                "rationale": (
                    "80%+ of text normalization tokens have deterministic mappings. "
                    "Rule-based approaches achieve near-perfect accuracy on these classes "
                    "while being infinitely faster than neural approaches. Focus neural "
                    "compute budget on genuinely ambiguous cases."
                ),
                "code_outline": (
                    "Create class-specific normalization rules: "
                    "PUNCT='<self>', PLAIN='<self>', LETTERS=spell_out(), "
                    "CARDINAL=num2words(), ORDINAL=ordinal2words(), "
                    "MONEY=format_currency(), DATE/TIME=format_datetime(), "
                    "Use regex patterns to identify classes, apply transformations"
                ),
            },
            {
                "name": "t5_seq2seq_ambiguous",
                "component_type": "model",
                "description": (
                    "T5-small fine-tuned ONLY on ambiguous cases (DATE, CARDINAL, MEASURE). "
                    "Smaller model is faster and sufficient when most cases are handled by rules."
                ),
                "estimated_impact": 0.25,
                "rationale": (
                    "T5 excels at text-to-text tasks but is expensive to run on all tokens. "
                    "By filtering to only ambiguous cases (DATE with multiple formats, "
                    "CARDINAL with context-dependent pronunciation), we get neural accuracy "
                    "where needed without wasting compute on deterministic classes."
                ),
                "code_outline": (
                    "Filter training data to ambiguous classes only, "
                    "T5ForConditionalGeneration.from_pretrained('t5-small'), "
                    "Seq2SeqTrainer with DataCollatorForSeq2Seq, "
                    "Train with learning_rate=3e-4, batch_size=32, "
                    "Evaluate with exact match accuracy"
                ),
            },
            {
                "name": "hybrid_ensemble",
                "component_type": "ensemble",
                "description": (
                    "Rule-priority ensemble: apply rules first, use T5 for unhandled/ambiguous cases. "
                    "This achieves high accuracy with fast inference."
                ),
                "estimated_impact": 0.05,
                "rationale": (
                    "Deterministic rules should always override neural predictions for reliability. "
                    "Neural model only fills gaps where rules don't apply or are ambiguous. "
                    "This hybrid approach is the winning strategy for text normalization competitions."
                ),
                "code_outline": (
                    "For each (class, token) pair: "
                    "1. Check if deterministic rule exists for class "
                    "2. If yes, apply rule (faster, reliable) "
                    "3. If no, use T5 prediction (neural fallback) "
                    "4. Save final predictions in submission format"
                ),
            },
        ]

    # Generic seq2seq plan (translation, summarization, etc.)
    return [
        {
            "name": "t5_base_seq2seq",
            "component_type": "model",
            "description": (
                "T5-base fine-tuned for seq2seq task using HuggingFace Seq2SeqTrainer. "
                "T5 uses text-to-text format ideal for translation, summarization, and generation."
            ),
            "estimated_impact": 0.35,
            "rationale": (
                "T5 (Text-to-Text Transfer Transformer) achieves SOTA on multiple seq2seq benchmarks. "
                "The text-to-text format unifies different NLP tasks into a single paradigm. "
                "Base size provides good balance between performance and training speed."
            ),
            "code_outline": (
                "T5ForConditionalGeneration.from_pretrained('t5-base'), "
                "T5Tokenizer, Seq2SeqTrainer with DataCollatorForSeq2Seq, "
                "learning_rate=1e-4, per_device_train_batch_size=8, "
                "evaluation_strategy='steps', metric: BLEU or ROUGE depending on task"
            ),
        },
        {
            "name": "bart_seq2seq",
            "component_type": "model",
            "description": (
                "BART-base as an alternative encoder-decoder architecture for ensemble diversity. "
                "BART uses denoising autoencoder pre-training which excels at text generation."
            ),
            "estimated_impact": 0.25,
            "rationale": (
                "BART provides architectural diversity from T5 (denoising vs text-to-text pre-training). "
                "Ensemble of different architectures reduces overfitting to single model biases. "
                "BART is particularly strong for abstractive summarization and text generation."
            ),
            "code_outline": (
                "BartForConditionalGeneration.from_pretrained('facebook/bart-base'), "
                "BartTokenizer, similar Seq2SeqTrainer setup as T5, "
                "num_beams=4 for beam search decoding"
            ),
        },
        {
            "name": "seq2seq_ensemble",
            "component_type": "ensemble",
            "description": (
                "Ensemble T5 and BART predictions using validation BLEU/ROUGE scores as weights. "
                "Use majority voting for discrete outputs or weighted average for continuous."
            ),
            "estimated_impact": 0.10,
            "rationale": (
                "Different seq2seq architectures capture different aspects of the mapping. "
                "Ensemble reduces variance and provides more robust predictions. "
                "Weight by validation performance to favor the better-performing model."
            ),
            "code_outline": (
                "Load predictions from both models, "
                "For each sample: compute weighted average of beam scores, "
                "Or use majority voting if predictions differ, "
                "Save final predictions"
            ),
        },
    ]
