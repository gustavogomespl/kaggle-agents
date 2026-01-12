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
        # PRIORITY: Lookup-first strategy for 80%+ coverage, neural only for ambiguous
        return [
            {
                "name": "lookup_baseline",
                "component_type": "preprocessing",
                "description": (
                    "Frequency-based lookup table for (class, before) -> after mappings. "
                    "Handles 80%+ of tokens with zero inference cost. "
                    "Identifies ambiguous samples that need neural refinement."
                ),
                "estimated_impact": 0.75,
                "rationale": (
                    "Text normalization has high determinism: PUNCT/PLAIN/VERBATIM always map to <self>, "
                    "LETTERS always spell out, common CARDINALs have fixed mappings. "
                    "Building lookup from training data captures these patterns perfectly. "
                    "Neural models are overkill for deterministic classes."
                ),
                "code_outline": (
                    "from kaggle_agents.utils.text_normalization import LookupBaseline, create_hybrid_pipeline; "
                    "pipeline = create_hybrid_pipeline(train_df, fast_mode=FAST_MODE); "
                    "lookup = pipeline['lookup']; "
                    "lookup.save(MODELS_DIR / 'lookup_baseline.json'); "
                    "Identify ambiguous samples via pipeline['ambiguous_indices']"
                ),
            },
            {
                "name": "rule_based_normalizer",
                "component_type": "preprocessing",
                "description": (
                    "Class-specific fallback rules for deterministic text normalization. "
                    "Handles PUNCT (keep as-is), LETTERS (spell out), PLAIN (keep), "
                    "VERBATIM (keep), TRANS (transliteration), ELECTRONIC (URLs/emails). "
                    "Complements lookup for unseen (class, before) pairs."
                ),
                "estimated_impact": 0.10,
                "rationale": (
                    "Fallback rules handle cases not seen in training data. "
                    "For deterministic classes like PUNCT/PLAIN, keep as-is is always correct. "
                    "LETTERS can be reliably spelled out. Rules provide guaranteed coverage."
                ),
                "code_outline": (
                    "Class-specific fallbacks already built into LookupBaseline; "
                    "For additional coverage: use num2words for unseen CARDINALs, "
                    "spell_out for LETTERS, keep-as-is for PLAIN/PUNCT/VERBATIM"
                ),
            },
            {
                "name": "t5_small_ambiguous_only",
                "component_type": "model",
                "description": (
                    "T5-small fine-tuned ONLY on ambiguous cases (DATE, TIME, MEASURE). "
                    "CRITICAL: max_steps=2000 guard prevents runaway training. "
                    "Uses HF compatibility wrapper for eval_strategy parameter."
                ),
                "estimated_impact": 0.12,
                "rationale": (
                    "T5-small (60M params) is faster than T5-base (220M) and sufficient "
                    "when most tokens are already handled by lookup/rules. "
                    "Training only on ambiguous samples (~10-20% of data) dramatically "
                    "reduces training time from 140 hours to ~30 minutes."
                ),
                "code_outline": (
                    "from kaggle_agents.utils.hf_compat import get_training_args_kwargs; "
                    "from kaggle_agents.utils.text_normalization import get_neural_training_config; "
                    "config = get_neural_training_config(n_ambiguous, fast_mode=FAST_MODE); "
                    "model = T5ForConditionalGeneration.from_pretrained('t5-small'); "
                    "args = Seq2SeqTrainingArguments(max_steps=config['max_steps'], "
                    "**get_training_args_kwargs(eval_strategy='steps', eval_steps=500)); "
                    "Train ONLY on ambiguous_df from pipeline"
                ),
            },
            {
                "name": "hybrid_ensemble",
                "component_type": "ensemble",
                "description": (
                    "Lookup-priority ensemble: use lookup/rules first, T5 only for failures. "
                    "Achieves high accuracy with sub-second inference per sample."
                ),
                "estimated_impact": 0.03,
                "rationale": (
                    "Deterministic lookup is always correct when available. "
                    "Neural model fills gaps for ambiguous patterns. "
                    "This hybrid achieves winning-level accuracy with 100x faster inference."
                ),
                "code_outline": (
                    "from kaggle_agents.utils.text_normalization import apply_hybrid_predictions; "
                    "lookup = LookupBaseline.load(MODELS_DIR / 'lookup_baseline.json'); "
                    "final_preds = apply_hybrid_predictions(test_df, lookup, neural_preds, neural_indices); "
                    "submission['after'] = final_preds"
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
                "T5 uses text-to-text format ideal for translation, summarization, and generation. "
                "CRITICAL: Uses max_steps=2000 guard and HF compatibility wrapper."
            ),
            "estimated_impact": 0.35,
            "rationale": (
                "T5 (Text-to-Text Transfer Transformer) achieves SOTA on multiple seq2seq benchmarks. "
                "The text-to-text format unifies different NLP tasks into a single paradigm. "
                "max_steps=2000 prevents timeout in constrained environments."
            ),
            "code_outline": (
                "from kaggle_agents.utils.hf_compat import get_training_args_kwargs; "
                "T5ForConditionalGeneration.from_pretrained('t5-base'), "
                "T5Tokenizer, Seq2SeqTrainer with DataCollatorForSeq2Seq, "
                "args = Seq2SeqTrainingArguments(max_steps=2000, learning_rate=1e-4, "
                "**get_training_args_kwargs(eval_strategy='steps', eval_steps=500)), "
                "metric: BLEU or ROUGE depending on task"
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
