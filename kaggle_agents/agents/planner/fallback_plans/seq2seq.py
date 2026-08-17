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

    Text-normalization tasks often benefit from a data-driven HYBRID approach:
    1. Learn repeated input-to-output mappings from the training split.
    2. Measure mapping ambiguity on held-out folds.
    3. Use a neural seq2seq model only where the empirical mapping is uncertain.

    Args:
        domain: Competition domain (seq_to_seq, text_normalization)
        sota_analysis: SOTA analysis results
        competition_name: Retained for API compatibility; never used for routing

    Returns:
        List of component dictionaries (3 components for hybrid approach)
    """
    del competition_name

    # Select the mapping-heavy plan only from an explicit task family produced
    # by data/domain inspection. Competition titles are not evidence about the
    # schema and must not alter the plan.
    metadata = sota_analysis.get("canonical_metadata") or sota_analysis.get(
        "data_contract"
    )
    if not isinstance(metadata, dict):
        metadata = {}
    task_family = str(
        sota_analysis.get("task_type")
        or metadata.get("task_type")
        or domain
    ).strip().lower()
    # A lookup/rule candidate is cheap to evaluate for any seq2seq mapping.
    # Its OOF coverage gate makes it a no-op for translation/summarization when
    # inputs do not repeat, while preserving the strong data-derived path for
    # tasks with recurring transformations.
    use_validated_mapping_candidate = task_family in {
        "seq2seq",
        "seq_to_seq",
        "text_normalization",
        "translation",
        "summarization",
    }

    if use_validated_mapping_candidate:
        # The lookup path is selected from the detected domain, while its
        # coverage and ambiguity thresholds are learned from the supplied data.
        return [
            {
                "name": "lookup_baseline",
                "component_type": "preprocessing",
                "description": (
                    "Cross-validated frequency lookup for the detected input, optional "
                    "type/context, and target columns. Measure lookup coverage and mapping "
                    "purity from training folds, then flag uncertain samples for refinement."
                ),
                "estimated_impact": 0.75,
                "rationale": (
                    "Repeated transformations can be learned cheaply from the supplied "
                    "training data. Cross-validation determines whether lookup generalizes "
                    "instead of assuming that named categories follow fixed rules."
                ),
                "code_outline": (
                    "from kaggle_agents.utils.text_normalization import LookupBaseline, create_hybrid_pipeline; "
                    "pipeline = create_hybrid_pipeline("
                    "train_df, test_df=test_df, column_contract=CANONICAL_METADATA, "
                    "sample_submission=SAMPLE_SUBMISSION_PATH, fast_mode=FAST_MODE); "
                    "lookup = pipeline['lookup']; "
                    "lookup.save(MODELS_DIR / 'lookup_baseline.json'); "
                    "Report held-out lookup coverage and identify ambiguous samples via "
                    "pipeline['ambiguous_indices']"
                ),
            },
            {
                "name": "validated_mapping_rules",
                "component_type": "preprocessing",
                "description": (
                    "Infer conservative class-conditional fallback rules from training rows. "
                    "Apply identity or character-level transformations only when held-out "
                    "validation confirms that the rule is deterministic."
                ),
                "estimated_impact": 0.10,
                "rationale": (
                    "Validated fallbacks cover unseen inputs without encoding target-specific "
                    "class behavior in the prompt. Low-purity groups remain neural-model cases."
                ),
                "code_outline": (
                    "Estimate per-group identity rate, mapping entropy, and character-rule "
                    "accuracy on held-out folds; enable only rules that pass the chosen "
                    "validation threshold"
                ),
            },
            {
                "name": "t5_small_ambiguous_only",
                "component_type": "model",
                "description": (
                    "T5-small fine-tuned only on samples marked ambiguous by the "
                    "cross-validated lookup/rule stage. "
                    "A budget-derived max_steps guard prevents runaway training. "
                    "Uses HF compatibility wrapper for eval_strategy parameter."
                ),
                "estimated_impact": 0.12,
                "rationale": (
                    "A compact model limits cost when a validated lookup handles a meaningful "
                    "fraction of rows. The actual routed fraction and runtime must be measured "
                    "on this dataset rather than assumed."
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
                    "Use the validated deterministic prediction when available and the neural "
                    "prediction otherwise; report routing coverage and fold performance."
                ),
                "estimated_impact": 0.03,
                "rationale": (
                    "The lookup reduces variance for mappings that repeat across folds, while "
                    "the neural model handles uncertain mappings. Weights and routing thresholds "
                    "come only from validation data."
                ),
                "code_outline": (
                    "from kaggle_agents.utils.text_normalization import apply_hybrid_predictions; "
                    "lookup = LookupBaseline.load(MODELS_DIR / 'lookup_baseline.json'); "
                    "final_preds = apply_hybrid_predictions(test_df, lookup, neural_preds, neural_indices); "
                    "assert len(SUBMISSION_TARGET_COLS) == 1, "
                    "'hybrid output requires one submission target'; "
                    "write_submission(final_preds)"
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
                "Uses a budget-derived max_steps guard and HF compatibility wrapper."
            ),
            "estimated_impact": 0.35,
            "rationale": (
                "A compact text-to-text model is a candidate for rows the OOF router "
                "marks uncertain. Retain it only when held-out sequence metrics improve "
                "within the measured runtime budget."
            ),
            "code_outline": (
                "from kaggle_agents.utils.hf_compat import get_training_args_kwargs; "
                "from kaggle_agents.utils.text_normalization import get_neural_training_config; "
                "config = get_neural_training_config(len(train_df), fast_mode=FAST_MODE); "
                "T5ForConditionalGeneration.from_pretrained('t5-base'), "
                "T5Tokenizer, Seq2SeqTrainer with DataCollatorForSeq2Seq, "
                "args = Seq2SeqTrainingArguments(max_steps=config['max_steps'], learning_rate=1e-4, "
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
