"""
Text/NLP competition fallback plan.

Uses a fold-local sparse baseline unless the task is sequence-to-sequence.
"""

from typing import Any


def create_text_fallback_plan(
    domain: str,
    sota_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create a bounded fallback plan for text/NLP competitions.

    Args:
        domain: Competition domain (text_classification, seq_to_seq, etc.)
        sota_analysis: SOTA analysis results

    Returns:
        List of component dictionaries (1 component)
    """
    if domain == "seq_to_seq":
        # Sequence-to-sequence tasks (translation, text normalization, summarization)
        return [
            {
                "name": "t5_base_seq2seq",
                "component_type": "model",
                "description": "T5-base fine-tuned for seq2seq task using HuggingFace Trainer API. T5 is designed for text-to-text tasks.",
                "estimated_impact": 0.30,
                "rationale": "T5 (Text-to-Text Transfer Transformer) is specifically designed for seq2seq tasks. Achieves SOTA on translation, summarization, and text normalization benchmarks.",
                "code_outline": "transformers.T5ForConditionalGeneration.from_pretrained('t5-base'), T5Tokenizer, Seq2SeqTrainer with DataCollatorForSeq2Seq, train with learning_rate=1e-4, evaluate with BLEU/ROUGE metrics",
            }
        ]
    if "regression" in domain.lower():
        return [
            {
                "name": "word_char_tfidf_ridge",
                "component_type": "model",
                "description": "A single word and character TF-IDF Ridge regression baseline fitted only on each fold's training rows.",
                "estimated_impact": 0.24,
                "rationale": "Sparse word and character features provide a deterministic continuous baseline without fitting vocabulary on validation or public test rows.",
                "code_outline": "Load train_df with align_train_to_canonical(pd.read_csv(TRAIN_PATH)); read text_feature_cols from CANONICAL_METADATA and use its declared text column only; use CANONICAL_Y and TARGET_COLS. For every canonical fold, create word_vectorizer=TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2)) and char_vectorizer=TfidfVectorizer(analyzer='char_wb', max_features=5000, min_df=2, ngram_range=(3, 5)); call fit_transform only on train_texts.iloc[train_idx], transform validation and test text with those fold-local vectorizers, hstack word+char matrices, and fit Ridge only on y[train_idx]. Accumulate continuous aligned OOF predictions and averaged test predictions; call save_component_artifacts(oof_preds, test_preds) and write_submission(test_preds).",
            }
        ]
    # Classification or regression tasks. One candidate avoids duplicate GPU
    # downloads and keeps the generated execution auditable.
    return [
        {
            "name": "word_char_tfidf_logreg",
            "component_type": "model",
            "description": "A single word and character TF-IDF LogisticRegression baseline fitted only on each fold's training rows.",
            "estimated_impact": 0.24,
            "rationale": "Sparse word and character features are fast, deterministic, and provide a valid baseline without downloading pretrained weights or fitting vocabulary on public test rows.",
            "code_outline": "Load train_df with align_train_to_canonical(pd.read_csv(TRAIN_PATH)); read text_feature_cols from CANONICAL_METADATA and use its declared text column only; use CANONICAL_Y, N_TARGETS, TARGET_COLS, and SUBMISSION_TARGET_COLS (never infer a target and never feed a non-declared metadata column to the vectorizer). For every canonical fold, create word_vectorizer=TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2)) and char_vectorizer=TfidfVectorizer(analyzer='char_wb', max_features=5000, min_df=2, ngram_range=(3, 5)); call fit_transform only on train_texts.iloc[train_idx], then transform val_texts and test_texts with those same fold-local vectorizers and hstack word+char matrices. For binary single-target classification fit LogisticRegression and save the positive-class probability as one output. For multiclass single-target classification keep all predict_proba columns as OOF/test artifact probabilities and save class_order. If len(SUBMISSION_TARGET_COLS) > 1, reorder those probabilities to the declared wide class_order and submit them; if len(SUBMISSION_TARGET_COLS) == 1, use argmax plus class_order to produce one column of submission labels while retaining full artifact probabilities. When N_TARGETS > 1, wrap LogisticRegression in OneVsRestClassifier and save exactly N_TARGETS multilabel probabilities without flattening y. Call save_component_artifacts(oof_preds, test_preds, class_order=class_order when multiclass) and write_submission(submission_predictions).",
        },
    ]
