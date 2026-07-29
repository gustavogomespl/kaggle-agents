"""
NLP/Text task constraints.
"""

NLP_CONSTRAINTS = """## NLP/TEXT REQUIREMENTS:

### 1. Text Preprocessing
- Use standard tokenization (BERT, RoBERTa tokenizers)
- Handle max_length: truncate or use sliding window
- Clean text: lowercase, remove special chars (if appropriate)

### 2. Pretrained Models
- PREFERRED backbone: DeBERTa-v3 ('microsoft/deberta-v3-small' fast /
  'microsoft/deberta-v3-base' when budget allows) - consistently stronger than
  BERT/RoBERTa on classification. Fallbacks: 'roberta-base', 'distilbert-base-uncased'.
- Fine-tune with fp16 (torch.cuda.amp) on GPU
- Use appropriate attention mask and token type ids

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
model = AutoModel.from_pretrained('microsoft/deberta-v3-small')

encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)
```

### 3. Classification Head
```python
class TextClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(self.base.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)
```

### 4. TF-IDF + Traditional ML Baseline
For quick baselines or when transformers are too slow, fit a fresh vocabulary
inside every canonical fold. Never fit IDF/vocabulary on validation or test
text:
```python
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():
    word_vectorizer = TfidfVectorizer(
        max_features=5000, min_df=2, ngram_range=(1, 2)
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", max_features=5000, min_df=2, ngram_range=(3, 5)
    )
    X_train = hstack([
        word_vectorizer.fit_transform(train_texts.iloc[train_idx]),
        char_vectorizer.fit_transform(train_texts.iloc[train_idx]),
    ])
    X_val = hstack([
        word_vectorizer.transform(train_texts.iloc[val_idx]),
        char_vectorizer.transform(train_texts.iloc[val_idx]),
    ])
    X_test = hstack([
        word_vectorizer.transform(test_texts),
        char_vectorizer.transform(test_texts),
    ])
    base_model = LogisticRegression(max_iter=1000, random_state=RUN_SEED)
    model = (
        OneVsRestClassifier(base_model)
        if N_TARGETS > 1
        else base_model
    )
    model.fit(X_train, CANONICAL_Y[train_idx])
```

### 5. Memory Efficiency
- Use gradient accumulation for large models
- Use mixed precision (fp16) training
- Consider DistilBERT for faster inference

### 6. NAIVE BAYES FEATURE REQUIREMENTS (CRITICAL)
MultinomialNB requires **NON-NEGATIVE features**. Violating this causes worse-than-random performance.

**COMPATIBLE with MultinomialNB:**
- Raw TF-IDF: `TfidfVectorizer()` output is always non-negative
- Count vectors: `CountVectorizer()` output

**INCOMPATIBLE with MultinomialNB (produces negative values):**
- TruncatedSVD/LSA: `TruncatedSVD(n_components=100)` produces negative values
- StandardScaler: Centers data around 0, producing negatives
- PCA: Can produce negative components

```python
# WRONG - MultinomialNB with LSA (negative features):
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_train_lsa = svd.fit_transform(X_train_tfidf)  # Contains negative values!
model = MultinomialNB()
model.fit(X_train_lsa, CANONICAL_Y[train_idx])  # Incompatible feature domain!

# CORRECT FOLD-LOCAL OPTIONS:
from sklearn.naive_bayes import GaussianNB, MultinomialNB

for fold_idx, train_idx, val_idx in iter_canonical_cv_splits():
    # Fit vocabulary and IDF only on this fold's training text.
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(train_texts.iloc[train_idx])
    X_val_tfidf = vectorizer.transform(train_texts.iloc[val_idx])
    X_test_tfidf = vectorizer.transform(test_texts)

    # Option 1: MultinomialNB with raw non-negative TF-IDF.
    # Validation/test use transform(), never fit_transform().
    model = MultinomialNB()
    model.fit(X_train_tfidf, CANONICAL_Y[train_idx])
    fold_val_predictions = model.predict_proba(X_val_tfidf)
    fold_test_predictions = model.predict_proba(X_test_tfidf)

# Option 2: GaussianNB for LSA/SVD features, also fit inside each fold.
# Fit SVD on X_train_tfidf only, then transform X_val_tfidf/X_test_tfidf
# with that same fitted transformer before fitting on CANONICAL_Y[train_idx].
# Do not shift validation/test features using their own minima: that changes
# feature semantics across partitions.
```

### 7. Text Column Resolution
For competitions with text in CSV files, resolve the text column from canonical
metadata. `feature_cols.json` is a raw train/test schema intersection and can
mix prose fields with timestamps or numeric metadata, so never pass every raw
feature to a numeric model and never choose a text column by position.
`TARGET_COLS` are labels and must never be candidate text inputs.
```python
text_feature_cols = list(CANONICAL_METADATA.get("text_feature_cols", []))
if not text_feature_cols:
    raise ValueError("Canonical metadata declares no text feature columns")
text_column = text_feature_cols[0]  # declared prose column, never positional
if text_column in TARGET_COLS:
    raise ValueError("A target column cannot be used as text input")
train_texts = train_df[text_column].fillna("").astype(str)
test_texts = test_df[text_column].fillna("").astype(str)
```
"""
