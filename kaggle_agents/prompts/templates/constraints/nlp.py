"""
NLP/Text task constraints.
"""

NLP_CONSTRAINTS = """## NLP/TEXT REQUIREMENTS:

### 1. Text Preprocessing
- Use standard tokenization (BERT, RoBERTa tokenizers)
- Handle max_length: truncate or use sliding window
- Clean text: lowercase, remove special chars (if appropriate)

### 2. Pretrained Models
- Use HuggingFace Transformers: BERT, RoBERTa, DistilBERT
- Fine-tune on task-specific data
- Use appropriate attention mask and token type ids

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

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
For quick baselines or when transformers are too slow:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
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
X_lsa = svd.fit_transform(X_tfidf)  # Contains negative values!
model = MultinomialNB()
model.fit(X_lsa, y)  # Will perform WORSE than random!

# CORRECT OPTIONS:

# Option 1: Use MultinomialNB with raw TF-IDF
from sklearn.naive_bayes import MultinomialNB
X_tfidf = TfidfVectorizer(max_features=10000).fit_transform(texts)
model = MultinomialNB()
model.fit(X_tfidf, y)  # Works correctly

# Option 2: Use GaussianNB for LSA/SVD features
from sklearn.naive_bayes import GaussianNB
X_lsa = TruncatedSVD(100).fit_transform(X_tfidf)
model = GaussianNB()  # Handles negative values
model.fit(X_lsa, y)

# Option 3: Shift LSA features to non-negative (NOT recommended)
X_shifted = X_lsa - X_lsa.min() + 1e-6
model = MultinomialNB()
model.fit(X_shifted, y)
```

### 7. Text Column Detection
For competitions with text in CSV files (not .txt directories), identify the text column:
```python
# Find text column by name or length
text_column = None
for col in df.columns:
    if col.lower() in ('text', 'sentence', 'content', 'body', 'review'):
        text_column = col
        break
    elif df[col].dtype == object:
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > 100:  # Long text content
            text_column = col
            break

if text_column is None:
    raise ValueError("No text column found in dataframe")
```
"""
