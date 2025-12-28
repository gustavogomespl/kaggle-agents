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
"""
