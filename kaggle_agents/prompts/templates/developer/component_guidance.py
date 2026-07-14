"""
Component-specific guidance for the Developer Agent.

Contains guidance strings for different component types.
"""

COMPONENT_GUIDANCE = {
    "model": """## Model Component Requirements
- IMPLEMENT soft-deadline pattern (see HARD_CONSTRAINTS #9)
  - For Keras/TensorFlow: MUST use DeadlineCallback in model.fit() callbacks
  - For sklearn/PyTorch manual loops: check _check_deadline() inside fold loop
- Train model with StratifiedKFold CV using n_splits=int(os.getenv("KAGGLE_AGENTS_CV_FOLDS","5"))
- Save OOF predictions to models/oof_{name}.npy for stacking
- Handle class imbalance if ratio > 2:1 (class_weight or scale_pos_weight)
- Print per-fold scores: [LOG:FOLD] fold={n} score={s:.6f}
- Use GPU if available (check torch.cuda.is_available())
- Create submission.csv in the metric's required representation: probabilities
  in [0,1] for AUC/LogLoss, original labels for hard-label metrics, and raw
  continuous predictions for regression (non-negative only when RMSLE requires it)
- SUBMISSION FORMAT: Target column is NOT always columns[1]!
  ```python
  sample_sub = pd.read_csv(sample_submission_path)
  print("Columns:", sample_sub.columns.tolist())  # ALWAYS check column names!
  # Use SAME target column name as in train.csv (e.g., 'Insult', 'target', 'label')
  target_col = train_df.columns[0]  # First column of train.csv is usually the target
  sample_sub[target_col] = predictions  # Fill ONLY the target column
  sample_sub.to_csv('submission.csv', index=False)
  ```
- ALWAYS print "Final Validation Performance: {score}" (a REAL computed score,
  never a fabricated one) even if stopped early due to deadline
- SAVE PyTorch checkpoints with TorchScript for ensemble compatibility (see HARD_CONSTRAINTS #10):
  ```python
  scripted_model = torch.jit.script(model)
  torch.jit.save(scripted_model, f"models/{component_name}_fold{fold_idx}.pt")
  ```
- PyTorch >= 2.6: torch.load() defaults to weights_only=True. NEVER save a whole
  model with torch.save(model) and reload with torch.load(path) - it raises
  UnpicklingError. Save/load state_dicts instead:
  ```python
  torch.save(model.state_dict(), path)
  model.load_state_dict(torch.load(path, map_location=device))
  ```
- FOLD CHECKPOINTING (MANDATORY for models slower than ~2 min/fold): after EACH
  completed fold, persist partial OOF + state so a timeout never loses finished
  folds (the ensemble recovers them automatically from this exact layout):
  ```python
  ckpt_dir = Path("models/checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
  state_path = ckpt_dir / f"{component_name}_checkpoint_state.json"
  partial_test_path = ckpt_dir / f"{component_name}_test_partial.npy"
  completed_folds = []
  test_pred_sum = None
  if state_path.exists() and partial_test_path.exists():
      with open(state_path) as f:
          completed_folds = [int(v) for v in json.load(f)["completed_folds"]]
      test_pred_sum = np.load(partial_test_path) * len(completed_folds)

  # At the START of each CV iteration, skip folds already in completed_folds.
  # Only accumulate after that fold's training and inference both succeed.
  if fold_idx in completed_folds:
      continue
  if test_pred_sum is None:
      test_pred_sum = np.zeros_like(fold_test_preds, dtype=np.float64)
  test_pred_sum += fold_test_preds
  completed_folds.append(int(fold_idx))
  np.save(ckpt_dir / f"{component_name}_oof_partial.npy", oof_preds)
  np.save(partial_test_path,
          test_pred_sum / len(completed_folds))
  with open(state_path, "w") as f:
      json.dump({"component_name": component_name,
                 "completed_folds": completed_folds,
                 "min_folds": 2}, f)
  ```""",

    "feature_engineering": """## Feature Engineering Requirements
- Transform train and test consistently
- NO model training in this component
- Save to train_engineered.csv, test_engineered.csv if creating new files
- Fast execution (<30 seconds)
- Print "Final Validation Performance: 1.0" on completion""",

    "ensemble": """## Ensemble Requirements

### LOADING PREVIOUS MODELS (CRITICAL - READ CAREFULLY):
1. **TorchScript Loading** (PREFERRED - no class definition needed):
   ```python
   model = torch.jit.load(checkpoint_path, map_location=device)
   model.eval()
   ```

2. **State Dict Fallback** (ONLY if TorchScript fails):
   - You MUST define the EXACT same model class as used in training
   - Inspect checkpoint keys to determine architecture:
   ```python
   state_dict = torch.load(path, map_location=device)
   # Look at key names: "net.0.weight" means self.net, NOT self.model
   # Look at number of layers to infer depth
   print([k for k in state_dict.keys()][:10])
   ```

### COMMON PITFALLS (WILL CAUSE state_dict LOADING TO FAIL):
- Defining model with `self.model` when checkpoint uses `self.net`
- Using different depth/channels than training component
- Missing dropout layers that exist in original
- Different layer ordering or architecture

### OOF-Based Stacking (no checkpoint loading needed):
- Load OOF predictions from models/oof_*.npy files
- Preferred: Stacking with LogisticRegression/Ridge meta-learner
- LEAKAGE RULE (guardrail-enforced): NEVER fit the meta-learner on the full OOF
  matrix and score it on those same rows. Score stacking with K-fold CV over the
  OOF matrix (fit on K-1 folds, score the held-out fold) and report the mean CV
  score as Final Validation Performance. Refit on all rows ONLY to produce the
  final test predictions.
- If only ONE model has valid OOF/test artifacts, SKIP the meta-learner: reuse
  that model's test predictions and report its OOF score directly.
- ANTI-FABRICATION RULE (guardrail-enforced): NEVER print a hardcoded, estimated,
  mock, or placeholder number as "Final Validation Performance". The printed score
  MUST be computed from real predictions on real held-out data produced in THIS
  run. If a real validation score cannot be computed, do NOT print the line at all.
- Fallback: Weighted average if OOF files missing
- Can use correlation analysis to select diverse models
- MUST validate shapes:
  - Load test.csv (or sample_submission) to get n_test
  - Skip any model where oof.shape[0] != n_train or test.shape[0] != n_test

### Final Output:
- Create submission.csv with final ensemble predictions
- Print "Final Validation Performance: {score}" at the end""",

    "preprocessing": """## Preprocessing Requirements
- Clean data, handle missing values, encode categoricals
- NO model training
- Fast execution (<10 seconds) for tabular data
- EXCEPTION - image/audio decode-once caching: for datasets > 5 GB (or DICOM),
  this component SHOULD decode + resize every image ONCE into models/img_cache/
  (resumable: skip files that already exist). This may take longer than 10s and
  is what makes large competitions converge - downstream model components then
  read ONLY from the cache.
- Save processed data for subsequent components
- Print "Final Validation Performance: 1.0" on completion""",

    "image_to_image_model": """## Image-to-Image Model Requirements (CRITICAL)
This is a PIXEL-LEVEL prediction task. Your model must output FULL IMAGES, not single values.

### DATA PIPELINE FIXES (MANDATORY - PREVENTS COMMON CRASHES):

1. **VARIABLE IMAGE DIMENSIONS** (torch.stack error):
   Images often have different sizes. Use these solutions:
   ```python
   # TRAINING: Use RandomCrop for consistent tensor sizes
   train_transform = transforms.Compose([
       transforms.RandomCrop(256, 256),  # Fixed size for batching
       transforms.ToTensor(),
   ])
   train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # batch_size > 1 OK

   # VALIDATION/TEST: Use batch_size=1 to handle any size
   val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
   test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
   ```

2. **NEGATIVE STRIDES** (numpy/torch conversion error):
   ```python
   # WRONG - creates negative strides:
   img = np.flip(img, axis=1)
   tensor = torch.from_numpy(img)  # CRASHES!

   # CORRECT - fix strides after augmentation:
   img = np.flip(img, axis=1)
   img = np.ascontiguousarray(img)  # FIX STRIDES
   tensor = torch.from_numpy(img)   # Now works!
   ```

3. **NO TRAIN.CSV** (FileNotFoundError):
   Many image-to-image competitions have NO CSV. Load from directories:
   ```python
   # DO NOT: pd.read_csv('train.csv')

   # DO THIS:
   train_dir = Path('/path/to/train')
   clean_dir = Path('/path/to/train_cleaned')
   noisy_files = sorted(train_dir.glob('*.png'))
   pairs = [(nf, clean_dir / nf.name) for nf in noisy_files if (clean_dir / nf.name).exists()]
   print(f"Found {len(pairs)} paired training samples.")
   ```

### Architecture (MUST USE):
- U-Net: encoder-decoder with skip connections
- Autoencoder: encoder-decoder without skip connections
- DnCNN: deep CNN with residual learning
- Fully Convolutional Network (FCN)

### Architecture (DO NOT USE):
- EfficientNet, ResNet, VGG with classification head
- Any model with global average pooling + dense layers
- Any model that outputs a single value per image

### Model Output:
- Input: Image of shape (H, W, C) or (H, W)
- Output: Image of shape (H, W, C) or (H, W) - SAME spatial dimensions
- Loss: MSE, L1, SSIM, or perceptual loss

### Submission Format (CRITICAL):
Sample submission has MILLIONS of rows (one per pixel), NOT one per image!

```python
# CORRECT pattern for pixel-level submission:
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)  # e.g., 5,789,880 rows

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem
    pred = model(load_image(img_path))  # OUTPUT: (H, W) image
    H, W = pred.shape

    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"  # Match sample format
            submission_rows.append({"id": pixel_id, "value": pred[row, col]})

# VERIFY before saving
assert len(submission_rows) == expected_rows, f"WRONG: {len(submission_rows)} vs {expected_rows}"
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
```

### Common Mistake to Avoid:
If your submission has ~29 rows instead of ~5.8M rows, you used a CLASSIFIER instead of an encoder-decoder.
The number of rows = number of test images means WRONG architecture.""",
}
