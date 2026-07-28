"""
Component-specific guidance for the Developer Agent.

Contains guidance strings for different component types.
"""

COMPONENT_GUIDANCE = {
    "model": """## Model Component Requirements
- IMPLEMENT soft-deadline pattern (see HARD_CONSTRAINTS #9)
  - For Keras/TensorFlow: MUST use DeadlineCallback in model.fit() callbacks
  - For sklearn/PyTorch manual loops: check _check_deadline() inside fold loop
- Train and validate exclusively with the injected `CANONICAL_FOLDS`; never
  instantiate a new splitter inside the component
- Assert `len(CANONICAL_FOLDS) == len(CANONICAL_TRAIN_IDS) == len(y)` and fail
  when the canonical contract is unavailable or does not match the loaded rows
- Save OOF predictions to models/oof_{name}.npy for stacking
- Handle class imbalance if ratio > 2:1 (class_weight or scale_pos_weight)
- Print per-fold scores: [LOG:FOLD] fold={n} score={s:.6f}
- Use GPU if available (check torch.cuda.is_available())
- Create submission.csv in the metric's required representation: probabilities
  in [0,1] for AUC/LogLoss, original labels for hard-label metrics, and raw
  continuous predictions for regression (non-negative only when RMSLE requires it)
- TARGET/SUBMISSION ROLES: Never infer a target or ID from column position.
  `TARGET_COLS`, `TARGET_TYPE`, and `N_TARGETS` come from canonical metadata;
  `TARGET_COL` is only the backward-compatible first target. Submission
  ID/prediction columns
  come from the injected `submission_format_info`; copy those exact names into
  the script and verify them against the public template before assignment.
  ```python
  if CANONICAL_FOLDS_AVAILABLE:
      target_cols = list(TARGET_COLS)
      missing_targets = [col for col in target_cols if col not in train_df.columns]
      if missing_targets:
          raise ValueError(f"Canonical target columns are absent: {missing_targets!r}")
      # Put the frame in canonical row order before extracting targets. The
      # helper handles both a real ID column and the positional naming used
      # when the competition supplies none - do NOT index by ID_COL yourself,
      # because that name may not exist in any CSV.
      train_df = align_train_to_canonical(train_df)
      y = (
          train_df[target_cols[0]].to_numpy()
          if TARGET_TYPE == "single"
          else train_df.loc[:, target_cols].to_numpy()
      )
      expected_y_shape = (
          (len(train_df),)
          if TARGET_TYPE == "single"
          else (len(train_df), N_TARGETS)
      )
      if y.shape != expected_y_shape or CANONICAL_Y.shape != expected_y_shape:
          raise ValueError("Training targets do not match canonical target shape")
      if not np.array_equal(y, CANONICAL_Y):
          raise ValueError("Training targets do not match canonical target values/order")
  else:
      # No canonical contract exists for this domain. Derive targets from the
      # observed training data and build a leak-free CV split seeded with
      # RUN_SEED. Never fabricate a validation score.
      pass

  def write_submission(
      sample_submission_path,
      submission_id_col,
      prediction_cols,
      predictions,
  ):
      # Pass submission_id_col/prediction_cols as literal values copied from
      # the injected submission_format_info. Do not derive them by position.
      sample_sub = pd.read_csv(sample_submission_path)
      prediction_cols = list(prediction_cols)
      required_cols = [submission_id_col, *prediction_cols]
      if not prediction_cols or any(
          col not in sample_sub.columns for col in required_cols
      ):
          raise ValueError("Injected submission roles do not match template")
      values = np.asarray(predictions)
      if values.shape[0] != len(sample_sub):
          raise ValueError("Prediction rows do not match sample_submission")
      if len(prediction_cols) == 1:
          if values.ndim == 2 and values.shape[1] == 1:
              values = values[:, 0]
          if values.ndim != 1:
              raise ValueError("Single-output template requires one prediction per row")
          sample_sub[prediction_cols[0]] = values
      else:
          if values.ndim != 2 or values.shape[1] != len(prediction_cols):
              raise ValueError("Wide template requires one prediction per output column")
          sample_sub.loc[:, prediction_cols] = values
      sample_sub.to_csv("submission.csv", index=False)
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
  partial_oof_path = ckpt_dir / f"{component_name}_oof_partial.npy"
  partial_test_path = ckpt_dir / f"{component_name}_test_partial.npy"
  completed_folds = []
  test_pred_sum = None
  checkpoint_paths = (state_path, partial_oof_path, partial_test_path)
  if any(path.exists() for path in checkpoint_paths):
      if not all(path.exists() for path in checkpoint_paths):
          raise RuntimeError("Incomplete fold checkpoint; refusing unsafe resume")
      with open(state_path) as f:
          checkpoint_state = json.load(f)
      if checkpoint_state.get("component_name") != component_name:
          raise RuntimeError("Checkpoint component mismatch")
      completed_folds = [int(v) for v in checkpoint_state["completed_folds"]]
      configured_folds = int(N_FOLDS)
      if len(completed_folds) != len(set(completed_folds)) or any(
          fold < 0 or fold >= configured_folds for fold in completed_folds
      ):
          raise RuntimeError("Invalid completed_folds checkpoint metadata")
      partial_oof = np.load(partial_oof_path)
      partial_test = np.load(partial_test_path)
      expected_oof_shape = tuple(checkpoint_state.get("oof_shape", ()))
      expected_test_shape = tuple(checkpoint_state.get("test_shape", ()))
      if partial_oof.shape != oof_preds.shape or (
          expected_oof_shape and partial_oof.shape != expected_oof_shape
      ):
          raise RuntimeError("Checkpoint OOF shape mismatch")
      if checkpoint_state.get("n_samples") != int(oof_preds.shape[0]):
          raise RuntimeError("Checkpoint sample-count mismatch")
      if checkpoint_state.get("n_folds") != configured_folds:
          raise RuntimeError("Checkpoint fold-count mismatch")
      if expected_test_shape and partial_test.shape != expected_test_shape:
          raise RuntimeError("Checkpoint test shape mismatch")
      completed_mask = np.isin(CANONICAL_FOLDS, completed_folds)
      if not np.all(np.isfinite(partial_oof[completed_mask])):
          raise RuntimeError("Completed checkpoint folds contain NaN or Inf")
      if np.any(np.isfinite(partial_oof[~CANONICAL_OOF_ELIGIBLE_MASK])):
          raise RuntimeError("Temporal warm-up checkpoint rows must remain NaN")
      if not np.all(np.isfinite(partial_test)):
          raise RuntimeError("Checkpoint test average contains NaN or Inf")
      oof_preds[...] = partial_oof
      test_pred_sum = partial_test * len(completed_folds)

  # At the START of each CV iteration, skip folds already in completed_folds.
  # Only accumulate after that fold's training and inference both succeed.
  if fold_idx in completed_folds:
      continue
  if test_pred_sum is None:
      test_pred_sum = np.zeros_like(fold_test_preds, dtype=np.float64)
  test_pred_sum += fold_test_preds
  completed_folds.append(int(fold_idx))
  np.save(partial_oof_path, oof_preds)
  np.save(partial_test_path,
          test_pred_sum / len(completed_folds))
  with open(state_path, "w") as f:
      json.dump({"component_name": component_name,
                 "completed_folds": completed_folds,
                 "min_folds": 2,
                 "n_samples": int(oof_preds.shape[0]),
                 "n_folds": int(N_FOLDS),
                 "oof_shape": list(oof_preds.shape),
                 "test_shape": list((test_pred_sum / len(completed_folds)).shape)},
                f)
  ```""",

    "feature_engineering": """## Feature Engineering Requirements
- Transform train and test consistently
- NO model training in this component
- Save to train_engineered.csv, test_engineered.csv if creating new files
- Fast execution (<30 seconds)
- Print "[LOG:COMPONENT] status=success type=feature_engineering" on completion;
  do not print a validation score because this component has no held-out metric""",

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
- Print "[LOG:COMPONENT] status=success type=preprocessing" on completion;
  do not print a validation score because this component has no held-out metric""",

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
Derive the row granularity and ID encoding from the supplied sample submission.
Do not assume a delimiter, coordinate base, image size, or ID layout.

```python
def write_pixel_submission(
    sample_submission_path,
    id_col,
    target_col,
    test_images,
    model,
):
    # Pass id_col/target_col as exact literal names copied from the injected
    # submission_format_info. Never derive either role from column position.
    sample_sub = pd.read_csv(sample_submission_path)
    if id_col not in sample_sub.columns or target_col not in sample_sub.columns:
        raise ValueError("Injected pixel submission roles do not match template")

    # Build this mapping only after inspecting and round-trip validating the
    # observed template IDs against the model's image/pixel coordinates.
    predictions_by_template_id = build_predictions_for_observed_ids(
        sample_sub[id_col].astype(str).tolist(),
        test_images,
        model,
    )
    template_ids = sample_sub[id_col].astype(str)
    missing = set(template_ids) - set(predictions_by_template_id)
    extra = set(predictions_by_template_id) - set(template_ids)
    if missing or extra:
        raise ValueError(
            f"Pixel-ID coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    submission = sample_sub.copy()
    submission[target_col] = template_ids.map(predictions_by_template_id)
    if submission[target_col].isna().any():
        raise ValueError("Missing pixel predictions after template alignment")
    submission.to_csv("submission.csv", index=False)
```

### Common Mistake to Avoid:
One prediction per test image is invalid when the local template requires
pixel-level rows; verify exact row and ID coverage against that template.""",
}
