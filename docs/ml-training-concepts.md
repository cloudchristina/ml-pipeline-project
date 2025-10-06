# Machine Learning Training Concepts

This document explains key concepts used in the ML training pipeline, specifically focused on transformer-based models and the HuggingFace ecosystem.

## Table of Contents
- [Training Pattern for Transformers](#training-pattern-for-transformers)
- [Preprocessing Function](#preprocessing-function)
- [Compute Metrics Function](#compute-metrics-function)
- [Evaluation Strategy](#evaluation-strategy)
- [Memory Optimization](#memory-optimization)

---

## Training Pattern for Transformers

### Standard Training Flow

```python
# 1. Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForSequenceClassification.from_pretrained("model-name")

# 2. Preprocess data (ONCE, before training)
dataset = load_dataset("imdb")
tokenized_dataset = dataset.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True  # Process in batches for efficiency
)

# 3. Configure training
training_args = TrainingArguments(...)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics  # Called during evaluation
)

# 4. Train
trainer.train()  # Uses preprocessed data
trainer.evaluate()  # Calls compute_metrics automatically
```

### Why This Pattern?

**Separation of concerns:**
- `preprocess_function` → Data preparation (done once)
- `Trainer` → Training loop (done many times)
- `compute_metrics` → Evaluation (done periodically)

**Efficiency:**
- Tokenization is slow, so do it once upfront
- During training, model just reads pre-tokenized data
- Metrics calculated only when needed (not every batch)

**Flexibility:**
- Easy to swap different tokenizers
- Easy to change evaluation metrics
- Easy to experiment with different preprocessing

---

## Preprocessing Function

### Purpose
Converts raw text into format the model understands (token IDs). Models can't process text directly - they need numerical inputs.

### Implementation

```python
def preprocess_function(examples, tokenizer):
    """Tokenize the texts."""
    # Tokenize: "I love this!" → [101, 1045, 2293, 2023, 102]
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )
```

### What It Does

1. **Splits text into tokens** (words/subwords)
   ```
   "I love this movie!" → ["I", "love", "this", "movie", "!"]
   ```

2. **Converts tokens to numerical IDs**
   ```
   ["I", "love", "this", "movie", "!"] → [101, 1045, 2293, 2023, 3185, 999, 102]
   ```

3. **Adds special tokens**
   - `[CLS]` (101) at start - classification token
   - `[SEP]` (102) at end - separation token

4. **Truncates/pads to consistent length**
   - Truncates if text > max_length
   - Pads if text < max_length

### Usage

Applied to entire dataset before training:

```python
train_dataset = train_dataset.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True  # Process multiple examples at once
)
```

**Without preprocessing:** Model receives raw text and crashes
**With preprocessing:** Model receives clean numerical tensors

---

## Compute Metrics Function

### Purpose
Evaluates model performance during/after training. Loss alone doesn't tell you about accuracy, F1, precision, recall, etc.

### Implementation

```python
def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Get predicted class

    # Calculate meaningful metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

### What It Does

1. **Receives model predictions and true labels**
   ```python
   predictions = [[0.2, 0.8], [0.9, 0.1], ...]  # Raw logits
   labels = [1, 0, ...]  # True labels
   ```

2. **Converts predictions to class labels**
   ```python
   predictions = np.argmax(predictions, axis=1)
   # [[0.2, 0.8], [0.9, 0.1]] → [1, 0]
   ```

3. **Calculates business-relevant metrics**
   - **Accuracy**: Overall correctness (90% = 9/10 correct)
   - **F1 Score**: Balance between precision and recall
   - **Precision**: Of predicted positives, how many are correct?
   - **Recall**: Of actual positives, how many did we find?

### When It's Called

- Automatically during `trainer.evaluate()`
- At each epoch end (if `evaluation_strategy="epoch"`)
- At specific step intervals (if `evaluation_strategy="steps"`)

**Without compute_metrics:** You only see loss (0.32, 0.28, ...) - less useful
**With compute_metrics:** You see accuracy (85%, 90%, ...), F1, precision, recall - much more meaningful

---

## Evaluation Strategy

### Overview
`evaluation_strategy` controls **when** the model evaluates performance on the validation/test dataset during training.

### Options

#### 1. `"no"` (default)
No evaluation during training. Only evaluates if you manually call `trainer.evaluate()`.

```python
evaluation_strategy="no"
# Training: train → train → train (no eval)
```

#### 2. `"steps"`
Evaluate every N training steps. Specify with `eval_steps` parameter.

```python
evaluation_strategy="steps",
eval_steps=500  # Evaluate every 500 batches
# Training: train 500 steps → eval → train 500 → eval → ...
```

#### 3. `"epoch"`
Evaluate at the end of each epoch (one complete pass through training data).

```python
evaluation_strategy="epoch"
# Training: epoch 1 → eval → epoch 2 → eval → epoch 3 → eval
```

### Example: Epoch Strategy

```python
training_args = TrainingArguments(
    evaluation_strategy="epoch",  # Evaluate after each epoch
    num_train_epochs=3,           # 3 epochs total
    load_best_model_at_end=True,  # Keep best model
    metric_for_best_model="f1",   # Use F1 to determine "best"
)
```

**What happens:**

1. **Epoch 1**: Train on all 5000 samples → Evaluate on 2000 test samples → Log metrics
2. **Epoch 2**: Train on all 5000 samples → Evaluate on 2000 test samples → Log metrics
3. **Epoch 3**: Train on all 5000 samples → Evaluate on 2000 test samples → Log metrics
4. **End**: Load best model based on F1 score

**Example output:**
```
Epoch 1: loss=0.45, eval_accuracy=0.85, eval_f1=0.84
Epoch 2: loss=0.32, eval_accuracy=0.90, eval_f1=0.89 ← Best!
Epoch 3: loss=0.25, eval_accuracy=0.89, eval_f1=0.88 ← Overfitting
→ Loads Epoch 2 model (best F1)
```

### Why Use Evaluation Strategy?

**Monitor progress:**
- See if accuracy/F1 improves each epoch
- Detect overfitting (training loss decreases but eval accuracy doesn't improve)

**Early stopping:**
- With `load_best_model_at_end=True`, automatically keeps the best checkpoint
- If epoch 2 has best F1, uses that model even if epoch 3 is worse

**Debugging:**
- Catch problems early (accuracy stuck at 50%, model not learning, etc.)

### When to Use Each Strategy

- **`"no"`** - Quick experiments, don't care about validation
- **`"steps"`** - Long training runs, want frequent feedback (every 500/1000 steps)
- **`"epoch"`** - **Standard choice**, balanced feedback without slowing training too much

---

## Memory Optimization

### Common Memory Issues

When training large models on limited hardware (Mac, consumer GPUs), you may encounter:

```
RuntimeError: MPS backend out of memory (MPS allocated: 18.02 GB, max allowed: 22.64 GB)
```

### Solutions

#### 1. Reduce Batch Size

```python
# Before (memory intensive)
per_device_train_batch_size=16,
per_device_eval_batch_size=32,

# After (memory efficient)
per_device_train_batch_size=4,   # 75% less memory
per_device_eval_batch_size=8,    # 75% less memory
```

#### 2. Gradient Accumulation

Simulate larger batch sizes without using more memory:

```python
per_device_train_batch_size=4,      # Small batch (low memory)
gradient_accumulation_steps=4,      # Accumulate 4 batches

# Effective batch size = 4 × 4 = 16 (same as before!)
```

**How it works:**
1. Process 4 samples → compute gradients → DON'T update weights yet
2. Process 4 more samples → accumulate gradients
3. Repeat 4 times (16 total samples)
4. Update weights with accumulated gradients

**Result:** Same training quality as batch size 16, but uses 75% less memory

#### 3. Gradient Checkpointing

Trade computation time for memory:

```python
gradient_checkpointing=True  # Save memory during backpropagation
```

**How it works:**
- During forward pass, don't save all intermediate activations
- During backward pass, recompute activations as needed
- **~40% memory savings** for ~20% slower training

#### 4. MPS Fallback (Mac only)

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

Allows PyTorch to fall back to CPU for operations that exceed MPS (Metal) memory limits.

### Complete Memory-Optimized Configuration

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,      # Reduced from 16
    per_device_eval_batch_size=8,       # Reduced from 32
    gradient_accumulation_steps=4,      # Simulate batch size 16
    gradient_checkpointing=True,        # Save memory
    fp16=False,                         # Disable mixed precision on Mac
    num_train_epochs=3,
    learning_rate=3e-5,
)
```

### Memory vs. Speed Tradeoffs

| Setting | Memory Usage | Training Speed | Quality |
|---------|-------------|----------------|---------|
| Batch size 16, no checkpointing | High | Fast | Baseline |
| Batch size 4, accumulation 4 | Low | ~Same | Same |
| Batch size 4, accumulation 4, checkpointing | Very Low | Slower (~20%) | Same |

---

## Common Issues & Troubleshooting

This section documents common issues you may encounter during ML training and their solutions.

### Issue 1: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'eval_strategy'`

**Error Message:**
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'eval_strategy'
```

**Cause:**
- Parameter name changed between transformers versions
- Older versions use `evaluation_strategy`
- Newer versions (4.41.0+) use `eval_strategy`

**Solution:**
```python
# For transformers < 4.41.0 (like 4.40.0)
training_args = TrainingArguments(
    evaluation_strategy="epoch",  # Use full name
    ...
)

# For transformers >= 4.41.0
training_args = TrainingArguments(
    eval_strategy="epoch",  # Shortened name
    ...
)
```

**How to Check Your Version:**
```python
import transformers
print(transformers.__version__)  # e.g., '4.40.0'
```

---

### Issue 2: `ImportError: Using the Trainer with PyTorch requires accelerate>=0.21.0`

**Error Message:**
```
ImportError: Using the Trainer with PyTorch requires accelerate>=0.21.0:
Please run pip install transformers[torch] or pip install accelerate -U
```

**Cause:**
- Missing `accelerate` package
- `Trainer` class requires accelerate for distributed training support

**Solution:**
```bash
# Install accelerate
pip install accelerate>=0.21.0

# Or install with transformers extras
pip install transformers[torch]
```

**Add to requirements.txt:**
```
accelerate==0.27.0  # Required for transformers Trainer
```

---

### Issue 3: `RuntimeError: MPS backend out of memory`

**Error Message:**
```
RuntimeError: MPS backend out of memory (MPS allocated: 18.02 GB, other allocations: 4.46 GB,
max allowed: 22.64 GB). Tried to allocate 192.00 MB on private pool.
```

**Cause:**
- Training on Mac with Apple Silicon (M1/M2/M3)
- MPS (Metal Performance Shaders) GPU has limited memory
- Batch size too large for available GPU memory

**Solutions (in order of preference):**

**Option 1: Reduce Batch Size + Gradient Accumulation**
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,      # Reduce from 16
    per_device_eval_batch_size=8,       # Reduce from 32
    gradient_accumulation_steps=4,      # Maintain effective batch size
    gradient_checkpointing=True,        # Trade compute for memory
    ...
)
```

**Option 2: Enable MPS Fallback**
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

**Option 3: Force CPU Usage** (Most Reliable)
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_default_device('cpu')

training_args = TrainingArguments(
    use_cpu=True,  # Explicitly use CPU
    per_device_train_batch_size=8,      # Can increase on CPU
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    ...
)
```

**Tradeoffs:**
- **GPU (MPS)**: Fast but memory-limited, may crash
- **CPU**: Slower (~3-4x) but stable, won't crash
- **GPU with small batches**: Moderate speed, stable

**Performance Comparison:**
| Device | Batch Size | Time per Epoch | Stability |
|--------|-----------|----------------|-----------|
| MPS | 16 | ~5 min | ❌ Crashes |
| MPS | 4 + accumulation | ~7 min | ⚠️ May crash |
| CPU | 8 + accumulation | ~15 min | ✅ Stable |

---

### Issue 4: Package Dependency Conflicts (fsspec, datasets, evidently)

**Error Message:**
```
ERROR: Cannot install -r requirements.txt because these package versions have conflicting dependencies.
The conflict is caused by:
    fsspec==2024.6.1
    datasets 2.16.0 depends on fsspec<=2023.10.0 and >=2023.1.0
    evidently 0.4.27 depends on fsspec>=2024.2.0
```

**Cause:**
- Different packages require incompatible versions of dependencies
- `datasets` requires older `fsspec` (<= 2023.10.0)
- `evidently` requires newer `fsspec` (>= 2024.2.0)
- Cannot satisfy both requirements simultaneously

**Solution:**

**Step 1: Identify Compatible Versions**
```bash
# Check which version of evidently is compatible with old fsspec
pip download --no-deps evidently==0.4.16
unzip -q -c evidently-0.4.16-py3-none-any.whl evidently-0.4.16.dist-info/METADATA | grep fsspec
# Output: Requires-Dist: fsspec (no strict version requirement)
```

**Step 2: Update requirements.txt**
```python
# Change this:
fsspec==2024.6.1
evidently==0.4.27

# To this:
fsspec<=2023.10.0,>=2023.1.0  # Compatible with datasets 2.16.0
evidently==0.4.16  # Last version compatible with old fsspec
```

**Step 3: Verify No Conflicts**
```bash
pip install -r requirements.txt --dry-run
```

**General Strategy for Dependency Conflicts:**
1. Identify the conflicting packages and version ranges
2. Find the overlap or downgrade one package
3. Test with `--dry-run` before actual installation
4. Document why you pinned specific versions in comments

---

### Issue 5: Training Loss Not Decreasing / Model Not Learning

**Symptoms:**
```
Epoch 1: loss=0.69, accuracy=0.50
Epoch 2: loss=0.69, accuracy=0.50
Epoch 3: loss=0.69, accuracy=0.50
```

**Causes & Solutions:**

**Cause 1: Learning Rate Too High**
```python
# Problem: Learning rate too aggressive
learning_rate=1e-3  # Too high for fine-tuning

# Solution: Use standard fine-tuning rate
learning_rate=3e-5  # Good for BERT/DistilBERT
learning_rate=5e-5  # Alternative
```

**Cause 2: Data Not Preprocessed Correctly**
```python
# Problem: Forgot to rename label column
train_dataset = dataset.map(preprocess_function)
# Model looks for "labels" but dataset has "label"

# Solution: Rename the column
train_dataset = train_dataset.rename_column("label", "labels")
```

**Cause 3: Model Output Doesn't Match Labels**
```python
# Problem: Binary classification but model has wrong num_labels
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3  # Wrong! Should be 2 for binary
)

# Solution: Match your task
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # 2 for binary (positive/negative)
)
```

**Cause 4: Frozen Model Weights**
```python
# Problem: Accidentally froze model layers
for param in model.parameters():
    param.requires_grad = False

# Solution: Ensure weights are trainable
for param in model.parameters():
    param.requires_grad = True  # Allow gradient updates
```

---

### Issue 6: Training is Extremely Slow

**Symptoms:**
- Training takes hours for small datasets
- Progress bar shows very slow iterations (e.g., 0.1 it/s)

**Solutions:**

**Solution 1: Reduce Dataset Size**
```python
# For quick experiments, use smaller subset
train_dataset = dataset["train"].select(range(2500))  # Instead of 25000
eval_dataset = dataset["test"].select(range(1000))    # Instead of 25000
```

**Solution 2: Increase Batch Size (if memory allows)**
```python
# Larger batches = fewer iterations per epoch
per_device_train_batch_size=16,  # Instead of 4
```

**Solution 3: Use Dataloader Workers**
```python
training_args = TrainingArguments(
    dataloader_num_workers=4,  # Parallel data loading
    ...
)
```

**Solution 4: Disable Unnecessary Logging**
```python
training_args = TrainingArguments(
    logging_steps=500,  # Log less frequently (instead of every step)
    report_to="none",   # Disable wandb/mlflow/tensorboard
    ...
)
```

---

### Issue 7: Model Overfitting (High Train Accuracy, Low Eval Accuracy)

**Symptoms:**
```
Epoch 1: train_loss=0.45, eval_loss=0.48, eval_accuracy=0.85
Epoch 2: train_loss=0.20, eval_loss=0.55, eval_accuracy=0.84  ← Eval getting worse
Epoch 3: train_loss=0.08, eval_loss=0.72, eval_accuracy=0.82  ← Clear overfitting
```

**Solutions:**

**Solution 1: Use Early Stopping**
```python
training_args = TrainingArguments(
    load_best_model_at_end=True,       # Restore best checkpoint
    metric_for_best_model="eval_loss", # Or "f1", "accuracy"
    save_strategy="epoch",
    evaluation_strategy="epoch",
    ...
)
```

**Solution 2: Increase Regularization**
```python
training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization (increase to 0.05)
    ...
)
```

**Solution 3: Reduce Training Epochs**
```python
num_train_epochs=2,  # Instead of 5 or 10
```

**Solution 4: Add Dropout (Model Architecture)**
```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(MODEL_NAME)
config.hidden_dropout_prob = 0.2      # Increase from default 0.1
config.attention_probs_dropout_prob = 0.2

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    num_labels=2
)
```

---

### Issue 8: CUDA/GPU Not Being Used (Training on CPU Unintentionally)

**Symptoms:**
- Training is slow even though you have a GPU
- Logs show `device: cpu` instead of `device: cuda:0`

**Check Current Device:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")
```

**Solutions:**

**Solution 1: Remove CPU-forcing Code**
```python
# Remove these lines if you want to use GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ❌ Remove this
torch.set_default_device('cpu')           # ❌ Remove this
training_args = TrainingArguments(
    use_cpu=True,  # ❌ Remove this or set to False
    ...
)
```

**Solution 2: Install PyTorch with GPU Support**
```bash
# For NVIDIA GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Mac (MPS)
pip install torch torchvision torchaudio  # Should auto-detect MPS
```

---

### Issue 9: ValueError: Attention Mask and Input IDs Shape Mismatch

**Error Message:**
```
ValueError: The `input_ids` and `attention_mask` should have the same shape,
but got shapes torch.Size([8, 512]) and torch.Size([8, 256])
```

**Cause:**
- Inconsistent tokenization settings
- Different `max_length` values in preprocessing and model forward pass

**Solution:**
```python
# Ensure consistent max_length throughout
MAX_LENGTH = 512

def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,  # Same value everywhere
        padding="max_length",    # Ensure consistent padding
    )
```

---

### Issue 10: Out of Disk Space (Model Checkpoints)

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Cause:**
- Trainer saves checkpoints every epoch/step
- Checkpoints are large (500MB - 2GB each)
- Multiple checkpoints accumulate quickly

**Solutions:**

**Solution 1: Limit Checkpoint Count**
```python
training_args = TrainingArguments(
    save_total_limit=2,  # Keep only 2 most recent checkpoints
    ...
)
```

**Solution 2: Disable Intermediate Checkpoints**
```python
training_args = TrainingArguments(
    save_strategy="no",  # Don't save intermediate checkpoints
    # Final model is still saved at the end
    ...
)
```

**Solution 3: Change Output Directory**
```python
# Save to external drive or larger partition
output_dir="/path/to/external/drive/models"
```

**Solution 4: Clean Up Old Checkpoints**
```bash
# Remove old checkpoint directories
rm -rf models/final_model/checkpoint-*
```

---

## Best Practices Summary

1. **Always preprocess data before training** - Don't tokenize on-the-fly
2. **Use meaningful metrics** - Don't rely on loss alone
3. **Evaluate regularly** - Use `evaluation_strategy="epoch"` as default
4. **Optimize for your hardware** - Reduce batch size, use gradient accumulation
5. **Monitor for overfitting** - Check if eval metrics plateau or degrade
6. **Save best model** - Use `load_best_model_at_end=True` with `metric_for_best_model`
7. **Test on small dataset first** - Verify pipeline works before full training
8. **Version pin critical dependencies** - Avoid breaking changes from auto-updates
9. **Document known issues** - Keep a troubleshooting log for your project
10. **Use CPU for stability** - If GPU memory is unreliable, CPU is safer

---

## References

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Training Arguments Reference](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [Tokenization Guide](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Troubleshooting Training Issues](https://huggingface.co/docs/transformers/troubleshooting)
