# Complete AWS SageMaker ML Pipeline Guide

**End-to-End Guide:** Setup → Training → Deployment → Testing

Compatible with: **SageMaker Distribution 3.4.2**

---

## Table of Contents

1. [Environment Setup](#step-1-environment-setup)
2. [Installation & Verification](#step-2-installation--verification)
3. [Data Loading & Exploration](#step-3-data-loading--exploration)
4. [Data Processing & Tokenization](#step-4-data-processing--tokenization)
5. [Model Training](#step-5-model-training)
6. [Model Evaluation](#step-6-model-evaluation)
7. [Save to S3](#step-7-save-model-to-s3)
8. [Register in SageMaker](#step-8-register-model-in-sagemaker)
9. [Deploy to Endpoint](#step-9-deploy-to-endpoint)
10. [Test Endpoint](#step-10-test-endpoint)
11. [Cleanup](#step-11-cleanup-resources)

---

## Prerequisites

### Required
- AWS SageMaker Studio Lab or Notebook Instance
- Instance type: `ml.t3.medium` (CPU) or `ml.g4dn.xlarge` (GPU recommended)
- Python 3.10 or 3.12
- IAM permissions: S3, SageMaker, ECR

### Recommended Instance Types

| Type | GPU | vCPUs | RAM | Cost/hr | Use Case |
|------|-----|-------|-----|---------|----------|
| ml.t3.medium | No | 2 | 4GB | ~$0.05 | Setup/testing |
| ml.g4dn.xlarge | Yes (T4) | 4 | 16GB | ~$0.53 | Training |
| ml.p3.2xlarge | Yes (V100) | 8 | 61GB | ~$3.06 | Fast training |

---

## Step 1: Environment Setup

### 1.1 Clone Repository

Open terminal in SageMaker JupyterLab:

```bash
cd ~
git clone <your-repository-url>
cd ml-pipeline-project
```

### 1.2 Upload Source Files

Ensure `src/models/` exists (may be in `.gitignore`):

```bash
ls -la src/models/
```

If missing, upload these files to `src/models/`:
- `__init__.py`
- `model_trainer.py`
- `training_pipeline.py`

### 1.3 Configure AWS Region

```bash
# Set your AWS region
export AWS_DEFAULT_REGION=ap-southeast-2

# Verify
echo $AWS_DEFAULT_REGION
```

---

## Step 2: Installation & Verification

### 2.1 Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected install time:** 5-10 minutes

### 2.2 Restart Kernel

**CRITICAL:** In Jupyter - **Kernel → Restart Kernel**

### 2.3 Verify Installation

Create new notebook: `sagemaker_pipeline.ipynb`

**Cell 1: Quick Verification**

```python
!python test_installation.py
```

Expected output:
```
✅ INSTALLATION SUCCESSFUL
All packages installed and working correctly!
```

**Cell 2: Check Versions**

```python
import sys
import torch
import transformers
import datasets
import numpy as np
import mlflow
import boto3

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"NumPy: {np.__version__}")
print(f"MLflow: {mlflow.__version__}")
print(f"Boto3: {boto3.__version__}")
```

**Expected versions:**
- PyTorch: 2.2.0
- Transformers: 4.40.0
- Datasets: 2.16.0
- NumPy: 1.26.4

---

## Step 3: Data Loading & Exploration

### 3.1 Setup Project

**Cell 3: Import Project Modules**

```python
import sys
from pathlib import Path

# Add project to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Import project modules
from src.utils.config import Config
from src.data.data_pipeline import DataPipeline
from src.models.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger

print("✓ Project modules imported successfully")
```

### 3.2 Initialize Configuration

**Cell 4: Setup Config**

```python
# Initialize config
config = Config()
config.aws_region = 'ap-southeast-2'
config.s3_bucket = 'ml-pipeline-artifacts'  # Change to your bucket

# Create directories
config.setup_directories(data_pipeline=True)

logger = get_logger(__name__)

print(f"✓ Configuration ready")
print(f"  Region: {config.aws_region}")
print(f"  S3 Bucket: {config.s3_bucket}")
print(f"  Data dir: {config.data_dir}")
print(f"  Model dir: {config.model_dir}")
```

### 3.3 Load Dataset

**Cell 5: Load IMDB Dataset**

```python
from datasets import load_dataset

print("Loading IMDB dataset (this takes 2-5 minutes)...")
raw_dataset = load_dataset("imdb")

print(f"\n✓ Dataset loaded")
print(f"  Splits: {list(raw_dataset.keys())}")
print(f"  Train: {len(raw_dataset['train']):,} examples")
print(f"  Test: {len(raw_dataset['test']):,} examples")

# Show sample
sample = raw_dataset['train'][0]
print(f"\nSample:")
print(f"  Text: {sample['text'][:150]}...")
print(f"  Label: {sample['label']} ({'positive' if sample['label'] == 1 else 'negative'})")
```

### 3.4 Dataset Statistics

**Cell 6: Analyze Data**

```python
from src.data.dataset_loader import HuggingFaceDatasetLoader

loader = HuggingFaceDatasetLoader(config)
stats = loader.get_dataset_statistics(raw_dataset)

print("Dataset Statistics:\n")
for split_name, split_stats in stats.items():
    print(f"{split_name.upper()}:")
    print(f"  Examples: {split_stats['num_examples']:,}")

    if 'label_distribution' in split_stats:
        dist = split_stats['label_distribution']
        print(f"  Negative (0): {dist.get(0, 0):,}")
        print(f"  Positive (1): {dist.get(1, 0):,}")

    if 'text_stats' in split_stats:
        ts = split_stats['text_stats']
        print(f"  Avg length: {ts['avg_length']:.0f} chars")
        print(f"  Range: {ts['min_length']} - {ts['max_length']} chars")
    print()
```

---

## Step 4: Data Processing & Tokenization

### 4.1 Execute Data Pipeline

**Cell 7: Process Data**

```python
# Initialize pipeline
data_pipeline = DataPipeline(config)

print("="*60)
print("DATA PIPELINE EXECUTION")
print("="*60)
print("Steps:")
print("  1. Load IMDB dataset")
print("  2. Create validation split (20%)")
print("  3. Tokenize text with DistilBERT")
print("  4. Save processed data")
print("\nEstimated time: 10-15 minutes")
print("="*60 + "\n")

# Execute
dataset = data_pipeline.execute_pipeline(
    dataset_name="imdb",
    model_name="distilbert-base-uncased",
    max_length=512,
    validation_split=0.2,
    force_reload=False  # Set True to reprocess
)

print(f"\n✓ Pipeline completed!\n")
print("Processed splits:")
for split_name, split_data in dataset.items():
    print(f"  {split_name}: {len(split_data):,} examples")
```

### 4.2 Verify Data Quality

**Cell 8: Quality Check**

```python
quality_report = data_pipeline.get_data_quality_report(dataset)

print("Data Quality Report:")
print(f"  Overall Score: {quality_report['overall_quality_score']:.1%}\n")

for split_name, checks in quality_report['checks'].items():
    score = checks['quality_score']
    status = "✓" if score >= 0.95 else "⚠"

    print(f"{status} {split_name.upper()}: {score:.1%}")
    print(f"    Completeness: {'✓' if checks['completeness']['passed'] else '✗'}")
    print(f"    Consistency: {'✓' if checks['consistency']['passed'] else '✗'}")
    print(f"    Validity: {'✓' if checks['validity']['passed'] else '✗'}")
```

### 4.3 Inspect Tokenized Data

**Cell 9: View Sample**

```python
# Examine tokenized example
sample = dataset['train'][0]

print("Tokenized Example:\n")
print(f"Keys: {list(sample.keys())}")
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Input IDs (first 20): {sample['input_ids'][:20]}")
print(f"Attention mask (first 20): {sample['attention_mask'][:20]}")
print(f"Label: {sample['label']}")
```

---

## Step 5: Model Training

### 5.1 Quick Training Test (Recommended First)

**Cell 10: Quick Test**

```python
import mlflow
from datetime import datetime

# Initialize training pipeline
training_pipeline = TrainingPipeline(config)

# Setup MLflow
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment(config.mlflow_experiment_name)

print("="*60)
print("QUICK TRAINING TEST")
print("="*60)
print("Dataset: 1,000 samples")
print("Epochs: 1")
print("Time: ~5-10 minutes")
print("="*60 + "\n")

# Quick test
quick_results = training_pipeline.quick_training_run(
    dataset_name="imdb",
    model_name="distilbert-base-uncased",
    max_samples=1000
)

print(f"\n✓ Quick test completed!")
print(f"  Loss: {quick_results['train_loss']:.4f}")
print(f"\nMetrics:")
for metric, value in quick_results['eval_metrics'].items():
    print(f"  {metric}: {value:.4f}")
```

**If quick test succeeds, proceed to full training. If it fails, debug before continuing.**

### 5.2 Full Model Training

**Cell 11: Full Training**

```python
from datetime import datetime

print("="*60)
print("FULL MODEL TRAINING")
print("="*60)
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
print("Dataset: Full IMDB (25k train)")
print("Model: DistilBERT-base-uncased")
print("Epochs: 3")
print("Duration: 20-60 minutes (depends on instance)")
print("="*60 + "\n")

# Execute full training
results = training_pipeline.execute_full_pipeline(
    dataset_name="imdb",
    model_name="distilbert-base-uncased",
    optimize_hyperparameters=False,
    experiment_name="sentiment_analysis",
    run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"End: {datetime.now().strftime('%H:%M:%S')}")
print(f"Duration: {results['pipeline_duration']/60:.1f} minutes")
print(f"Model: {results['training_results']['model_path']}")
print("="*60)
```

---

## Step 6: Model Evaluation

### 6.1 View Training Results

**Cell 12: Results Summary**

```python
print("TRAINING RESULTS")
print("="*60)

print(f"\nData Quality: {results['data_quality_score']:.1%}")

print(f"\nValidation Metrics:")
for metric, value in results['training_results']['eval_metrics'].items():
    print(f"  {metric}: {value:.4f}")

if results['test_results']:
    print(f"\nTest Set Metrics:")
    for metric, value in results['test_results'].items():
        print(f"  {metric}: {value:.4f}")

val = results['validation_results']
if 'model_complexity' in val:
    comp = val['model_complexity']
    print(f"\nModel Info:")
    print(f"  Parameters: {comp['total_parameters']:,}")
    print(f"  Trainable: {comp['trainable_parameters']:,}")
    print(f"  Size: {comp['model_size_mb']:.1f} MB")

if 'confidence_stats' in val:
    conf = val['confidence_stats']
    print(f"\nPrediction Confidence:")
    print(f"  Mean: {conf['mean_confidence']:.1%}")
    print(f"  Range: {conf['min_confidence']:.1%} - {conf['max_confidence']:.1%}")

print("="*60)
```

### 6.2 Test Predictions

**Cell 13: Make Predictions**

```python
from src.models.model_trainer import ModelTrainer

# Load trained model
trainer = ModelTrainer(config)
model_path = results['training_results']['model_path']
trainer.load_model(model_path)

# Test sentences
test_texts = [
    "This movie was absolutely fantastic! Best film I've seen this year.",
    "Terrible waste of time. I want my money back.",
    "It was okay, nothing special but watchable.",
    "Incredible performances and stunning cinematography!",
    "Boring and predictable. Fell asleep halfway through."
]

print("Model Predictions:\n")
predictions = trainer.predict(test_texts)

for i, pred in enumerate(predictions['predictions'], 1):
    text = pred['text'][:60]
    sentiment = pred['predicted_sentiment']
    confidence = pred['confidence']

    emoji = "😊" if sentiment == "POSITIVE" else "😞"
    print(f"{i}. {text}...")
    print(f"   {emoji} {sentiment} ({confidence:.1%})\n")
```

---

## Step 7: Save Model to S3

### 7.1 Create Model Archive

**Cell 14: Package Model**

```python
import tarfile
import os

model_name = 'sentiment-model'
model_version = 'v1'
archive_name = f'{model_name}-{model_version}.tar.gz'

print(f"Creating model archive: {archive_name}")

# Create tar.gz archive
with tarfile.open(archive_name, 'w:gz') as tar:
    tar.add(model_path, arcname=model_name)

# Check file size
size_mb = os.path.getsize(archive_name) / (1024 * 1024)
print(f"✓ Archive created: {size_mb:.1f} MB")
```

### 7.2 Upload to S3

**Cell 15: Upload**

```python
import boto3

s3_client = boto3.client('s3', region_name=config.aws_region)
s3_prefix = 'sentiment-analysis'
s3_model_path = f'{s3_prefix}/models/{archive_name}'

try:
    print(f"Uploading to S3...")
    s3_client.upload_file(archive_name, config.s3_bucket, s3_model_path)

    model_uri = f's3://{config.s3_bucket}/{s3_model_path}'
    print(f"✓ Upload successful!")
    print(f"  Location: {model_uri}")

    # Store for later use
    MODEL_URI = model_uri

except Exception as e:
    print(f"✗ Upload failed: {e}")
    print(f"  Check S3 bucket exists: {config.s3_bucket}")
    print(f"  Check IAM permissions for S3")
```

---

## Step 8: Register Model in SageMaker

### 8.1 Create Model Package Group

**Cell 16: Setup Model Registry**

```python
sagemaker_client = boto3.client('sagemaker', region_name=config.aws_region)
model_package_group_name = 'sentiment-analysis-models'

try:
    response = sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription='Sentiment analysis models for production'
    )
    print(f"✓ Model package group created: {model_package_group_name}")
except sagemaker_client.exceptions.ResourceInUse:
    print(f"✓ Model package group exists: {model_package_group_name}")
except Exception as e:
    print(f"✗ Error: {e}")
```

### 8.2 Create Inference Script

**Cell 17: Inference Code**

```python
# Create inference script for SageMaker
inference_code = '''
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def model_fn(model_dir):
    """Load model from directory"""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {'model': model, 'tokenizer': tokenizer}

def input_fn(request_body, content_type='application/json'):
    """Parse input data"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data.get('inputs', data.get('text', ''))
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction"""
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']

    # Tokenize
    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Format output
    predicted_class = predictions.argmax(dim=-1).item()
    confidence = predictions.max(dim=-1).values.item()

    return {
        'sentiment': 'POSITIVE' if predicted_class == 1 else 'NEGATIVE',
        'confidence': float(confidence),
        'probabilities': {
            'NEGATIVE': float(predictions[0][0]),
            'POSITIVE': float(predictions[0][1])
        }
    }

def output_fn(prediction, accept='application/json'):
    """Format output"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
'''

# Save inference script
with open('inference.py', 'w') as f:
    f.write(inference_code)

print("✓ Inference script created: inference.py")
```

### 8.3 Register Model

**Cell 18: Register in SageMaker**

```python
import sagemaker
from sagemaker.pytorch import PyTorchModel
from datetime import datetime

# Get execution role
try:
    role = sagemaker.get_execution_role()
except:
    # If not in SageMaker, use IAM role ARN
    role = "arn:aws:iam::<account-id>:role/<sagemaker-role>"
    print(f"⚠ Using specified role: {role}")

session = sagemaker.Session()

# Create PyTorch model
pytorch_model = PyTorchModel(
    model_data=MODEL_URI,
    role=role,
    framework_version='2.2.0',
    py_version='py310',
    entry_point='inference.py',
)

# Register model
print("Registering model (this may take 2-5 minutes)...")

model_package = pytorch_model.register(
    content_types=['application/json'],
    response_types=['application/json'],
    inference_instances=['ml.t2.medium', 'ml.m5.large', 'ml.m5.xlarge'],
    transform_instances=['ml.m5.large'],
    model_package_group_name=model_package_group_name,
    approval_status='PendingManualApproval',
    description=f'Sentiment Analysis - DistilBERT - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
)

print(f"✓ Model registered!")
print(f"  ARN: {model_package.model_package_arn}")

# Store for later
MODEL_PACKAGE_ARN = model_package.model_package_arn
```

---

## Step 9: Deploy to Endpoint

### 9.1 Approve Model

**Cell 19: Approve for Deployment**

```python
# Approve model for production
sagemaker_client.update_model_package(
    ModelPackageArn=MODEL_PACKAGE_ARN,
    ModelApprovalStatus='Approved',
    ApprovalDescription='Approved for production deployment'
)

print("✓ Model approved for production")
```

### 9.2 Deploy Endpoint

**Cell 20: Create Endpoint**

```python
from datetime import datetime

endpoint_name = f'sentiment-endpoint-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

print("="*60)
print("DEPLOYING MODEL TO ENDPOINT")
print("="*60)
print(f"Endpoint name: {endpoint_name}")
print("Instance: ml.m5.large")
print("Estimated time: 5-10 minutes")
print("="*60 + "\n")

print("Starting deployment...")

# Deploy
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name
)

print("\n" + "="*60)
print("DEPLOYMENT COMPLETE!")
print("="*60)
print(f"Endpoint: {endpoint_name}")
print(f"Status: InService")
print("="*60)

# Store for testing
ENDPOINT_NAME = endpoint_name
```

---

## Step 10: Test Endpoint

### 10.1 Test Predictions

**Cell 21: Test Endpoint**

```python
import json

# Test data
test_cases = [
    "This movie was absolutely fantastic! Best film ever!",
    "Terrible waste of time. Complete garbage.",
    "It was okay, nothing special.",
    "Amazing performances and stunning visuals!",
    "So boring, I fell asleep."
]

print("Testing Endpoint:\n")

for i, text in enumerate(test_cases, 1):
    # Prepare input
    payload = {'inputs': text}

    # Invoke endpoint
    response = predictor.predict(payload)

    # Parse response
    sentiment = response['sentiment']
    confidence = response['confidence']

    emoji = "😊" if sentiment == "POSITIVE" else "😞"

    print(f"{i}. {text[:50]}...")
    print(f"   {emoji} {sentiment} ({confidence:.1%})\n")
```

### 10.2 Endpoint Information

**Cell 22: Endpoint Details**

```python
# Get endpoint details
endpoint_desc = sagemaker_client.describe_endpoint(
    EndpointName=ENDPOINT_NAME
)

print("Endpoint Information:")
print(f"  Name: {endpoint_desc['EndpointName']}")
print(f"  Status: {endpoint_desc['EndpointStatus']}")
print(f"  Created: {endpoint_desc['CreationTime']}")
print(f"  ARN: {endpoint_desc['EndpointArn']}")

# Get endpoint config
config_desc = sagemaker_client.describe_endpoint_config(
    EndpointConfigName=endpoint_desc['EndpointConfigName']
)

variant = config_desc['ProductionVariants'][0]
print(f"\nDeployment:")
print(f"  Instance: {variant['InstanceType']}")
print(f"  Count: {variant['InitialInstanceCount']}")
print(f"  Model: {variant['ModelName']}")
```

---

## Step 11: Cleanup Resources

### 11.1 Delete Endpoint (Important!)

**Cell 23: Delete Endpoint**

```python
# IMPORTANT: Delete endpoint to stop charges
print("Deleting endpoint to stop charges...")

try:
    predictor.delete_endpoint()
    print(f"✓ Endpoint deleted: {ENDPOINT_NAME}")
except Exception as e:
    print(f"✗ Error deleting endpoint: {e}")
    print(f"  Manually delete: {ENDPOINT_NAME}")
```

### 11.2 Optional: Delete Model and Config

**Cell 24: Cleanup Model Resources**

```python
# Optional: Delete model and config (keeps registry entry)
try:
    # Get endpoint config name
    endpoint_config_name = endpoint_desc['EndpointConfigName']

    # Delete endpoint config
    sagemaker_client.delete_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    print(f"✓ Endpoint config deleted")

except Exception as e:
    print(f"Note: {e}")

print("\nModel package in registry preserved for future deployments")
```

---

## Summary & Next Steps

### What You've Accomplished

✅ Environment setup and package installation
✅ Data loading and exploration (IMDB dataset)
✅ Data processing and tokenization
✅ Model training (DistilBERT sentiment classifier)
✅ Model evaluation and testing
✅ Model saved to S3
✅ Model registered in SageMaker Model Registry
✅ Model deployed to real-time endpoint
✅ Endpoint testing and validation
✅ Resource cleanup

### Model Performance

Check your results:
- Validation accuracy: ~90-92%
- Test accuracy: ~90-92%
- Inference time: ~100-200ms per request

### Cost Optimization

**Training:**
- Use Spot instances for training (70% cost savings)
- Use smaller dataset for experiments
- Delete notebooks when not in use

**Inference:**
- Use smallest instance that meets latency requirements
- Consider SageMaker Serverless Inference for variable traffic
- Use batch transform for bulk predictions
- Always delete endpoints when not in use

### Redeployment

To redeploy your model later:

```python
# List approved models
approved_packages = sagemaker_client.list_model_packages(
    ModelPackageGroupName='sentiment-analysis-models',
    ModelApprovalStatus='Approved'
)

# Deploy from registry
model_package_arn = approved_packages['ModelPackageSummaryList'][0]['ModelPackageArn']
# ... deploy code ...
```

### Production Considerations

1. **Monitoring:** Set up CloudWatch alarms for endpoint metrics
2. **Autoscaling:** Configure autoscaling for variable load
3. **A/B Testing:** Deploy multiple model versions
4. **Model Updates:** Retrain periodically with new data
5. **Data Drift:** Monitor for distribution changes
6. **Cost Tracking:** Tag resources and monitor costs

---

## Troubleshooting

### Common Issues

**Problem: Out of Memory during training**
```python
# Reduce batch size
config.batch_size = 8  # Default is 16
config.max_length = 256  # Default is 512
```

**Problem: Endpoint deployment fails**
```python
# Check endpoint status
response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
print(response['FailureReason'])
```

**Problem: Slow training**
```python
# Check if using GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
# If False, request GPU instance (ml.g4dn.xlarge)
```

**Problem: S3 access denied**
- Check IAM role has S3 permissions
- Verify bucket name is correct
- Ensure bucket is in same region

---

## Additional Resources

- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Model Registry Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [Endpoint Deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html)
- [Project README](README.md)

---

**Guide Version:** 1.0
**Last Updated:** 2025-10-04
**Compatible With:** SageMaker Distribution 3.4.2
**Instance Types:** ml.t3.medium, ml.g4dn.xlarge, ml.p3.2xlarge
