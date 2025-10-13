# AWS SageMaker Deployment Guide

This guide explains how to deploy your trained sentiment analysis model to AWS SageMaker as a real-time inference endpoint.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Testing the Endpoint](#testing-the-endpoint)
- [Monitoring and Management](#monitoring-and-management)
- [Costs and Optimization](#costs-and-optimization)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is SageMaker Real-Time Inference?

SageMaker provides managed infrastructure for hosting ML models as REST API endpoints:

```
Client → API Gateway → SageMaker Endpoint → Your Model → Response
```

**Benefits:**
- ✅ Fully managed (no server maintenance)
- ✅ Auto-scaling (handles traffic spikes)
- ✅ High availability (99.9% SLA)
- ✅ Monitoring built-in (CloudWatch metrics)
- ✅ A/B testing support (model variants)

**Architecture:**
```
┌─────────────────┐
│  Your Model     │
│  (PyTorch)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inference Code  │  ← inference.py (you write this)
│ (inference.py)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SageMaker       │  ← Managed container
│ PyTorch         │
│ Container       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SageMaker       │  ← HTTPS endpoint
│ Endpoint        │
└─────────────────┘
```

---

## Prerequisites

### 1. AWS Account Setup

**Required:**
- AWS Account with admin access
- AWS CLI installed and configured
- IAM role with SageMaker permissions

**Install AWS CLI:**
```bash
# Mac
brew install awscli

# Linux
pip install awscli

# Configure
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)
```

**Verify Setup:**
```bash
aws sts get-caller-identity
# Should show your account ID
```

---

### 2. Python Dependencies

**Install SageMaker SDK:**
```bash
pip install sagemaker boto3 awscli
```

**Add to requirements.txt:**
```
sagemaker>=2.200.0
boto3>=1.34.0
```

---

### 3. Trained Model

**Required files:**
```
models/final_model/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── tokenizer_config.json    # Tokenizer configuration
├── vocab.txt                # Vocabulary
└── special_tokens_map.json  # Special tokens
```

**Verify model exists:**
```bash
ls -la models/final_model/
```

---

## Deployment Steps

### Step 1: Create Inference Script

Create `scripts/sagemaker_inference.py`:

```python
"""
SageMaker inference script for sentiment analysis model.
This script handles model loading and prediction requests.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global variables (loaded once when container starts)
tokenizer = None
model = None
device = None


def model_fn(model_dir):
    """
    Load the model from the model_dir.
    Called once when the container starts.

    Args:
        model_dir: Path to the model artifacts

    Returns:
        dict: Dictionary containing model and tokenizer
    """
    global tokenizer, model, device

    print(f"Loading model from {model_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()  # Set to evaluation mode

    print("Model loaded successfully")

    return {"model": model, "tokenizer": tokenizer, "device": device}


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.

    Args:
        request_body: The request payload
        request_content_type: The content type of the request

    Returns:
        dict: Parsed input data
    """
    print(f"Received content type: {request_content_type}")

    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """
    Perform prediction on the input data.

    Args:
        input_data: Preprocessed input
        model_dict: Dictionary containing model and tokenizer

    Returns:
        dict: Prediction results
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    # Extract text from input
    if isinstance(input_data, dict):
        text = input_data.get("text", "")
    elif isinstance(input_data, str):
        text = input_data
    else:
        raise ValueError("Input must be a dict with 'text' key or a string")

    if not text:
        raise ValueError("Empty text provided")

    print(f"Processing text: {text[:100]}...")  # Log first 100 chars

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Map class to sentiment
    sentiment = "positive" if predicted_class == 1 else "negative"

    result = {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "negative": round(probabilities[0][0].item(), 4),
            "positive": round(probabilities[0][1].item(), 4)
        }
    }

    print(f"Prediction: {result}")

    return result


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output.

    Args:
        prediction: The prediction result
        response_content_type: The desired response content type

    Returns:
        str: Serialized prediction
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
```

---

### Step 2: Create Deployment Script

Create `scripts/deploy_to_sagemaker.py`:

```python
"""
Deploy sentiment analysis model to AWS SageMaker.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from datetime import datetime
import time

# Configuration
MODEL_DIR = "models/final_model"
SCRIPT_PATH = "scripts/sagemaker_inference.py"
ROLE_NAME = "SageMakerExecutionRole"  # Your SageMaker IAM role
INSTANCE_TYPE = "ml.t2.medium"  # Cheapest option for testing
ENDPOINT_NAME = f"sentiment-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print("=" * 60)
print("SAGEMAKER DEPLOYMENT")
print("=" * 60)

# Step 1: Get SageMaker session and role
print("\n[1/6] Setting up SageMaker session...")
session = sagemaker.Session()
region = session.boto_region_name
print(f"Region: {region}")

# Get IAM role
try:
    iam = boto3.client('iam')
    role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
    print(f"IAM Role: {role_arn}")
except Exception as e:
    print(f"Error getting IAM role: {e}")
    print(f"\nPlease create a SageMaker execution role named '{ROLE_NAME}'")
    print("Or update ROLE_NAME in this script to match your existing role")
    exit(1)

# Step 2: Upload model to S3
print("\n[2/6] Uploading model to S3...")
model_data = session.upload_data(
    path=MODEL_DIR,
    key_prefix="sentiment-analysis/model"
)
print(f"Model uploaded to: {model_data}")

# Step 3: Create PyTorch Model
print("\n[3/6] Creating SageMaker PyTorch model...")
pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role_arn,
    entry_point=SCRIPT_PATH,
    framework_version="2.2.0",  # Match your PyTorch version
    py_version="py312",
    sagemaker_session=session,
)
print("Model created successfully")

# Step 4: Deploy model to endpoint
print(f"\n[4/6] Deploying model to endpoint '{ENDPOINT_NAME}'...")
print(f"Instance type: {INSTANCE_TYPE}")
print("This will take 5-10 minutes...")

start_time = time.time()

predictor = pytorch_model.deploy(
    instance_type=INSTANCE_TYPE,
    initial_instance_count=1,
    endpoint_name=ENDPOINT_NAME,
    wait=True,  # Wait for deployment to complete
)

elapsed_time = time.time() - start_time
print(f"Deployment completed in {elapsed_time:.1f} seconds")

# Step 5: Test the endpoint
print("\n[5/6] Testing endpoint...")
test_data = {
    "text": "This movie was absolutely fantastic! Best film I've seen this year."
}

result = predictor.predict(test_data)
print(f"Test input: {test_data['text']}")
print(f"Prediction: {result}")

# Step 6: Display endpoint information
print("\n[6/6] Deployment complete!")
print("=" * 60)
print("ENDPOINT INFORMATION")
print("=" * 60)
print(f"Endpoint name: {ENDPOINT_NAME}")
print(f"Endpoint ARN: {predictor.endpoint_name}")
print(f"Region: {region}")
print(f"Instance type: {INSTANCE_TYPE}")
print(f"Status: InService")
print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. Test the endpoint:")
print(f"   python scripts/test_sagemaker_endpoint.py --endpoint {ENDPOINT_NAME}")
print("\n2. Monitor costs in AWS Console:")
print(f"   https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{ENDPOINT_NAME}")
print("\n3. Delete endpoint when done (to stop billing):")
print(f"   aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}")
print("=" * 60)

# Save endpoint info
with open("sagemaker_endpoint_info.txt", "w") as f:
    f.write(f"Endpoint Name: {ENDPOINT_NAME}\n")
    f.write(f"Region: {region}\n")
    f.write(f"Instance Type: {INSTANCE_TYPE}\n")
    f.write(f"Deployed: {datetime.now().isoformat()}\n")

print(f"\nEndpoint info saved to: sagemaker_endpoint_info.txt")
```

---

### Step 3: Create Test Script

Create `scripts/test_sagemaker_endpoint.py`:

```python
"""
Test SageMaker endpoint with sample predictions.
"""

import boto3
import json
import argparse

# Test cases
TEST_CASES = [
    {
        "text": "This movie was absolutely fantastic! Best film I've seen this year.",
        "expected": "positive"
    },
    {
        "text": "Terrible waste of time. I want my money back.",
        "expected": "negative"
    },
    {
        "text": "The acting was superb and the plot kept me engaged throughout.",
        "expected": "positive"
    },
    {
        "text": "Boring, predictable, and poorly executed.",
        "expected": "negative"
    },
    {
        "text": "An absolute masterpiece! Every scene was beautifully crafted.",
        "expected": "positive"
    }
]


def test_endpoint(endpoint_name, region="us-east-1"):
    """Test the SageMaker endpoint with sample inputs."""

    print("=" * 60)
    print(f"TESTING SAGEMAKER ENDPOINT: {endpoint_name}")
    print("=" * 60)

    # Create SageMaker runtime client
    client = boto3.client('sagemaker-runtime', region_name=region)

    correct = 0
    total = len(TEST_CASES)

    for i, test_case in enumerate(TEST_CASES, 1):
        text = test_case["text"]
        expected = test_case["expected"]

        print(f"\n[Test {i}/{total}]")
        print(f"Input: {text}")
        print(f"Expected: {expected}")

        # Prepare request
        payload = json.dumps({"text": text})

        try:
            # Invoke endpoint
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload
            )

            # Parse response
            result = json.loads(response['Body'].read().decode())

            predicted = result['sentiment']
            confidence = result['confidence']

            print(f"Predicted: {predicted} (confidence: {confidence:.2%})")

            # Check if correct
            if predicted == expected:
                print("✅ CORRECT")
                correct += 1
            else:
                print("❌ INCORRECT")

        except Exception as e:
            print(f"❌ ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {correct/total:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SageMaker endpoint")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")

    args = parser.parse_args()

    test_endpoint(args.endpoint, args.region)
```

---

### Step 4: Create IAM Role (If Needed)

If you don't have a SageMaker execution role, create one:

**Using AWS Console:**
1. Go to IAM → Roles → Create role
2. Select "SageMaker" as trusted entity
3. Attach policy: `AmazonSageMakerFullAccess`
4. Name: `SageMakerExecutionRole`
5. Click "Create role"

**Using AWS CLI:**
```bash
# Create role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policy
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

---

### Step 5: Deploy the Model

**Run deployment script:**
```bash
python scripts/deploy_to_sagemaker.py
```

**Expected output:**
```
============================================================
SAGEMAKER DEPLOYMENT
============================================================

[1/6] Setting up SageMaker session...
Region: us-east-1
IAM Role: arn:aws:iam::123456789012:role/SageMakerExecutionRole

[2/6] Uploading model to S3...
Model uploaded to: s3://sagemaker-us-east-1-123456789012/sentiment-analysis/model

[3/6] Creating SageMaker PyTorch model...
Model created successfully

[4/6] Deploying model to endpoint 'sentiment-analysis-20250106-143022'...
Instance type: ml.t2.medium
This will take 5-10 minutes...
Deployment completed in 487.3 seconds

[5/6] Testing endpoint...
Test input: This movie was absolutely fantastic! Best film I've seen this year.
Prediction: {'sentiment': 'positive', 'confidence': 0.9654, 'probabilities': {'negative': 0.0346, 'positive': 0.9654}}

[6/6] Deployment complete!
============================================================
ENDPOINT INFORMATION
============================================================
Endpoint name: sentiment-analysis-20250106-143022
Region: us-east-1
Instance type: ml.t2.medium
Status: InService
```

---

## Testing the Endpoint

### Method 1: Using Test Script

```bash
python scripts/test_sagemaker_endpoint.py --endpoint sentiment-analysis-20250106-143022
```

---

### Method 2: Using Python SDK

```python
import boto3
import json

# Create client
client = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Prepare request
payload = json.dumps({
    "text": "This movie was amazing!"
})

# Invoke endpoint
response = client.invoke_endpoint(
    EndpointName='sentiment-analysis-20250106-143022',
    ContentType='application/json',
    Body=payload
)

# Parse response
result = json.loads(response['Body'].read().decode())
print(result)
# Output: {'sentiment': 'positive', 'confidence': 0.9654, ...}
```

---

### Method 3: Using AWS CLI

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name sentiment-analysis-20250106-143022 \
  --content-type application/json \
  --body '{"text":"This movie was terrible!"}' \
  output.json

cat output.json
# {"sentiment": "negative", "confidence": 0.9234, ...}
```

---

### Method 4: Using cURL (via API Gateway)

If you set up API Gateway in front of SageMaker:

```bash
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was great!"}'
```

---

## Monitoring and Management

### View Endpoint Status

**AWS Console:**
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints
```

**AWS CLI:**
```bash
aws sagemaker describe-endpoint --endpoint-name sentiment-analysis-20250106-143022
```

---

### Monitor Metrics (CloudWatch)

**Key metrics:**
- `ModelLatency` - Time to generate prediction
- `Invocations` - Number of requests
- `ModelInvocationErrors` - Failed predictions
- `CPUUtilization` - Instance CPU usage

**View in Console:**
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#metricsV2:graph=~()
```

---

### Update Endpoint

**Deploy new model version:**
```python
# Update model
new_model = PyTorchModel(...)

# Update endpoint (zero downtime)
predictor.update_endpoint(
    model_name=new_model.name,
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

---

### Auto-scaling Configuration

**Enable auto-scaling:**
```python
from sagemaker.predictor import Predictor

predictor = Predictor(endpoint_name=ENDPOINT_NAME)

# Configure auto-scaling
predictor.update_endpoint_weights_and_capacities(
    desired_weights_and_capacities=[
        {
            'DesiredWeight': 1,
            'VariantName': 'AllTraffic',
            'DesiredInstanceCount': 2  # Scale to 2 instances
        }
    ]
)
```

---

### Delete Endpoint (Stop Billing)

**Important:** Endpoints bill per hour even when idle!

**Delete via Python:**
```python
predictor.delete_endpoint()
```

**Delete via CLI:**
```bash
aws sagemaker delete-endpoint --endpoint-name sentiment-analysis-20250106-143022
```

**Verify deletion:**
```bash
aws sagemaker list-endpoints
```

---

## Costs and Optimization

### Pricing (as of 2025)

| Instance Type | vCPUs | Memory | Price/Hour | Monthly (24/7) |
|---------------|-------|--------|------------|----------------|
| ml.t2.medium | 2 | 4 GB | $0.065 | ~$47 |
| ml.t2.large | 2 | 8 GB | $0.13 | ~$94 |
| ml.m5.large | 2 | 8 GB | $0.134 | ~$97 |
| ml.c5.xlarge | 4 | 8 GB | $0.238 | ~$172 |

**Additional costs:**
- Data transfer: $0.09/GB (outbound)
- S3 storage: $0.023/GB/month
- CloudWatch logs: $0.50/GB

---

### Cost Optimization Strategies

**1. Use smallest instance that meets performance needs**
```python
# Start with cheapest
instance_type = "ml.t2.medium"  # $0.065/hr

# Scale up if needed
instance_type = "ml.m5.large"   # $0.134/hr
```

**2. Delete endpoint when not in use**
```bash
# Delete at night, recreate in morning
aws sagemaker delete-endpoint --endpoint-name my-endpoint

# Recreate when needed
python scripts/deploy_to_sagemaker.py
```

**3. Use Serverless Inference (for low traffic)**
```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=10,
)

predictor = pytorch_model.deploy(
    serverless_inference_config=serverless_config
)
```

**Serverless pricing:**
- $0.20 per 1M requests
- $0.0000133 per second of compute time
- **Much cheaper for < 1000 requests/day**

**4. Use SageMaker Savings Plans**
- 1-year commitment: 20% discount
- 3-year commitment: 40% discount

---

### Performance Optimization

**1. Batch predictions**
```python
# Instead of 100 single calls
for text in texts:
    predict(text)  # 100 network calls

# Use batch
predict(texts)  # 1 network call
```

**2. Enable model caching**
```python
# In inference.py
model.eval()
torch.set_grad_enabled(False)  # Disable gradient computation
```

**3. Use GPU for high throughput**
```python
# For > 100 requests/second
instance_type = "ml.g4dn.xlarge"  # GPU instance
```

---

## Troubleshooting

### Issue 1: Deployment Fails

**Error:**
```
ModelError: An error occurred (ModelError) when calling the CreateEndpoint operation
```

**Solutions:**
1. Check IAM role has SageMaker permissions
2. Verify model artifacts are valid
3. Check CloudWatch logs for details:
   ```bash
   aws logs tail /aws/sagemaker/Endpoints/sentiment-analysis-20250106-143022
   ```

---

### Issue 2: Endpoint Returns 5xx Errors

**Error:**
```
ModelError: An error occurred (ValidationError) when calling the InvokeEndpoint operation
```

**Solutions:**
1. Check inference.py has all required functions
2. Verify input format matches `input_fn`
3. Check model loads correctly in `model_fn`
4. View logs:
   ```bash
   aws logs filter-pattern "ERROR" --log-group-name /aws/sagemaker/Endpoints/my-endpoint
   ```

---

### Issue 3: Slow Predictions

**Symptoms:**
- Latency > 5 seconds per request

**Solutions:**
1. Use GPU instance: `ml.g4dn.xlarge`
2. Enable model compilation with SageMaker Neo
3. Reduce max_length in tokenizer
4. Use batch predictions
5. Enable multi-model endpoint

---

### Issue 4: High Costs

**Symptoms:**
- Unexpected bill > $100/month

**Solutions:**
1. Delete unused endpoints
2. Switch to Serverless Inference
3. Use smaller instance type
4. Enable auto-scaling with min=0
5. Set up billing alerts in AWS Console

---

## Next Steps

### 1. Production Best Practices

- ✅ Set up CloudWatch alarms for errors
- ✅ Enable endpoint auto-scaling
- ✅ Configure VPC for security
- ✅ Set up A/B testing for model updates
- ✅ Implement request throttling
- ✅ Add authentication (API Gateway + Cognito)

---

### 2. Integration with Application

**FastAPI Integration:**
```python
import boto3
from fastapi import FastAPI

app = FastAPI()
sagemaker_client = boto3.client('sagemaker-runtime')

@app.post("/predict")
def predict(text: str):
    response = sagemaker_client.invoke_endpoint(
        EndpointName='sentiment-analysis-20250106-143022',
        ContentType='application/json',
        Body=json.dumps({"text": text})
    )
    return json.loads(response['Body'].read())
```

---

### 3. CI/CD Pipeline

**Automate deployment:**
```yaml
# .github/workflows/deploy-model.yml
name: Deploy to SageMaker
on:
  push:
    branches: [main]
    paths: ['models/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to SageMaker
        run: python scripts/deploy_to_sagemaker.py
```

---

## Summary

### Deployment Checklist

- [ ] AWS account configured
- [ ] IAM role created
- [ ] Model trained and saved
- [ ] Inference script created (`sagemaker_inference.py`)
- [ ] Deployment script created (`deploy_to_sagemaker.py`)
- [ ] Model deployed to endpoint
- [ ] Endpoint tested successfully
- [ ] Monitoring configured
- [ ] Cost alerts set up
- [ ] Documentation updated

### Key Commands

```bash
# Deploy
python scripts/deploy_to_sagemaker.py

# Test
python scripts/test_sagemaker_endpoint.py --endpoint <name>

# Monitor
aws sagemaker describe-endpoint --endpoint-name <name>

# Delete (stop billing!)
aws sagemaker delete-endpoint --endpoint-name <name>
```

---

## Resources

- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [PyTorch on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/index.html)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
