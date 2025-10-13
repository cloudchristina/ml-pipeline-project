#!/usr/bin/env python3
"""
Deploy sentiment analysis model to AWS SageMaker.

Usage:
    python scripts/deploy_to_sagemaker.py

Requirements:
    - AWS CLI configured (aws configure)
    - SageMaker execution role exists
    - Model trained and saved in models/final_model/
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from datetime import datetime
import time
import os
import sys

# Configuration
MODEL_DIR = "models/final_model"
SCRIPT_PATH = "scripts/sagemaker_inference.py"
ROLE_NAME = "SageMakerExecutionRole"  # Update this to your IAM role name
INSTANCE_TYPE = "ml.t2.medium"  # Cheapest option: ~$0.065/hour
ENDPOINT_NAME = f"sentiment-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print("=" * 60)
print("SAGEMAKER DEPLOYMENT")
print("=" * 60)

# Validate prerequisites
print("\n[0/6] Validating prerequisites...")

# Check if model exists
if not os.path.exists(MODEL_DIR):
    print(f"❌ Error: Model directory not found: {MODEL_DIR}")
    print("Please train the model first: python scripts/train_better_model.py")
    sys.exit(1)

# Check if inference script exists
if not os.path.exists(SCRIPT_PATH):
    print(f"❌ Error: Inference script not found: {SCRIPT_PATH}")
    sys.exit(1)

required_files = [
    "pytorch_model.bin",
    "config.json",
    "tokenizer_config.json",
    "vocab.txt"
]

for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        print(f"❌ Error: Required file not found: {MODEL_DIR}/{file}")
        sys.exit(1)

print("✅ All prerequisites met")

# Step 1: Get SageMaker session and role
print("\n[1/6] Setting up SageMaker session...")
try:
    session = sagemaker.Session()
    region = session.boto_region_name
    print(f"Region: {region}")
except Exception as e:
    print(f"❌ Error: Could not create SageMaker session")
    print(f"Make sure AWS CLI is configured: aws configure")
    print(f"Error details: {e}")
    sys.exit(1)

# Get IAM role
print(f"Looking for IAM role: {ROLE_NAME}")
try:
    iam = boto3.client('iam')
    role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
    print(f"✅ IAM Role found: {role_arn}")
except Exception as e:
    print(f"❌ Error: Could not find IAM role '{ROLE_NAME}'")
    print(f"\nTo create the role, run:")
    print(f"aws iam create-role --role-name {ROLE_NAME} \\")
    print(f'  --assume-role-policy-document \'{{"Version":"2012-10-17","Statement":[{{"Effect":"Allow","Principal":{{"Service":"sagemaker.amazonaws.com"}},"Action":"sts:AssumeRole"}}]}}\'')
    print(f"\naws iam attach-role-policy --role-name {ROLE_NAME} \\")
    print(f"  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
    print(f"\nOr update ROLE_NAME in this script to match your existing role")
    sys.exit(1)

# Step 2: Upload model to S3
print("\n[2/6] Uploading model to S3...")
print(f"Model directory: {MODEL_DIR}")
try:
    model_data = session.upload_data(
        path=MODEL_DIR,
        key_prefix="sentiment-analysis/model"
    )
    print(f"✅ Model uploaded to: {model_data}")
except Exception as e:
    print(f"❌ Error uploading model to S3: {e}")
    sys.exit(1)

# Step 3: Create PyTorch Model
print("\n[3/6] Creating SageMaker PyTorch model...")
try:
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role_arn,
        entry_point=SCRIPT_PATH,
        framework_version="2.2.0",  # Match your PyTorch version
        py_version="py312",
        sagemaker_session=session,
    )
    print("✅ Model created successfully")
except Exception as e:
    print(f"❌ Error creating PyTorch model: {e}")
    sys.exit(1)

# Step 4: Deploy model to endpoint
print(f"\n[4/6] Deploying model to endpoint '{ENDPOINT_NAME}'...")
print(f"Instance type: {INSTANCE_TYPE} (~$0.065/hour)")
print("⏳ This will take 5-10 minutes...")
print("Note: You can safely Ctrl+C - deployment will continue in background")

start_time = time.time()

try:
    predictor = pytorch_model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=1,
        endpoint_name=ENDPOINT_NAME,
        wait=True,  # Wait for deployment to complete
    )

    elapsed_time = time.time() - start_time
    print(f"✅ Deployment completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

except KeyboardInterrupt:
    print("\n⚠️  Deployment interrupted but will continue in background")
    print(f"Check status: aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error deploying model: {e}")
    print("\nCheck CloudWatch logs for details:")
    print(f"aws logs tail /aws/sagemaker/Endpoints/{ENDPOINT_NAME} --follow")
    sys.exit(1)

# Step 5: Test the endpoint
print("\n[5/6] Testing endpoint...")
test_data = {
    "text": "This movie was absolutely fantastic! Best film I've seen this year."
}

try:
    result = predictor.predict(test_data)
    print(f"✅ Test successful!")
    print(f"   Input: {test_data['text']}")
    print(f"   Prediction: {result}")
except Exception as e:
    print(f"⚠️  Warning: Endpoint test failed: {e}")
    print("Endpoint is deployed but may need a few more seconds to be ready")

# Step 6: Display endpoint information
print("\n[6/6] Deployment complete!")
print("=" * 60)
print("ENDPOINT INFORMATION")
print("=" * 60)
print(f"Endpoint name: {ENDPOINT_NAME}")
print(f"Region: {region}")
print(f"Instance type: {INSTANCE_TYPE}")
print(f"Status: InService")
print(f"\n💰 Cost: ~$0.065/hour (~$47/month if running 24/7)")
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. Test the endpoint:")
print(f"   python scripts/test_sagemaker_endpoint.py --endpoint {ENDPOINT_NAME}")
print("\n2. Monitor endpoint in AWS Console:")
console_url = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{ENDPOINT_NAME}"
print(f"   {console_url}")
print("\n3. View CloudWatch logs:")
print(f"   aws logs tail /aws/sagemaker/Endpoints/{ENDPOINT_NAME} --follow")
print("\n4. ⚠️  IMPORTANT: Delete endpoint when done to stop billing:")
print(f"   aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}")
print("=" * 60)

# Save endpoint info
info_file = "sagemaker_endpoint_info.txt"
with open(info_file, "w") as f:
    f.write(f"Endpoint Name: {ENDPOINT_NAME}\n")
    f.write(f"Region: {region}\n")
    f.write(f"Instance Type: {INSTANCE_TYPE}\n")
    f.write(f"Deployed: {datetime.now().isoformat()}\n")
    f.write(f"Cost: ~$0.065/hour\n")
    f.write(f"\nTo delete:\n")
    f.write(f"aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}\n")

print(f"\n📝 Endpoint info saved to: {info_file}")
print("\n✅ Deployment successful!")
