#!/usr/bin/env python3
"""
Test script to verify all packages are installed correctly
Run this after: pip install -r requirements.txt
"""

import sys
from packaging import version

def test_imports():
    """Test all critical imports work"""
    print("Testing imports...")
    errors = []

    # Core ML packages
    try:
        import torch
        import transformers
        import datasets
        import numpy as np
        import pandas as pd
        import sklearn
        print("✓ Core ML packages imported")
    except ImportError as e:
        errors.append(f"Core ML import error: {e}")

    # MLOps
    try:
        import mlflow
        import optuna
        print("✓ MLOps packages imported")
    except ImportError as e:
        errors.append(f"MLOps import error: {e}")

    # API
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✓ API packages imported")
    except ImportError as e:
        errors.append(f"API import error: {e}")

    # Database
    try:
        import sqlalchemy
        import psycopg2
        import alembic
        print("✓ Database packages imported")
    except ImportError as e:
        errors.append(f"Database import error: {e}")

    # Monitoring
    try:
        import evidently
        from prometheus_client import Counter
        print("✓ Monitoring packages imported")
    except ImportError as e:
        errors.append(f"Monitoring import error: {e}")

    # Utilities
    try:
        from dotenv import load_dotenv
        import yaml
        from loguru import logger
        print("✓ Utility packages imported")
    except ImportError as e:
        errors.append(f"Utility import error: {e}")

    # AWS
    try:
        import boto3
        print("✓ AWS packages imported")
    except ImportError as e:
        errors.append(f"AWS import error: {e}")

    return errors


def test_versions():
    """Test package versions are compatible"""
    print("\nChecking versions...")

    import torch
    import transformers
    import datasets
    import numpy as np
    import pandas as pd
    import mlflow
    import boto3

    versions = {
        'Python': sys.version.split()[0],
        'PyTorch': torch.__version__,
        'Transformers': transformers.__version__,
        'Datasets': datasets.__version__,
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'MLflow': mlflow.__version__,
        'Boto3': boto3.__version__,
    }

    print("\nInstalled Versions:")
    for name, ver in versions.items():
        print(f"  {name}: {ver}")

    # Critical version checks
    warnings = []

    # NumPy must be < 2.0
    if version.parse(np.__version__) >= version.parse("2.0.0"):
        warnings.append("⚠️  NumPy >= 2.0 is not compatible with SageMaker 3.4.2")

    # PyTorch should be 2.1-2.2
    torch_ver = version.parse(torch.__version__.split('+')[0])
    if torch_ver < version.parse("2.1.0") or torch_ver >= version.parse("2.3.0"):
        warnings.append("⚠️  PyTorch version may not be compatible with SageMaker 3.4.2")

    # Transformers should be 4.35-4.40
    trans_ver = version.parse(transformers.__version__)
    if trans_ver < version.parse("4.35.0") or trans_ver >= version.parse("4.41.0"):
        warnings.append("⚠️  Transformers version may not be compatible with SageMaker 3.4.2")

    return warnings


def test_functionality():
    """Test basic functionality"""
    print("\nTesting functionality...")

    # Test PyTorch
    import torch
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"✓ PyTorch tensor operations work")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    # Test Transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    result = tokenizer("Hello world")
    print(f"✓ Transformers tokenization works")

    # Test Datasets
    from datasets import Dataset
    data = {"text": ["Hello", "World"], "label": [0, 1]}
    dataset = Dataset.from_dict(data)
    print(f"✓ Datasets creation works ({len(dataset)} examples)")

    # Test NumPy/Pandas
    import numpy as np
    import pandas as pd
    arr = np.array([1, 2, 3])
    df = pd.DataFrame({"col": [1, 2, 3]})
    print(f"✓ NumPy/Pandas operations work")

    return True


def test_project_imports():
    """Test project-specific imports"""
    print("\nTesting project imports...")

    try:
        from src.utils.config import Config
        from src.utils.logger import get_logger
        print("✓ Project utils imported")
    except ImportError as e:
        print(f"⚠️  Project utils import failed: {e}")
        print("   (This is OK if running outside project directory)")
        return False

    try:
        from src.data.data_pipeline import DataPipeline
        from src.data.dataset_loader import HuggingFaceDatasetLoader
        print("✓ Project data modules imported")
    except ImportError as e:
        print(f"⚠️  Project data import failed: {e}")
        return False

    try:
        from src.models.model_trainer import ModelTrainer
        from src.models.training_pipeline import TrainingPipeline
        print("✓ Project model modules imported")
    except ImportError as e:
        print(f"⚠️  Project models import failed: {e}")
        return False

    return True


def main():
    print("="*60)
    print("PACKAGE INSTALLATION VERIFICATION")
    print("="*60)

    # Test imports
    import_errors = test_imports()

    if import_errors:
        print("\n❌ IMPORT ERRORS:")
        for error in import_errors:
            print(f"  {error}")
        return 1

    # Test versions
    version_warnings = test_versions()

    if version_warnings:
        print("\n⚠️  VERSION WARNINGS:")
        for warning in version_warnings:
            print(f"  {warning}")

    # Test functionality
    try:
        test_functionality()
    except Exception as e:
        print(f"\n❌ FUNCTIONALITY TEST FAILED: {e}")
        return 1

    # Test project imports (optional)
    project_ok = test_project_imports()

    # Summary
    print("\n" + "="*60)
    if import_errors:
        print("❌ INSTALLATION FAILED")
        print("="*60)
        return 1
    elif version_warnings:
        print("⚠️  INSTALLATION COMPLETE WITH WARNINGS")
        print("="*60)
        print("Packages installed but may have compatibility issues")
        return 0
    else:
        print("✅ INSTALLATION SUCCESSFUL")
        print("="*60)
        print("All packages installed and working correctly!")
        if project_ok:
            print("Project imports also verified ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
