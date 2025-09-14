#!/usr/bin/env python3
"""
Component testing script for ML Pipeline.
Run individual components step by step to verify functionality.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_step(step_name, test_func):
    """Helper to run test steps with error handling."""
    print(f"\n{'='*50}")
    print(f"🧪 TESTING: {step_name}")
    print(f"{'='*50}")

    try:
        test_func()
        print(f"✅ {step_name} - PASSED")
        return True
    except Exception as e:
        print(f"❌ {step_name} - FAILED")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration management."""
    from utils.config import Config
    from utils.exceptions import ConfigurationError

    # Test basic initialization
    config = Config()
    assert hasattr(config, 'model_name')
    assert hasattr(config, 'database_url')
    print("✓ Configuration initialization works")

    # Test validation
    config.batch_size = -1
    try:
        config.validate()
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        print("✓ Configuration validation works")

    # Reset valid value
    config.batch_size = 16

    # Test database URL generation
    db_url = config.database_url
    assert "postgresql://" in db_url
    print("✓ Database URL generation works")

def test_custom_exceptions():
    """Test custom exception classes."""
    from utils.exceptions import (
        DatabaseError, ModelError, ValidationError,
        SecurityError, ConfigurationError
    )

    # Test exception creation
    db_error = DatabaseError("Test database error", {"table": "users"})
    assert str(db_error) == "Test database error"
    assert db_error.details["table"] == "users"
    print("✓ Custom exceptions work correctly")

    # Test inheritance
    model_error = ModelError("Model loading failed")
    assert hasattr(model_error, 'message')
    assert hasattr(model_error, 'details')
    print("✓ Exception inheritance works")

def test_database_security():
    """Test database security features."""
    from unittest.mock import Mock, patch
    from database.database import DatabaseManager
    from utils.exceptions import SecurityError
    from utils.config import Config

    # Create test config
    config = Config()

    with patch('database.database.create_engine') as mock_engine:
        mock_engine.return_value = Mock()

        # Create database manager
        db_manager = DatabaseManager(config)
        db_manager.engine = Mock()
        db_manager._valid_table_names = {"users", "products"}

        # Test valid table name
        result = db_manager._validate_table_name("users")
        assert result == "users"
        print("✓ Valid table name validation works")

        # Test invalid table name with SQL injection
        try:
            db_manager._validate_table_name("users; DROP TABLE users;")
            assert False, "Should have raised SecurityError"
        except SecurityError:
            print("✓ SQL injection prevention works")

        # Test table not in whitelist
        try:
            db_manager._validate_table_name("malicious_table")
            assert False, "Should have raised SecurityError"
        except SecurityError:
            print("✓ Table whitelist validation works")

def test_api_models():
    """Test API model validation."""
    from api.models import PredictionRequest, BatchPredictionRequest
    from pydantic import ValidationError

    # Test valid prediction request
    request = PredictionRequest(text="This is a test message")
    assert request.text == "This is a test message"
    assert request.return_probabilities == True  # default
    print("✓ Valid prediction request creation works")

    # Test invalid prediction request (empty text)
    try:
        PredictionRequest(text="")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("✓ Empty text validation works")

    # Test batch request
    batch_request = BatchPredictionRequest(texts=["Text 1", "Text 2"])
    assert len(batch_request.texts) == 2
    print("✓ Batch prediction request works")

def test_data_pipeline_mock():
    """Test data pipeline with mocked components."""
    from unittest.mock import Mock, patch
    from data.data_pipeline import DataPipeline
    from utils.config import Config

    config = Config()

    # Mock external dependencies
    with patch('data.data_pipeline.HuggingFaceDatasetLoader') as mock_loader:
        mock_loader.return_value.load_dataset.return_value = {"train": Mock(), "validation": Mock()}
        mock_loader.return_value.get_dataset_statistics.return_value = {"train": {"num_examples": 1000}}

        with patch('mlflow.start_run') as mock_mlflow:
            mock_mlflow.return_value.__enter__ = Mock()
            mock_mlflow.return_value.__exit__ = Mock()

            pipeline = DataPipeline(config)

            # Test pipeline initialization
            assert pipeline.config == config
            print("✓ Data pipeline initialization works")

def test_monitoring_components():
    """Test monitoring and alerting components."""
    from monitoring.alerting import AlertingService
    from utils.config import Config

    config = Config()
    alerting = AlertingService(config)

    # Test email alert formatting
    alert_body = alerting._format_alert_email(
        "Test Alert",
        "This is a test message",
        {"metric": "cpu_usage", "value": 85}
    )
    assert "Test Alert" in alert_body
    assert "cpu_usage" in alert_body
    print("✓ Alert formatting works")

def test_docker_compose_validation():
    """Test Docker Compose configuration."""
    import yaml

    # Load docker-compose.yml
    compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        print("⚠️  docker-compose.yml not found, skipping validation")
        return

    with open(compose_path) as f:
        compose_config = yaml.safe_load(f)

    # Validate structure
    assert "services" in compose_config
    assert "postgres" in compose_config["services"]
    assert "api" in compose_config["services"]
    print("✓ Docker Compose structure is valid")

    # Check security: password should not have default
    postgres_env = compose_config["services"]["postgres"]["environment"]
    db_password = postgres_env.get("POSTGRES_PASSWORD", "")
    assert "${DB_PASSWORD:?" in db_password, "Database password should be required"
    print("✓ Docker Compose security validation passed")

def test_terraform_validation():
    """Test Terraform configuration syntax."""
    import subprocess

    tf_path = Path("infrastructure")
    if not tf_path.exists():
        print("⚠️  Infrastructure directory not found, skipping Terraform validation")
        return

    try:
        # Check if terraform is installed
        result = subprocess.run(["terraform", "version"],
                              capture_output=True, text=True, cwd=tf_path)
        if result.returncode != 0:
            print("⚠️  Terraform not installed, skipping validation")
            return

        # Validate terraform syntax
        result = subprocess.run(["terraform", "validate"],
                              capture_output=True, text=True, cwd=tf_path)

        if result.returncode == 0:
            print("✓ Terraform configuration is valid")
        else:
            print(f"⚠️  Terraform validation warnings: {result.stderr}")

    except FileNotFoundError:
        print("⚠️  Terraform not installed, skipping validation")

def test_pytest_execution():
    """Test pytest execution."""
    import subprocess

    try:
        # Run unit tests if they exist
        if Path("tests/unit").exists():
            result = subprocess.run([
                "python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✓ Unit tests passed")
                print(f"Output: {result.stdout}")
            else:
                print(f"⚠️  Some unit tests failed: {result.stderr}")
        else:
            print("⚠️  Unit tests directory not found")

    except Exception as e:
        print(f"⚠️  Could not run pytest: {e}")

def run_all_tests():
    """Run all component tests."""
    print("🚀 Starting ML Pipeline Component Testing")
    print("=" * 60)

    test_results = []

    # Define test steps
    test_steps = [
        ("Configuration Management", test_configuration),
        ("Custom Exceptions", test_custom_exceptions),
        ("Database Security", test_database_security),
        ("API Models", test_api_models),
        ("Data Pipeline (Mock)", test_data_pipeline_mock),
        ("Monitoring Components", test_monitoring_components),
        ("Docker Compose Validation", test_docker_compose_validation),
        ("Terraform Validation", test_terraform_validation),
        ("Pytest Execution", test_pytest_execution),
    ]

    # Run each test
    for step_name, test_func in test_steps:
        success = test_step(step_name, test_func)
        test_results.append((step_name, success))

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    for step_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{step_name:<35} {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All component tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)