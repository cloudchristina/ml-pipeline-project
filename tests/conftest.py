"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure src is in path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from database.database import DatabaseManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    config = Config()
    config.data_dir = str(temp_dir / "data")
    config.model_dir = str(temp_dir / "models")
    config.logs_dir = str(temp_dir / "logs")
    config.db_host = "localhost"
    config.db_port = 5432
    config.db_name = "test_ml_pipeline"
    config.db_user = "test_user"
    config.db_password = "test_password"
    config.mlflow_tracking_uri = "sqlite:///:memory:"

    # Create directories
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)

    return config


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    with patch('src.database.database.create_engine') as mock_engine:
        mock_engine.return_value = Mock()
        yield mock_engine


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This movie is absolutely fantastic! I loved it.",
        "The worst film I've ever seen. Terrible acting.",
        "It was okay, nothing special but watchable.",
        "Brilliant cinematography and outstanding performances."
    ]


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return [
        {
            "predicted_label": 1,
            "predicted_sentiment": "POSITIVE",
            "confidence": 0.95,
            "probabilities": {"NEGATIVE": 0.05, "POSITIVE": 0.95}
        },
        {
            "predicted_label": 0,
            "predicted_sentiment": "NEGATIVE",
            "confidence": 0.88,
            "probabilities": {"NEGATIVE": 0.88, "POSITIVE": 0.12}
        }
    ]


@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL for integration tests."""
    # Use SQLite in memory for fast tests
    return "sqlite:///:memory:"


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = {
        "predictions": [
            {
                "text": "test text",
                "predicted_label": 1,
                "predicted_sentiment": "POSITIVE",
                "confidence": 0.95,
                "probabilities": {"NEGATIVE": 0.05, "POSITIVE": 0.95}
            }
        ]
    }
    return mock_model


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env_vars = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "DB_PASSWORD": "test_password"
    }

    # Store original values
    original_values = {}
    for key, value in test_env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value