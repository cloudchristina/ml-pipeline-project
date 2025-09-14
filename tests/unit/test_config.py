"""
Unit tests for configuration management.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.config import Config
from utils.exceptions import ConfigurationError


class TestConfig:
    """Test configuration class."""

    def test_config_initialization(self, temp_dir):
        """Test basic configuration initialization."""
        config = Config()
        assert config.model_name == "distilbert-base-uncased"
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3

    def test_config_from_env_vars(self, temp_dir):
        """Test configuration loading from environment variables."""
        with patch.dict(os.environ, {
            "MODEL_NAME": "bert-base-uncased",
            "BATCH_SIZE": "32",
            "LEARNING_RATE": "1e-4",
            "NUM_EPOCHS": "5"
        }):
            config = Config()
            assert config.model_name == "bert-base-uncased"
            assert config.batch_size == 32
            assert config.learning_rate == 1e-4
            assert config.num_epochs == 5

    def test_database_url_generation(self):
        """Test database URL generation."""
        config = Config()
        config.db_host = "localhost"
        config.db_port = 5432
        config.db_name = "test_db"
        config.db_user = "test_user"
        config.db_password = "test_pass"

        expected_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
        assert config.database_url == expected_url

    def test_config_validation_positive_values(self):
        """Test validation of positive numeric values."""
        config = Config()

        # Test negative batch size
        config.batch_size = -1
        with pytest.raises(ConfigurationError, match="batch_size must be positive"):
            config.validate()

        # Test zero learning rate
        config.batch_size = 16
        config.learning_rate = 0
        with pytest.raises(ConfigurationError, match="learning_rate must be positive"):
            config.validate()

        # Test negative num_epochs
        config.learning_rate = 2e-5
        config.num_epochs = -1
        with pytest.raises(ConfigurationError, match="num_epochs must be positive"):
            config.validate()

    def test_config_validation_production_secrets(self, temp_dir):
        """Test validation of required secrets in production."""
        config = Config()
        config.data_dir = str(temp_dir)

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            # Test missing DB password
            config.db_password = ""
            with pytest.raises(ConfigurationError, match="DB_PASSWORD is required"):
                config.validate()

            # Test missing alert email with monitoring enabled
            config.db_password = "secure_password"
            config.monitoring_enabled = True
            config.alert_email = ""
            with pytest.raises(ConfigurationError, match="ALERT_EMAIL is required"):
                config.validate()

    def test_config_validation_directories(self):
        """Test validation of required directories."""
        config = Config()
        config.data_dir = "/nonexistent/directory"

        with pytest.raises(ConfigurationError, match="data_dir does not exist"):
            config.validate()

    def test_config_to_dict(self, test_config):
        """Test configuration serialization to dictionary."""
        config_dict = test_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "batch_size" in config_dict
        assert "db_host" in config_dict

    def test_config_update_from_dict(self, test_config):
        """Test configuration update from dictionary."""
        updates = {
            "batch_size": 64,
            "learning_rate": 1e-3,
            "model_name": "roberta-base"
        }

        test_config.update_from_dict(updates)

        assert test_config.batch_size == 64
        assert test_config.learning_rate == 1e-3
        assert test_config.model_name == "roberta-base"

    def test_config_save_and_load_file(self, temp_dir, test_config):
        """Test configuration save and load from file."""
        config_file = temp_dir / "config.yml"

        # Save configuration
        test_config.save_to_file(str(config_file))
        assert config_file.exists()

        # Load configuration
        loaded_config = Config.from_file(str(config_file))
        assert loaded_config.model_name == test_config.model_name
        assert loaded_config.batch_size == test_config.batch_size

    def test_config_directory_creation(self, temp_dir):
        """Test automatic directory creation."""
        config = Config()
        config.data_dir = str(temp_dir / "new_data")
        config.model_dir = str(temp_dir / "new_models")
        config.logs_dir = str(temp_dir / "new_logs")

        # Trigger directory creation
        config.__post_init__()

        assert Path(config.data_dir).exists()
        assert Path(config.model_dir).exists()
        assert Path(config.logs_dir).exists()
        assert Path(config.data_dir + "/raw").exists()
        assert Path(config.data_dir + "/processed").exists()

    def test_model_serving_url(self, test_config):
        """Test model serving URL generation."""
        test_config.api_host = "0.0.0.0"
        test_config.api_port = 8080

        expected_url = "http://0.0.0.0:8080"
        assert test_config.model_serving_url == expected_url