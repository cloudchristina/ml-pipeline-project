"""
Unit tests for custom exceptions.
"""

import pytest
from utils.exceptions import (
    MLPipelineException,
    ConfigurationError,
    DatabaseError,
    ModelError,
    ValidationError,
    SecurityError,
    AuthenticationError,
    PredictionError
)


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_base_exception_with_message(self):
        """Test base exception with message only."""
        error = MLPipelineException("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_base_exception_with_details(self):
        """Test base exception with message and details."""
        details = {"error_code": 123, "context": "test"}
        error = MLPipelineException("Test error", details)

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == details

    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError("Invalid configuration")

        assert isinstance(error, MLPipelineException)
        assert str(error) == "Invalid configuration"

    def test_database_error_with_details(self):
        """Test database error with details."""
        details = {"table": "users", "operation": "insert"}
        error = DatabaseError("Database connection failed", details)

        assert isinstance(error, MLPipelineException)
        assert error.details["table"] == "users"
        assert error.details["operation"] == "insert"

    def test_model_error_inheritance(self):
        """Test model error inheritance."""
        error = ModelError("Model loading failed")

        assert isinstance(error, MLPipelineException)
        assert str(error) == "Model loading failed"

    def test_prediction_error_inheritance(self):
        """Test prediction error inherits from model error."""
        error = PredictionError("Prediction failed")

        assert isinstance(error, ModelError)
        assert isinstance(error, MLPipelineException)

    def test_validation_error(self):
        """Test validation error."""
        details = {"field": "email", "value": "invalid_email"}
        error = ValidationError("Email validation failed", details)

        assert isinstance(error, MLPipelineException)
        assert error.details["field"] == "email"

    def test_security_error(self):
        """Test security error."""
        details = {"attack_type": "sql_injection", "blocked": True}
        error = SecurityError("Security violation detected", details)

        assert isinstance(error, MLPipelineException)
        assert error.details["attack_type"] == "sql_injection"

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Invalid credentials")

        assert isinstance(error, MLPipelineException)
        assert str(error) == "Invalid credentials"

    def test_exception_details_default_empty(self):
        """Test that exception details default to empty dict."""
        error = DatabaseError("Test error")

        assert error.details == {}
        assert isinstance(error.details, dict)

    def test_exception_chaining(self):
        """Test exception chaining works correctly."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            config_error = ConfigurationError("Configuration failed")

            # Test that we can access the original exception context
            assert isinstance(config_error, MLPipelineException)
            assert str(config_error) == "Configuration failed"

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from base exception."""
        exception_classes = [
            ConfigurationError,
            DatabaseError,
            ModelError,
            ValidationError,
            SecurityError,
            AuthenticationError,
            PredictionError
        ]

        for exc_class in exception_classes:
            error = exc_class("Test message")
            assert isinstance(error, MLPipelineException)
            assert hasattr(error, 'message')
            assert hasattr(error, 'details')