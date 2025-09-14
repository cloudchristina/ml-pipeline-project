"""
Custom exception classes for the ML pipeline.

This module defines custom exception classes that provide more specific error handling
and better error messages throughout the application.
"""

from typing import Any, Dict, Optional


class MLPipelineException(Exception):
    """Base exception class for ML Pipeline errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(MLPipelineException):
    """Raised when there are configuration-related errors."""
    pass


class DatabaseError(MLPipelineException):
    """Raised when database operations fail."""
    pass


class ModelError(MLPipelineException):
    """Raised when model operations fail."""
    pass


class DataPipelineError(MLPipelineException):
    """Raised when data pipeline operations fail."""
    pass


class ValidationError(MLPipelineException):
    """Raised when input validation fails."""
    pass


class AuthenticationError(MLPipelineException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(MLPipelineException):
    """Raised when authorization fails."""
    pass


class ExternalServiceError(MLPipelineException):
    """Raised when external service calls fail."""
    pass


class MLflowError(ExternalServiceError):
    """Raised when MLflow operations fail."""
    pass


class DriftDetectionError(MLPipelineException):
    """Raised when drift detection operations fail."""
    pass


class MonitoringError(MLPipelineException):
    """Raised when monitoring operations fail."""
    pass


class PredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class TrainingError(ModelError):
    """Raised when model training fails."""
    pass


class DataQualityError(DataPipelineError):
    """Raised when data quality checks fail."""
    pass


class SecurityError(MLPipelineException):
    """Raised when security violations are detected."""
    pass


class RateLimitError(MLPipelineException):
    """Raised when rate limits are exceeded."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class DeploymentError(MLPipelineException):
    """Raised when deployment operations fail."""
    pass


class AlertError(MonitoringError):
    """Raised when alerting operations fail."""
    pass