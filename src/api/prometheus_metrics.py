"""
Prometheus metrics for the ML API.
Exposes metrics in Prometheus format for monitoring and alerting.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import generate_latest
from prometheus_client.core import CollectorRegistry

# Create custom registry to avoid conflicts
custom_registry = CollectorRegistry()

# Request metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=custom_registry
)

# Prediction metrics
api_predictions_total = Counter(
    'api_predictions_total',
    'Total number of predictions made',
    ['sentiment'],
    registry=custom_registry
)

api_prediction_duration_seconds = Histogram(
    'api_prediction_duration_seconds',
    'Prediction duration in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=custom_registry
)

api_prediction_confidence = Histogram(
    'api_prediction_confidence',
    'Distribution of prediction confidence scores',
    ['sentiment'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
    registry=custom_registry
)

# Error metrics
api_errors_total = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint'],
    registry=custom_registry
)

# Model metrics
model_loaded = Gauge(
    'model_loaded',
    'Whether the model is currently loaded (1=loaded, 0=not loaded)',
    registry=custom_registry
)

# System metrics
active_requests = Gauge(
    'api_active_requests',
    'Number of requests currently being processed',
    registry=custom_registry
)

# Database metrics
database_queries_total = Counter(
    'database_queries_total',
    'Total number of database queries',
    ['operation', 'table'],
    registry=custom_registry
)

# Info metrics
model_info = Info(
    'model',
    'Information about the currently loaded model',
    registry=custom_registry
)


def get_metrics():
    """
    Generate Prometheus metrics output.

    Returns:
        bytes: Prometheus-formatted metrics
    """
    return generate_latest(custom_registry)


def record_prediction(sentiment: str, confidence: float, duration_seconds: float, endpoint: str = 'predict'):
    """
    Record a prediction event with metrics.

    Args:
        sentiment: The predicted sentiment (POSITIVE/NEGATIVE)
        confidence: The confidence score (0-1)
        duration_seconds: Time taken for prediction in seconds
        endpoint: The API endpoint used
    """
    api_predictions_total.labels(sentiment=sentiment).inc()
    api_prediction_confidence.labels(sentiment=sentiment).observe(confidence)
    api_prediction_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)


def record_request(method: str, endpoint: str, status: int):
    """
    Record an API request.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status: HTTP status code
    """
    api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()


def record_error(error_type: str, endpoint: str):
    """
    Record an API error.

    Args:
        error_type: Type of error (e.g., ValidationError, ModelError)
        endpoint: API endpoint where error occurred
    """
    api_errors_total.labels(error_type=error_type, endpoint=endpoint).inc()


def set_model_loaded_status(loaded: bool):
    """
    Set the model loaded status.

    Args:
        loaded: True if model is loaded, False otherwise
    """
    model_loaded.set(1 if loaded else 0)


def set_model_info(model_name: str, version: str, **kwargs):
    """
    Set model information.

    Args:
        model_name: Name of the model
        version: Model version
        **kwargs: Additional model metadata
    """
    info_dict = {
        'model_name': model_name,
        'version': version,
        **kwargs
    }
    model_info.info(info_dict)


def increment_active_requests():
    """Increment the active requests counter."""
    active_requests.inc()


def decrement_active_requests():
    """Decrement the active requests counter."""
    active_requests.dec()


def record_database_query(operation: str, table: str):
    """
    Record a database query.

    Args:
        operation: Database operation (SELECT, INSERT, UPDATE, DELETE)
        table: Table name
    """
    database_queries_total.labels(operation=operation, table=table).inc()
