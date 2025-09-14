from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, validator


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    return_probabilities: bool = Field(default=True, description="Whether to return class probabilities")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    return_probabilities: bool = Field(default=True, description="Whether to return class probabilities")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")

    @validator('texts')
    def validate_texts(cls, v):
        cleaned_texts = []
        for text in v:
            if not text.strip():
                raise ValueError('All texts must be non-empty')
            cleaned_texts.append(text.strip())
        return cleaned_texts


class PredictionResult(BaseModel):
    text: str
    predicted_label: int
    predicted_sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = None
    prediction_time_ms: float
    model_version: str
    timestamp: datetime


class BatchPredictionResult(BaseModel):
    request_id: Optional[str]
    predictions: List[PredictionResult]
    total_predictions: int
    total_time_ms: float
    model_name: str
    timestamp: datetime


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    mlflow_connected: bool
    version: str
    uptime_seconds: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    tokenizer_name: str
    max_length: int
    num_labels: int
    training_dataset: str
    training_date: Optional[datetime]
    performance_metrics: Optional[Dict[str, float]]


class ApiStatus(BaseModel):
    api_version: str
    model_info: ModelInfo
    health: HealthCheckResponse
    statistics: Dict[str, Any]


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., description="ID of the prediction to provide feedback for")
    is_correct: bool = Field(..., description="Whether the prediction was correct")
    feedback_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Feedback score (0-1)")
    comments: Optional[str] = Field(None, max_length=1000, description="Optional feedback comments")


class FeedbackResponse(BaseModel):
    prediction_id: str
    feedback_recorded: bool
    message: str
    timestamp: datetime


class ModelMetrics(BaseModel):
    deployment_id: str
    model_name: str
    version: str

    # Performance metrics
    total_predictions: int
    avg_prediction_time_ms: float
    predictions_per_hour: float

    # Quality metrics
    avg_confidence_score: float
    low_confidence_predictions: int
    error_rate: float

    # Distribution metrics
    sentiment_distribution: Dict[str, int]

    # Time period
    time_period_hours: int
    last_updated: datetime


class DriftAlert(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    deployment_id: Optional[str] = None


class AlertsResponse(BaseModel):
    alerts: List[DriftAlert]
    total_alerts: int
    active_alerts: int
    critical_alerts: int
    last_updated: datetime


class ExperimentSummary(BaseModel):
    experiment_name: str
    run_count: int
    best_accuracy: Optional[float]
    best_f1_score: Optional[float]
    avg_training_duration: Optional[float]
    last_run_date: str
    status_distribution: Dict[str, int]


class ModelDeploymentInfo(BaseModel):
    deployment_id: str
    deployment_name: str
    version: str
    status: str
    endpoint_url: Optional[str]
    health_status: str
    deployed_at: Optional[datetime]
    last_health_check: Optional[datetime]

    # Resource information
    replicas: int
    cpu_request: Optional[str]
    memory_request: Optional[str]

    # Performance
    request_count: int
    avg_response_time: Optional[float]
    error_rate: float


class DataQualityReport(BaseModel):
    dataset_name: str
    version: Optional[str]
    overall_quality_score: float
    timestamp: str

    checks: Dict[str, Dict[str, Any]]

    # Split-specific metrics
    train_metrics: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    dataset_name: str = Field(default="imdb", description="Dataset to use for training")
    model_name: Optional[str] = Field(None, description="Model to fine-tune (default from config)")
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name")
    run_name: Optional[str] = Field(None, description="MLflow run name")

    # Training parameters
    num_epochs: Optional[int] = Field(None, ge=1, le=10)
    batch_size: Optional[int] = Field(None, ge=1, le=64)
    learning_rate: Optional[float] = Field(None, gt=0.0, lt=1.0)

    # Optimization
    optimize_hyperparameters: bool = Field(default=False, description="Whether to perform hyperparameter optimization")
    n_trials: Optional[int] = Field(10, ge=1, le=50, description="Number of optimization trials")

    # Quick training for testing
    quick_run: bool = Field(default=False, description="Quick training run with limited data")
    max_samples: Optional[int] = Field(1000, ge=100, le=10000, description="Max samples for quick run")


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    mlflow_run_id: Optional[str]
    started_at: datetime
    estimated_duration: Optional[str]


class TrainingStatus(BaseModel):
    job_id: str
    status: str  # RUNNING, COMPLETED, FAILED, CANCELLED
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_step: Optional[str]

    # Results (if completed)
    mlflow_run_id: Optional[str]
    final_metrics: Optional[Dict[str, float]]
    model_path: Optional[str]

    # Timestamps
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    # Error information (if failed)
    error_message: Optional[str]
    error_details: Optional[Dict[str, Any]]


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None