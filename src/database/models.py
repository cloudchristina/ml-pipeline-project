from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mlflow_run_id = Column(String(255), unique=True, nullable=False)
    experiment_name = Column(String(255), nullable=False)
    run_name = Column(String(255), nullable=False)
    status = Column(String(50), default="RUNNING")

    # Model information
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(50))
    dataset_name = Column(String(255), nullable=False)

    # Performance metrics
    accuracy = Column(Float)
    f1_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    loss = Column(Float)

    # Training parameters
    hyperparameters = Column(JSON)
    training_duration = Column(Float)  # seconds

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Additional metadata
    metadata = Column(JSON)
    artifacts_path = Column(Text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "mlflow_run_id": self.mlflow_run_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "status": self.status,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
            "hyperparameters": self.hyperparameters,
            "training_duration": self.training_duration,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
            "artifacts_path": self.artifacts_path
        }


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_run_id = Column(UUID(as_uuid=True), nullable=False)

    # Deployment information
    deployment_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    status = Column(String(50), default="DEPLOYING")  # DEPLOYING, ACTIVE, INACTIVE, FAILED

    # Endpoint information
    endpoint_url = Column(String(500))
    api_version = Column(String(50), default="v1")

    # Resource allocation
    cpu_request = Column(String(50))
    memory_request = Column(String(50))
    cpu_limit = Column(String(50))
    memory_limit = Column(String(50))
    replicas = Column(Integer, default=1)

    # Performance tracking
    request_count = Column(Integer, default=0)
    avg_response_time = Column(Float)
    error_rate = Column(Float, default=0.0)

    # Timestamps
    deployed_at = Column(DateTime)
    last_health_check = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Additional metadata
    deployment_config = Column(JSON)
    health_status = Column(String(50), default="UNKNOWN")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "experiment_run_id": str(self.experiment_run_id),
            "deployment_name": self.deployment_name,
            "version": self.version,
            "status": self.status,
            "endpoint_url": self.endpoint_url,
            "api_version": self.api_version,
            "cpu_request": self.cpu_request,
            "memory_request": self.memory_request,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "replicas": self.replicas,
            "request_count": self.request_count,
            "avg_response_time": self.avg_response_time,
            "error_rate": self.error_rate,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deployment_config": self.deployment_config,
            "health_status": self.health_status
        }


class DatasetMetrics(Base):
    __tablename__ = "dataset_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name = Column(String(255), nullable=False)
    version = Column(String(50))
    split_name = Column(String(50), nullable=False)  # train, validation, test

    # Basic statistics
    num_examples = Column(Integer, nullable=False)
    avg_text_length = Column(Float)
    min_text_length = Column(Integer)
    max_text_length = Column(Integer)

    # Quality metrics
    quality_score = Column(Float)
    completeness_score = Column(Float)
    consistency_score = Column(Float)
    validity_score = Column(Float)

    # Label distribution
    label_distribution = Column(JSON)

    # Data drift metrics
    drift_score = Column(Float)
    is_drift_detected = Column(Boolean, default=False)

    # Timestamps
    computed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Additional metadata
    preprocessing_config = Column(JSON)
    statistics = Column(JSON)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "dataset_name": self.dataset_name,
            "version": self.version,
            "split_name": self.split_name,
            "num_examples": self.num_examples,
            "avg_text_length": self.avg_text_length,
            "min_text_length": self.min_text_length,
            "max_text_length": self.max_text_length,
            "quality_score": self.quality_score,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "validity_score": self.validity_score,
            "label_distribution": self.label_distribution,
            "drift_score": self.drift_score,
            "is_drift_detected": self.is_drift_detected,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "preprocessing_config": self.preprocessing_config,
            "statistics": self.statistics
        }


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_id = Column(UUID(as_uuid=True), nullable=False)

    # Request information
    request_id = Column(String(255), unique=True, nullable=False)
    input_text = Column(Text, nullable=False)
    input_hash = Column(String(255))  # Hash of input for deduplication

    # Prediction results
    predicted_label = Column(Integer)
    predicted_sentiment = Column(String(50))
    confidence_score = Column(Float)
    probabilities = Column(JSON)

    # Performance metrics
    prediction_time = Column(Float)  # milliseconds
    model_version = Column(String(50))

    # Request metadata
    user_agent = Column(String(500))
    ip_address = Column(String(45))  # IPv6 compatible
    api_version = Column(String(50))

    # Timestamps
    predicted_at = Column(DateTime, default=datetime.utcnow)

    # Feedback and monitoring
    feedback_score = Column(Float)  # User feedback if available
    is_correct = Column(Boolean)  # Ground truth if available
    drift_score = Column(Float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "deployment_id": str(self.deployment_id),
            "request_id": self.request_id,
            "input_text": self.input_text,
            "input_hash": self.input_hash,
            "predicted_label": self.predicted_label,
            "predicted_sentiment": self.predicted_sentiment,
            "confidence_score": self.confidence_score,
            "probabilities": self.probabilities,
            "prediction_time": self.prediction_time,
            "model_version": self.model_version,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "api_version": self.api_version,
            "predicted_at": self.predicted_at.isoformat() if self.predicted_at else None,
            "feedback_score": self.feedback_score,
            "is_correct": self.is_correct,
            "drift_score": self.drift_score
        }


class MonitoringAlert(Base):
    __tablename__ = "monitoring_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Alert information
    alert_type = Column(String(100), nullable=False)  # DATA_DRIFT, MODEL_DRIFT, PERFORMANCE, ERROR
    severity = Column(String(50), default="MEDIUM")  # LOW, MEDIUM, HIGH, CRITICAL
    status = Column(String(50), default="ACTIVE")  # ACTIVE, ACKNOWLEDGED, RESOLVED

    # Related entities
    deployment_id = Column(UUID(as_uuid=True))
    experiment_run_id = Column(UUID(as_uuid=True))

    # Alert details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    metric_name = Column(String(255))
    metric_value = Column(Float)
    threshold_value = Column(Float)

    # Resolution
    resolved_by = Column(String(255))
    resolution_notes = Column(Text)

    # Timestamps
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)

    # Additional data
    alert_data = Column(JSON)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "status": self.status,
            "deployment_id": str(self.deployment_id) if self.deployment_id else None,
            "experiment_run_id": str(self.experiment_run_id) if self.experiment_run_id else None,
            "title": self.title,
            "description": self.description,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "alert_data": self.alert_data
        }