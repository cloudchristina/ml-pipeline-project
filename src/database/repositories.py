from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from .models import ExperimentRun, ModelDeployment, DatasetMetrics, PredictionLog, MonitoringAlert
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_run(self, run_data: Dict[str, Any]) -> ExperimentRun:
        """Create a new experiment run."""
        run = ExperimentRun(**run_data)
        self.session.add(run)
        self.session.flush()  # Get the ID without committing
        return run

    def get_run_by_id(self, run_id: UUID) -> Optional[ExperimentRun]:
        """Get experiment run by ID."""
        return self.session.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()

    def get_run_by_mlflow_id(self, mlflow_run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run by MLflow run ID."""
        return self.session.query(ExperimentRun).filter(
            ExperimentRun.mlflow_run_id == mlflow_run_id
        ).first()

    def update_run(self, run_id: UUID, update_data: Dict[str, Any]) -> Optional[ExperimentRun]:
        """Update experiment run."""
        run = self.get_run_by_id(run_id)
        if run:
            for key, value in update_data.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            run.updated_at = datetime.utcnow()
            self.session.flush()
        return run

    def get_best_runs(
        self,
        experiment_name: str = None,
        metric: str = "f1_score",
        limit: int = 10
    ) -> List[ExperimentRun]:
        """Get best performing runs based on a metric."""
        query = self.session.query(ExperimentRun)

        if experiment_name:
            query = query.filter(ExperimentRun.experiment_name == experiment_name)

        # Order by metric (assuming higher is better for most metrics)
        if hasattr(ExperimentRun, metric):
            query = query.filter(getattr(ExperimentRun, metric).is_not(None))
            query = query.order_by(desc(getattr(ExperimentRun, metric)))

        return query.limit(limit).all()

    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Get summary statistics for an experiment."""
        runs = self.session.query(ExperimentRun).filter(
            ExperimentRun.experiment_name == experiment_name
        ).all()

        if not runs:
            return {"experiment_name": experiment_name, "run_count": 0}

        # Calculate summary statistics
        accuracies = [r.accuracy for r in runs if r.accuracy is not None]
        f1_scores = [r.f1_score for r in runs if r.f1_score is not None]
        training_durations = [r.training_duration for r in runs if r.training_duration is not None]

        return {
            "experiment_name": experiment_name,
            "run_count": len(runs),
            "best_accuracy": max(accuracies) if accuracies else None,
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "best_f1_score": max(f1_scores) if f1_scores else None,
            "avg_f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else None,
            "avg_training_duration": sum(training_durations) / len(training_durations) if training_durations else None,
            "last_run_date": max(r.created_at for r in runs).isoformat(),
            "status_distribution": {
                status: len([r for r in runs if r.status == status])
                for status in set(r.status for r in runs)
            }
        }


class DeploymentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_deployment(self, deployment_data: Dict[str, Any]) -> ModelDeployment:
        """Create a new model deployment."""
        deployment = ModelDeployment(**deployment_data)
        self.session.add(deployment)
        self.session.flush()
        return deployment

    def get_deployment_by_id(self, deployment_id: UUID) -> Optional[ModelDeployment]:
        """Get deployment by ID."""
        return self.session.query(ModelDeployment).filter(
            ModelDeployment.id == deployment_id
        ).first()

    def get_active_deployments(self) -> List[ModelDeployment]:
        """Get all active deployments."""
        return self.session.query(ModelDeployment).filter(
            ModelDeployment.status == "ACTIVE"
        ).all()

    def update_deployment_status(
        self,
        deployment_id: UUID,
        status: str,
        health_status: str = None
    ) -> Optional[ModelDeployment]:
        """Update deployment status."""
        deployment = self.get_deployment_by_id(deployment_id)
        if deployment:
            deployment.status = status
            deployment.last_health_check = datetime.utcnow()
            if health_status:
                deployment.health_status = health_status
            self.session.flush()
        return deployment

    def update_deployment_metrics(
        self,
        deployment_id: UUID,
        request_count: int = None,
        avg_response_time: float = None,
        error_rate: float = None
    ) -> Optional[ModelDeployment]:
        """Update deployment performance metrics."""
        deployment = self.get_deployment_by_id(deployment_id)
        if deployment:
            if request_count is not None:
                deployment.request_count = request_count
            if avg_response_time is not None:
                deployment.avg_response_time = avg_response_time
            if error_rate is not None:
                deployment.error_rate = error_rate
            deployment.updated_at = datetime.utcnow()
            self.session.flush()
        return deployment

    def get_deployment_metrics(self, deployment_id: UUID, days: int = 7) -> Dict[str, Any]:
        """Get deployment metrics over specified time period."""
        deployment = self.get_deployment_by_id(deployment_id)
        if not deployment:
            return {}

        # Get prediction logs for the time period
        since_date = datetime.utcnow() - timedelta(days=days)
        predictions = self.session.query(PredictionLog).filter(
            PredictionLog.deployment_id == deployment_id,
            PredictionLog.predicted_at >= since_date
        ).all()

        if not predictions:
            return {"deployment_id": str(deployment_id), "metrics": {}}

        # Calculate metrics
        total_predictions = len(predictions)
        avg_prediction_time = sum(p.prediction_time for p in predictions if p.prediction_time) / total_predictions
        avg_confidence = sum(p.confidence_score for p in predictions if p.confidence_score) / total_predictions

        # Prediction distribution
        sentiment_counts = {}
        for pred in predictions:
            sentiment = pred.predicted_sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        return {
            "deployment_id": str(deployment_id),
            "time_period_days": days,
            "metrics": {
                "total_predictions": total_predictions,
                "avg_prediction_time_ms": avg_prediction_time,
                "avg_confidence_score": avg_confidence,
                "predictions_per_day": total_predictions / days,
                "sentiment_distribution": sentiment_counts
            }
        }


class DatasetRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_dataset_metrics(self, metrics_data: Dict[str, Any]) -> DatasetMetrics:
        """Create new dataset metrics entry."""
        metrics = DatasetMetrics(**metrics_data)
        self.session.add(metrics)
        self.session.flush()
        return metrics

    def get_latest_metrics(self, dataset_name: str, split_name: str) -> Optional[DatasetMetrics]:
        """Get latest metrics for a dataset split."""
        return self.session.query(DatasetMetrics).filter(
            DatasetMetrics.dataset_name == dataset_name,
            DatasetMetrics.split_name == split_name
        ).order_by(desc(DatasetMetrics.computed_at)).first()

    def get_drift_history(
        self,
        dataset_name: str,
        days: int = 30
    ) -> List[DatasetMetrics]:
        """Get data drift history for a dataset."""
        since_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(DatasetMetrics).filter(
            DatasetMetrics.dataset_name == dataset_name,
            DatasetMetrics.computed_at >= since_date,
            DatasetMetrics.drift_score.is_not(None)
        ).order_by(DatasetMetrics.computed_at).all()

    def detect_quality_degradation(
        self,
        dataset_name: str,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Detect quality degradation in dataset."""
        # Get recent metrics
        recent_metrics = self.session.query(DatasetMetrics).filter(
            DatasetMetrics.dataset_name == dataset_name
        ).order_by(desc(DatasetMetrics.computed_at)).limit(5).all()

        if len(recent_metrics) < 2:
            return []

        alerts = []
        current = recent_metrics[0]
        previous = recent_metrics[1]

        # Check for significant drops in quality scores
        quality_metrics = ["quality_score", "completeness_score", "consistency_score", "validity_score"]

        for metric in quality_metrics:
            current_val = getattr(current, metric)
            previous_val = getattr(previous, metric)

            if current_val and previous_val and (previous_val - current_val) > threshold:
                alerts.append({
                    "dataset_name": dataset_name,
                    "metric": metric,
                    "current_value": current_val,
                    "previous_value": previous_val,
                    "degradation": previous_val - current_val,
                    "timestamp": current.computed_at.isoformat()
                })

        return alerts


class PredictionRepository:
    def __init__(self, session: Session):
        self.session = session

    def log_prediction(self, prediction_data: Dict[str, Any]) -> PredictionLog:
        """Log a prediction."""
        prediction = PredictionLog(**prediction_data)
        self.session.add(prediction)
        self.session.flush()
        return prediction

    def get_predictions_by_deployment(
        self,
        deployment_id: UUID,
        limit: int = 1000,
        offset: int = 0
    ) -> List[PredictionLog]:
        """Get predictions for a specific deployment."""
        return self.session.query(PredictionLog).filter(
            PredictionLog.deployment_id == deployment_id
        ).order_by(desc(PredictionLog.predicted_at)).offset(offset).limit(limit).all()

    def get_prediction_statistics(
        self,
        deployment_id: UUID,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get prediction statistics for a deployment."""
        since_time = datetime.utcnow() - timedelta(hours=hours)

        predictions = self.session.query(PredictionLog).filter(
            PredictionLog.deployment_id == deployment_id,
            PredictionLog.predicted_at >= since_time
        ).all()

        if not predictions:
            return {"total_predictions": 0}

        # Calculate statistics
        total = len(predictions)
        confidence_scores = [p.confidence_score for p in predictions if p.confidence_score]
        prediction_times = [p.prediction_time for p in predictions if p.prediction_time]

        sentiment_counts = {}
        for pred in predictions:
            sentiment = pred.predicted_sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        return {
            "total_predictions": total,
            "time_period_hours": hours,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0,
            "max_confidence": max(confidence_scores) if confidence_scores else 0,
            "avg_prediction_time": sum(prediction_times) / len(prediction_times) if prediction_times else 0,
            "sentiment_distribution": sentiment_counts,
            "predictions_per_hour": total / hours
        }

    def detect_prediction_anomalies(
        self,
        deployment_id: UUID,
        confidence_threshold: float = 0.5
    ) -> List[PredictionLog]:
        """Detect predictions with low confidence scores."""
        return self.session.query(PredictionLog).filter(
            PredictionLog.deployment_id == deployment_id,
            PredictionLog.confidence_score < confidence_threshold
        ).order_by(desc(PredictionLog.predicted_at)).limit(100).all()


class MonitoringRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_alert(self, alert_data: Dict[str, Any]) -> MonitoringAlert:
        """Create a new monitoring alert."""
        alert = MonitoringAlert(**alert_data)
        self.session.add(alert)
        self.session.flush()
        return alert

    def get_active_alerts(self, severity: str = None) -> List[MonitoringAlert]:
        """Get active alerts, optionally filtered by severity."""
        query = self.session.query(MonitoringAlert).filter(
            MonitoringAlert.status == "ACTIVE"
        )

        if severity:
            query = query.filter(MonitoringAlert.severity == severity)

        return query.order_by(desc(MonitoringAlert.triggered_at)).all()

    def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str) -> Optional[MonitoringAlert]:
        """Acknowledge an alert."""
        alert = self.session.query(MonitoringAlert).filter(
            MonitoringAlert.id == alert_id
        ).first()

        if alert and alert.status == "ACTIVE":
            alert.status = "ACKNOWLEDGED"
            alert.acknowledged_at = datetime.utcnow()
            # Store who acknowledged in resolution_notes for now
            alert.resolution_notes = f"Acknowledged by: {acknowledged_by}"
            self.session.flush()

        return alert

    def resolve_alert(
        self,
        alert_id: UUID,
        resolved_by: str,
        resolution_notes: str = None
    ) -> Optional[MonitoringAlert]:
        """Resolve an alert."""
        alert = self.session.query(MonitoringAlert).filter(
            MonitoringAlert.id == alert_id
        ).first()

        if alert:
            alert.status = "RESOLVED"
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            if resolution_notes:
                alert.resolution_notes = resolution_notes
            self.session.flush()

        return alert

    def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get alert summary for the specified time period."""
        since_date = datetime.utcnow() - timedelta(days=days)

        alerts = self.session.query(MonitoringAlert).filter(
            MonitoringAlert.triggered_at >= since_date
        ).all()

        if not alerts:
            return {"total_alerts": 0, "time_period_days": days}

        # Group by status and severity
        status_counts = {}
        severity_counts = {}
        type_counts = {}

        for alert in alerts:
            # Status distribution
            status_counts[alert.status] = status_counts.get(alert.status, 0) + 1
            # Severity distribution
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            # Type distribution
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1

        # Calculate resolution metrics
        resolved_alerts = [a for a in alerts if a.status == "RESOLVED" and a.resolved_at]
        if resolved_alerts:
            resolution_times = []
            for alert in resolved_alerts:
                resolution_time = (alert.resolved_at - alert.triggered_at).total_seconds() / 3600  # hours
                resolution_times.append(resolution_time)

            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        else:
            avg_resolution_time = None

        return {
            "total_alerts": len(alerts),
            "time_period_days": days,
            "status_distribution": status_counts,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "avg_resolution_time_hours": avg_resolution_time,
            "alerts_per_day": len(alerts) / days
        }