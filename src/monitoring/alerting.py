import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest

from ..database.database import get_db_manager
from ..database.repositories import MonitoringRepository, DeploymentRepository
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AlertingService:
    """Service for managing and sending alerts."""

    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()

    def send_email_alert(
        self,
        to_email: str,
        subject: str,
        message: str,
        alert_data: Dict[str, Any] = None
    ) -> bool:
        """Send email alert."""
        if not self.config.alert_email:
            logger.warning("No alert email configured, skipping email notification")
            return False

        try:
            # This is a simplified email sender
            # In production, you'd use a proper email service like SendGrid, SES, etc.
            logger.info(f"Sending email alert to {to_email}: {subject}")

            # Format alert message
            email_body = self._format_alert_email(subject, message, alert_data)

            # For demonstration, just log the email content
            # In production, implement actual email sending
            logger.info(f"Email alert content:\n{email_body}")

            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def _format_alert_email(
        self,
        subject: str,
        message: str,
        alert_data: Dict[str, Any] = None
    ) -> str:
        """Format alert email content."""
        email_body = f"""
ML Pipeline Alert: {subject}

Alert Details:
{message}

Timestamp: {datetime.utcnow().isoformat()}
"""

        if alert_data:
            email_body += f"\nAdditional Information:\n"
            for key, value in alert_data.items():
                email_body += f"- {key}: {value}\n"

        email_body += f"\nPlease check the monitoring dashboard for more details."

        return email_body

    def send_slack_alert(
        self,
        webhook_url: str,
        message: str,
        channel: str = "#alerts",
        alert_data: Dict[str, Any] = None
    ) -> bool:
        """Send Slack alert via webhook."""
        try:
            slack_message = {
                "channel": channel,
                "text": f"🚨 ML Pipeline Alert",
                "attachments": [
                    {
                        "color": "danger",
                        "fields": [
                            {
                                "title": "Alert Message",
                                "value": message,
                                "short": False
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.utcnow().isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }

            if alert_data:
                for key, value in alert_data.items():
                    slack_message["attachments"][0]["fields"].append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True
                    })

            response = requests.post(webhook_url, json=slack_message, timeout=10)
            response.raise_for_status()

            logger.info("Slack alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def process_alert_queue(self):
        """Process pending alerts and send notifications."""
        try:
            with self.db_manager.get_session() as session:
                monitoring_repo = MonitoringRepository(session)

                # Get active alerts that haven't been processed
                active_alerts = monitoring_repo.get_active_alerts()

                for alert in active_alerts:
                    self._process_single_alert(alert, monitoring_repo)

                logger.info(f"Processed {len(active_alerts)} alerts")

        except Exception as e:
            logger.error(f"Failed to process alert queue: {str(e)}")

    def _process_single_alert(self, alert, monitoring_repo: MonitoringRepository):
        """Process a single alert."""
        try:
            # Determine notification method based on severity
            if alert.severity in ["HIGH", "CRITICAL"]:
                # Send email for high severity alerts
                if self.config.alert_email:
                    self.send_email_alert(
                        to_email=self.config.alert_email,
                        subject=alert.title,
                        message=alert.description,
                        alert_data={
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "metric_value": alert.metric_value,
                            "threshold": alert.threshold_value
                        }
                    )

            # Log alert processing
            logger.info(f"Processed alert {alert.id}: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to process alert {alert.id}: {str(e)}")


class PrometheusExporter:
    """Prometheus metrics exporter for ML pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.registry = CollectorRegistry()
        self.db_manager = get_db_manager()

        # Define metrics
        self.prediction_count = Counter(
            'ml_predictions_total',
            'Total number of predictions made',
            ['deployment_id', 'sentiment'],
            registry=self.registry
        )

        self.prediction_time = Histogram(
            'ml_prediction_duration_seconds',
            'Time spent on predictions',
            ['deployment_id'],
            registry=self.registry
        )

        self.model_confidence = Gauge(
            'ml_model_confidence_score',
            'Average model confidence score',
            ['deployment_id'],
            registry=self.registry
        )

        self.drift_score = Gauge(
            'ml_drift_score',
            'Data/Model drift score',
            ['deployment_id', 'drift_type'],
            registry=self.registry
        )

        self.active_alerts = Gauge(
            'ml_active_alerts_total',
            'Number of active alerts',
            ['severity', 'alert_type'],
            registry=self.registry
        )

        self.api_requests = Counter(
            'ml_api_requests_total',
            'Total API requests',
            ['endpoint', 'status_code'],
            registry=self.registry
        )

    def update_metrics(self):
        """Update all Prometheus metrics."""
        try:
            logger.info("Updating Prometheus metrics")

            with self.db_manager.get_session() as session:
                # Update deployment metrics
                deployment_repo = DeploymentRepository(session)
                active_deployments = deployment_repo.get_active_deployments()

                for deployment in active_deployments:
                    deployment_id = str(deployment.id)

                    # Update prediction metrics
                    self.model_confidence.labels(deployment_id=deployment_id).set(
                        deployment.avg_response_time or 0
                    )

                # Update alert metrics
                monitoring_repo = MonitoringRepository(session)
                alerts = monitoring_repo.get_active_alerts()

                # Clear previous alert metrics
                self.active_alerts.clear()

                # Group alerts by severity and type
                alert_groups = {}
                for alert in alerts:
                    key = (alert.severity, alert.alert_type)
                    alert_groups[key] = alert_groups.get(key, 0) + 1

                for (severity, alert_type), count in alert_groups.items():
                    self.active_alerts.labels(severity=severity, alert_type=alert_type).set(count)

            logger.info("Prometheus metrics updated successfully")

        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {str(e)}")

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        try:
            self.update_metrics()
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {str(e)}")
            return ""

    def record_prediction(self, deployment_id: str, sentiment: str, prediction_time: float):
        """Record a prediction event."""
        try:
            self.prediction_count.labels(
                deployment_id=deployment_id,
                sentiment=sentiment
            ).inc()

            self.prediction_time.labels(
                deployment_id=deployment_id
            ).observe(prediction_time / 1000.0)  # Convert ms to seconds

        except Exception as e:
            logger.error(f"Failed to record prediction metric: {str(e)}")

    def record_api_request(self, endpoint: str, status_code: int):
        """Record an API request."""
        try:
            self.api_requests.labels(
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record API request metric: {str(e)}")


class HealthChecker:
    """Health checking and monitoring service."""

    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()

    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }

        try:
            # Database health
            db_health = self._check_database_health()
            health_status["components"]["database"] = db_health

            # Model service health
            model_health = self._check_model_service_health()
            health_status["components"]["model_service"] = model_health

            # Alert system health
            alert_health = self._check_alert_system_health()
            health_status["components"]["alert_system"] = alert_health

            # Determine overall status
            component_statuses = [comp["status"] for comp in health_status["components"].values()]
            if "critical" in component_statuses:
                health_status["overall_status"] = "critical"
            elif "unhealthy" in component_statuses:
                health_status["overall_status"] = "unhealthy"
            elif "degraded" in component_statuses:
                health_status["overall_status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "critical",
                "error": str(e)
            }

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            db_health = self.db_manager.health_check()
            return {
                "status": "healthy" if db_health["status"] == "healthy" else "unhealthy",
                "details": db_health
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e)
            }

    def _check_model_service_health(self) -> Dict[str, Any]:
        """Check model service health."""
        try:
            # This would typically make a request to your model service
            # For now, we'll simulate it
            return {
                "status": "healthy",
                "model_loaded": True,
                "last_prediction": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _check_alert_system_health(self) -> Dict[str, Any]:
        """Check alerting system health."""
        try:
            with self.db_manager.get_session() as session:
                monitoring_repo = MonitoringRepository(session)

                # Check for stuck alerts (active for too long)
                stuck_threshold = datetime.utcnow() - timedelta(hours=24)
                stuck_alerts = monitoring_repo.session.query(monitoring_repo.session.bind.execute(
                    """
                    SELECT COUNT(*)
                    FROM monitoring_alerts
                    WHERE status = 'ACTIVE'
                    AND triggered_at < %s
                    """,
                    (stuck_threshold,)
                )).scalar()

                status = "degraded" if stuck_alerts > 5 else "healthy"

                return {
                    "status": status,
                    "stuck_alerts": stuck_alerts,
                    "email_configured": bool(self.config.alert_email)
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def run_periodic_checks(self):
        """Run periodic health checks and create alerts if needed."""
        try:
            logger.info("Running periodic health checks")

            health = self.check_system_health()

            if health["overall_status"] in ["critical", "unhealthy"]:
                # Create system health alert
                with self.db_manager.get_session() as session:
                    monitoring_repo = MonitoringRepository(session)

                    alert_data = {
                        "alert_type": "SYSTEM_HEALTH",
                        "severity": "CRITICAL" if health["overall_status"] == "critical" else "HIGH",
                        "title": f"System Health: {health['overall_status'].upper()}",
                        "description": f"System health status: {health['overall_status']}",
                        "alert_data": health
                    }

                    monitoring_repo.create_alert(alert_data)

            logger.info(f"Health check completed: {health['overall_status']}")

        except Exception as e:
            logger.error(f"Periodic health check failed: {str(e)}")


# Global instances
alerting_service: Optional[AlertingService] = None
prometheus_exporter: Optional[PrometheusExporter] = None
health_checker: Optional[HealthChecker] = None


def get_alerting_service() -> AlertingService:
    """Get alerting service instance."""
    global alerting_service
    if alerting_service is None:
        from ..utils.config import config
        alerting_service = AlertingService(config)
    return alerting_service


def get_prometheus_exporter() -> PrometheusExporter:
    """Get Prometheus exporter instance."""
    global prometheus_exporter
    if prometheus_exporter is None:
        from ..utils.config import config
        prometheus_exporter = PrometheusExporter(config)
    return prometheus_exporter


def get_health_checker() -> HealthChecker:
    """Get health checker instance."""
    global health_checker
    if health_checker is None:
        from ..utils.config import config
        health_checker = HealthChecker(config)
    return health_checker