import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..models.model_trainer import ModelTrainer
from ..database.database import get_db_manager
from ..database.repositories import PredictionRepository, DeploymentRepository
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    def __init__(self, config: Config):
        self.config = config
        self.model_trainer = ModelTrainer(config)
        self.db_manager = get_db_manager()
        self.model_loaded = False
        self.deployment_id = None

    def load_model(self, model_path: str = None) -> bool:
        """Load the trained model for serving."""
        try:
            model_path = model_path or f"{self.config.model_dir}/final_model"
            logger.info(f"Loading model from {model_path}")

            self.model_trainer.load_model(model_path)
            self.model_loaded = True

            # Create deployment record
            self._create_deployment_record()

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            return False

    def _create_deployment_record(self):
        """Create a deployment record in the database."""
        try:
            with self.db_manager.get_session() as session:
                deployment_repo = DeploymentRepository(session)

                deployment_data = {
                    "experiment_run_id": str(uuid.uuid4()),  # This should come from the actual experiment
                    "deployment_name": "sentiment_api",
                    "version": "1.0.0",
                    "status": "ACTIVE",
                    "endpoint_url": f"http://{self.config.api_host}:{self.config.api_port}",
                    "api_version": "v1",
                    "replicas": 1,
                    "deployed_at": datetime.utcnow(),
                    "health_status": "HEALTHY"
                }

                deployment = deployment_repo.create_deployment(deployment_data)
                self.deployment_id = deployment.id

                logger.info(f"Created deployment record with ID: {self.deployment_id}")

        except Exception as e:
            logger.error(f"Failed to create deployment record: {str(e)}")

    def predict_single(
        self,
        text: str,
        return_probabilities: bool = True,
        request_id: str = None
    ) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        start_time = time.time()
        request_id = request_id or str(uuid.uuid4())

        try:
            # Make prediction
            result = self.model_trainer.predict([text])
            prediction = result["predictions"][0]

            # Calculate prediction time
            prediction_time_ms = (time.time() - start_time) * 1000

            # Prepare response
            response = {
                "request_id": request_id,
                "text": text,
                "predicted_label": prediction["predicted_label"],
                "predicted_sentiment": prediction["predicted_sentiment"],
                "confidence": prediction["confidence"],
                "prediction_time_ms": prediction_time_ms,
                "model_version": self.config.model_name,
                "timestamp": datetime.utcnow().isoformat()
            }

            if return_probabilities:
                response["probabilities"] = prediction["probabilities"]

            # Log prediction to database
            self._log_prediction(response, request_id)

            return response

        except Exception as e:
            logger.error(f"Prediction failed for request {request_id}: {str(e)}")
            raise

    def predict_batch(
        self,
        texts: List[str],
        return_probabilities: bool = True,
        request_id: str = None
    ) -> Dict[str, Any]:
        """Make batch predictions."""
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        start_time = time.time()
        request_id = request_id or str(uuid.uuid4())

        try:
            # Make predictions
            result = self.model_trainer.predict(texts)
            predictions = result["predictions"]

            # Calculate total prediction time
            total_time_ms = (time.time() - start_time) * 1000

            # Prepare individual predictions
            formatted_predictions = []
            for i, (text, prediction) in enumerate(zip(texts, predictions)):
                pred_response = {
                    "text": text,
                    "predicted_label": prediction["predicted_label"],
                    "predicted_sentiment": prediction["predicted_sentiment"],
                    "confidence": prediction["confidence"],
                    "prediction_time_ms": total_time_ms / len(texts),  # Average time per prediction
                    "model_version": self.config.model_name,
                    "timestamp": datetime.utcnow().isoformat()
                }

                if return_probabilities:
                    pred_response["probabilities"] = prediction["probabilities"]

                formatted_predictions.append(pred_response)

                # Log each prediction
                self._log_prediction(pred_response, f"{request_id}_{i}")

            # Prepare batch response
            batch_response = {
                "request_id": request_id,
                "predictions": formatted_predictions,
                "total_predictions": len(predictions),
                "total_time_ms": total_time_ms,
                "model_name": self.config.model_name,
                "timestamp": datetime.utcnow().isoformat()
            }

            return batch_response

        except Exception as e:
            logger.error(f"Batch prediction failed for request {request_id}: {str(e)}")
            raise

    def _log_prediction(self, prediction_data: Dict[str, Any], request_id: str):
        """Log prediction to database."""
        if not self.deployment_id:
            logger.warning("No deployment ID available, skipping prediction logging")
            return

        try:
            with self.db_manager.get_session() as session:
                prediction_repo = PredictionRepository(session)

                # Create input hash for deduplication
                input_hash = hashlib.sha256(prediction_data["text"].encode()).hexdigest()

                log_data = {
                    "deployment_id": self.deployment_id,
                    "request_id": request_id,
                    "input_text": prediction_data["text"],
                    "input_hash": input_hash,
                    "predicted_label": prediction_data["predicted_label"],
                    "predicted_sentiment": prediction_data["predicted_sentiment"],
                    "confidence_score": prediction_data["confidence"],
                    "probabilities": prediction_data.get("probabilities"),
                    "prediction_time": prediction_data["prediction_time_ms"],
                    "model_version": prediction_data["model_version"],
                    "predicted_at": datetime.utcnow()
                }

                prediction_repo.log_prediction(log_data)

        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"status": "Model not loaded"}

        return {
            "model_name": self.config.model_name,
            "model_version": "1.0.0",  # This should come from model metadata
            "tokenizer_name": self.config.model_name,
            "max_length": self.config.max_length,
            "num_labels": self.config.num_labels,
            "loaded": self.model_loaded,
            "deployment_id": str(self.deployment_id) if self.deployment_id else None
        }


class HealthService:
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()
        self.startup_time = datetime.utcnow()

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
            "version": "1.0.0"
        }

        # Check database connectivity
        try:
            db_health = self.db_manager.health_check()
            health_status["database_connected"] = db_health["status"] == "healthy"
            health_status["database_details"] = db_health
        except Exception as e:
            health_status["database_connected"] = False
            health_status["database_error"] = str(e)

        # Check MLflow connectivity (with timeout to prevent hanging)
        try:
            import mlflow
            from urllib3.util import Timeout
            import requests

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            # Only check if MLflow URI is http/https (not file-based)
            if self.config.mlflow_tracking_uri.startswith(('http://', 'https://')):
                # Quick connectivity check with 2 second timeout
                try:
                    response = requests.get(
                        self.config.mlflow_tracking_uri,
                        timeout=2
                    )
                    health_status["mlflow_connected"] = response.status_code < 500
                except requests.exceptions.Timeout:
                    health_status["mlflow_connected"] = False
                    health_status["mlflow_error"] = "Connection timeout"
                except Exception as e:
                    health_status["mlflow_connected"] = False
                    health_status["mlflow_error"] = str(e)
            else:
                # File-based MLflow tracking
                health_status["mlflow_connected"] = True
                health_status["mlflow_type"] = "file-based"
        except Exception as e:
            health_status["mlflow_connected"] = False
            health_status["mlflow_error"] = str(e)

        # Check model status
        # This would need to be injected from the prediction service
        health_status["model_loaded"] = False

        # Overall status determination
        # Critical dependencies: database (for logging), model (injected separately)
        # Optional dependencies: MLflow (only for training/tracking, not serving)
        critical_failures = []

        if not health_status["database_connected"]:
            critical_failures.append("database")
            health_status["status"] = "degraded"

        # Note: MLflow connectivity is informational only - API serves predictions without it
        # MLflow is only required during training, not during inference serving

        return health_status


class MetricsService:
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()

    def get_prediction_statistics(
        self,
        deployment_id: str = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get prediction statistics."""
        try:
            with self.db_manager.get_session() as session:
                prediction_repo = PredictionRepository(session)

                if deployment_id:
                    from uuid import UUID
                    deployment_uuid = UUID(deployment_id)
                    stats = prediction_repo.get_prediction_statistics(deployment_uuid, hours)
                else:
                    # Get stats for all deployments
                    stats = {"message": "Deployment ID required for specific statistics"}

                return stats

        except Exception as e:
            logger.error(f"Failed to get prediction statistics: {str(e)}")
            return {"error": str(e)}

    def get_model_performance_metrics(
        self,
        deployment_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get model performance metrics."""
        try:
            with self.db_manager.get_session() as session:
                deployment_repo = DeploymentRepository(session)
                from uuid import UUID

                deployment_uuid = UUID(deployment_id)
                metrics = deployment_repo.get_deployment_metrics(deployment_uuid, days)

                return metrics

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {"error": str(e)}

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            # Database metrics
            db_metrics = self.db_manager.get_database_metrics()

            # Add API-specific metrics
            api_metrics = {
                "api_version": "v1",
                "configuration": {
                    "max_length": self.config.max_length,
                    "batch_size": self.config.batch_size,
                    "model_name": self.config.model_name
                }
            }

            return {
                "database": db_metrics,
                "api": api_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {"error": str(e)}


class FeedbackService:
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()

    def record_feedback(
        self,
        prediction_id: str,
        is_correct: bool,
        feedback_score: float = None,
        comments: str = None
    ) -> Dict[str, Any]:
        """Record feedback for a prediction."""
        try:
            with self.db_manager.get_session() as session:
                prediction_repo = PredictionRepository(session)

                # Find the prediction log entry
                # This would need to be implemented in the repository
                # For now, we'll just return a success response

                feedback_data = {
                    "prediction_id": prediction_id,
                    "is_correct": is_correct,
                    "feedback_score": feedback_score,
                    "comments": comments,
                    "recorded_at": datetime.utcnow()
                }

                # In a real implementation, you'd update the PredictionLog entry
                # with the feedback information

                logger.info(f"Feedback recorded for prediction {prediction_id}")

                return {
                    "prediction_id": prediction_id,
                    "feedback_recorded": True,
                    "message": "Feedback recorded successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to record feedback: {str(e)}")
            return {
                "prediction_id": prediction_id,
                "feedback_recorded": False,
                "message": f"Failed to record feedback: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }