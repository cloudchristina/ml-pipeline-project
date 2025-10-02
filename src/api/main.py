import time
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from .models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResult,
    BatchPredictionResult,
    HealthCheckResponse,
    ModelInfo,
    ApiStatus,
    FeedbackRequest,
    FeedbackResponse,
    ModelMetrics,
    ErrorResponse
)
from .services import PredictionService, HealthService, MetricsService, FeedbackService
from . import prometheus_metrics
from ..database.database import get_db_session
from ..utils.config import config
from ..utils.logger import get_logger, setup_logging

# Setup directories for API service (only basic directories, no data pipeline)
config.setup_directories(data_pipeline=False)

# Setup logging
setup_logging(config.log_level, f"{config.logs_dir}/api.log")
logger = get_logger(__name__)

# Global services
prediction_service: PredictionService = None
health_service: HealthService = None
metrics_service: MetricsService = None
feedback_service: FeedbackService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global prediction_service, health_service, metrics_service, feedback_service

    # Startup
    logger.info("Starting up ML API service")

    try:
        # Initialize services
        prediction_service = PredictionService(config)
        health_service = HealthService(config)
        metrics_service = MetricsService(config)
        feedback_service = FeedbackService(config)

        # Load model (make this non-blocking)
        try:
            model_loaded = prediction_service.load_model()
            if not model_loaded:
                logger.warning("Model not loaded successfully, some endpoints may not work")
                prometheus_metrics.set_model_loaded_status(False)
            else:
                prometheus_metrics.set_model_loaded_status(True)
                model_info = prediction_service.get_model_info()
                prometheus_metrics.set_model_info(
                    model_name=model_info.get("model_name", "unknown"),
                    version=model_info.get("model_version", "1.0.0")
                )
        except Exception as model_error:
            logger.warning(f"Model loading failed, continuing without model: {str(model_error)}")
            prometheus_metrics.set_model_loaded_status(False)

        logger.info("ML API service started successfully")

    except Exception as e:
        logger.error(f"Failed to start ML API service: {str(e)}")
        # Don't raise - allow the API to start even with some failures
        logger.warning("API starting with limited functionality")

    yield

    # Shutdown
    logger.info("Shutting down ML API service")


# Create FastAPI app
app = FastAPI(
    title="ML Pipeline API",
    description="A comprehensive ML API for sentiment analysis with monitoring and drift detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS based on environment
if config.log_level.upper() == "DEBUG":
    # Development environment - more permissive
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
else:
    # Production environment - restrict to specific domains
    allowed_origins = [
        # Add your production domains here
        # "https://yourdomain.com",
        # "https://api.yourdomain.com"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specific methods only
    allow_headers=["Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=str(exc.detail),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            detail=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    try:
        metrics_output = prometheus_metrics.get_metrics()
        return Response(content=metrics_output, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Get API health status."""
    try:
        health_status = health_service.get_health_status()

        # Add model status
        if prediction_service:
            health_status["model_loaded"] = prediction_service.model_loaded

        return HealthCheckResponse(
            status=health_status["status"],
            model_loaded=health_status.get("model_loaded", False),
            database_connected=health_status.get("database_connected", False),
            mlflow_connected=health_status.get("mlflow_connected", False),
            version=health_status["version"],
            uptime_seconds=health_status["uptime_seconds"],
            timestamp=datetime.fromisoformat(health_status["timestamp"].replace('Z', '+00:00')),
            details=health_status
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/", response_model=ApiStatus)
async def root():
    """Get API status and information."""
    try:
        health = await health_check()
        model_info = prediction_service.get_model_info() if prediction_service else {}
        statistics = metrics_service.get_system_metrics() if metrics_service else {}

        return ApiStatus(
            api_version="v1",
            model_info=ModelInfo(
                model_name=model_info.get("model_name", "unknown"),
                model_version=model_info.get("model_version", "unknown"),
                tokenizer_name=model_info.get("tokenizer_name", "unknown"),
                max_length=model_info.get("max_length", 0),
                num_labels=model_info.get("num_labels", 0),
                training_dataset="imdb",
                training_date=None,
                performance_metrics=None
            ),
            health=health,
            statistics=statistics
        )

    except Exception as e:
        logger.error(f"Failed to get API status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get API status")


# Prediction endpoints
@app.post("/predict", response_model=PredictionResult)
async def predict_sentiment(request: PredictionRequest):
    """Make a single sentiment prediction."""
    if not prediction_service or not prediction_service.model_loaded:
        prometheus_metrics.record_error("ModelNotLoaded", "/predict")
        raise HTTPException(status_code=503, detail="Model not loaded")

    prometheus_metrics.increment_active_requests()
    start_time = time.time()

    try:
        logger.info(f"Processing prediction request for text length: {len(request.text)}")

        result = prediction_service.predict_single(
            text=request.text,
            return_probabilities=request.return_probabilities,
            request_id=request.request_id
        )

        # Record metrics
        duration_seconds = time.time() - start_time
        prometheus_metrics.record_prediction(
            sentiment=result["predicted_sentiment"],
            confidence=result["confidence"],
            duration_seconds=duration_seconds,
            endpoint="predict"
        )
        prometheus_metrics.record_request("POST", "/predict", 200)

        return PredictionResult(
            text=result["text"],
            predicted_label=result["predicted_label"],
            predicted_sentiment=result["predicted_sentiment"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
            prediction_time_ms=result["prediction_time_ms"],
            model_version=result["model_version"],
            timestamp=datetime.fromisoformat(result["timestamp"])
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        prometheus_metrics.record_error("PredictionError", "/predict")
        prometheus_metrics.record_request("POST", "/predict", 500)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        prometheus_metrics.decrement_active_requests()


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch sentiment predictions."""
    if not prediction_service or not prediction_service.model_loaded:
        prometheus_metrics.record_error("ModelNotLoaded", "/predict/batch")
        raise HTTPException(status_code=503, detail="Model not loaded")

    prometheus_metrics.increment_active_requests()
    start_time = time.time()

    try:
        logger.info(f"Processing batch prediction request for {len(request.texts)} texts")

        result = prediction_service.predict_batch(
            texts=request.texts,
            return_probabilities=request.return_probabilities,
            request_id=request.request_id
        )

        # Convert predictions to PredictionResult objects and record metrics
        predictions = []
        for pred in result["predictions"]:
            predictions.append(
                PredictionResult(
                    text=pred["text"],
                    predicted_label=pred["predicted_label"],
                    predicted_sentiment=pred["predicted_sentiment"],
                    confidence=pred["confidence"],
                    probabilities=pred.get("probabilities"),
                    prediction_time_ms=pred["prediction_time_ms"],
                    model_version=pred["model_version"],
                    timestamp=datetime.fromisoformat(pred["timestamp"])
                )
            )

            # Record individual prediction metrics
            prometheus_metrics.record_prediction(
                sentiment=pred["predicted_sentiment"],
                confidence=pred["confidence"],
                duration_seconds=pred["prediction_time_ms"] / 1000.0,
                endpoint="batch"
            )

        # Record batch request
        prometheus_metrics.record_request("POST", "/predict/batch", 200)

        return BatchPredictionResult(
            request_id=result["request_id"],
            predictions=predictions,
            total_predictions=result["total_predictions"],
            total_time_ms=result["total_time_ms"],
            model_name=result["model_name"],
            timestamp=datetime.fromisoformat(result["timestamp"])
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        prometheus_metrics.record_error("PredictionError", "/predict/batch")
        prometheus_metrics.record_request("POST", "/predict/batch", 500)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    finally:
        prometheus_metrics.decrement_active_requests()


# Feedback endpoint
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a prediction."""
    if not feedback_service:
        raise HTTPException(status_code=503, detail="Feedback service not available")

    try:
        logger.info(f"Recording feedback for prediction: {request.prediction_id}")

        result = feedback_service.record_feedback(
            prediction_id=request.prediction_id,
            is_correct=request.is_correct,
            feedback_score=request.feedback_score,
            comments=request.comments
        )

        return FeedbackResponse(
            prediction_id=result["prediction_id"],
            feedback_recorded=result["feedback_recorded"],
            message=result["message"],
            timestamp=datetime.fromisoformat(result["timestamp"])
        )

    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


# Metrics endpoints
@app.get("/metrics/model/{deployment_id}", response_model=ModelMetrics)
async def get_model_metrics(deployment_id: str, hours: int = 24):
    """Get model performance metrics."""
    if not metrics_service:
        raise HTTPException(status_code=503, detail="Metrics service not available")

    try:
        stats = metrics_service.get_prediction_statistics(deployment_id, hours)
        perf_metrics = metrics_service.get_model_performance_metrics(deployment_id)

        return ModelMetrics(
            deployment_id=deployment_id,
            model_name=config.model_name,
            version="1.0.0",
            total_predictions=stats.get("total_predictions", 0),
            avg_prediction_time_ms=stats.get("avg_prediction_time", 0.0),
            predictions_per_hour=stats.get("predictions_per_hour", 0.0),
            avg_confidence_score=stats.get("avg_confidence", 0.0),
            low_confidence_predictions=0,  # Would need to calculate
            error_rate=0.0,  # Would need to calculate
            sentiment_distribution=stats.get("sentiment_distribution", {}),
            time_period_hours=hours,
            last_updated=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Failed to get model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")


@app.get("/metrics/system")
async def get_system_metrics():
    """Get system metrics."""
    if not metrics_service:
        raise HTTPException(status_code=503, detail="Metrics service not available")

    try:
        return metrics_service.get_system_metrics()

    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


# Admin endpoints
@app.post("/admin/reload-model")
async def reload_model(model_path: str = None):
    """Reload the model (admin endpoint)."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")

    try:
        logger.info("Reloading model...")
        success = prediction_service.load_model(model_path)

        if success:
            return {"message": "Model reloaded successfully", "timestamp": datetime.utcnow()}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")

    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/admin/config")
async def get_config():
    """Get current configuration (admin endpoint)."""
    try:
        return {
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "api_version": "v1",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


# Database session dependency for other endpoints that might need it
def get_db():
    return next(get_db_session())


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level=config.log_level.lower(),
        reload=False  # Set to True for development
    )