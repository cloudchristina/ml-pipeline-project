import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    TextDescriptorsDistribution,
    TextDescriptorsDriftMetric
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..database.database import get_db_manager
from ..database.repositories import DatasetRepository, PredictionRepository, MonitoringRepository
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    is_drift_detected: bool
    drift_score: float
    drift_type: str  # "data", "model", "concept"
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime


class TextDataDriftDetector:
    """Drift detection for text data using various methods."""

    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()
        self.reference_data = None
        self.reference_features = None
        self.vectorizer = None

    def set_reference_data(self, texts: List[str], labels: List[int] = None):
        """Set reference data for drift comparison."""
        logger.info(f"Setting reference data with {len(texts)} samples")

        self.reference_data = {
            'texts': texts,
            'labels': labels
        }

        # Extract text features for comparison
        self._extract_text_features(texts)

        logger.info("Reference data set successfully")

    def _extract_text_features(self, texts: List[str]):
        """Extract features from text data."""
        # TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )

        self.reference_features = self.vectorizer.fit_transform(texts)

        logger.info(f"Extracted features: {self.reference_features.shape}")

    def detect_distribution_drift(
        self,
        new_texts: List[str],
        threshold: float = 0.1
    ) -> DriftDetectionResult:
        """Detect drift in text distribution using TF-IDF similarity."""
        if self.reference_features is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        logger.info(f"Detecting distribution drift for {len(new_texts)} samples")

        try:
            # Extract features from new data
            new_features = self.vectorizer.transform(new_texts)

            # Calculate distribution similarity
            ref_mean = self.reference_features.mean(axis=0)
            new_mean = new_features.mean(axis=0)

            # Cosine similarity between reference and new distributions
            similarity = cosine_similarity(ref_mean, new_mean)[0, 0]
            drift_score = 1.0 - similarity

            # Statistical tests on text statistics
            ref_lengths = [len(text.split()) for text in self.reference_data['texts']]
            new_lengths = [len(text.split()) for text in new_texts]

            # Length distribution comparison
            from scipy import stats
            length_stat, length_p_value = stats.ks_2samp(ref_lengths, new_lengths)

            is_drift = (drift_score > threshold) or (length_p_value < 0.05)

            details = {
                'cosine_similarity': similarity,
                'distribution_drift_score': drift_score,
                'length_ks_statistic': length_stat,
                'length_p_value': length_p_value,
                'reference_avg_length': np.mean(ref_lengths),
                'new_avg_length': np.mean(new_lengths),
                'threshold': threshold
            }

            return DriftDetectionResult(
                is_drift_detected=is_drift,
                drift_score=drift_score,
                drift_type="data",
                confidence=max(drift_score, 1.0 - length_p_value) if is_drift else 1.0 - drift_score,
                details=details,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error in distribution drift detection: {str(e)}")
            raise

    def detect_concept_drift(
        self,
        new_texts: List[str],
        new_predictions: List[Dict[str, Any]],
        threshold: float = 0.1
    ) -> DriftDetectionResult:
        """Detect concept drift based on prediction confidence and patterns."""
        logger.info(f"Detecting concept drift for {len(new_predictions)} predictions")

        try:
            # Analyze confidence scores
            confidences = [pred['confidence'] for pred in new_predictions]
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            low_confidence_ratio = sum(1 for c in confidences if c < 0.6) / len(confidences)

            # Analyze prediction distribution
            sentiment_counts = {}
            for pred in new_predictions:
                sentiment = pred['predicted_sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

            # Calculate entropy of predictions
            total_preds = len(new_predictions)
            entropy = 0
            for count in sentiment_counts.values():
                p = count / total_preds
                if p > 0:
                    entropy -= p * np.log2(p)

            # Drift indicators
            drift_indicators = {
                'low_avg_confidence': avg_confidence < 0.7,
                'high_confidence_variance': confidence_std > 0.3,
                'high_low_confidence_ratio': low_confidence_ratio > 0.3,
                'unusual_prediction_entropy': entropy < 0.5 or entropy > 0.95
            }

            drift_score = sum(drift_indicators.values()) / len(drift_indicators)
            is_drift = drift_score > threshold

            details = {
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'low_confidence_ratio': low_confidence_ratio,
                'prediction_entropy': entropy,
                'sentiment_distribution': sentiment_counts,
                'drift_indicators': drift_indicators,
                'threshold': threshold
            }

            return DriftDetectionResult(
                is_drift_detected=is_drift,
                drift_score=drift_score,
                drift_type="concept",
                confidence=drift_score,
                details=details,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error in concept drift detection: {str(e)}")
            raise

    def detect_model_performance_drift(
        self,
        deployment_id: str,
        lookback_days: int = 7,
        threshold: float = 0.05
    ) -> DriftDetectionResult:
        """Detect drift in model performance over time."""
        logger.info(f"Detecting model performance drift for deployment {deployment_id}")

        try:
            with self.db_manager.get_session() as session:
                prediction_repo = PredictionRepository(session)

                # Get recent predictions
                since_date = datetime.utcnow() - timedelta(days=lookback_days)
                from uuid import UUID
                predictions = prediction_repo.session.query(prediction_repo.session.bind.execute(
                    """
                    SELECT
                        DATE(predicted_at) as prediction_date,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(*) as prediction_count,
                        AVG(prediction_time) as avg_prediction_time
                    FROM prediction_logs
                    WHERE deployment_id = %s
                    AND predicted_at >= %s
                    GROUP BY DATE(predicted_at)
                    ORDER BY prediction_date
                    """,
                    (str(deployment_id), since_date)
                )).fetchall()

                if len(predictions) < 3:
                    logger.warning("Insufficient data for performance drift detection")
                    return DriftDetectionResult(
                        is_drift_detected=False,
                        drift_score=0.0,
                        drift_type="model",
                        confidence=0.0,
                        details={"error": "Insufficient data"},
                        timestamp=datetime.utcnow()
                    )

                # Analyze trends
                confidence_values = [p[1] for p in predictions]  # avg_confidence
                response_times = [p[3] for p in predictions if p[3]]  # avg_prediction_time

                # Linear regression to detect trends
                from scipy import stats
                days = list(range(len(confidence_values)))

                # Confidence trend
                confidence_slope, confidence_intercept, confidence_r, confidence_p, _ = stats.linregress(days, confidence_values)

                # Response time trend (if available)
                response_trend = {}
                if response_times and len(response_times) == len(days):
                    response_slope, response_intercept, response_r, response_p, _ = stats.linregress(days, response_times)
                    response_trend = {
                        'slope': response_slope,
                        'r_value': response_r,
                        'p_value': response_p
                    }

                # Determine drift
                confidence_declining = confidence_slope < -threshold and confidence_p < 0.05
                performance_degrading = (response_trend.get('slope', 0) > threshold * 1000 and
                                       response_trend.get('p_value', 1) < 0.05)

                is_drift = confidence_declining or performance_degrading
                drift_score = abs(confidence_slope) if confidence_declining else abs(response_trend.get('slope', 0) / 1000)

                details = {
                    'confidence_trend': {
                        'slope': confidence_slope,
                        'r_value': confidence_r,
                        'p_value': confidence_p
                    },
                    'response_time_trend': response_trend,
                    'confidence_declining': confidence_declining,
                    'performance_degrading': performance_degrading,
                    'lookback_days': lookback_days,
                    'data_points': len(predictions)
                }

                return DriftDetectionResult(
                    is_drift_detected=is_drift,
                    drift_score=drift_score,
                    drift_type="model",
                    confidence=1.0 - min(confidence_p, response_trend.get('p_value', 1)),
                    details=details,
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            logger.error(f"Error in model performance drift detection: {str(e)}")
            raise


class EvidencelyDriftDetector:
    """Drift detection using Evidently AI library."""

    def __init__(self, config: Config):
        self.config = config
        self.db_manager = get_db_manager()

    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label"
    ) -> Dict[str, Any]:
        """Generate comprehensive drift report using Evidently."""
        logger.info("Generating Evidently drift report")

        try:
            # Column mapping for text data
            column_mapping = ColumnMapping()
            column_mapping.text_features = [text_column]
            if label_column in reference_data.columns:
                column_mapping.target = label_column

            # Create drift report
            data_drift_report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
                TextDescriptorsDistribution(),
                TextDescriptorsDriftMetric()
            ])

            data_drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )

            # Extract results
            drift_results = data_drift_report.as_dict()

            return {
                "report": drift_results,
                "summary": self._extract_drift_summary(drift_results),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating Evidently report: {str(e)}")
            raise

    def _extract_drift_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from Evidently report."""
        try:
            summary = {
                "dataset_drift_detected": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "drift_details": {}
            }

            # Extract dataset-level drift
            for metric in report.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    result = metric.get("result", {})
                    summary["dataset_drift_detected"] = result.get("dataset_drift", False)
                    summary["drift_score"] = result.get("drift_share", 0.0)

                # Extract feature-level drift
                elif metric.get("metric") == "DataDriftTable":
                    result = metric.get("result", {})
                    drift_by_columns = result.get("drift_by_columns", {})

                    for column, drift_info in drift_by_columns.items():
                        if drift_info.get("drift_detected", False):
                            summary["drifted_features"].append({
                                "feature": column,
                                "drift_score": drift_info.get("drift_score", 0.0),
                                "stattest": drift_info.get("stattest_name", "unknown")
                            })

            return summary

        except Exception as e:
            logger.error(f"Error extracting drift summary: {str(e)}")
            return {"error": str(e)}


class DriftMonitor:
    """Main drift monitoring orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self.text_detector = TextDataDriftDetector(config)
        self.evidently_detector = EvidencelyDriftDetector(config)
        self.db_manager = get_db_manager()

    def initialize_reference_data(self, dataset_name: str = "imdb"):
        """Initialize reference data from training dataset."""
        logger.info(f"Initializing reference data from {dataset_name}")

        try:
            # Load reference data from database or dataset
            # This is a simplified version - in practice, you'd load from your training data
            from ..data.data_pipeline import DataPipeline

            data_pipeline = DataPipeline(self.config)
            dataset = data_pipeline.execute_pipeline(dataset_name=dataset_name)

            # Extract sample for reference
            reference_texts = []
            reference_labels = []

            # Take a sample from validation set
            validation_data = dataset.get("validation", dataset.get("train"))
            sample_size = min(1000, len(validation_data))

            for i in range(sample_size):
                if "text" in validation_data[i]:
                    reference_texts.append(validation_data[i]["text"])
                    reference_labels.append(validation_data[i].get("label", 0))

            self.text_detector.set_reference_data(reference_texts, reference_labels)

            logger.info(f"Reference data initialized with {len(reference_texts)} samples")

        except Exception as e:
            logger.error(f"Failed to initialize reference data: {str(e)}")
            raise

    def run_drift_detection(
        self,
        deployment_id: str,
        hours_lookback: int = 24
    ) -> List[DriftDetectionResult]:
        """Run comprehensive drift detection."""
        logger.info(f"Running drift detection for deployment {deployment_id}")

        results = []

        try:
            with self.db_manager.get_session() as session:
                prediction_repo = PredictionRepository(session)
                monitoring_repo = MonitoringRepository(session)

                # Get recent predictions
                since_time = datetime.utcnow() - timedelta(hours=hours_lookback)
                from uuid import UUID
                deployment_uuid = UUID(deployment_id)

                recent_predictions = prediction_repo.session.query(prediction_repo.session.bind.execute(
                    """
                    SELECT input_text, predicted_sentiment, confidence_score, probabilities
                    FROM prediction_logs
                    WHERE deployment_id = %s
                    AND predicted_at >= %s
                    ORDER BY predicted_at DESC
                    LIMIT 1000
                    """,
                    (str(deployment_id), since_time)
                )).fetchall()

                if len(recent_predictions) < 10:
                    logger.warning("Insufficient recent predictions for drift detection")
                    return results

                # Extract data
                texts = [p[0] for p in recent_predictions]
                predictions = [
                    {
                        'predicted_sentiment': p[1],
                        'confidence': p[2],
                        'probabilities': p[3]
                    }
                    for p in recent_predictions
                ]

                # 1. Data distribution drift
                try:
                    data_drift = self.text_detector.detect_distribution_drift(texts)
                    results.append(data_drift)

                    if data_drift.is_drift_detected:
                        self._create_drift_alert(
                            monitoring_repo, deployment_id, data_drift, "DATA_DRIFT"
                        )
                except Exception as e:
                    logger.error(f"Data drift detection failed: {str(e)}")

                # 2. Concept drift
                try:
                    concept_drift = self.text_detector.detect_concept_drift(texts, predictions)
                    results.append(concept_drift)

                    if concept_drift.is_drift_detected:
                        self._create_drift_alert(
                            monitoring_repo, deployment_id, concept_drift, "CONCEPT_DRIFT"
                        )
                except Exception as e:
                    logger.error(f"Concept drift detection failed: {str(e)}")

                # 3. Model performance drift
                try:
                    perf_drift = self.text_detector.detect_model_performance_drift(deployment_id)
                    results.append(perf_drift)

                    if perf_drift.is_drift_detected:
                        self._create_drift_alert(
                            monitoring_repo, deployment_id, perf_drift, "MODEL_DRIFT"
                        )
                except Exception as e:
                    logger.error(f"Performance drift detection failed: {str(e)}")

                logger.info(f"Drift detection completed. {len(results)} checks performed")
                return results

        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            raise

    def _create_drift_alert(
        self,
        monitoring_repo: MonitoringRepository,
        deployment_id: str,
        drift_result: DriftDetectionResult,
        alert_type: str
    ):
        """Create an alert for detected drift."""
        try:
            severity = "HIGH" if drift_result.drift_score > 0.3 else "MEDIUM"

            alert_data = {
                "alert_type": alert_type,
                "severity": severity,
                "deployment_id": deployment_id,
                "title": f"{alert_type.replace('_', ' ').title()} Detected",
                "description": f"Drift detected with score {drift_result.drift_score:.3f}",
                "metric_name": f"{drift_result.drift_type}_drift_score",
                "metric_value": drift_result.drift_score,
                "threshold_value": self.config.drift_detection_threshold,
                "alert_data": {
                    "drift_type": drift_result.drift_type,
                    "confidence": drift_result.confidence,
                    "details": drift_result.details,
                    "timestamp": drift_result.timestamp.isoformat()
                }
            }

            monitoring_repo.create_alert(alert_data)
            logger.info(f"Created {alert_type} alert for deployment {deployment_id}")

        except Exception as e:
            logger.error(f"Failed to create drift alert: {str(e)}")

    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get drift detection summary."""
        try:
            with self.db_manager.get_session() as session:
                monitoring_repo = MonitoringRepository(session)

                # Get drift-related alerts
                since_date = datetime.utcnow() - timedelta(days=days)
                alerts = monitoring_repo.session.query(monitoring_repo.session.bind.execute(
                    """
                    SELECT alert_type, severity, COUNT(*) as count
                    FROM monitoring_alerts
                    WHERE alert_type IN ('DATA_DRIFT', 'MODEL_DRIFT', 'CONCEPT_DRIFT')
                    AND triggered_at >= %s
                    GROUP BY alert_type, severity
                    """,
                    (since_date,)
                )).fetchall()

                summary = {
                    "time_period_days": days,
                    "total_drift_alerts": sum(alert[2] for alert in alerts),
                    "drift_alerts_by_type": {},
                    "drift_alerts_by_severity": {},
                    "last_updated": datetime.utcnow().isoformat()
                }

                for alert_type, severity, count in alerts:
                    summary["drift_alerts_by_type"][alert_type] = summary["drift_alerts_by_type"].get(alert_type, 0) + count
                    summary["drift_alerts_by_severity"][severity] = summary["drift_alerts_by_severity"].get(severity, 0) + count

                return summary

        except Exception as e:
            logger.error(f"Failed to get drift summary: {str(e)}")
            return {"error": str(e)}