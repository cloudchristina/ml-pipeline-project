#!/usr/bin/env python3
"""
Comprehensive test script for the ML pipeline project.
Tests all components and populates database with sample data.
"""

import sys
import requests
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.database import get_db_manager
from src.database.models import PredictionLog, ExperimentRun, DatasetMetrics
from src.utils.config import config
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def test_api_health():
    """Test API health endpoint"""
    print_header("TEST 1: API Health Check")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_predictions():
    """Test prediction endpoints and populate prediction_logs table"""
    print_header("TEST 2: Making Predictions (Populating prediction_logs)")

    test_texts = [
        "This movie is absolutely amazing! Best film I've ever seen!",
        "Terrible waste of time. I want my money back.",
        "Pretty good, I enjoyed it overall.",
        "Worst movie ever made. Absolutely horrible.",
        "Fantastic performance by the actors!",
        "Boring and predictable. Not recommended.",
        "A masterpiece! Incredible cinematography!",
        "Mediocre at best, nothing special.",
        "I loved every minute of it!",
        "Disappointing and forgettable."
    ]

    predictions = []
    for i, text in enumerate(test_texts, 1):
        try:
            payload = {
                "text": text,
                "return_probabilities": True
            }
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                predictions.append(result)
                sentiment = result.get('sentiment', 'N/A')
                confidence = result.get('confidence', 0)
                print(f"{i}. [{sentiment}] ({confidence:.2%}) - {text[:50]}...")
            else:
                print(f"{i}. ❌ Failed: {response.status_code}")

        except Exception as e:
            print(f"{i}. ❌ Error: {e}")

    print(f"\n✅ Made {len(predictions)} predictions")
    return len(predictions) > 0

def test_batch_predictions():
    """Test batch prediction endpoint"""
    print_header("TEST 3: Batch Predictions")

    batch_texts = [
        "Great movie, highly recommend!",
        "Not worth watching.",
        "Amazing storyline and acting!"
    ]

    try:
        payload = {
            "texts": batch_texts,
            "return_probabilities": True
        }
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=payload,
            timeout=15
        )

        if response.status_code == 200:
            results = response.json()
            print(f"Batch size: {len(results.get('predictions', []))}")
            for i, pred in enumerate(results.get('predictions', []), 1):
                sentiment = pred.get('sentiment', 'N/A')
                confidence = pred.get('confidence', 0)
                print(f"{i}. [{sentiment}] ({confidence:.2%})")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print_header("TEST 4: Model Information")

    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"Model Name: {info.get('model_name', 'N/A')}")
            print(f"Model Path: {info.get('model_path', 'N/A')}")
            print(f"Labels: {info.get('labels', 'N/A')}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def populate_sample_data():
    """Populate database with additional sample data"""
    print_header("TEST 5: Populating Database with Sample Data")

    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Add experiment runs
            print("Adding experiment runs...")
            experiment1 = ExperimentRun(
                mlflow_run_id="exp_001_initial_training",
                experiment_name="sentiment_analysis",
                run_name="initial_training_run",
                model_name="distilbert-base-uncased",
                dataset_name="imdb",
                hyperparameters={
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "num_epochs": 3
                },
                accuracy=0.856,
                f1_score=0.851,
                precision=0.849,
                recall=0.853,
                loss=0.324,
                training_duration=3600.5,
                artifacts_path="models/final_model/checkpoint-313",
                status="COMPLETED"
            )

            experiment2 = ExperimentRun(
                mlflow_run_id="exp_002_hyperparameter_tuning",
                experiment_name="sentiment_analysis",
                run_name="hyperparameter_tuning_run",
                model_name="distilbert-base-uncased",
                dataset_name="imdb",
                hyperparameters={
                    "learning_rate": 3e-5,
                    "batch_size": 32,
                    "num_epochs": 5
                },
                accuracy=0.872,
                f1_score=0.869,
                precision=0.865,
                recall=0.873,
                loss=0.298,
                training_duration=7200.3,
                artifacts_path="models/improved_model",
                status="COMPLETED"
            )

            session.add(experiment1)
            session.add(experiment2)
            print("✅ Added 2 experiment runs")

            # Add dataset metrics
            print("Adding dataset metrics...")
            dataset1 = DatasetMetrics(
                dataset_name="imdb",
                version="v1.0",
                split_name="train",
                num_examples=5000,
                avg_text_length=234.5,
                min_text_length=10,
                max_text_length=512,
                label_distribution={"positive": 2500, "negative": 2500},
                quality_score=0.95,
                completeness_score=1.0,
                consistency_score=0.92,
                validity_score=0.98,
                drift_score=0.05,
                is_drift_detected=False,
                statistics={
                    "avg_text_length": 234.5,
                    "max_text_length": 512,
                    "min_text_length": 10
                }
            )

            dataset2 = DatasetMetrics(
                dataset_name="imdb",
                version="v1.0",
                split_name="test",
                num_examples=2000,
                avg_text_length=228.3,
                min_text_length=15,
                max_text_length=512,
                label_distribution={"positive": 1000, "negative": 1000},
                quality_score=0.94,
                completeness_score=1.0,
                consistency_score=0.91,
                validity_score=0.97,
                drift_score=0.06,
                is_drift_detected=False,
                statistics={
                    "avg_text_length": 228.3,
                    "max_text_length": 512,
                    "min_text_length": 15
                }
            )

            session.add(dataset1)
            session.add(dataset2)
            print("✅ Added 2 dataset metrics")

            session.commit()

        print("\n✅ Database populated successfully!")
        return True

    except Exception as e:
        print(f"❌ Error populating database: {e}")
        return False

def check_database_contents():
    """Check and display database contents"""
    print_header("TEST 6: Checking Database Contents")

    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Count records in each table
            from sqlalchemy import func

            prediction_count = session.query(func.count(PredictionLog.id)).scalar()
            experiment_count = session.query(func.count(ExperimentRun.id)).scalar()
            dataset_count = session.query(func.count(DatasetMetrics.id)).scalar()

            print(f"📊 Database Statistics:")
            print(f"  - Prediction Logs: {prediction_count} records")
            print(f"  - Experiment Runs: {experiment_count} records")
            print(f"  - Dataset Metrics: {dataset_count} records")

            # Show sample predictions
            if prediction_count > 0:
                print(f"\n📝 Recent Predictions (last 5):")
                predictions = session.query(PredictionLog).order_by(
                    PredictionLog.predicted_at.desc()
                ).limit(5).all()

                for i, pred in enumerate(predictions, 1):
                    text_preview = pred.input_text[:50] + "..." if len(pred.input_text) > 50 else pred.input_text
                    print(f"  {i}. [{pred.predicted_sentiment}] {text_preview}")

            return True

    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ML Pipeline Full System Test" + " "*30 + "║")
    print("╚" + "="*78 + "╝")

    results = []

    # Run all tests
    results.append(("API Health Check", test_api_health()))
    results.append(("Single Predictions", test_predictions()))
    results.append(("Batch Predictions", test_batch_predictions()))
    results.append(("Model Info", test_model_info()))
    results.append(("Populate Sample Data", populate_sample_data()))
    results.append(("Database Contents", check_database_contents()))

    # Print summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*80}\n")

    if passed == total:
        print("🎉 All tests passed! Your ML pipeline is working correctly!")
        print("\n📋 Next Steps:")
        print("  1. Open TablePlus and refresh to see the populated data")
        print("  2. Visit MLflow UI: http://localhost:5001")
        print("  3. Visit API docs: http://localhost:8000/docs")
        print("  4. Visit Frontend: http://localhost:3000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
