#!/usr/bin/env python3
"""
Training Pipeline Script

This script demonstrates how to run the complete ML training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.training_pipeline import TrainingPipeline
from src.utils.config import config
from src.utils.logger import get_logger, setup_logging

# Setup directories for training pipeline (includes data directories)
config.setup_directories(data_pipeline=True)

# Setup logging
setup_logging(config.log_level, f"{config.logs_dir}/training.log")
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ML training pipeline")

    # Dataset options
    parser.add_argument("--dataset", default="imdb", help="Dataset name to use")
    parser.add_argument("--model", default=None, help="Model name to use")

    # Training options
    parser.add_argument("--experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name")

    # Hyperparameter optimization
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials")

    # Quick training for testing
    parser.add_argument("--quick", action="store_true", help="Quick training run with limited data")

    args = parser.parse_args()

    try:
        logger.info("Starting ML training pipeline")
        logger.info(f"Arguments: {vars(args)}")

        # Initialize training pipeline
        pipeline = TrainingPipeline(config)

        if args.quick:
            logger.info("Running quick training pipeline")
            results = pipeline.quick_training_run(
                dataset_name=args.dataset,
                model_name=args.model
            )
        else:
            logger.info("Running full training pipeline")
            results = pipeline.execute_full_pipeline(
                dataset_name=args.dataset,
                model_name=args.model,
                optimize_hyperparameters=args.optimize,
                n_trials=args.n_trials,
                experiment_name=args.experiment,
                run_name=args.run_name
            )

        # Print results summary
        print("\n" + "="*50)
        print("TRAINING PIPELINE RESULTS")
        print("="*50)
        print(f"Pipeline Duration: {results['pipeline_duration']:.2f} seconds")
        print(f"Data Quality Score: {results.get('data_quality_score', 'N/A')}")
        print(f"Final F1 Score: {results['training_results']['eval_metrics']['eval_f1']:.4f}")
        print(f"Final Accuracy: {results['training_results']['eval_metrics']['eval_accuracy']:.4f}")
        print(f"MLflow Run ID: {results['pipeline_run_id']}")

        if results['hyperparameter_optimization']['enabled']:
            print(f"Best Hyperparameters: {results['hyperparameter_optimization']['best_params']}")

        print("="*50)

        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        print(f"ERROR: Training pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()