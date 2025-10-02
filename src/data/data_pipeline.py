import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import pandas as pd
from datasets import DatasetDict

from .dataset_loader import HuggingFaceDatasetLoader
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    def __init__(self, config: Config):
        self.config = config
        # Ensure data directories exist for pipeline operations
        self.config.setup_directories(data_pipeline=True)
        self.loader = HuggingFaceDatasetLoader(config)
        self.run_metadata = {}

    def execute_pipeline(
        self,
        dataset_name: str = "imdb",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        validation_split: float = 0.2,
        force_reload: bool = False
    ) -> DatasetDict:
        """Execute the complete data pipeline."""
        pipeline_start = datetime.now()

        try:
            logger.info("Starting data pipeline execution")

            # Start MLflow run for data pipeline tracking
            with mlflow.start_run(run_name=f"data_pipeline_{dataset_name}") as run:
                # Log parameters
                mlflow.log_params({
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "max_length": max_length,
                    "validation_split": validation_split,
                    "force_reload": force_reload
                })

                # Check if processed dataset exists and force_reload is False
                processed_path = Path(self.config.data_dir) / "processed" / dataset_name

                if processed_path.exists() and not force_reload:
                    logger.info("Loading existing processed dataset")
                    dataset = self.loader.load_processed_dataset(str(processed_path))
                else:
                    logger.info("Processing dataset from scratch")
                    dataset = self._process_fresh_dataset(
                        dataset_name, model_name, max_length, validation_split
                    )

                # Generate and log statistics
                stats = self.loader.get_dataset_statistics(dataset)
                self._log_dataset_metrics(stats)

                # Save metadata
                self.run_metadata = {
                    "pipeline_run_id": run.info.run_id,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "execution_time": (datetime.now() - pipeline_start).total_seconds(),
                    "timestamp": datetime.now().isoformat(),
                    "statistics": stats
                }

                self._save_metadata()

                logger.info(f"Data pipeline completed in {self.run_metadata['execution_time']:.2f}s")
                mlflow.log_metric("pipeline_duration_seconds", self.run_metadata['execution_time'])

                return dataset

        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            mlflow.log_metric("pipeline_success", 0)
            raise

    def _process_fresh_dataset(
        self,
        dataset_name: str,
        model_name: str,
        max_length: int,
        validation_split: float
    ) -> DatasetDict:
        """Process dataset from scratch."""

        # Step 1: Load raw dataset
        logger.info("Step 1: Loading raw dataset")
        raw_dataset = self.loader.load_dataset(dataset_name)
        mlflow.log_metric("raw_dataset_size", len(raw_dataset))

        # Step 2: Create validation split if needed
        if "validation" not in raw_dataset and validation_split > 0:
            logger.info("Step 2: Creating validation split")
            splits = self.loader.create_validation_split(
                raw_dataset["train"],
                test_size=validation_split
            )
            # Reconstruct dataset with validation split
            processed_splits = {
                "train": splits["train"],
                "validation": splits["validation"]
            }
            if "test" in raw_dataset:
                processed_splits["test"] = raw_dataset["test"]

            raw_dataset = DatasetDict(processed_splits)

        # Step 3: Initialize tokenizer
        logger.info("Step 3: Initializing tokenizer")
        self.loader.prepare_tokenizer(model_name)

        # Step 4: Tokenize dataset
        logger.info("Step 4: Tokenizing dataset")
        tokenized_dataset = self.loader.tokenize_dataset(raw_dataset, max_length)

        # Step 5: Save processed dataset
        processed_path = Path(self.config.data_dir) / "processed" / dataset_name
        logger.info("Step 5: Saving processed dataset")
        self.loader.save_processed_dataset(tokenized_dataset, str(processed_path))

        return tokenized_dataset

    def _log_dataset_metrics(self, stats: Dict[str, Any]):
        """Log dataset statistics to MLflow."""
        for split_name, split_stats in stats.items():
            # Log basic metrics
            mlflow.log_metric(f"{split_name}_num_examples", split_stats["num_examples"])

            # Log text statistics if available
            if "text_stats" in split_stats:
                text_stats = split_stats["text_stats"]
                mlflow.log_metric(f"{split_name}_avg_text_length", text_stats["avg_length"])
                mlflow.log_metric(f"{split_name}_min_text_length", text_stats["min_length"])
                mlflow.log_metric(f"{split_name}_max_text_length", text_stats["max_length"])

            # Log label distribution if available
            if "label_distribution" in split_stats:
                label_dist = split_stats["label_distribution"]
                for label, count in label_dist.items():
                    mlflow.log_metric(f"{split_name}_label_{label}_count", count)

        # Log overall metrics
        total_examples = sum(stats[split]["num_examples"] for split in stats)
        mlflow.log_metric("total_examples", total_examples)

    def _save_metadata(self):
        """Save pipeline metadata to file."""
        metadata_path = Path(self.config.data_dir) / "metadata" / "pipeline_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(self.run_metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved to {metadata_path}")

    def get_data_quality_report(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_quality_score": 0.0,
            "checks": {}
        }

        total_checks = 0
        passed_checks = 0

        for split_name, split_dataset in dataset.items():
            split_checks = {
                "completeness": self._check_completeness(split_dataset),
                "consistency": self._check_consistency(split_dataset),
                "validity": self._check_validity(split_dataset)
            }

            # Calculate split quality score
            split_passed = sum(1 for check in split_checks.values() if check["passed"])
            split_total = len(split_checks)
            split_score = split_passed / split_total if split_total > 0 else 0

            quality_report["checks"][split_name] = {
                **split_checks,
                "quality_score": split_score
            }

            total_checks += split_total
            passed_checks += split_passed

        # Calculate overall quality score
        quality_report["overall_quality_score"] = passed_checks / total_checks if total_checks > 0 else 0

        logger.info(f"Data quality score: {quality_report['overall_quality_score']:.2f}")
        return quality_report

    def _check_completeness(self, dataset) -> Dict[str, Any]:
        """Check data completeness."""
        total_examples = len(dataset)

        # Check for missing values in key columns
        missing_counts = {}
        for column in dataset.column_names:
            if column in ["input_ids", "attention_mask", "label"]:
                # For tokenized data, check for None values
                column_data = dataset[column]
                missing_counts[column] = sum(1 for item in column_data if item is None)

        total_missing = sum(missing_counts.values())
        completeness_score = 1.0 - (total_missing / (total_examples * len(missing_counts))) if total_examples > 0 else 0

        return {
            "passed": completeness_score > 0.95,
            "score": completeness_score,
            "missing_counts": missing_counts,
            "total_missing": total_missing
        }

    def _check_consistency(self, dataset) -> Dict[str, Any]:
        """Check data consistency."""
        consistency_issues = []

        # Check input_ids and attention_mask length consistency
        if "input_ids" in dataset.column_names and "attention_mask" in dataset.column_names:
            input_ids = dataset["input_ids"]
            attention_mask = dataset["attention_mask"]

            inconsistent_lengths = 0
            for i, (input_id, attention) in enumerate(zip(input_ids, attention_mask)):
                if len(input_id) != len(attention):
                    inconsistent_lengths += 1

            if inconsistent_lengths > 0:
                consistency_issues.append(f"{inconsistent_lengths} examples with mismatched input_ids/attention_mask lengths")

        consistency_score = 1.0 if len(consistency_issues) == 0 else 0.8

        return {
            "passed": len(consistency_issues) == 0,
            "score": consistency_score,
            "issues": consistency_issues
        }

    def _check_validity(self, dataset) -> Dict[str, Any]:
        """Check data validity."""
        validity_issues = []

        # Check label validity
        if "label" in dataset.column_names:
            labels = dataset["label"]
            unique_labels = set(labels)

            # For binary classification (like IMDB), expect labels 0 and 1
            expected_labels = {0, 1}
            if not unique_labels.issubset(expected_labels):
                validity_issues.append(f"Unexpected labels found: {unique_labels - expected_labels}")

        validity_score = 1.0 if len(validity_issues) == 0 else 0.7

        return {
            "passed": len(validity_issues) == 0,
            "score": validity_score,
            "issues": validity_issues
        }