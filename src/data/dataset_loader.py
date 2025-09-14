import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceDatasetLoader:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.raw_dataset = None
        self.processed_dataset = None

    def load_dataset(self, dataset_name: str = "imdb") -> DatasetDict:
        """Load dataset from Hugging Face Hub with caching."""
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            cache_dir = Path(self.config.data_dir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Load dataset with caching
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )

            self.raw_dataset = dataset
            logger.info(f"Dataset loaded successfully: {len(dataset)} splits")
            return dataset

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise

    def prepare_tokenizer(self, model_name: str = "distilbert-base-uncased") -> AutoTokenizer:
        """Initialize and configure tokenizer."""
        try:
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=Path(self.config.data_dir) / "cache" / "tokenizers"
            )

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Tokenizer loaded successfully")
            return self.tokenizer

        except Exception as e:
            logger.error(f"Error loading tokenizer {model_name}: {str(e)}")
            raise

    def tokenize_dataset(self, dataset: DatasetDict, max_length: int = 512) -> DatasetDict:
        """Tokenize text data for model training."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_tokenizer() first.")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

        try:
            logger.info("Starting tokenization process")

            # Tokenize all splits
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
                desc="Tokenizing"
            )

            # Set format for PyTorch
            tokenized_dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"]
            )

            self.processed_dataset = tokenized_dataset
            logger.info("Tokenization completed successfully")
            return tokenized_dataset

        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise

    def create_validation_split(
        self,
        dataset: Dataset,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dataset]:
        """Create train/validation split if not already present."""
        try:
            logger.info(f"Creating validation split with test_size={test_size}")

            # Convert to pandas for sklearn split
            df = dataset.to_pandas()

            train_df, val_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df["label"] if "label" in df.columns else None
            )

            # Convert back to Dataset
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)

            logger.info(f"Split created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

            return {
                "train": train_dataset,
                "validation": val_dataset
            }

        except Exception as e:
            logger.error(f"Error creating validation split: {str(e)}")
            raise

    def get_dataset_statistics(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Generate dataset statistics and metadata."""
        try:
            stats = {}

            for split_name, split_dataset in dataset.items():
                split_stats = {
                    "num_examples": len(split_dataset),
                    "columns": split_dataset.column_names,
                    "features": str(split_dataset.features)
                }

                # Text length statistics if text column exists
                if "text" in split_dataset.column_names:
                    texts = split_dataset["text"]
                    text_lengths = [len(text.split()) for text in texts]
                    split_stats["text_stats"] = {
                        "avg_length": sum(text_lengths) / len(text_lengths),
                        "min_length": min(text_lengths),
                        "max_length": max(text_lengths)
                    }

                # Label distribution if label column exists
                if "label" in split_dataset.column_names:
                    labels = split_dataset["label"]
                    label_counts = pd.Series(labels).value_counts().to_dict()
                    split_stats["label_distribution"] = label_counts

                stats[split_name] = split_stats

            logger.info("Dataset statistics generated")
            return stats

        except Exception as e:
            logger.error(f"Error generating statistics: {str(e)}")
            raise

    def save_processed_dataset(self, dataset: DatasetDict, output_path: str):
        """Save processed dataset to disk."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving processed dataset to {output_path}")
            dataset.save_to_disk(str(output_path))
            logger.info("Dataset saved successfully")

        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise

    def load_processed_dataset(self, dataset_path: str) -> DatasetDict:
        """Load previously processed dataset from disk."""
        try:
            logger.info(f"Loading processed dataset from {dataset_path}")
            dataset = DatasetDict.load_from_disk(dataset_path)
            self.processed_dataset = dataset
            logger.info("Processed dataset loaded successfully")
            return dataset

        except Exception as e:
            logger.error(f"Error loading processed dataset: {str(e)}")
            raise