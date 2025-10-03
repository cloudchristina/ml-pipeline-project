import os
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
    """
    Handles loading and preprocessing datasets from Hugging Face Hub.

    This class manages the entire dataset lifecycle including:
    - Loading raw datasets from Hugging Face
    - Tokenization for model consumption
    - Train/validation splits
    - Dataset statistics and caching
    """

    def __init__(self, config: Config):
        """
        Initialize the dataset loader.

        Args:
            config: Configuration object containing data directories and settings
        """
        self.config = config
        self.tokenizer = None
        self.raw_dataset = None
        self.processed_dataset = None

    def load_dataset(self, dataset_name: str = "imdb") -> DatasetDict:
        """
        Load dataset from Hugging Face Hub with caching.

        Downloads and caches datasets from Hugging Face for offline use.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub (default: "imdb")

        Returns:
            DatasetDict containing train/test/validation splits

        Raises:
            Exception: If dataset cannot be loaded from Hugging Face
        """
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
        """
        Initialize and configure tokenizer for the specified model.

        Loads the tokenizer corresponding to the model architecture and sets up
        padding tokens if needed.

        Args:
            model_name: Hugging Face model name (default: "distilbert-base-uncased")

        Returns:
            Configured AutoTokenizer instance

        Raises:
            Exception: If tokenizer cannot be loaded
        """
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
        """
        Tokenize text data for model training.

        Converts raw text into token IDs with attention masks, ready for model input.
        Applies truncation and padding to ensure uniform sequence length.

        Args:
            dataset: DatasetDict containing raw text in "text" field
            max_length: Maximum sequence length for tokenization (default: 512)

        Returns:
            Tokenized DatasetDict with input_ids, attention_mask, and label fields

        Raises:
            ValueError: If tokenizer is not initialized
            Exception: If tokenization fails
        """
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

            # Tokenize all splits - only remove 'text' column, keep 'label'
            columns_to_remove = [col for col in dataset["train"].column_names if col != "label"]
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=columns_to_remove,
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
        """
        Create train/validation split if not already present.

        Splits training data into separate train and validation sets using
        stratified sampling to maintain label distribution.

        Args:
            dataset: Dataset to split (typically the training set)
            test_size: Proportion of data for validation (default: 0.2 = 20%)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Dictionary with "train" and "validation" Dataset objects

        Raises:
            Exception: If split creation fails
        """
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

            # Convert back to Dataset (reset index to avoid __index_level_0__ column)
            train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
            val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

            logger.info(f"Split created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

            return {
                "train": train_dataset,
                "validation": val_dataset
            }

        except Exception as e:
            logger.error(f"Error creating validation split: {str(e)}")
            raise

    def get_dataset_statistics(self, dataset: DatasetDict) -> Dict[str, Any]:
        """
        Generate dataset statistics and metadata.

        Computes comprehensive statistics including:
        - Number of examples per split
        - Text length statistics (avg, min, max)
        - Label distribution

        Args:
            dataset: DatasetDict to analyze

        Returns:
            Dictionary containing statistics for each split

        Raises:
            Exception: If statistics generation fails
        """
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
                    # Convert to list to handle both regular lists and torch tensors
                    labels = split_dataset["label"]
                    if hasattr(labels[0], 'item'):  # It's a torch tensor
                        labels = [label.item() for label in labels]
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