import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for the ML pipeline."""

    # Paths
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "models"))
    logs_dir: str = field(default_factory=lambda: os.getenv("LOGS_DIR", "logs"))

    # MLflow Configuration
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow_experiment_name: str = field(default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment_analysis"))

    # Database Configuration
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "ml_pipeline"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    # Model Configuration
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "distilbert-base-uncased"))
    max_length: int = field(default_factory=lambda: int(os.getenv("MAX_LENGTH", "512")))
    num_labels: int = field(default_factory=lambda: int(os.getenv("NUM_LABELS", "2")))

    # Training Configuration
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "16")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "2e-5")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "3")))
    warmup_steps: int = field(default_factory=lambda: int(os.getenv("WARMUP_STEPS", "500")))
    weight_decay: float = field(default_factory=lambda: float(os.getenv("WEIGHT_DECAY", "0.01")))

    # API Configuration
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))

    # Monitoring Configuration
    monitoring_enabled: bool = field(default_factory=lambda: os.getenv("MONITORING_ENABLED", "true").lower() == "true")
    drift_detection_threshold: float = field(default_factory=lambda: float(os.getenv("DRIFT_DETECTION_THRESHOLD", "0.1")))
    alert_email: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL", ""))

    # AWS Configuration
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-west-2"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "ml-pipeline-artifacts"))
    ecr_repository: str = field(default_factory=lambda: os.getenv("ECR_REPOSITORY", "ml-pipeline"))

    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def __post_init__(self):
        """Create necessary directories after initialization."""
        directories = [
            self.data_dir,
            self.model_dir,
            self.logs_dir,
            f"{self.data_dir}/raw",
            f"{self.data_dir}/processed",
            f"{self.data_dir}/cache",
            f"{self.data_dir}/metadata"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def database_url(self) -> str:
        """Generate database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def model_serving_url(self) -> str:
        """Generate model serving URL."""
        return f"http://{self.api_host}:{self.api_port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from file."""
        import yaml

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = cls()
        config.update_from_dict(config_dict)
        return config

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        import yaml

        with open(config_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def validate(self) -> None:
        """Validate configuration parameters."""
        from .exceptions import ConfigurationError

        # Validate positive numeric values
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")

        if self.num_epochs <= 0:
            raise ConfigurationError("num_epochs must be positive")

        if self.max_length <= 0:
            raise ConfigurationError("max_length must be positive")

        if self.num_labels <= 0:
            raise ConfigurationError("num_labels must be positive")

        # Check required secrets in production
        if os.getenv("ENVIRONMENT", "development").lower() == "production":
            if not self.db_password:
                raise ConfigurationError("DB_PASSWORD is required in production environment")

            if not self.alert_email:
                raise ConfigurationError("ALERT_EMAIL is required when monitoring is enabled in production")

        # Validate directories
        if not Path(self.data_dir).exists():
            raise ConfigurationError(f"data_dir does not exist: {self.data_dir}")

        # Warnings for non-critical issues
        if self.monitoring_enabled and not self.alert_email:
            print("Warning: Monitoring enabled but no alert email configured")


# Global configuration instance
config = Config()