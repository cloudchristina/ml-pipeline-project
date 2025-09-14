import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Setup logging configuration using loguru."""

    # Remove default logger
    loguru_logger.remove()

    # Default format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    loguru_logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        loguru_logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )


def get_logger(name: str) -> loguru_logger.__class__:
    """Get a logger instance."""
    return loguru_logger.bind(name=name)


class InterceptHandler(logging.Handler):
    """Intercept standard logging records and redirect to loguru."""

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logging call
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_intercept_handler():
    """Setup handler to intercept standard logging and redirect to loguru."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Silence some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


# Initialize logging on import
setup_logging()
setup_intercept_handler()