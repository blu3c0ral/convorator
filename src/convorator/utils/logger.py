import logging
import sys
import os
from typing import Any, Callable, Optional
import time
from functools import wraps

# For more advanced structured logging
import structlog

# For performance metrics
import prometheus_client


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    console_output: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger instance with appropriate handlers.

    Args:
        name: Name of the logger
        log_file: Optional specific log file path
        log_dir: Directory for log files if log_file not specified
        console_output: Whether to output logs to console

    Returns:
        Configured logger
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Create log directory if needed
    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates and ensure files are closed
    if logger.handlers:
        for handler in logger.handlers:
            handler.close()  # Properly close file handlers
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def cleanup_logger(name: str):
    """
    Properly close all handlers for a given logger.

    Args:
        name: Name of the logger to clean up
    """
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def get_all_loggers() -> list[str]:
    """
    Get all logger instances that have been created.

    Returns:
        list: Names of all loggers with handlers
    """
    loggers: list[str] = []

    # Get the logger manager dictionary that contains all loggers
    logger_dict = logging.Logger.manager.loggerDict

    # Filter for loggers that have handlers (these are the ones that need cleanup)
    for logger_name, logger_instance in logger_dict.items():
        # Skip PlaceHolder objects which aren't actual loggers
        if isinstance(logger_instance, logging.Logger):
            if logger_instance.handlers:
                loggers.append(logger_name)

    # Also check the root logger
    root_logger = logging.getLogger()
    if root_logger.handlers:
        loggers.append("root")

    return loggers


def cleanup_all_loggers():
    """
    Close handlers for all loggers that have been created.
    This is useful for cleaning up resources in test tearDown methods.
    """
    # Get all logger names
    logger_names = get_all_loggers()

    # Clean up each logger
    for name in logger_names:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing handler for logger '{name}': {e}")
        logger.handlers.clear()

    return logger_names


# Configure structlog for structured logging
def configure_structured_logging():
    """Configure structlog for the application."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Simple performance timing decorator
def time_function(
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to time function execution.

    Args:
        logger: Optional logger to use (uses print if None)

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            if logger:
                logger.info(f"Function {func.__name__} executed in {elapsed:.4f}s")
            else:
                print(f"Function {func.__name__} executed in {elapsed:.4f}s")

            return result

        return wrapper

    return decorator


# Initialize metrics for prometheus (if needed)
def setup_prometheus_metrics(
    export_port: int = 8000,
) -> dict[str, prometheus_client.Summary | prometheus_client.Counter]:
    """
    Set up Prometheus metrics exporter.

    Args:
        export_port: Port to expose metrics

    Returns:
        Dict of metric collectors
    """
    # Start metrics server
    prometheus_client.start_http_server(export_port)

    # Create some basic metrics
    metrics: dict[str, prometheus_client.Summary | prometheus_client.Counter] = {
        "function_duration": prometheus_client.Summary(
            "function_duration_seconds", "Time spent in functions", ["function"]
        ),
        "errors_total": prometheus_client.Counter(
            "errors_total", "Total number of errors", ["type"]
        ),
        "requests_total": prometheus_client.Counter(
            "requests_total", "Total number of requests", ["endpoint"]
        ),
    }

    return metrics


# Configure root logger
def configure_root_logger(level: int = logging.INFO):
    """Configure the root logger with sensible defaults."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
