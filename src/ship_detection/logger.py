"""
File: logger.py
Author: Adam Muhtar <adam.muhtar23@imperial.ac.uk>
Description: Structured logging utilities.
"""

# Standard library imports
import os

# Third party imports
import structlog


def structlog_logger(enable_json: bool = None) -> structlog.BoundLogger:
    """Return a configured structlog logger.
    
    Args:
        enable_json (`bool`): Enable JSON logging if True.
    
    Returns:
        `structlog.BoundLogger`: Structured logger instance.
    """
    if enable_json is None:
        enable_json = os.getenv("JSON_LOGS", "false").lower() == "true"
    
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer()
    ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True
    )
    
    return structlog.get_logger()