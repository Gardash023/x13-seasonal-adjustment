"""
Logging configuration for X13 Seasonal Adjustment library.

This module provides a centralized logging configuration with different
log levels and formatters for development and production environments.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


class X13Logger:
    """
    Centralized logger for X13 Seasonal Adjustment library.
    
    Provides structured logging with configurable levels and outputs.
    Supports both console and file logging with different formats.
    """
    
    _instance: Optional['X13Logger'] = None
    _configured: bool = False
    
    def __new__(cls) -> 'X13Logger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configured:
            self.configure_logging()
            self._configured = True
    
    @staticmethod
    def configure_logging(
        level: str = "INFO",
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
        format_type: str = "standard"
    ) -> None:
        """
        Configure logging for the X13 library.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file in addition to console
            log_file_path: Path to log file (if None, uses default)
            format_type: Format type ('standard', 'detailed', 'json')
        """
        
        # Define log formats
        formats = {
            "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            "json": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", "line": %(lineno)d, "function": "%(funcName)s", "message": "%(message)s"}'
        }
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(
            formats.get(format_type, formats["standard"]),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_to_file:
            if log_file_path is None:
                log_file_path = Path.cwd() / "x13_seasonal_adjustment.log"
            
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # File gets everything
            file_formatter = logging.Formatter(
                formats["detailed"],
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Suppress warnings from dependencies if not in DEBUG mode
        if level.upper() != "DEBUG":
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
            # Suppress specific library warnings
            logging.getLogger("statsmodels").setLevel(logging.WARNING)
            logging.getLogger("matplotlib").setLevel(logging.WARNING)
            logging.getLogger("sklearn").setLevel(logging.WARNING)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        return logging.getLogger(f"x13_seasonal_adjustment.{name}")
    
    @staticmethod
    def set_level(level: str) -> None:
        """
        Change the logging level at runtime.
        
        Args:
            level: New logging level
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(getattr(logging, level.upper()))
    
    @staticmethod
    def enable_debug() -> None:
        """Enable debug logging."""
        X13Logger.set_level("DEBUG")
        # Show warnings in debug mode
        warnings.filterwarnings("default")
    
    @staticmethod
    def disable_logging() -> None:
        """Disable all logging."""
        logging.getLogger().disabled = True
        warnings.filterwarnings("ignore")
    
    @staticmethod
    def enable_logging() -> None:
        """Re-enable logging after disabling."""
        logging.getLogger().disabled = False


class LoggingContextManager:
    """Context manager for temporary logging level changes."""
    
    def __init__(self, level: str):
        self.level = level.upper()
        self.original_level = None
    
    def __enter__(self):
        self.original_level = logging.getLogger().level
        X13Logger.set_level(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            logging.getLogger().setLevel(self.original_level)


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Decorated function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = X13Logger.get_logger(func.__module__)
        start_time = time.time()
        
        logger.debug(f"Starting execution of {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed successfully in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}")
            raise
    
    return wrapper


def log_method_calls(cls):
    """
    Class decorator to log method calls.
    
    Args:
        cls: Class to be decorated
        
    Returns:
        Decorated class
    """
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_execution_time(attr_value))
    return cls


# Configuration dictionaries for different environments

DEVELOPMENT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "x13_development.log",
            "mode": "w"
        }
    },
    "loggers": {
        "x13_seasonal_adjustment": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

PRODUCTION_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "production": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "production",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "production",
            "filename": "x13_seasonal_adjustment.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "x13_seasonal_adjustment": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False
        }
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"]
    }
}


def setup_logging_from_config(config_dict: Dict[str, Any]) -> None:
    """
    Setup logging from a configuration dictionary.
    
    Args:
        config_dict: Logging configuration dictionary
    """
    logging.config.dictConfig(config_dict)


def setup_development_logging() -> None:
    """Setup logging for development environment."""
    setup_logging_from_config(DEVELOPMENT_CONFIG)


def setup_production_logging() -> None:
    """Setup logging for production environment."""
    setup_logging_from_config(PRODUCTION_CONFIG)


# Initialize default logging
X13Logger()
