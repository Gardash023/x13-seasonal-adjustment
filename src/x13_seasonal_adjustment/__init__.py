"""
X13 Seasonal Adjustment Library

Professional implementation of the X13-ARIMA-SEATS seasonal adjustment algorithm
for detecting and removing seasonal effects from time series data.

This library provides a comprehensive, production-ready Python implementation 
following the methodology of the US Census Bureau's X13-ARIMA-SEATS program,
with robust error handling, professional logging, and extensive quality diagnostics.

Key Features:
- Automatic seasonality detection with multiple statistical tests
- Full X13-ARIMA-SEATS algorithm implementation
- Robust error handling and comprehensive logging
- Professional quality diagnostics and reporting
- High-performance optimized computations
- Flexible API for simple and advanced use cases

Basic Usage:
    >>> import pandas as pd
    >>> from x13_seasonal_adjustment import X13SeasonalAdjustment
    >>> 
    >>> # Load your time series data
    >>> data = pd.Series([100, 110, 95, 105, 120, 108, 90, 98, 125, 115, 88, 102])
    >>> 
    >>> # Apply seasonal adjustment with default settings
    >>> x13 = X13SeasonalAdjustment()
    >>> result = x13.fit_transform(data)
    >>> 
    >>> # Access results
    >>> print(f"Seasonally adjusted series: {result.seasonally_adjusted}")
    >>> print(f"Seasonality strength: {result.seasonality_strength:.3f}")
    >>> 
    >>> # Visualize results
    >>> result.plot()

Advanced Usage:
    >>> # Customize the seasonal adjustment process
    >>> x13 = X13SeasonalAdjustment(
    ...     freq='M',
    ...     transform='auto',
    ...     outlier_detection=True,
    ...     enable_logging=True,
    ...     log_level='DEBUG'
    ... )
    >>> result = x13.fit_transform(monthly_data)
    >>> 
    >>> # Access comprehensive diagnostics
    >>> print(f"Quality measures: {result.quality_measures}")
    >>> print(f"ARIMA model: {result.arima_model_info}")
"""

# Core functionality
from .core.x13 import X13SeasonalAdjustment
from .core.result import SeasonalAdjustmentResult

# Specialized components
from .tests.seasonality_tests import SeasonalityTests
from .arima.auto_arima import AutoARIMA
from .diagnostics.quality import QualityDiagnostics

# Exception classes for error handling
from .exceptions import (
    X13SeasonalAdjustmentError,
    DataValidationError,
    TimeSeriesError,
    FrequencyError,
    SeasonalityError,
    ARIMAModelError,
    DecompositionError,
    TransformationError,
    OutlierDetectionError,
    ForecastError,
    QualityAssessmentError,
    ConfigurationError,
    InsufficientDataError,
    ModelNotFittedError,
    ConvergenceError
)

# Logging configuration
from .logging_config import X13Logger, LoggingContextManager

# Version and metadata
__version__ = "0.1.3"
__author__ = "Gardash Abbasov"
__email__ = "gardash.abbasov@gmail.com"
__license__ = "MIT"
__description__ = "Professional X13-ARIMA-SEATS seasonal adjustment for Python"
__url__ = "https://github.com/Gardash023/x13-seasonal-adjustment"

# Public API
__all__ = [
    # Core classes
    "X13SeasonalAdjustment",
    "SeasonalAdjustmentResult",
    
    # Specialized components
    "SeasonalityTests",
    "AutoARIMA", 
    "QualityDiagnostics",
    
    # Exception classes
    "X13SeasonalAdjustmentError",
    "DataValidationError",
    "TimeSeriesError",
    "FrequencyError",
    "SeasonalityError",
    "ARIMAModelError",
    "DecompositionError",
    "TransformationError",
    "OutlierDetectionError",
    "ForecastError",
    "QualityAssessmentError",
    "ConfigurationError",
    "InsufficientDataError", 
    "ModelNotFittedError",
    "ConvergenceError",
    
    # Logging
    "X13Logger",
    "LoggingContextManager",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]

# Configure default logging
try:
    X13Logger()
except Exception:
    # Silently continue if logging setup fails
    pass
