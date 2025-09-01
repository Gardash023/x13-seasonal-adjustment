"""
Custom exceptions for X13 Seasonal Adjustment library.

This module defines custom exception classes for better error handling
and debugging throughout the X13 seasonal adjustment process.
"""

from typing import Optional, Any


class X13SeasonalAdjustmentError(Exception):
    """Base exception class for X13 Seasonal Adjustment library."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, **kwargs):
        self.message = message
        self.error_code = error_code
        self.additional_info = kwargs
        super().__init__(self.message)
    
    def __str__(self) -> str:
        error_str = f"X13 Error: {self.message}"
        if self.error_code:
            error_str = f"X13 Error [{self.error_code}]: {self.message}"
        return error_str


class DataValidationError(X13SeasonalAdjustmentError):
    """Raised when input data validation fails."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs):
        self.data_type = data_type
        super().__init__(message, error_code="DATA_VALIDATION", data_type=data_type, **kwargs)


class TimeSeriesError(X13SeasonalAdjustmentError):
    """Raised for time series specific errors."""
    
    def __init__(self, message: str, series_info: Optional[dict] = None, **kwargs):
        self.series_info = series_info or {}
        super().__init__(message, error_code="TIME_SERIES", series_info=series_info, **kwargs)


class FrequencyError(TimeSeriesError):
    """Raised when frequency detection or validation fails."""
    
    def __init__(self, message: str, detected_freq: Optional[str] = None, 
                 expected_freq: Optional[str] = None, **kwargs):
        self.detected_freq = detected_freq
        self.expected_freq = expected_freq
        super().__init__(
            message, 
            series_info={
                "detected_frequency": detected_freq,
                "expected_frequency": expected_freq
            },
            **kwargs
        )


class SeasonalityError(X13SeasonalAdjustmentError):
    """Raised when seasonality detection or processing fails."""
    
    def __init__(self, message: str, seasonality_tests: Optional[dict] = None, **kwargs):
        self.seasonality_tests = seasonality_tests or {}
        super().__init__(message, error_code="SEASONALITY", tests=seasonality_tests, **kwargs)


class ARIMAModelError(X13SeasonalAdjustmentError):
    """Raised when ARIMA model fitting or prediction fails."""
    
    def __init__(self, message: str, model_order: Optional[tuple] = None, 
                 seasonal_order: Optional[tuple] = None, **kwargs):
        self.model_order = model_order
        self.seasonal_order = seasonal_order
        super().__init__(
            message, 
            error_code="ARIMA_MODEL", 
            model_order=model_order,
            seasonal_order=seasonal_order,
            **kwargs
        )


class DecompositionError(X13SeasonalAdjustmentError):
    """Raised when seasonal decomposition fails."""
    
    def __init__(self, message: str, decomposition_mode: Optional[str] = None, 
                 component: Optional[str] = None, **kwargs):
        self.decomposition_mode = decomposition_mode
        self.component = component
        super().__init__(
            message, 
            error_code="DECOMPOSITION", 
            mode=decomposition_mode,
            component=component,
            **kwargs
        )


class OutlierDetectionError(X13SeasonalAdjustmentError):
    """Raised when outlier detection fails."""
    
    def __init__(self, message: str, outlier_types: Optional[list] = None, **kwargs):
        self.outlier_types = outlier_types or []
        super().__init__(message, error_code="OUTLIER_DETECTION", types=outlier_types, **kwargs)


class TransformationError(X13SeasonalAdjustmentError):
    """Raised when data transformation fails."""
    
    def __init__(self, message: str, transform_type: Optional[str] = None, **kwargs):
        self.transform_type = transform_type
        super().__init__(message, error_code="TRANSFORMATION", transform_type=transform_type, **kwargs)


class ForecastError(X13SeasonalAdjustmentError):
    """Raised when forecasting fails."""
    
    def __init__(self, message: str, forecast_steps: Optional[int] = None, **kwargs):
        self.forecast_steps = forecast_steps
        super().__init__(message, error_code="FORECAST", steps=forecast_steps, **kwargs)


class QualityAssessmentError(X13SeasonalAdjustmentError):
    """Raised when quality assessment fails."""
    
    def __init__(self, message: str, quality_measures: Optional[dict] = None, **kwargs):
        self.quality_measures = quality_measures or {}
        super().__init__(message, error_code="QUALITY_ASSESSMENT", measures=quality_measures, **kwargs)


class ConfigurationError(X13SeasonalAdjustmentError):
    """Raised when configuration parameters are invalid."""
    
    def __init__(self, message: str, parameter: Optional[str] = None, value: Any = None, **kwargs):
        self.parameter = parameter
        self.value = value
        super().__init__(
            message, 
            error_code="CONFIGURATION", 
            parameter=parameter,
            value=value,
            **kwargs
        )


class InsufficientDataError(DataValidationError):
    """Raised when there is insufficient data for processing."""
    
    def __init__(self, message: str, required_length: Optional[int] = None, 
                 actual_length: Optional[int] = None, **kwargs):
        self.required_length = required_length
        self.actual_length = actual_length
        super().__init__(
            message, 
            data_type="insufficient_data",
            required_length=required_length,
            actual_length=actual_length,
            **kwargs
        )


class ModelNotFittedError(X13SeasonalAdjustmentError):
    """Raised when trying to use a model that hasn't been fitted."""
    
    def __init__(self, message: str = "Model must be fitted before calling this method"):
        super().__init__(message, error_code="MODEL_NOT_FITTED")


class ConvergenceError(ARIMAModelError):
    """Raised when model convergence fails."""
    
    def __init__(self, message: str, iterations: Optional[int] = None, **kwargs):
        self.iterations = iterations
        super().__init__(message, iterations=iterations, **kwargs)


# Utility functions for error handling

def handle_pandas_errors(func):
    """Decorator to handle common pandas errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pd.errors.ParserError as e:
            raise DataValidationError(f"Data parsing failed: {str(e)}", data_type="parsing")
        except pd.errors.EmptyDataError as e:
            raise DataValidationError(f"Empty data provided: {str(e)}", data_type="empty")
        except KeyError as e:
            raise DataValidationError(f"Missing required column or index: {str(e)}", data_type="key_error")
    return wrapper


def handle_numpy_errors(func):
    """Decorator to handle common numpy errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except np.linalg.LinAlgError as e:
            raise ARIMAModelError(f"Linear algebra error in model fitting: {str(e)}")
        except ValueError as e:
            if "NaN" in str(e) or "inf" in str(e):
                raise DataValidationError(f"Invalid numerical values: {str(e)}", data_type="numerical")
            raise
    return wrapper


def handle_statsmodels_errors(func):
    """Decorator to handle common statsmodels errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "convergence" in error_msg:
                raise ConvergenceError(f"Model convergence failed: {str(e)}")
            elif "singular" in error_msg or "rank" in error_msg:
                raise ARIMAModelError(f"Matrix singularity error: {str(e)}")
            elif "parameter" in error_msg:
                raise ConfigurationError(f"Invalid model parameters: {str(e)}")
            raise ARIMAModelError(f"Statsmodels error: {str(e)}")
    return wrapper


# Import pandas and numpy for decorators
import pandas as pd
import numpy as np
