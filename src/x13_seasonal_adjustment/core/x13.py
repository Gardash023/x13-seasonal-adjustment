"""
Main X13 Seasonal Adjustment class.

This module implements the core X13-ARIMA-SEATS seasonal adjustment algorithm
with comprehensive error handling and logging capabilities.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin

# Internal imports
from .result import SeasonalAdjustmentResult
from .decomposition import SeasonalDecomposition
from ..arima.auto_arima import AutoARIMA
from ..tests.seasonality_tests import SeasonalityTests
from ..utils.validation import validate_time_series
from ..utils.preprocessing import preprocess_series
from ..exceptions import (
    X13SeasonalAdjustmentError,
    DataValidationError,
    TimeSeriesError,
    FrequencyError,
    SeasonalityError,
    ARIMAModelError,
    DecompositionError,
    TransformationError,
    ConfigurationError,
    ModelNotFittedError,
    InsufficientDataError,
    handle_pandas_errors,
    handle_numpy_errors,
    handle_statsmodels_errors
)
from ..logging_config import X13Logger, log_execution_time, LoggingContextManager


class X13SeasonalAdjustment(BaseEstimator, TransformerMixin):
    """
    Main X13-ARIMA-SEATS seasonal adjustment class.
    
    This class implements the methodology of the US Census Bureau's X13-ARIMA-SEATS 
    program to detect and remove seasonal effects from time series data with
    comprehensive error handling and professional logging.
    
    Parameters:
        freq (str): Data frequency ('M'=Monthly, 'Q'=Quarterly, 'A'=Annual, 'auto'=Auto-detect)
        transform (str): Logarithmic transformation ('auto', 'log', 'none')
        outlier_detection (bool): Whether to perform outlier detection
        outlier_types (List[str]): Types of outliers to detect ['AO', 'LS', 'TC', 'SO']
        trading_day (bool): Whether to model trading day effects
        easter (bool): Whether to model Easter effects
        arima_order (Union[Tuple, str]): ARIMA model order or 'auto'
        seasonal_arima_order (Union[Tuple, str]): Seasonal ARIMA order or 'auto'
        max_seasonal_ma (int): Maximum seasonal MA order
        x11_mode (str): X11 decomposition mode ('multiplicative', 'additive', 'auto')
        forecast_maxlead (int): Maximum forecast length
        backcast_maxlead (int): Maximum backcast length
        enable_logging (bool): Whether to enable detailed logging
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Raises:
        ConfigurationError: If parameters are invalid
        ValueError: If parameter combinations are incompatible
    
    Examples:
        >>> from x13_seasonal_adjustment import X13SeasonalAdjustment
        >>> import pandas as pd
        >>> 
        >>> # Basic usage with auto-configuration
        >>> x13 = X13SeasonalAdjustment()
        >>> result = x13.fit_transform(data)
        >>> 
        >>> # Advanced configuration
        >>> x13 = X13SeasonalAdjustment(
        ...     freq='M',
        ...     transform='log',
        ...     outlier_detection=True,
        ...     arima_order=(1, 1, 1),
        ...     seasonal_arima_order=(0, 1, 1)
        ... )
        >>> result = x13.fit_transform(data)
    """
    
    def __init__(
        self,
        freq: str = 'auto',
        transform: str = 'auto',
        outlier_detection: bool = True,
        outlier_types: Optional[List[str]] = None,
        trading_day: bool = True,
        easter: bool = True,
        arima_order: Union[Tuple[int, int, int], str] = 'auto',
        seasonal_arima_order: Union[Tuple[int, int, int], str] = 'auto',
        max_seasonal_ma: int = 2,
        x11_mode: str = 'auto',
        forecast_maxlead: int = 12,
        backcast_maxlead: int = 12,
        enable_logging: bool = True,
        log_level: str = 'INFO',
        **kwargs
    ):
        # Initialize logger
        self.logger = X13Logger.get_logger(self.__class__.__name__)
        
        # Configure logging if enabled
        if enable_logging:
            X13Logger.set_level(log_level.upper())
            self.logger.info("Initializing X13SeasonalAdjustment instance")
        
        try:
            # Store configuration parameters
            self.freq = freq
            self.transform_mode = transform
            self.outlier_detection = outlier_detection
            self.outlier_types = outlier_types or ['AO', 'LS', 'TC']
            self.trading_day = trading_day
            self.easter = easter
            self.arima_order = arima_order
            self.seasonal_arima_order = seasonal_arima_order
            self.max_seasonal_ma = max_seasonal_ma
            self.x11_mode = x11_mode
            self.forecast_maxlead = forecast_maxlead
            self.backcast_maxlead = backcast_maxlead
            self.enable_logging = enable_logging
            self.log_level = log_level.upper()
            
            # Validate configuration parameters
            self._validate_parameters()
            
            # Internal state variables
            self._is_fitted = False
            self._arima_model = None
            self._seasonal_decomposer = None
            self._seasonality_tester = None
            self._original_series = None
            self._preprocessing_info = None
            self._model_diagnostics = None
            self._fit_timestamp = None
            
            # Performance tracking
            self._timing_info = {}
            
            if enable_logging:
                self.logger.info("X13SeasonalAdjustment initialized successfully")
                self.logger.debug(f"Configuration: freq={freq}, transform={transform}, "
                                f"outlier_detection={outlier_detection}")
                
        except Exception as e:
            if enable_logging:
                self.logger.error(f"Failed to initialize X13SeasonalAdjustment: {str(e)}")
            raise ConfigurationError(f"Initialization failed: {str(e)}") from e
    
    def _validate_parameters(self) -> None:
        """
        Validates all configuration parameters.
        
        Raises:
            ConfigurationError: If any parameter is invalid
        """
        try:
            # Frequency validation
            valid_freqs = ['auto', 'M', 'ME', 'Q', 'QE', 'A', 'AE', 'Y', 'YE', 'D', 'W']
            if not isinstance(self.freq, str):
                raise ConfigurationError("freq must be a string", parameter="freq", value=self.freq)
            
            if self.freq not in valid_freqs:
                raise ConfigurationError(
                    f"freq '{self.freq}' is invalid. Valid values: {valid_freqs[:4]} + variants",
                    parameter="freq",
                    value=self.freq
                )
            
            # Transform validation
            valid_transforms = ['auto', 'log', 'none']
            if self.transform_mode not in valid_transforms:
                raise ConfigurationError(
                    f"transform '{self.transform_mode}' is invalid. Valid values: {valid_transforms}",
                    parameter="transform",
                    value=self.transform_mode
                )
            
            # X11 mode validation
            valid_x11_modes = ['auto', 'multiplicative', 'additive']
            if self.x11_mode not in valid_x11_modes:
                raise ConfigurationError(
                    f"x11_mode '{self.x11_mode}' is invalid. Valid values: {valid_x11_modes}",
                    parameter="x11_mode",
                    value=self.x11_mode
                )
            
            # Outlier types validation
            valid_outlier_types = ['AO', 'LS', 'TC', 'SO']
            if not isinstance(self.outlier_types, list):
                raise ConfigurationError(
                    "outlier_types must be a list",
                    parameter="outlier_types",
                    value=type(self.outlier_types)
                )
            
            for otype in self.outlier_types:
                if otype not in valid_outlier_types:
                    raise ConfigurationError(
                        f"Outlier type '{otype}' is invalid. Valid values: {valid_outlier_types}",
                        parameter="outlier_types",
                        value=otype
                    )
            
            # ARIMA order validation
            if self.arima_order != 'auto':
                if not isinstance(self.arima_order, (tuple, list)) or len(self.arima_order) != 3:
                    raise ConfigurationError(
                        "arima_order must be 'auto' or a tuple/list of 3 integers",
                        parameter="arima_order",
                        value=self.arima_order
                    )
                
                p, d, q = self.arima_order
                if not all(isinstance(x, int) and x >= 0 for x in [p, d, q]):
                    raise ConfigurationError(
                        "ARIMA order components must be non-negative integers",
                        parameter="arima_order",
                        value=self.arima_order
                    )
                
                if p > 5 or q > 5 or d > 2:
                    raise ConfigurationError(
                        "ARIMA order components are too large (max: p=5, d=2, q=5)",
                        parameter="arima_order",
                        value=self.arima_order
                    )
            
            # Seasonal ARIMA order validation
            if self.seasonal_arima_order != 'auto':
                if not isinstance(self.seasonal_arima_order, (tuple, list)) or len(self.seasonal_arima_order) != 3:
                    raise ConfigurationError(
                        "seasonal_arima_order must be 'auto' or a tuple/list of 3 integers",
                        parameter="seasonal_arima_order",
                        value=self.seasonal_arima_order
                    )
                
                P, D, Q = self.seasonal_arima_order
                if not all(isinstance(x, int) and x >= 0 for x in [P, D, Q]):
                    raise ConfigurationError(
                        "Seasonal ARIMA order components must be non-negative integers",
                        parameter="seasonal_arima_order",
                        value=self.seasonal_arima_order
                    )
                
                if P > 3 or Q > 3 or D > 1:
                    raise ConfigurationError(
                        "Seasonal ARIMA order components are too large (max: P=3, D=1, Q=3)",
                        parameter="seasonal_arima_order",
                        value=self.seasonal_arima_order
                    )
            
            # Numeric parameter validation
            if not isinstance(self.max_seasonal_ma, int) or self.max_seasonal_ma < 0:
                raise ConfigurationError(
                    "max_seasonal_ma must be a non-negative integer",
                    parameter="max_seasonal_ma",
                    value=self.max_seasonal_ma
                )
            
            if not isinstance(self.forecast_maxlead, int) or self.forecast_maxlead < 1:
                raise ConfigurationError(
                    "forecast_maxlead must be a positive integer",
                    parameter="forecast_maxlead",
                    value=self.forecast_maxlead
                )
            
            if not isinstance(self.backcast_maxlead, int) or self.backcast_maxlead < 1:
                raise ConfigurationError(
                    "backcast_maxlead must be a positive integer",
                    parameter="backcast_maxlead",
                    value=self.backcast_maxlead
                )
            
            # Boolean parameter validation
            for param_name in ['outlier_detection', 'trading_day', 'easter', 'enable_logging']:
                param_value = getattr(self, param_name)
                if not isinstance(param_value, bool):
                    raise ConfigurationError(
                        f"{param_name} must be a boolean",
                        parameter=param_name,
                        value=param_value
                    )
            
            # Log level validation
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if self.log_level not in valid_log_levels:
                raise ConfigurationError(
                    f"log_level '{self.log_level}' is invalid. Valid values: {valid_log_levels}",
                    parameter="log_level",
                    value=self.log_level
                )
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Parameter validation failed: {str(e)}") from e
    
    def _determine_seasonal_period(self, series: pd.Series) -> int:
        """Determines seasonal period length based on frequency."""
        freq_to_period = {
            'M': 12,   # Monthly
            'Q': 4,    # Quarterly  
            'A': 1,    # Annual
            'D': 365,  # Daily
            'W': 52    # Weekly
        }
        return freq_to_period.get(self.freq, 12)
    
    def _determine_transform(self, series: pd.Series) -> str:
        """Determines automatic logarithmic transformation decision."""
        if self.transform_mode != 'auto':
            return self.transform_mode
        
        # Decide using coefficient of variation
        cv = np.std(series) / np.mean(series)
        
        # Log-level test
        if cv > 0.2:  # High variation
            return 'log'
        else:
            return 'none'
    
    def _apply_transform(self, series: pd.Series, transform_type: str) -> Tuple[pd.Series, Dict]:
        """Veri dönüşümü uygular."""
        transform_info = {'type': transform_type}
        
        if transform_type == 'log':
            if (series <= 0).any():
                warnings.warn("Negatif değerler logaritmik dönüşüm için düzeltiliyor")
                min_val = series.min()
                shift = abs(min_val) + 1 if min_val <= 0 else 0
                series = series + shift
                transform_info['shift'] = shift
            
            transformed = np.log(series)
            transform_info['applied'] = True
        else:
            transformed = series.copy()
            transform_info['applied'] = False
        
        return transformed, transform_info
    
    def _reverse_transform(self, series: pd.Series, transform_info: Dict) -> pd.Series:
        """Dönüşümü tersine çevirir."""
        if not transform_info.get('applied', False):
            return series
        
        # Log dönüşümünü tersine çevir
        result = np.exp(series)
        
        # Shift varsa tersine çevir
        if 'shift' in transform_info:
            result = result - transform_info['shift']
        
        return result
    
    @log_execution_time
    @handle_pandas_errors
    @handle_numpy_errors
    @handle_statsmodels_errors
    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'X13SeasonalAdjustment':
        """
        Fit the X13-ARIMA-SEATS model to the time series data.
        
        This method performs the complete X13 seasonal adjustment model fitting process
        including data validation, preprocessing, seasonality testing, ARIMA model
        selection, and seasonal decomposition preparation.
        
        Args:
            X (Union[pd.Series, pd.DataFrame]): Time series data to fit
            y (Any, optional): Ignored, present for sklearn compatibility
            
        Returns:
            X13SeasonalAdjustment: Fitted model instance
            
        Raises:
            DataValidationError: If input data is invalid
            TimeSeriesError: If time series processing fails
            SeasonalityError: If seasonality detection fails
            ARIMAModelError: If ARIMA model fitting fails
            InsufficientDataError: If insufficient data for modeling
            
        Examples:
            >>> x13 = X13SeasonalAdjustment()
            >>> x13.fit(monthly_data)
            >>> print(f"Model fitted successfully: {x13._is_fitted}")
        """
        import time
        fit_start_time = time.time()
        
        try:
            if self.enable_logging:
                self.logger.info("Starting X13 model fitting process")
                self.logger.debug(f"Input data shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
            
            # Reset fitting state
            self._is_fitted = False
            self._fit_timestamp = time.time()
            self._timing_info = {}
            
            # Data validation and preparation
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Validating and preparing input data")
                
            if self.freq and self.freq != 'auto':
                try:
                    series = validate_time_series(X, freq=self.freq, min_length=24)
                except DataValidationError as e:
                    self.logger.error(f"Data validation failed: {str(e)}")
                    raise
            else:
                try:
                    series = validate_time_series(X, min_length=24)
                    # Auto-detect frequency
                    from ..utils.validation import validate_frequency
                    detected_freq = validate_frequency(series)
                    self.freq = detected_freq
                    if self.enable_logging:
                        self.logger.info(f"Auto-detected frequency: {detected_freq}")
                except (DataValidationError, FrequencyError) as e:
                    self.logger.error(f"Frequency detection failed: {str(e)}")
                    raise
                    
            self._original_series = series.copy()
            self._timing_info['data_validation'] = time.time() - step_start
            
            # Data preprocessing
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Preprocessing time series data")
                
            try:
                preprocessed_series, self._preprocessing_info = preprocess_series(
                    series, 
                    handle_missing=True,
                    detect_outliers=self.outlier_detection
                )
                
                if self.enable_logging and self._preprocessing_info.get('outliers'):
                    outlier_count = len(self._preprocessing_info['outliers'])
                    self.logger.info(f"Detected {outlier_count} outliers during preprocessing")
                    
            except Exception as e:
                raise DataValidationError(f"Data preprocessing failed: {str(e)}") from e
                
            self._timing_info['preprocessing'] = time.time() - step_start
            
            # Seasonality testing
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Testing for seasonality patterns")
                
            try:
                seasonal_period = self._determine_seasonal_period(series)
                self._seasonality_tester = SeasonalityTests(
                    seasonal_period=seasonal_period
                )
                seasonality_result = self._seasonality_tester.run_all_tests(preprocessed_series)
                
                if self.enable_logging:
                    self.logger.info(f"Seasonality detected: {seasonality_result.has_seasonality}")
                    if hasattr(seasonality_result, 'test_results'):
                        for test_name, result in seasonality_result.test_results.items():
                            self.logger.debug(f"  {test_name}: {result}")
                
                if not seasonality_result.has_seasonality:
                    warning_msg = "No clear seasonality detected. Results may be unreliable."
                    if self.enable_logging:
                        self.logger.warning(warning_msg)
                    warnings.warn(warning_msg, UserWarning)
                    
            except Exception as e:
                raise SeasonalityError(f"Seasonality testing failed: {str(e)}") from e
                
            self._timing_info['seasonality_testing'] = time.time() - step_start
            
            # Data transformation
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Applying data transformation")
                
            try:
                transform_type = self._determine_transform(preprocessed_series)
                transformed_series, transform_info = self._apply_transform(preprocessed_series, transform_type)
                self._preprocessing_info['transform'] = transform_info
                
                if self.enable_logging:
                    self.logger.info(f"Applied transformation: {transform_type}")
                    if transform_info.get('shift'):
                        self.logger.debug(f"Applied data shift: {transform_info['shift']}")
                        
            except Exception as e:
                raise TransformationError(f"Data transformation failed: {str(e)}") from e
                
            self._timing_info['transformation'] = time.time() - step_start
            
            # ARIMA model fitting
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Fitting ARIMA model")
                
            try:
                self._arima_model = AutoARIMA(
                    seasonal_period=seasonal_period,
                    max_p=3, max_q=3, max_P=2, max_Q=2,
                    max_d=2, max_D=1,
                    information_criterion='aicc',
                    seasonal=True,
                    suppress_warnings=not self.enable_logging
                )
                
                self._arima_model.fit(transformed_series)
                
                if self.enable_logging:
                    self.logger.info(f"ARIMA model fitted: {self._arima_model.order_} x {self._arima_model.seasonal_order_}")
                    self.logger.debug(f"Model AIC: {self._arima_model.aic_:.3f}, BIC: {self._arima_model.bic_:.3f}")
                    
            except Exception as e:
                raise ARIMAModelError(f"ARIMA model fitting failed: {str(e)}") from e
                
            self._timing_info['arima_fitting'] = time.time() - step_start
            
            # Seasonal decomposer preparation
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Preparing seasonal decomposer")
                
            try:
                self._seasonal_decomposer = SeasonalDecomposition(
                    mode=self.x11_mode,
                    seasonal_period=seasonal_period,
                    arima_model=self._arima_model,
                    forecast_maxlead=self.forecast_maxlead,
                    backcast_maxlead=self.backcast_maxlead
                )
                
            except Exception as e:
                raise DecompositionError(f"Seasonal decomposer initialization failed: {str(e)}") from e
                
            self._timing_info['decomposer_setup'] = time.time() - step_start
            
            # Mark as fitted
            self._is_fitted = True
            total_time = time.time() - fit_start_time
            self._timing_info['total_fit_time'] = total_time
            
            if self.enable_logging:
                self.logger.info(f"X13 model fitting completed successfully in {total_time:.3f} seconds")
                self.logger.debug("Timing breakdown: " + 
                    ", ".join([f"{k}: {v:.3f}s" for k, v in self._timing_info.items() if k != 'total_fit_time']))
            
            return self
            
        except (DataValidationError, TimeSeriesError, SeasonalityError, 
                ARIMAModelError, DecompositionError, TransformationError) as e:
            # Re-raise known exceptions
            if self.enable_logging:
                self.logger.error(f"Model fitting failed: {str(e)}")
            raise
            
        except Exception as e:
            # Catch unexpected exceptions
            error_msg = f"Unexpected error during model fitting: {str(e)}"
            if self.enable_logging:
                self.logger.error(error_msg)
            raise X13SeasonalAdjustmentError(error_msg) from e
    
    @log_execution_time
    @handle_pandas_errors
    @handle_numpy_errors
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> SeasonalAdjustmentResult:
        """
        Perform seasonal adjustment transformation on the time series.
        
        This method applies the fitted X13-ARIMA-SEATS model to perform seasonal
        adjustment, returning comprehensive results including seasonally adjusted
        series, trend, seasonal factors, and quality diagnostics.
        
        Args:
            X (Union[pd.Series, pd.DataFrame]): Time series data to transform
            
        Returns:
            SeasonalAdjustmentResult: Complete seasonal adjustment results
            
        Raises:
            ModelNotFittedError: If model hasn't been fitted yet
            DataValidationError: If input data is invalid
            DecompositionError: If seasonal decomposition fails
            
        Examples:
            >>> x13 = X13SeasonalAdjustment()
            >>> x13.fit(data)
            >>> result = x13.transform(data)
            >>> print(f"Seasonality strength: {result.seasonality_strength:.3f}")
        """
        import time
        transform_start_time = time.time()
        
        try:
            # Check if model is fitted
            if not self._is_fitted:
                raise ModelNotFittedError("Model must be fitted before calling transform()")
            
            if self.enable_logging:
                self.logger.info("Starting seasonal adjustment transformation")
                self.logger.debug(f"Input data shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
            
            # Data validation
            step_start = time.time()
            try:
                series = validate_time_series(X, freq=self.freq)
                if self.enable_logging:
                    self.logger.debug(f"Validated input series with {len(series)} observations")
            except Exception as e:
                raise DataValidationError(f"Input validation failed: {str(e)}") from e
            
            # Data preprocessing (consistent with fit)
            if self.enable_logging:
                self.logger.debug("Applying consistent preprocessing")
            
            try:
                preprocessed_series, _ = preprocess_series(
                    series,
                    handle_missing=True,
                    outlier_info=self._preprocessing_info.get('outliers')
                )
            except Exception as e:
                raise DataValidationError(f"Data preprocessing failed: {str(e)}") from e
            
            # Apply transformation
            if self.enable_logging:
                self.logger.debug("Applying data transformation")
            
            try:
                transform_info = self._preprocessing_info['transform']
                transformed_series, _ = self._apply_transform(preprocessed_series, transform_info['type'])
            except Exception as e:
                raise TransformationError(f"Data transformation failed: {str(e)}") from e
            
            preprocessing_time = time.time() - step_start
            
            # Seasonal decomposition
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Performing seasonal decomposition")
                
            try:
                decomposition_result = self._seasonal_decomposer.decompose(transformed_series)
                
                if self.enable_logging:
                    mode = decomposition_result.get('mode', 'unknown')
                    self.logger.info(f"Seasonal decomposition completed using {mode} mode")
                    
            except Exception as e:
                raise DecompositionError(f"Seasonal decomposition failed: {str(e)}") from e
            
            decomposition_time = time.time() - step_start
            
            # Transform results back to original scale
            step_start = time.time()
            if self.enable_logging:
                self.logger.debug("Transforming results to original scale")
                
            try:
                seasonally_adjusted = self._reverse_transform(
                    decomposition_result['seasonally_adjusted'], 
                    transform_info
                )
                trend = self._reverse_transform(decomposition_result['trend'], transform_info)
                
                # Handle seasonal factors and irregular component
                if transform_info['type'] == 'log':
                    # Multiplicative model - keep as factors/ratios
                    seasonal_factors = decomposition_result['seasonal'] 
                    irregular = decomposition_result['irregular']
                else:
                    # Additive model - keep as differences
                    seasonal_factors = decomposition_result['seasonal']
                    irregular = decomposition_result['irregular']
                    
            except Exception as e:
                raise TransformationError(f"Scale transformation failed: {str(e)}") from e
            
            # Calculate strength measures
            if self.enable_logging:
                self.logger.debug("Calculating seasonality and trend strength")
                
            try:
                seasonality_strength = self._calculate_seasonality_strength(series, seasonal_factors)
                trend_strength = self._calculate_trend_strength(series, trend)
                
                if self.enable_logging:
                    self.logger.info(f"Seasonality strength: {seasonality_strength:.3f}")
                    self.logger.info(f"Trend strength: {trend_strength:.3f}")
                    
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Strength calculation failed: {str(e)}")
                seasonality_strength = 0.0
                trend_strength = 0.0
            
            # Calculate quality measures
            if self.enable_logging:
                self.logger.debug("Calculating quality measures")
                
            try:
                quality_measures = self._calculate_quality_measures(
                    series, seasonally_adjusted, seasonal_factors, irregular
                )
                
                if self.enable_logging and quality_measures:
                    q_stat = quality_measures.get('Q', 'N/A')
                    self.logger.info(f"Overall quality measure (Q): {q_stat}")
                    
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Quality measure calculation failed: {str(e)}")
                quality_measures = {}
            
            computation_time = time.time() - step_start
            
            # Prepare ARIMA model information
            try:
                arima_info = {
                    'order': self._arima_model.order_,
                    'seasonal_order': self._arima_model.seasonal_order_,
                    'aic': self._arima_model.aic_,
                    'bic': self._arima_model.bic_,
                }
                
                # Add AICC if available
                if hasattr(self._arima_model, 'aicc_'):
                    arima_info['aicc'] = self._arima_model.aicc_
                    
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"ARIMA info extraction failed: {str(e)}")
                arima_info = {}
            
            # Create result object
            total_time = time.time() - transform_start_time
            
            if self.enable_logging:
                self.logger.info(f"Seasonal adjustment completed in {total_time:.3f} seconds")
                self.logger.debug(f"Timing: preprocessing={preprocessing_time:.3f}s, "
                                f"decomposition={decomposition_time:.3f}s, "
                                f"computation={computation_time:.3f}s")
            
            return SeasonalAdjustmentResult(
                original=series,
                seasonally_adjusted=seasonally_adjusted,
                seasonal_factors=seasonal_factors,
                trend=trend,
                irregular=irregular,
                seasonality_strength=seasonality_strength,
                trend_strength=trend_strength,
                trading_day_factors=decomposition_result.get('trading_day'),
                easter_factors=decomposition_result.get('easter'),
                outliers=self._preprocessing_info.get('outliers'),
                arima_model_info=arima_info,
                quality_measures=quality_measures
            )
            
        except (ModelNotFittedError, DataValidationError, DecompositionError, TransformationError) as e:
            # Re-raise known exceptions
            if self.enable_logging:
                self.logger.error(f"Transform failed: {str(e)}")
            raise
            
        except Exception as e:
            # Catch unexpected exceptions
            error_msg = f"Unexpected error during transformation: {str(e)}"
            if self.enable_logging:
                self.logger.error(error_msg)
            raise X13SeasonalAdjustmentError(error_msg) from e
    
    def fit_transform(self, X: Union[pd.Series, pd.DataFrame], y=None) -> SeasonalAdjustmentResult:
        """
        Modeli eğitir ve dönüştürür.
        
        Args:
            X (Union[pd.Series, pd.DataFrame]): Zaman serisi verisi
            y: İgnore edilir
            
        Returns:
            SeasonalAdjustmentResult: Mevsimsellikten arındırma sonuçları
        """
        return self.fit(X, y).transform(X)
    
    def _calculate_seasonality_strength(self, original: pd.Series, seasonal: pd.Series) -> float:
        """Mevsimsellik gücünü hesaplar."""
        seasonal_var = np.var(seasonal)
        total_var = np.var(original)
        
        if total_var == 0:
            return 0.0
        
        return min(1.0, max(0.0, seasonal_var / total_var))
    
    def _calculate_trend_strength(self, original: pd.Series, trend: pd.Series) -> float:
        """Trend gücünü hesaplar."""
        detrended = original - trend
        detrended_var = np.var(detrended)
        total_var = np.var(original)
        
        if total_var == 0:
            return 0.0
        
        return min(1.0, max(0.0, 1 - (detrended_var / total_var)))
    
    def _calculate_quality_measures(
        self, 
        original: pd.Series, 
        seasonally_adjusted: pd.Series,
        seasonal: pd.Series,
        irregular: pd.Series
    ) -> Dict[str, float]:
        """X13 kalite ölçütlerini hesaplar (M ve Q istatistikleri)."""
        quality = {}
        
        # M1 - Contribution of the irregular to the variance of the stationary portion
        if len(irregular) > 12:
            m1 = np.var(irregular) / np.var(seasonally_adjusted)
            quality['M1'] = m1
        
        # M7 - Amount of month-to-month change in the irregular component
        if len(irregular) > 1:
            irregular_diff = irregular.diff().dropna()
            if len(irregular_diff) > 0:
                m7 = np.std(irregular_diff) / np.std(irregular)
                quality['M7'] = m7
        
        # Q - Overall quality measure
        m_stats = [v for k, v in quality.items() if k.startswith('M')]
        if m_stats:
            quality['Q'] = np.mean(m_stats)
        
        return quality
    
    def forecast(self, steps: int = 12) -> pd.Series:
        """
        Mevsimsellikten arındırılmış seri için öngörü yapar.
        
        Args:
            steps (int): Öngörü adım sayısı
            
        Returns:
            pd.Series: Öngörü değerleri
        """
        if not self._is_fitted:
            raise ValueError("Model henüz eğitilmemiş.")
        
        return self._arima_model.forecast(steps=steps)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Model özetini döndürür.
        
        Returns:
            Dict[str, Any]: Model bilgileri
        """
        if not self._is_fitted:
            raise ValueError("Model henüz eğitilmemiş.")
        
        summary = {
            'parameters': {
                'freq': self.freq,
                'transform': self.transform_mode,
                'outlier_detection': self.outlier_detection,
                'trading_day': self.trading_day,
                'easter': self.easter,
            },
            'arima_model': {
                'order': self._arima_model.order_,
                'seasonal_order': self._arima_model.seasonal_order_,
                'aic': self._arima_model.aic_,
                'bic': self._arima_model.bic_,
            },
            'preprocessing': self._preprocessing_info,
        }
        
        return summary
