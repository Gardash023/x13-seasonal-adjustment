"""
Comprehensive professional tests for X13 Seasonal Adjustment library.

This module contains integration tests that verify the complete functionality
of the X13 seasonal adjustment library with professional-grade error handling,
logging, and quality assurance.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict

from x13_seasonal_adjustment import (
    X13SeasonalAdjustment,
    SeasonalAdjustmentResult,
    X13Logger,
    LoggingContextManager
)
from x13_seasonal_adjustment.exceptions import (
    X13SeasonalAdjustmentError,
    DataValidationError,
    TimeSeriesError,
    SeasonalityError,
    ARIMAModelError,
    ModelNotFittedError,
    ConfigurationError,
    InsufficientDataError
)


class TestProfessionalX13Integration:
    """Comprehensive integration tests for professional X13 implementation."""
    
    @pytest.fixture
    def monthly_data(self) -> pd.Series:
        """Create realistic monthly time series with seasonal patterns."""
        dates = pd.date_range('2010-01-01', periods=120, freq='M')
        
        # Create realistic seasonal pattern
        trend = np.linspace(100, 200, 120)
        seasonal = 15 * np.sin(2 * np.pi * np.arange(120) / 12)
        irregular = np.random.normal(0, 5, 120)
        
        # Add some realistic features
        series = trend + seasonal + irregular
        
        # Add a few outliers
        series[30] += 50  # Additive outlier
        series[60:65] += 20  # Level shift
        
        return pd.Series(series, index=dates, name='monthly_values')
    
    @pytest.fixture
    def quarterly_data(self) -> pd.Series:
        """Create quarterly time series data."""
        dates = pd.date_range('2000-Q1', periods=60, freq='Q')
        
        trend = np.linspace(1000, 1500, 60)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(60) / 4)
        noise = np.random.normal(0, 25, 60)
        
        return pd.Series(trend + seasonal + noise, index=dates)
    
    @pytest.fixture
    def short_series(self) -> pd.Series:
        """Create series with insufficient data for testing error handling."""
        dates = pd.date_range('2023-01-01', periods=10, freq='M')
        return pd.Series(np.random.randn(10), index=dates)
    
    def test_comprehensive_monthly_adjustment(self, monthly_data: pd.Series) -> None:
        """Test complete monthly seasonal adjustment workflow."""
        # Test with full logging enabled
        x13 = X13SeasonalAdjustment(
            freq='M',
            transform='auto',
            outlier_detection=True,
            enable_logging=True,
            log_level='DEBUG'
        )
        
        # Perform seasonal adjustment
        result = x13.fit_transform(monthly_data)
        
        # Validate result structure
        assert isinstance(result, SeasonalAdjustmentResult)
        assert len(result.original) == len(monthly_data)
        assert len(result.seasonally_adjusted) == len(monthly_data)
        assert len(result.seasonal_factors) == len(monthly_data)
        assert len(result.trend) == len(monthly_data)
        assert len(result.irregular) == len(monthly_data)
        
        # Validate strength measures
        assert 0.0 <= result.seasonality_strength <= 1.0
        assert 0.0 <= result.trend_strength <= 1.0
        
        # Validate ARIMA model info
        assert 'order' in result.arima_model_info
        assert 'seasonal_order' in result.arima_model_info
        assert 'aic' in result.arima_model_info
        
        # Test quality measures exist
        assert result.quality_measures is not None
        assert isinstance(result.quality_measures, dict)
    
    def test_quarterly_seasonal_adjustment(self, quarterly_data: pd.Series) -> None:
        """Test quarterly data seasonal adjustment."""
        x13 = X13SeasonalAdjustment(
            freq='Q',
            transform='log',
            outlier_detection=True,
            enable_logging=False  # Test with logging disabled
        )
        
        result = x13.fit_transform(quarterly_data)
        
        # Validate quarterly-specific features
        assert len(result.original) == len(quarterly_data)
        assert result.seasonality_strength >= 0.0
        
        # Test seasonal pattern analysis
        seasonal_pattern = result.seasonal_factors.groupby(
            result.seasonal_factors.index.quarter
        ).mean()
        assert len(seasonal_pattern) == 4  # Four quarters
    
    def test_error_handling_insufficient_data(self, short_series: pd.Series) -> None:
        """Test error handling for insufficient data."""
        x13 = X13SeasonalAdjustment()
        
        with pytest.raises(InsufficientDataError) as exc_info:
            x13.fit(short_series)
        
        # Validate exception details
        assert "insufficient" in str(exc_info.value).lower()
        assert exc_info.value.actual_length == len(short_series)
        assert exc_info.value.required_length is not None
    
    def test_invalid_configuration_error_handling(self) -> None:
        """Test comprehensive parameter validation."""
        
        # Test invalid frequency
        with pytest.raises(ConfigurationError) as exc_info:
            X13SeasonalAdjustment(freq='INVALID')
        assert exc_info.value.parameter == 'freq'
        
        # Test invalid transform
        with pytest.raises(ConfigurationError):
            X13SeasonalAdjustment(transform='INVALID')
        
        # Test invalid ARIMA order
        with pytest.raises(ConfigurationError):
            X13SeasonalAdjustment(arima_order=(10, 5, 10))  # Too large
        
        # Test invalid outlier types
        with pytest.raises(ConfigurationError):
            X13SeasonalAdjustment(outlier_types=['INVALID'])
    
    def test_model_not_fitted_error(self, monthly_data: pd.Series) -> None:
        """Test error handling for unfitted model."""
        x13 = X13SeasonalAdjustment()
        
        # Test transform without fitting
        with pytest.raises(ModelNotFittedError):
            x13.transform(monthly_data)
        
        # Test forecast without fitting
        with pytest.raises(ModelNotFittedError):
            x13.forecast(steps=12)
    
    def test_logging_functionality(self, monthly_data: pd.Series) -> None:
        """Test comprehensive logging functionality."""
        
        # Test with different log levels
        for log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            x13 = X13SeasonalAdjustment(
                enable_logging=True,
                log_level=log_level
            )
            
            # This should not raise any errors
            result = x13.fit_transform(monthly_data)
            assert isinstance(result, SeasonalAdjustmentResult)
        
        # Test logging context manager
        with LoggingContextManager('DEBUG'):
            x13 = X13SeasonalAdjustment()
            result = x13.fit_transform(monthly_data)
            assert isinstance(result, SeasonalAdjustmentResult)
    
    def test_data_validation_edge_cases(self) -> None:
        """Test data validation with various edge cases."""
        x13 = X13SeasonalAdjustment()
        
        # Test with NaN values
        data_with_nan = pd.Series([1, 2, np.nan, 4, 5] * 10)
        data_with_nan.index = pd.date_range('2020-01-01', periods=50, freq='M')
        
        # Should handle NaN appropriately
        with pytest.raises(DataValidationError):
            x13.fit(data_with_nan)
        
        # Test with constant series
        constant_series = pd.Series([100] * 50)
        constant_series.index = pd.date_range('2020-01-01', periods=50, freq='M')
        
        with pytest.raises(DataValidationError):
            x13.fit(constant_series)
    
    def test_forecast_functionality(self, monthly_data: pd.Series) -> None:
        """Test forecasting capabilities."""
        x13 = X13SeasonalAdjustment()
        x13.fit(monthly_data)
        
        # Test basic forecasting
        forecast = x13.forecast(steps=12)
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 12
        
        # Test forecast with confidence intervals
        forecast_with_ci = x13.forecast(steps=6, alpha=0.05)
        if isinstance(forecast_with_ci, tuple):
            forecast_values, conf_intervals = forecast_with_ci
            assert len(forecast_values) == 6
            assert isinstance(conf_intervals, pd.DataFrame)
            assert conf_intervals.shape[0] == 6
    
    def test_model_summary_and_diagnostics(self, monthly_data: pd.Series) -> None:
        """Test model summary and diagnostic functionality."""
        x13 = X13SeasonalAdjustment(enable_logging=True)
        result = x13.fit_transform(monthly_data)
        
        # Test model summary
        summary = x13.get_model_summary()
        assert isinstance(summary, dict)
        assert 'parameters' in summary
        assert 'arima_model' in summary
        assert 'preprocessing' in summary
        
        # Test result summary
        result_summary = result.summary()
        assert isinstance(result_summary, pd.DataFrame)
        assert len(result_summary) > 0
        
        # Test decomposition quality assessment
        quality = result.decomposition_quality
        assert quality in ['Excellent', 'Good', 'Fair', 'Poor', 'Unknown']
    
    def test_different_data_frequencies(self) -> None:
        """Test handling of different data frequencies."""
        frequencies = ['M', 'Q', 'A']
        
        for freq in frequencies:
            if freq == 'M':
                periods, seasonal_period = 60, 12
            elif freq == 'Q':
                periods, seasonal_period = 40, 4
            else:  # 'A'
                periods, seasonal_period = 20, 1
            
            # Create test data
            dates = pd.date_range('2000-01-01', periods=periods, freq=freq)
            if seasonal_period > 1:
                seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / seasonal_period)
            else:
                seasonal = np.zeros(periods)
            
            trend = np.linspace(100, 200, periods)
            noise = np.random.normal(0, 5, periods)
            data = pd.Series(trend + seasonal + noise, index=dates)
            
            # Test seasonal adjustment
            x13 = X13SeasonalAdjustment(freq=freq)
            result = x13.fit_transform(data)
            
            assert isinstance(result, SeasonalAdjustmentResult)
            assert len(result.seasonally_adjusted) == periods
    
    def test_performance_and_memory_efficiency(self, monthly_data: pd.Series) -> None:
        """Test performance and memory efficiency."""
        import time
        import psutil
        import os
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the operation
        start_time = time.time()
        
        x13 = X13SeasonalAdjustment()
        result = x13.fit_transform(monthly_data)
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance assertions
        execution_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        # Should complete reasonably quickly
        assert execution_time < 30.0  # seconds
        
        # Should not use excessive memory
        assert memory_increase < 100  # MB
        
        # Result should be valid
        assert isinstance(result, SeasonalAdjustmentResult)
    
    def test_reproducibility(self, monthly_data: pd.Series) -> None:
        """Test that results are reproducible."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run twice with same parameters
        x13_1 = X13SeasonalAdjustment(
            freq='M',
            transform='auto',
            outlier_detection=True
        )
        result_1 = x13_1.fit_transform(monthly_data)
        
        # Reset random seed
        np.random.seed(42)
        
        x13_2 = X13SeasonalAdjustment(
            freq='M',
            transform='auto', 
            outlier_detection=True
        )
        result_2 = x13_2.fit_transform(monthly_data)
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(
            result_1.seasonally_adjusted.values,
            result_2.seasonally_adjusted.values,
            rtol=1e-10
        )
    
    @pytest.mark.slow
    def test_large_dataset_handling(self) -> None:
        """Test handling of larger datasets."""
        # Create larger dataset (10 years of daily data)
        dates = pd.date_range('2010-01-01', periods=3650, freq='D')
        
        # Create realistic daily pattern
        trend = np.linspace(100, 200, 3650)
        yearly_seasonal = 20 * np.sin(2 * np.pi * np.arange(3650) / 365.25)
        weekly_seasonal = 5 * np.sin(2 * np.pi * np.arange(3650) / 7)
        noise = np.random.normal(0, 10, 3650)
        
        data = pd.Series(
            trend + yearly_seasonal + weekly_seasonal + noise,
            index=dates
        )
        
        # Test with daily frequency
        x13 = X13SeasonalAdjustment(freq='D')
        
        # This should handle large data efficiently
        result = x13.fit_transform(data)
        
        assert isinstance(result, SeasonalAdjustmentResult)
        assert len(result.seasonally_adjusted) == len(data)
    
    def test_api_compatibility(self, monthly_data: pd.Series) -> None:
        """Test scikit-learn API compatibility."""
        x13 = X13SeasonalAdjustment()
        
        # Test sklearn-style methods
        fitted_model = x13.fit(monthly_data)
        assert fitted_model is x13  # Should return self
        
        # Test transform method
        result = x13.transform(monthly_data)
        assert isinstance(result, SeasonalAdjustmentResult)
        
        # Test fit_transform method
        result2 = X13SeasonalAdjustment().fit_transform(monthly_data)
        assert isinstance(result2, SeasonalAdjustmentResult)
    
    def test_warning_handling(self) -> None:
        """Test appropriate warning generation."""
        # Create data with weak seasonality
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        weak_seasonal_data = pd.Series(
            np.random.normal(100, 5, 50),  # Mostly noise
            index=dates
        )
        
        # Should generate warning about weak seasonality
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            x13 = X13SeasonalAdjustment()
            result = x13.fit_transform(weak_seasonal_data)
            
            # Should have warnings but still produce result
            assert len(warning_list) > 0
            assert isinstance(result, SeasonalAdjustmentResult)
    
    def test_edge_case_data_types(self) -> None:
        """Test handling of different input data types."""
        # Test with DataFrame input
        dates = pd.date_range('2020-01-01', periods=60, freq='M')
        df = pd.DataFrame({
            'value': np.random.randn(60) * 10 + 100
        }, index=dates)
        
        x13 = X13SeasonalAdjustment()
        result = x13.fit_transform(df)
        assert isinstance(result, SeasonalAdjustmentResult)
        
        # Test with numpy array (should work with date index)
        array_data = np.random.randn(60) * 10 + 100
        series = pd.Series(array_data, index=dates)
        
        result2 = x13.fit_transform(series)
        assert isinstance(result2, SeasonalAdjustmentResult)


# Additional utility tests
class TestX13UtilityFunctions:
    """Test utility functions and edge cases."""
    
    def test_logger_initialization(self) -> None:
        """Test logger initialization and configuration."""
        logger = X13Logger.get_logger('test_module')
        assert logger is not None
        assert logger.name == 'x13_seasonal_adjustment.test_module'
    
    def test_exception_hierarchy(self) -> None:
        """Test exception hierarchy and information."""
        # Test base exception
        base_exc = X13SeasonalAdjustmentError("Test error", error_code="TEST")
        assert base_exc.error_code == "TEST"
        assert "Test error" in str(base_exc)
        
        # Test specific exceptions
        data_exc = DataValidationError("Data invalid", data_type="test_data")
        assert data_exc.data_type == "test_data"
        assert "DATA_VALIDATION" in str(data_exc)
        
        config_exc = ConfigurationError("Invalid param", parameter="test_param", value="invalid")
        assert config_exc.parameter == "test_param"
        assert config_exc.value == "invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
