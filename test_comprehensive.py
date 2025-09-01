#!/usr/bin/env python3
"""
Comprehensive test suite for X13 Seasonal Adjustment library.
Tests various data types, edge cases, and error conditions.
"""

import numpy as np
import pandas as pd
import warnings
from x13_seasonal_adjustment import X13SeasonalAdjustment, SeasonalityTests

def create_test_data():
    """Create various test datasets."""
    np.random.seed(42)
    
    # 1. Monthly data with strong seasonality
    dates_monthly = pd.date_range('2019-01-01', periods=48, freq='ME')
    trend = np.linspace(100, 200, 48)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = np.random.normal(0, 5, 48)
    monthly_data = pd.Series(trend + seasonal + noise, index=dates_monthly, name='Monthly Sales')
    
    # 2. Quarterly data (longer series)
    dates_quarterly = pd.date_range('2019-01-01', periods=32, freq='QE')  # 8 years
    trend_q = np.linspace(1000, 1500, 32)
    seasonal_q = 100 * np.sin(2 * np.pi * np.arange(32) / 4)
    noise_q = np.random.normal(0, 20, 32)
    quarterly_data = pd.Series(trend_q + seasonal_q + noise_q, index=dates_quarterly, name='Quarterly GDP')
    
    # 3. Data with outliers
    outlier_data = monthly_data.copy()
    outlier_data.iloc[10] += 100  # Add outlier
    outlier_data.iloc[25] -= 80   # Add another outlier
    
    # 4. Non-seasonal data (random walk)
    random_walk = pd.Series(np.cumsum(np.random.randn(48)), index=dates_monthly, name='Random Walk')
    
    # 5. Very short data (edge case)
    short_data = monthly_data.iloc[:15].copy()
    
    # 6. Data with missing values
    missing_data = monthly_data.copy()
    missing_data.iloc[5:8] = np.nan
    missing_data.iloc[20] = np.nan
    
    return {
        'monthly': monthly_data,
        'quarterly': quarterly_data,
        'outliers': outlier_data,
        'random_walk': random_walk,
        'short': short_data,
        'missing': missing_data
    }

def test_basic_functionality():
    """Test basic X13 functionality."""
    print("🧪 Testing Basic Functionality...")
    
    data = create_test_data()
    
    # Test monthly data
    x13 = X13SeasonalAdjustment()
    try:
        result = x13.fit_transform(data['monthly'])
        print(f"✅ Monthly data: Success (trend_strength: {result.trend_strength:.2f})")
        assert hasattr(result, 'seasonally_adjusted')
        assert hasattr(result, 'seasonal_factors')
        assert hasattr(result, 'trend')
    except Exception as e:
        print(f"❌ Monthly data failed: {e}")
        return False
    
    return True

def test_different_frequencies():
    """Test different data frequencies."""
    print("🧪 Testing Different Frequencies...")
    
    data = create_test_data()
    
    # Test quarterly data
    try:
        x13_q = X13SeasonalAdjustment(freq='Q')
        result_q = x13_q.fit_transform(data['quarterly'])
        print(f"✅ Quarterly data: Success")
    except Exception as e:
        print(f"❌ Quarterly data failed: {e}")
        return False
    
    return True

def test_outlier_detection():
    """Test outlier detection."""
    print("🧪 Testing Outlier Detection...")
    
    data = create_test_data()
    
    try:
        x13_outlier = X13SeasonalAdjustment(outlier_detection=True)
        result = x13_outlier.fit_transform(data['outliers'])
        print(f"✅ Outlier detection: Success")
    except Exception as e:
        print(f"❌ Outlier detection failed: {e}")
        return False
    
    return True

def test_transforms():
    """Test different transformation types."""
    print("🧪 Testing Transformations...")
    
    data = create_test_data()
    
    # Test log transform
    try:
        x13_log = X13SeasonalAdjustment(transform='log')
        result_log = x13_log.fit_transform(data['monthly'])
        print(f"✅ Log transform: Success")
    except Exception as e:
        print(f"❌ Log transform failed: {e}")
        return False
    
    # Test no transform
    try:
        x13_none = X13SeasonalAdjustment(transform='none')
        result_none = x13_none.fit_transform(data['monthly'])
        print(f"✅ No transform: Success")
    except Exception as e:
        print(f"❌ No transform failed: {e}")
        return False
    
    return True

def test_seasonality_tests():
    """Test seasonality detection."""
    print("🧪 Testing Seasonality Detection...")
    
    data = create_test_data()
    
    try:
        seasonality_tester = SeasonalityTests()
        
        # Test seasonal data
        result_seasonal = seasonality_tester.run_all_tests(data['monthly'])
        print(f"✅ Seasonal data detected: {result_seasonal.has_seasonality} (confidence: {result_seasonal.confidence_level:.4f})")
        
        # Test non-seasonal data
        result_rw = seasonality_tester.run_all_tests(data['random_walk'])
        print(f"✅ Random walk seasonality: {result_rw.has_seasonality} (confidence: {result_rw.confidence_level:.4f})")
        
    except Exception as e:
        print(f"❌ Seasonality tests failed: {e}")
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("🧪 Testing Edge Cases...")
    
    data = create_test_data()
    
    # Test very short data
    try:
        x13_short = X13SeasonalAdjustment()
        result_short = x13_short.fit_transform(data['short'])
        print(f"✅ Short data: Success (length: {len(data['short'])})")
    except Exception as e:
        print(f"⚠️ Short data warning: {e}")
    
    # Test data with missing values
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x13_missing = X13SeasonalAdjustment()
            result_missing = x13_missing.fit_transform(data['missing'])
            print(f"✅ Missing data: Handled")
    except Exception as e:
        print(f"❌ Missing data failed: {e}")
        return False
    
    return True

def test_invalid_inputs():
    """Test invalid input handling."""
    print("🧪 Testing Invalid Inputs...")
    
    # Test invalid frequency
    try:
        x13_invalid = X13SeasonalAdjustment(freq='INVALID')
        # Try to fit - validation happens during fit
        data = create_test_data()
        x13_invalid.fit(data['monthly'])
        print(f"❌ Should have failed with invalid frequency")
        return False
    except ValueError:
        print(f"✅ Invalid frequency rejected correctly")
    
    # Test invalid transform
    try:
        x13_invalid = X13SeasonalAdjustment(transform='INVALID')
        # Try to fit - validation happens during fit
        data = create_test_data()
        x13_invalid.fit(data['monthly'])
        print(f"❌ Should have failed with invalid transform")
        return False
    except ValueError:
        print(f"✅ Invalid transform rejected correctly")
    
    return True

def test_performance():
    """Test performance with larger datasets."""
    print("🧪 Testing Performance...")
    
    # Create larger dataset
    dates_large = pd.date_range('2000-01-01', periods=300, freq='ME')  # 25 years
    trend_large = np.linspace(100, 500, 300)
    seasonal_large = 50 * np.sin(2 * np.pi * np.arange(300) / 12)
    noise_large = np.random.normal(0, 10, 300)
    large_data = pd.Series(trend_large + seasonal_large + noise_large, 
                          index=dates_large, name='Large Dataset')
    
    try:
        import time
        start_time = time.time()
        
        x13_large = X13SeasonalAdjustment()
        result_large = x13_large.fit_transform(large_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Large dataset (300 points): {duration:.2f} seconds")
        
        if duration > 30:  # More than 30 seconds is too slow
            print(f"⚠️ Performance warning: {duration:.2f}s is quite slow")
        
    except Exception as e:
        print(f"❌ Large dataset failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Comprehensive X13 Test Suite...")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_different_frequencies,
        test_outlier_detection,
        test_transforms,
        test_seasonality_tests,
        test_edge_cases,
        test_invalid_inputs,
        test_performance
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Package is ready for release.")
        return True
    else:
        print("⚠️ Some tests failed. Please review before release.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
