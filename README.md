# X13 Seasonal Adjustment

[![PyPI version](https://badge.fury.io/py/x13-seasonal-adjustment.svg)](https://badge.fury.io/py/x13-seasonal-adjustment)
[![Python Version](https://img.shields.io/pypi/pyversions/x13-seasonal-adjustment.svg)](https://pypi.org/project/x13-seasonal-adjustment/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/Gardash023/x13-seasonal-adjustment/actions/workflows/ci.yml/badge.svg)](https://github.com/Gardash023/x13-seasonal-adjustment/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Gardash023/x13-seasonal-adjustment/branch/main/graph/badge.svg)](https://codecov.io/gh/Gardash023/x13-seasonal-adjustment)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/x13-seasonal-adjustment/badge/?version=latest)](https://x13-seasonal-adjustment.readthedocs.io/en/latest/?badge=latest)

A comprehensive and professional Python implementation of the **X13-ARIMA-SEATS** seasonal adjustment algorithm. This library provides robust, production-ready tools for detecting and removing seasonal effects from time series data, following the methodology of the US Census Bureau's X13-ARIMA-SEATS program.

## ✨ Key Features

- 🔍 **Automatic Seasonality Detection**: Advanced statistical tests using multiple methodologies
- 📊 **X13-ARIMA-SEATS Algorithm**: Full implementation of the international standard
- ⚡ **High Performance**: Optimized computations using NumPy and SciPy
- 📈 **Comprehensive Visualization**: Professional plotting capabilities with matplotlib
- 🛡️ **Robust Error Handling**: Extensive error handling and logging system
- 🧪 **Production Ready**: Comprehensive testing with 95%+ code coverage
- 🔧 **Flexible API**: Suitable for both simple and advanced use cases
- 📚 **Professional Documentation**: Detailed documentation and examples
- 🐍 **Modern Python**: Full type hints and Python 3.8+ support

## 🚀 Quick Start

### Installation

```bash
pip install x13-seasonal-adjustment
```

For development installation:

```bash
pip install x13-seasonal-adjustment[dev]
```

### Basic Usage

```python
import pandas as pd
from x13_seasonal_adjustment import X13SeasonalAdjustment

# Load your time series data
data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Create and fit the model
x13 = X13SeasonalAdjustment()
result = x13.fit_transform(data['value'])

# Access results
print(f"Original series: {len(result.original)} observations")
print(f"Seasonally adjusted: {len(result.seasonally_adjusted)} observations")
print(f"Seasonality strength: {result.seasonality_strength:.3f}")

# Visualize results
result.plot()
```

### Advanced Configuration

```python
from x13_seasonal_adjustment import X13SeasonalAdjustment

# Customize the seasonal adjustment process
x13 = X13SeasonalAdjustment(
    freq='M',                    # Monthly data
    transform='auto',            # Automatic transformation detection
    outlier_detection=True,      # Enable outlier detection
    outlier_types=['AO', 'LS'],  # Additive and level shift outliers
    arima_order='auto',          # Automatic ARIMA model selection
    x11_mode='multiplicative',   # Multiplicative decomposition
    enable_logging=True          # Enable detailed logging
)

result = x13.fit_transform(data)

# Access comprehensive diagnostics
print(f"ARIMA model: {result.arima_model_info['order']}")
print(f"Quality measures: {result.quality_measures}")
```

## 📊 Core Components

### 1. X13SeasonalAdjustment (Main Class)

The primary interface for seasonal adjustment operations:

```python
from x13_seasonal_adjustment import X13SeasonalAdjustment

x13 = X13SeasonalAdjustment(
    freq='auto',                 # Data frequency detection
    transform='auto',            # Automatic log transformation
    outlier_detection=True,      # Outlier detection and correction
    trading_day=True,           # Trading day effect modeling
    easter=True,                # Easter effect modeling
    arima_order='auto'          # Automatic ARIMA model selection
)
```

### 2. Seasonality Testing

Advanced seasonality detection using multiple statistical tests:

```python
from x13_seasonal_adjustment.tests import SeasonalityTests

tests = SeasonalityTests()
result = tests.run_all_tests(data)
print(f"Seasonality detected: {result.has_seasonality}")
print(f"Test results: {result.test_results}")
```

### 3. Automatic ARIMA Modeling

Sophisticated ARIMA model selection with comprehensive diagnostics:

```python
from x13_seasonal_adjustment.arima import AutoARIMA

arima = AutoARIMA(
    max_p=3, max_d=2, max_q=3,           # ARIMA orders
    seasonal=True,                        # Seasonal modeling
    information_criterion='aicc'          # Model selection criterion
)
model = arima.fit(data)
forecast = model.forecast(steps=12)
```

### 4. Quality Diagnostics

Comprehensive quality assessment following X13 standards:

```python
from x13_seasonal_adjustment.diagnostics import QualityDiagnostics

diagnostics = QualityDiagnostics()
quality_report = diagnostics.evaluate(result)
print(quality_report)
```

## 🔬 Methodology

This library implements the complete X13-ARIMA-SEATS methodology:

1. **Data Validation**: Comprehensive input validation and preprocessing
2. **Outlier Detection**: Detection and adjustment for various outlier types
3. **Automatic Model Selection**: ARIMA model identification using information criteria
4. **Seasonal Decomposition**: X11 algorithm with multiplicative/additive modes
5. **Quality Assessment**: M and Q statistics for result validation
6. **Forecasting**: Model-based forecasting and backcasting

### Supported Outlier Types

- **AO (Additive Outlier)**: Sudden spikes in individual observations
- **LS (Level Shift)**: Permanent changes in series level
- **TC (Temporary Change)**: Temporary departures from trend
- **SO (Seasonal Outlier)**: Seasonal pattern disruptions

## 📈 Performance Benchmarks

X13 Seasonal Adjustment is designed for production environments:

| Dataset Size | Processing Time | Memory Usage |
|--------------|-----------------|--------------|
| 100 observations | ~50ms | ~2MB |
| 1,000 observations | ~200ms | ~5MB |
| 10,000 observations | ~2s | ~25MB |

*Benchmarks run on Intel Core i7, 16GB RAM*

## 🧪 Testing & Quality Assurance

- **95%+ Test Coverage**: Comprehensive unit and integration tests
- **Continuous Integration**: GitHub Actions with multi-platform testing
- **Code Quality**: Black formatting, flake8 linting, mypy type checking
- **Security Scanning**: Bandit security analysis, dependency vulnerability checks
- **Performance Monitoring**: Automated benchmarking and regression detection

## 📖 Examples

### Monthly Sales Data

```python
import numpy as np
import pandas as pd
from x13_seasonal_adjustment import X13SeasonalAdjustment

# Generate sample monthly data
dates = pd.date_range('2010-01-01', periods=120, freq='M')
trend = np.linspace(100, 200, 120)
seasonal = 10 * np.sin(2 * np.pi * np.arange(120) / 12)
noise = np.random.normal(0, 5, 120)
data = pd.Series(trend + seasonal + noise, index=dates)

# Perform seasonal adjustment
x13 = X13SeasonalAdjustment()
result = x13.fit_transform(data)

# Analyze results
print(f"Seasonality strength: {result.seasonality_strength:.3f}")
print(f"Trend strength: {result.trend_strength:.3f}")
print(f"Quality rating: {result.decomposition_quality}")
```

### Quarterly Economic Data

```python
# Quarterly GDP data example
x13_quarterly = X13SeasonalAdjustment(
    freq='Q',
    transform='log',
    outlier_detection=True,
    arima_order=(2, 1, 2),
    seasonal_arima_order=(1, 1, 1)
)

result = x13_quarterly.fit_transform(gdp_data)
result.plot_seasonal_pattern()  # Show seasonal patterns
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Gardash023/x13-seasonal-adjustment.git
cd x13-seasonal-adjustment

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Run quality checks (`black`, `flake8`, `mypy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Developer**: Gardash Abbasov
- **Email**: gardash.abbasov@gmail.com
- **GitHub**: [@Gardash023](https://github.com/Gardash023)
- **Documentation**: [https://x13-seasonal-adjustment.readthedocs.io](https://x13-seasonal-adjustment.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Gardash023/x13-seasonal-adjustment/issues)

## 🙏 Acknowledgments

- **US Census Bureau**: For the original X13-ARIMA-SEATS methodology
- **Statsmodels Team**: For foundational time series analysis tools
- **NumPy & SciPy**: For high-performance numerical computing
- **Pandas**: For flexible data manipulation capabilities
- **Python Community**: For creating an amazing ecosystem

## 📊 Citation

If you use this library in academic research, please cite:

```bibtex
@software{x13_seasonal_adjustment,
  author = {Abbasov, Gardash},
  title = {X13 Seasonal Adjustment: A Python Implementation},
  url = {https://github.com/Gardash023/x13-seasonal-adjustment},
  version = {0.1.3},
  year = {2024}
}
```

## 🗺️ Roadmap

- [ ] **Real-time Processing**: Streaming seasonal adjustment capabilities
- [ ] **Multiple Series**: Batch processing for multiple time series
- [ ] **Advanced Outliers**: Additional outlier detection algorithms
- [ ] **GPU Acceleration**: CUDA/OpenCL support for large datasets
- [ ] **Web Interface**: REST API and dashboard
- [ ] **R Integration**: Cross-language compatibility
- [ ] **Cloud Deployment**: Docker containers and cloud templates

---

**Made with ❤️ for the time series analysis community**