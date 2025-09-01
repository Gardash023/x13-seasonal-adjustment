# Contributing to X13 Seasonal Adjustment

Thank you for your interest in contributing to the X13 Seasonal Adjustment library! We welcome contributions from the community and appreciate your efforts to improve this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## 🤝 Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the community standards.

### Our Pledge

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Inclusive**: Welcome newcomers and be patient with questions
- **Be Collaborative**: Share knowledge and help others learn
- **Be Professional**: Maintain professional conduct in all interactions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of time series analysis concepts
- Familiarity with Python scientific computing stack (NumPy, Pandas, SciPy)

### Fork the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/x13-seasonal-adjustment.git
   cd x13-seasonal-adjustment
   ```

3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/Gardash023/x13-seasonal-adjustment.git
   ```

## 🛠️ Development Environment

### Setting Up Your Environment

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Development Dependencies

The development environment includes:
- **Testing**: pytest, pytest-cov, pytest-benchmark
- **Code Quality**: black, flake8, isort, mypy
- **Security**: bandit, safety  
- **Documentation**: sphinx, sphinx-rtd-theme
- **Development**: pre-commit, tox

## 📝 Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **🐛 Bug Reports**: Help us identify and fix issues
2. **✨ Feature Requests**: Suggest new functionality  
3. **🔧 Bug Fixes**: Submit fixes for known issues
4. **⚡ Performance Improvements**: Optimize existing code
5. **📚 Documentation**: Improve or add documentation
6. **🧪 Tests**: Add or improve test coverage
7. **🎨 Code Improvements**: Refactor or clean up code

### Contribution Workflow

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: For new features or significant changes
3. **Create a branch**: Use descriptive branch names
4. **Make changes**: Follow our coding standards
5. **Test thoroughly**: Ensure all tests pass
6. **Submit a pull request**: Follow our PR template

### Branch Naming Convention

Use descriptive branch names that indicate the type of change:

```bash
feature/add-outlier-detection     # New feature
fix/arima-convergence-issue      # Bug fix
docs/update-api-reference        # Documentation
refactor/improve-performance     # Code refactoring
test/add-integration-tests       # Test additions
```

## 🎨 Code Style

We maintain high code quality standards with automated tools:

### Python Style Guide

- **Formatting**: [Black](https://black.readthedocs.io/) with 88-character line length
- **Import Sorting**: [isort](https://pycqa.github.io/isort/) with Black profile
- **Linting**: [flake8](https://flake8.pycqa.org/) with specific configuration
- **Type Hints**: Required for all new code using [mypy](http://mypy-lang.org/)

### Code Quality Checks

Run these commands before submitting:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

### Automated Checks

Our CI pipeline automatically runs:
- Code formatting verification
- Import sorting verification
- Linting checks
- Type checking
- Security scanning
- Test suite execution

## 🧪 Testing

### Testing Philosophy

- **High Coverage**: Maintain >85% test coverage
- **Quality Tests**: Write meaningful tests, not just for coverage
- **Multiple Levels**: Unit tests, integration tests, performance tests
- **Real Data**: Test with realistic time series data

### Test Organization

```
tests/
├── unit/               # Unit tests for individual components
├── integration/        # Integration tests for full workflows  
├── performance/        # Performance benchmarks
├── data/              # Test datasets
└── fixtures/          # Shared test fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/x13_seasonal_adjustment

# Run specific test categories  
pytest -m "unit"          # Unit tests only
pytest -m "integration"   # Integration tests only
pytest -m "not slow"      # Skip slow tests

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

### Writing Tests

Follow these guidelines when writing tests:

1. **Use descriptive names**: Test names should clearly describe what's being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Test edge cases**: Include boundary conditions and error cases
4. **Use fixtures**: Share common setup using pytest fixtures
5. **Mock external dependencies**: Use mocking for external services

Example test:

```python
import pytest
from x13_seasonal_adjustment import X13SeasonalAdjustment, DataValidationError

def test_x13_raises_error_for_insufficient_data():
    """Test that X13 raises DataValidationError for insufficient data."""
    # Arrange
    short_series = pd.Series([1, 2, 3])  # Only 3 observations
    x13 = X13SeasonalAdjustment()
    
    # Act & Assert
    with pytest.raises(DataValidationError, match="insufficient data"):
        x13.fit(short_series)
```

## 📚 Documentation

### Documentation Standards

- **API Documentation**: All public methods must have comprehensive docstrings
- **Type Hints**: All function signatures must include type hints
- **Examples**: Include usage examples in docstrings
- **Tutorials**: Create tutorials for complex features

### Docstring Format

We use Google-style docstrings:

```python
def seasonal_adjustment(
    data: pd.Series, 
    freq: str = 'auto',
    transform: str = 'auto'
) -> SeasonalAdjustmentResult:
    """
    Perform seasonal adjustment on time series data.
    
    This function applies the X13-ARIMA-SEATS methodology to detect
    and remove seasonal effects from the input time series.
    
    Args:
        data: Time series data to adjust
        freq: Data frequency ('M', 'Q', 'A', or 'auto')
        transform: Transformation type ('auto', 'log', 'none')
        
    Returns:
        Complete seasonal adjustment results including seasonally
        adjusted series, seasonal factors, and quality diagnostics.
        
    Raises:
        DataValidationError: If input data is invalid
        InsufficientDataError: If insufficient observations
        
    Examples:
        >>> import pandas as pd
        >>> data = pd.Series([100, 110, 95, 105])
        >>> result = seasonal_adjustment(data)
        >>> print(result.seasonally_adjusted)
        
    Note:
        For optimal results, use at least 3 years of monthly data
        or 8 quarters of quarterly data.
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🔄 Pull Request Process

### Before Submitting

1. **Sync your fork**: Pull latest changes from upstream
2. **Run all tests**: Ensure everything passes locally
3. **Update documentation**: Add/update relevant documentation
4. **Add changelog entry**: Update CHANGELOG.md if applicable
5. **Self-review**: Review your changes carefully

### PR Template

When creating a pull request, please:

1. **Use descriptive title**: Clearly describe the change
2. **Fill out template**: Complete all sections of the PR template
3. **Link related issues**: Reference any related issues
4. **Add screenshots**: For UI changes, include before/after images
5. **Request reviews**: Tag relevant maintainers

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No merge conflicts
- [ ] PR description is clear and complete

### Review Process

1. **Automated checks**: CI pipeline must pass
2. **Maintainer review**: At least one maintainer approval required
3. **Community feedback**: Address any community comments
4. **Final approval**: Maintainer will merge when ready

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear title**: Descriptive summary of the issue
2. **Environment details**: OS, Python version, library version
3. **Reproduction steps**: Step-by-step instructions to reproduce
4. **Expected behavior**: What you expected to happen
5. **Actual behavior**: What actually happened
6. **Error messages**: Full error traces if applicable
7. **Sample data**: Minimal example that demonstrates the issue

### Feature Requests

For feature requests, please provide:

1. **Use case**: Describe the problem you're trying to solve
2. **Proposed solution**: Your idea for addressing the need
3. **Alternatives**: Other approaches you've considered
4. **Examples**: How the feature would be used
5. **Benefits**: Why this would be valuable to users

### Issue Labels

We use labels to categorize issues:

- **bug**: Something isn't working
- **enhancement**: New feature or improvement
- **documentation**: Documentation improvements
- **good first issue**: Good for newcomers
- **help wanted**: Community assistance needed
- **priority:high**: Critical issues
- **priority:low**: Nice-to-have improvements

## 🌟 Community

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: gardash.abbasov@gmail.com for private matters

### Recognition

We recognize contributors in several ways:

- **Contributors file**: All contributors listed in CONTRIBUTORS.md
- **Release notes**: Significant contributions highlighted
- **GitHub statistics**: Contributions tracked in GitHub insights

### Mentorship

New contributors can expect:

- **Guidance**: Help with development environment setup
- **Code review**: Constructive feedback on submissions
- **Mentoring**: Support for learning project conventions
- **Pair programming**: Collaborative development opportunities

## ❓ Getting Help

If you need help:

1. **Check documentation**: Review existing docs and examples
2. **Search issues**: Look for similar questions or problems
3. **Create discussion**: Start a GitHub Discussion for questions
4. **Contact maintainers**: Email for private or urgent matters

## 🎯 Development Roadmap

Current priorities include:

- **Performance optimization**: Faster processing for large datasets
- **Additional algorithms**: More seasonal adjustment methods
- **Real-time processing**: Streaming data capabilities
- **GPU acceleration**: CUDA/OpenCL support
- **Cloud integration**: AWS/Azure/GCP deployment guides

---

**Thank you for contributing to X13 Seasonal Adjustment! Your efforts help make time series analysis more accessible to the Python community.**
