# Makefile for X13 Seasonal Adjustment Library
# Professional development workflow automation

.PHONY: help install install-dev clean test test-all lint format type-check security docs build publish docker-build docker-run pre-commit setup-dev

# Default target
help:
	@echo "X13 Seasonal Adjustment - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup-dev     Set up development environment"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install with development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run flake8 linting"
	@echo "  type-check    Run mypy type checking"
	@echo "  security      Run security checks with bandit"
	@echo "  pre-commit    Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run basic test suite"
	@echo "  test-all      Run full test suite with coverage"
	@echo "  test-fast     Run tests excluding slow tests"
	@echo "  benchmark     Run performance benchmarks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Build documentation"
	@echo "  docs-serve    Build and serve documentation locally"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  build         Build distribution packages"
	@echo "  publish       Publish to PyPI (requires auth)"
	@echo "  publish-test  Publish to Test PyPI"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run container interactively"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Clean build artifacts"
	@echo "  clean-all     Clean all generated files"
	@echo ""

# Setup and Installation
setup-dev: install-dev pre-commit-install
	@echo "Development environment setup complete!"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

pre-commit-install:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Code Quality
format:
	@echo "Formatting code with black..."
	black src/ tests/ examples/
	@echo "Sorting imports with isort..."
	isort src/ tests/ examples/

lint:
	@echo "Running flake8 linting..."
	flake8 src/ tests/ examples/

type-check:
	@echo "Running mypy type checking..."
	mypy src/ --ignore-missing-imports

security:
	@echo "Running bandit security checks..."
	bandit -r src/ -f json -o reports/bandit-report.json || true
	@echo "Running safety vulnerability checks..."
	safety check --json --output reports/safety-report.json || true

pre-commit:
	@echo "Running all pre-commit hooks..."
	pre-commit run --all-files

# Testing
test:
	@echo "Running basic test suite..."
	pytest tests/ -v --tb=short

test-all:
	@echo "Running full test suite with coverage..."
	pytest tests/ -v --cov=src/x13_seasonal_adjustment --cov-report=html --cov-report=term --cov-report=xml

test-fast:
	@echo "Running fast tests (excluding slow tests)..."
	pytest tests/ -v -m "not slow" --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/ -v -m "integration" --tb=short

benchmark:
	@echo "Running performance benchmarks..."
	pytest tests/test_performance.py --benchmark-only --benchmark-json=reports/benchmark.json

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html

docs-serve:
	@echo "Building and serving documentation..."
	cd docs && make html && python -m http.server 8000 --directory _build/html

docs-clean:
	@echo "Cleaning documentation..."
	cd docs && make clean

# Build and Distribution
build: clean
	@echo "Building distribution packages..."
	python -m build
	@echo "Checking distribution..."
	twine check dist/*

publish-test: build
	@echo "Publishing to Test PyPI..."
	twine upload --repository testpypi dist/*

publish: build
	@echo "Publishing to PyPI..."
	twine upload dist/*

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t x13-seasonal-adjustment:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -it --rm x13-seasonal-adjustment:latest

docker-dev:
	@echo "Running development Docker container..."
	docker build --target development -t x13-seasonal-adjustment:dev .
	docker run -it --rm -p 8888:8888 -v $(PWD):/app x13-seasonal-adjustment:dev

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf reports/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean docs-clean
	@echo "Cleaning all generated files..."
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf .coverage.*
	rm -rf coverage.xml
	rm -rf bandit-report.json
	rm -rf safety-report.json
	rm -rf benchmark.json

# Development Workflow
check-all: format lint type-check security test-fast
	@echo "All checks passed!"

release-check: check-all test-all docs build
	@echo "Release checks completed successfully!"

# Continuous Integration Helpers
ci-setup:
	pip install --upgrade pip setuptools wheel
	pip install -e .[dev]

ci-test:
	pytest tests/ -v --cov=src/x13_seasonal_adjustment --cov-report=xml --junitxml=junit.xml

ci-quality:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	bandit -r src/ -f json

# Performance Monitoring
profile:
	@echo "Running performance profiler..."
	python -m cProfile -o profile_results.prof examples/basic_usage.py
	python -c "import pstats; pstats.Stats('profile_results.prof').sort_stats('cumulative').print_stats(20)"

memory-profile:
	@echo "Running memory profiler..."
	mprof run examples/basic_usage.py
	mprof plot --output memory_usage.png

# Environment Info
env-info:
	@echo "Environment Information:"
	@echo "========================"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'Not in git repo')"
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'Not in git repo')"
	@echo ""
	@echo "Installed packages:"
	pip list | grep -E "(x13|pandas|numpy|scipy|statsmodels|matplotlib|sklearn)"

# Report Generation
reports-dir:
	mkdir -p reports

generate-reports: reports-dir
	@echo "Generating all reports..."
	pytest tests/ --cov=src/x13_seasonal_adjustment --cov-report=html:reports/coverage --junitxml=reports/junit.xml
	bandit -r src/ -f json -o reports/bandit-report.json || true
	safety check --json --output reports/safety-report.json || true
	mypy src/ --html-report reports/mypy-report --ignore-missing-imports || true

# Version Management
version:
	@echo "Current version: $$(python -c 'import x13_seasonal_adjustment; print(x13_seasonal_adjustment.__version__)')"

bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

# Git Helpers
git-status:
	@echo "Git Status:"
	git status --porcelain

git-clean-check:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Working directory is not clean. Please commit or stash changes."; \
		exit 1; \
	fi

# Complete Development Workflow
dev-workflow: setup-dev format lint type-check security test-all docs
	@echo "Complete development workflow executed successfully!"
