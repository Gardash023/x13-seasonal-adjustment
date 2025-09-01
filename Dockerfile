# Multi-stage build for X13 Seasonal Adjustment
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create and set work directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install the package and dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e . && \
    pip install pytest pytest-cov

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r x13user && useradd -r -g x13user x13user

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy built packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /opt/venv/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /opt/venv/bin
COPY --from=builder /build/src /app/src

# Set work directory
WORKDIR /app

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R x13user:x13user /app /opt/venv

# Switch to non-root user
USER x13user

# Install the package in the virtual environment
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from x13_seasonal_adjustment import X13SeasonalAdjustment; print('OK')"

# Default command
CMD ["python", "-c", "from x13_seasonal_adjustment import X13SeasonalAdjustment; print('X13 Seasonal Adjustment is ready!')"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-benchmark \
    black \
    flake8 \
    mypy \
    isort \
    bandit \
    jupyter \
    ipython

# Install pre-commit
RUN pip install pre-commit

# Switch back to non-root user
USER x13user

# Expose port for Jupyter
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Labels for metadata
LABEL maintainer="Gardash Abbasov <gardash.abbasov@gmail.com>" \
      version="0.1.3" \
      description="X13-ARIMA-SEATS Seasonal Adjustment for Python" \
      org.opencontainers.image.source="https://github.com/Gardash023/x13-seasonal-adjustment" \
      org.opencontainers.image.documentation="https://x13-seasonal-adjustment.readthedocs.io" \
      org.opencontainers.image.licenses="MIT"
