# Multi-stage Dockerfile for Kaggle Agents
# Optimized for size and build speed

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime - Final lightweight image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY kaggle_agents/ ./kaggle_agents/
COPY examples/ ./examples/
COPY README.md ./

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KAGGLE_WORK_DIR=/app/data

# Create directories for data persistence
RUN mkdir -p /app/data /app/models /app/logs /app/.cache && \
    chmod -R 777 /app/data /app/models /app/logs /app/.cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import kaggle_agents; print('healthy')" || exit 1

# Default command - run CLI help
CMD ["python", "-m", "kaggle_agents.cli", "--help"]
