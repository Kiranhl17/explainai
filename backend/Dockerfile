# ==============================================================================
# ExplainAI — Backend Dockerfile
# ==============================================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python deps in an isolated layer
#   Stage 2 (runtime): Copy only necessary artifacts → smaller final image
#
# Base image: python:3.11-slim (Debian bookworm)
# Final image size: ~800MB (dominated by numpy/scipy/sklearn binaries)
#
# Build:  docker build -t explainai-backend .
# Run:    docker run -p 8000:8000 -e ALLOWED_ORIGINS="https://yourfrontend.vercel.app" explainai-backend
# ==============================================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for scientific Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

# Runtime system libs (libgomp1 required by sklearn/XGBoost/LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash explainai
USER explainai
WORKDIR /home/explainai/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=explainai:explainai . .

# Create uploads directory
RUN mkdir -p uploads

# Expose FastAPI port
EXPOSE 8000

# Environment defaults (override at runtime)
ENV ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start server
# --workers 1: Single worker to share in-memory session store
# Increase to 2+ workers only if using Redis-backed sessions
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
