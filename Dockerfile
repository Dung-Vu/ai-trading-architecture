# =============================================================================
# Stage 1: Builder — install build deps and compile wheels
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages into a custom prefix so we can copy only what's needed
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# Stage 2: Runtime — slim production image
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime-only system deps (curl for healthchecks, tini for signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Python defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports (8501 = Streamlit dashboard, 9000 = QuestDB web console proxy)
EXPOSE 8501 9000

# Use tini as entrypoint for proper signal handling
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "src.main"]
