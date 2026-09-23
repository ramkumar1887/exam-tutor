# Multi-stage production Dockerfile for Exam Tutor AI
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TUTOR_DB_PATH=/app/data/sessions/knowledge.db

WORKDIR /app

# Install system dependencies (build-essential for C-extensions, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create data directories with appropriate permissions
RUN mkdir -p /app/data/sessions /app/data/uploads /app/data/reports

# Create non-root user for security
RUN useradd -m -u 1000 tutoruser && \
    chown -R tutoruser:tutoruser /app
USER tutoruser

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default start FastAPI REST API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
