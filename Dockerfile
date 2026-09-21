FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY main.py .

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_RELOAD=false \
    PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000", "--no-reload"]
