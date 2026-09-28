FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and frontend
COPY app/ ./app/
COPY static/ ./static/

# Run with uvicorn
EXPOSE 8000
# One worker: paper trades and price data live in per-process memory (2 workers split state across requests), and 2 workers risk exceeding the 512MB free instance.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
