# SupplySense application image.
# This container runs the Python pipelines / (later) Streamlit app.
# PostgreSQL runs as a separate service (see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

# System deps needed by psycopg2 and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default command: run the Phase 1 data pipeline. Overridden in
# docker-compose for the Streamlit app service in later phases.
CMD ["python", "-m", "scripts.run_phase1_pipeline"]
