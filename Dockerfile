# ── Axiom backend: FastAPI + ML pipeline ─────────────────────────────────────
# Builds a lean, ARM/x86-portable image that runs the web API (api.py) on :8000.
FROM python:3.11-slim

# MPLBACKEND=Agg → matplotlib renders charts headlessly (no display).
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# libgomp1 is the OpenMP runtime XGBoost and LightGBM link against; without it
# they fail to import. Everything else ships as self-contained wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached across code changes.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Backend source only (frontend lives in its own image).
COPY api.py database.py main.py ./
COPY core ./core
COPY agents ./agents
COPY visualization ./visualization
COPY configs ./configs

# Writable runtime dirs (also provided as volumes in docker-compose for persistence).
RUN mkdir -p data/uploads artifacts reports logs

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
