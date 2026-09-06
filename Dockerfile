# Backend image: FastAPI + the ML/talk-to-data pipeline (src/, api/).
# libgomp1 is required by LightGBM's OpenMP-based training/inference.
# Pinned to 3.12, not 3.13: numpy==1.26.4 has no prebuilt wheel for 3.13,
# which forces a source build that fails without compiler tooling in the
# slim image. 3.12 has full wheel support for every pin in requirements.txt.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY documents/ documents/
COPY data/sample/ data/sample/

EXPOSE 8000

# data/sample (the committed demo dataset) ships baked into the image so
# the container is self-sufficient even without the ./data volume mount
# docker-compose.yml adds, see src/data/db.py's fallback chain.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
