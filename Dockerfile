# API server image: FastAPI + face models + the face index, CPU only.
# Thumbnails/crops are NOT included; they're served from R2 (see indexing/upload_to_r2.py).
#
#   uv run indexing/build_index.py          # data/index must exist before building
#   docker build -t paintmatch-api .
#   docker run -p 8080:8080 -e PAINTMATCH_IMAGE_BASE_URL=https://images.example.com paintmatch-api

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
# insightface compiles a small C++ extension at install time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Bake the face models into the image so containers start without a ~280 MB download.
# Only the detector and the recognizer are used; drop the rest of the model pack.
ENV PAINTMATCH_MODEL_ROOT=/app/insightface
RUN .venv/bin/python -c "from paintmatch.faces import FaceEmbedder; FaceEmbedder()" \
    && rm -f /app/insightface/models/buffalo_l.zip \
             /app/insightface/models/buffalo_l/1k3d68.onnx \
             /app/insightface/models/buffalo_l/2d106det.onnx \
             /app/insightface/models/buffalo_l/genderage.onnx


FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/insightface /app/insightface
COPY backend ./backend
COPY data/index ./data/index

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PAINTMATCH_MODEL_ROOT=/app/insightface \
    PAINTMATCH_DATA_DIR=/app/data \
    PORT=8080

USER app
EXPOSE 8080
# Cloud Run / Fly.io / Render set $PORT; default 8080.
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
