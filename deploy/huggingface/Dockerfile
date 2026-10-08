# Hugging Face Space image: the matching API plus everything it serves. Unlike the
# root Dockerfile, the face index and the painting thumbnails/crops are baked in,
# downloaded at build time from the assets dataset at an exact commit.
# deploy/publish.py fills in ASSETS_REPO and ASSETS_REVISION and pushes this file
# to the Space, which then builds it on Hugging Face's servers.

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
# insightface compiles a small C++ extension at install time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Bake in the face models (only the detector and recognizer are used).
ENV PAINTMATCH_MODEL_ROOT=/app/insightface
RUN .venv/bin/python -c "from paintmatch.faces import FaceEmbedder; FaceEmbedder()" \
    && rm -f /app/insightface/models/buffalo_l.zip \
             /app/insightface/models/buffalo_l/1k3d68.onnx \
             /app/insightface/models/buffalo_l/2d106det.onnx \
             /app/insightface/models/buffalo_l/genderage.onnx

# Face index + painting images, pinned to an exact dataset commit.
ARG ASSETS_REPO=__ASSETS_REPO__
ARG ASSETS_REVISION=__ASSETS_REVISION__
RUN .venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], repo_type='dataset', revision=sys.argv[2], local_dir='/app/data', \
allow_patterns=['index/*', 'thumbs/*', 'crops/*'], max_workers=16)" "$ASSETS_REPO" "$ASSETS_REVISION" \
    && rm -rf /app/data/.cache


FROM python:3.11-slim
# Spaces run containers as uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/insightface /app/insightface
COPY --from=builder /app/data /app/data
COPY backend ./backend

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PAINTMATCH_MODEL_ROOT=/app/insightface \
    PAINTMATCH_DATA_DIR=/app/data \
    PORT=8080

USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
