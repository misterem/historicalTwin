"""Matching API: POST a selfie, get back the most similar portrait paintings.

    uv run uvicorn backend.app:app --reload     # then open http://localhost:8000

Selfies are processed in memory only and are never written to disk.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from paintmatch.faces import FaceEmbedder, load_image
from paintmatch.index import FaceIndex, crop_name, thumb_name

DATA_DIR = Path(os.environ.get("PAINTMATCH_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
# Public base URL of the thumbs/ and crops/ folders (e.g. the R2 bucket's domain).
# Unset in local dev, where this server serves them from DATA_DIR instead.
IMAGE_BASE_URL = os.environ.get("PAINTMATCH_IMAGE_BASE_URL", "").rstrip("/")
# Comma-separated frontend origins allowed to call the API, e.g. "https://paintmatch.app".
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("PAINTMATCH_ALLOWED_ORIGINS", "").split(",") if o.strip()]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
SELFIE_MAX_SIDE = 1280
# Detector input size for selfies. 320 is ~1.7x faster than 640 but landmarks are a bit
# less precise, which shifts similarity scores slightly (0.975 -> 0.948 on a self-match).
SELFIE_DET_SIZE = int(os.environ.get("PAINTMATCH_SELFIE_DET_SIZE", "640"))

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["embedder"] = FaceEmbedder(det_size=SELFIE_DET_SIZE)
    state["index"] = FaceIndex(DATA_DIR / "index")
    yield
    state.clear()


app = FastAPI(title="paintmatch", lifespan=lifespan)
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"])


@app.get("/health")
def health() -> dict:
    return {"ok": True, "faces_indexed": len(state["index"])}


@app.post("/match")
def match(file: UploadFile, k: int = 6) -> dict:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    try:
        rgb = load_image(io.BytesIO(data), max_side=SELFIE_MAX_SIDE)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(400, "Could not read that image")

    faces = state["embedder"].faces(rgb)
    if not faces:
        raise HTTPException(422, "No face found in the photo")
    selfie = faces[0]  # largest face

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = selfie.bbox.tolist()
    matches = state["index"].search(selfie.embedding, k=max(1, min(k, 50)))
    return {
        "selfie_face": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
        "matches": [
            {
                **m,
                "thumb_url": f"{IMAGE_BASE_URL}/thumbs/{thumb_name(m['image_id'])}",
                "crop_url": f"{IMAGE_BASE_URL}/crops/{crop_name(m['image_id'], m['face_idx'])}",
            }
            for m in matches
        ],
    }


if not IMAGE_BASE_URL:
    if not (DATA_DIR / "thumbs").is_dir():
        raise RuntimeError(
            f"{DATA_DIR / 'thumbs'} not found: set PAINTMATCH_IMAGE_BASE_URL to where the images are hosted "
            "(the Docker image doesn't include them), or run indexing/build_index.py first"
        )
    app.mount("/thumbs", StaticFiles(directory=DATA_DIR / "thumbs"), name="thumbs")
    app.mount("/crops", StaticFiles(directory=DATA_DIR / "crops"), name="crops")
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
