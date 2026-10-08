"""On-disk face index: written by indexing/build_index.py, searched by the backend.

Everything lives in one file, <data_dir>/index/index.npz, so a write is a single
atomic rename and the embeddings can never get out of step with their metadata:

    embeddings        float32 (n_faces, 512), L2-normalized; row i <-> faces/* row i
    faces/<field>     one array per FACE_FIELDS entry, n_faces long
    images/<field>    one array per IMAGE_FIELDS entry, one row per painting seen

Painting metadata (artist/title/year) is kept separately in paintings.json, written by
indexing/build_metadata.py, so it can be rebuilt without re-embedding faces.

Thumbnails live in <data_dir>/thumbs/<image_id>.jpg and face crops in
<data_dir>/crops/<image_id>_<face_idx>.jpg.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from paintmatch.faces import EMBEDDING_DIM

INDEX_FILE = "index.npz"
# Optional painting metadata from indexing/build_metadata.py: {image_id: {artist, title, year, ...}}.
# Only ~3/4 of paintings have an entry; the rest are returned without these fields.
PAINTINGS_FILE = "paintings.json"
PAINTING_FIELDS = ("artist", "title", "year")

FACE_FIELDS = {
    "image_id": str,
    "image_path": str,  # repo-relative, e.g. images/0/<image_id>.jpg
    "face_idx": np.int16,  # 0 = largest face in the painting
    "x1": np.float32,  # bbox, 0-1 relative to the image
    "y1": np.float32,
    "x2": np.float32,
    "y2": np.float32,
    "det_score": np.float32,
    "face_px": np.float32,  # face width in the decoded image
}
IMAGE_FIELDS = {
    "image_id": str,
    "image_path": str,
    "n_faces": np.int16,
    "error": str,  # "" if the painting was indexed successfully
}

Columns = dict[str, np.ndarray]


def thumb_name(image_id: str) -> str:
    return f"{image_id}.jpg"


def crop_name(image_id: str, face_idx: int) -> str:
    return f"{image_id}_{face_idx}.jpg"


def rows_to_columns(rows: list[dict], fields: dict) -> Columns:
    return {name: np.array([r[name] for r in rows], dtype=dtype) for name, dtype in fields.items()}


def columns_to_rows(columns: Columns) -> list[dict]:
    n = len(next(iter(columns.values())))
    return [{name: col[i].item() for name, col in columns.items()} for i in range(n)]


def save_index(index_dir: Path, embeddings: np.ndarray, face_rows: list[dict], image_rows: list[dict]) -> None:
    assert len(embeddings) == len(face_rows), (len(embeddings), len(face_rows))
    arrays = {"embeddings": embeddings.astype(np.float32).reshape(-1, EMBEDDING_DIM)}
    arrays |= {f"faces/{k}": v for k, v in rows_to_columns(face_rows, FACE_FIELDS).items()}
    arrays |= {f"images/{k}": v for k, v in rows_to_columns(image_rows, IMAGE_FIELDS).items()}
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_dir / f".{INDEX_FILE}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, index_dir / INDEX_FILE)


def load_index(index_dir: Path) -> tuple[np.ndarray, Columns, Columns]:
    with np.load(index_dir / INDEX_FILE, allow_pickle=False) as z:
        embeddings = z["embeddings"]
        faces = {name: z[f"faces/{name}"] for name in FACE_FIELDS}
        images = {name: z[f"images/{name}"] for name in IMAGE_FIELDS}
    assert all(len(col) == len(embeddings) for col in faces.values())
    return embeddings, faces, images


class FaceIndex:
    """Brute-force cosine search. At ~15k faces this takes a few milliseconds."""

    def __init__(self, index_dir: Path):
        self.embeddings, self.faces, self.images = load_index(index_dir)
        paintings_path = index_dir / PAINTINGS_FILE
        self.paintings: dict[str, dict] = (
            json.loads(paintings_path.read_text(encoding="utf-8")) if paintings_path.exists() else {}
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        min_det_score: float = 0.0,
        one_per_image: bool = True,
    ) -> list[dict]:
        """Return the k most similar faces as dicts: FACE_FIELDS, face_id, score, and
        PAINTING_FIELDS when the painting's metadata is known."""
        scores = self.embeddings @ query.astype(np.float32)
        if min_det_score > 0:
            scores = np.where(self.faces["det_score"] >= min_det_score, scores, -np.inf)

        image_ids = self.faces["image_id"]
        results: list[dict] = []
        seen_images: set[str] = set()
        for face_id in np.argsort(-scores):
            if not np.isfinite(scores[face_id]):
                break
            if one_per_image and image_ids[face_id] in seen_images:
                continue
            seen_images.add(image_ids[face_id])
            fields = {name: col[face_id].item() for name, col in self.faces.items()}
            painting = self.paintings.get(fields["image_id"], {})
            fields |= {k: painting[k] for k in PAINTING_FIELDS if painting.get(k)}
            results.append({**fields, "face_id": int(face_id), "score": float(scores[face_id])})
            if len(results) == k:
                break
        return results
