"""On-disk face index: written by indexing/build_index.py, searched by the backend.

Everything lives in one file, <data_dir>/index/index.npz, so a write is a single
atomic rename and the embeddings can never get out of step with their metadata:

    embeddings        float32 (n_faces, 512), L2-normalized; row i <-> faces/* row i
    faces/<field>     one array per FACE_FIELDS entry, n_faces long
    images/<field>    one array per IMAGE_FIELDS entry, one row per painting seen

Two optional companion files, each rebuildable without re-embedding faces:
    paintings.json    artist/title/year per painting (indexing/build_metadata.py)
    appearance.npz    DINOv2 appearance embeddings per face (indexing/build_appearance.py),
                      row-aligned with index.npz and checked against it on load. With it,
                      search blends ArcFace identity and DINOv2 appearance similarity.

Thumbnails live in <data_dir>/thumbs/<image_id>.jpg and face crops in
<data_dir>/crops/<image_id>_<face_idx>.jpg.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from paintmatch.appearance import DIM as APPEARANCE_DIM
from paintmatch.appearance import MODEL_ID as APPEARANCE_MODEL_ID
from paintmatch.faces import EMBEDDING_DIM

INDEX_FILE = "index.npz"
# Optional painting metadata from indexing/build_metadata.py: {image_id: {artist, title, year, ...}}.
# Only ~3/4 of paintings have an entry; the rest are returned without these fields.
PAINTINGS_FILE = "paintings.json"
# museum/url only exist for paintings from museum open-access collections (fetch_museums.py).
PAINTING_FIELDS = ("artist", "title", "year", "museum", "url")
APPEARANCE_FILE = "appearance.npz"
# Weight of ArcFace (identity) vs DINOv2 (appearance) in the blended score. Chosen from a blind
# review where appearance-led rows won ~2:1; ArcFace keeps "this painting is actually you" hits.
DEFAULT_ARCFACE_WEIGHT = 0.3

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


def save_appearance(index_dir: Path, embeddings: np.ndarray, image_ids: list[str], face_idxs: list[int]) -> None:
    """Write appearance embeddings with the face keys they belong to, so a stale file is detectable."""
    assert len(embeddings) == len(image_ids) == len(face_idxs)
    tmp = index_dir / f".{APPEARANCE_FILE}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, embeddings=embeddings.astype(np.float32).reshape(-1, APPEARANCE_DIM),
                 image_id=np.array(image_ids, dtype=str), face_idx=np.array(face_idxs, dtype=np.int16),
                 model=np.array(APPEARANCE_MODEL_ID))
    os.replace(tmp, index_dir / APPEARANCE_FILE)


def load_appearance(index_dir: Path, faces: Columns) -> np.ndarray | None:
    """Appearance embeddings aligned with `faces`, or None if the file doesn't exist.

    Raises if the file was built for a different index or model: blending misaligned rows
    would quietly return wrong matches.
    """
    path = index_dir / APPEARANCE_FILE
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["model"]) != APPEARANCE_MODEL_ID:
            raise ValueError(f"{path} was built with {z['model']}, expected {APPEARANCE_MODEL_ID}: "
                             "re-run indexing/build_appearance.py")
        if not (np.array_equal(z["image_id"], faces["image_id"]) and np.array_equal(z["face_idx"], faces["face_idx"])):
            raise ValueError(f"{path} doesn't match {INDEX_FILE}: re-run indexing/build_appearance.py")
        return z["embeddings"]


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-8)


def blend_scores(arcface: np.ndarray, appearance: np.ndarray, arcface_weight: float) -> np.ndarray:
    """Cosine similarities live on different scales per model, so standardize each over all
    faces for this query before mixing."""
    return arcface_weight * zscore(arcface) + (1 - arcface_weight) * zscore(appearance)


class FaceIndex:
    """Brute-force search over all faces. At ~15k faces this takes a few milliseconds."""

    def __init__(self, index_dir: Path):
        self.embeddings, self.faces, self.images = load_index(index_dir)
        self.appearance = load_appearance(index_dir, self.faces)
        paintings_path = index_dir / PAINTINGS_FILE
        self.paintings: dict[str, dict] = (
            json.loads(paintings_path.read_text(encoding="utf-8")) if paintings_path.exists() else {}
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def has_appearance(self) -> bool:
        return self.appearance is not None

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        min_det_score: float = 0.0,
        one_per_image: bool = True,
        appearance_query: np.ndarray | None = None,
        arcface_weight: float = DEFAULT_ARCFACE_WEIGHT,
    ) -> list[dict]:
        """The k faces most like the query (see rank()).

        `query` is the selfie's ArcFace embedding. With an `appearance_query` (its DINOv2
        embedding) and appearance embeddings in the index, the ranking blends both; otherwise
        it's ArcFace alone. Results also carry the raw `arcface_score` and, when blending,
        `appearance_score` cosine similarities.
        """
        arcface = self.embeddings @ query.astype(np.float32)
        if appearance_query is None or self.appearance is None:
            return self.rank(arcface, k, min_det_score, one_per_image, extra_scores={"arcface_score": arcface})
        appearance = self.appearance @ appearance_query.astype(np.float32)
        return self.rank(
            blend_scores(arcface, appearance, arcface_weight), k, min_det_score, one_per_image,
            extra_scores={"arcface_score": arcface, "appearance_score": appearance},
        )

    def rank(
        self,
        scores: np.ndarray,
        k: int = 5,
        min_det_score: float = 0.0,
        one_per_image: bool = True,
        extra_scores: dict[str, np.ndarray] | None = None,
    ) -> list[dict]:
        """Turn per-face scores (higher = more similar) into the top k results as dicts:
        FACE_FIELDS, face_id, score, any `extra_scores` for that face, and PAINTING_FIELDS
        when the painting's metadata is known."""
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
            fields |= {name: float(v[face_id]) for name, v in (extra_scores or {}).items()}
            results.append({**fields, "face_id": int(face_id), "score": float(scores[face_id])})
            if len(results) == k:
                break
        return results
