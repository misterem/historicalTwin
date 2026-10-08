"""Shared fixtures: a tiny synthetic index and a stand-in for the InsightFace models.

Nothing here downloads models or data, so the suite runs in seconds anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from paintmatch.faces import EMBEDDING_DIM, Face
from paintmatch.index import save_index


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def embedding_for(seed: int) -> np.ndarray:
    return unit(np.random.default_rng(seed).normal(size=EMBEDDING_DIM))


def face_row(image_id: str, face_idx: int = 0, det_score: float = 0.8) -> dict:
    return {
        "image_id": image_id,
        "image_path": f"images/0/{image_id}.jpg",
        "face_idx": face_idx,
        "x1": 0.25, "y1": 0.2, "x2": 0.75, "y2": 0.7,
        "det_score": det_score,
        "face_px": 120.0,
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A data dir with 4 faces from 3 paintings ("aaa" has two faces), plus thumbs/crops folders.

    Face embeddings are embedding_for(0..3), so a query of embedding_for(i) matches row i exactly.
    """
    faces = [face_row("aaa", 0), face_row("aaa", 1), face_row("bbb"), face_row("ccc", det_score=0.3)]
    images = [{"image_id": i, "image_path": f"images/0/{i}.jpg", "n_faces": n, "error": ""}
              for i, n in [("aaa", 2), ("bbb", 1), ("ccc", 1)]]
    embeddings = np.stack([embedding_for(i) for i in range(len(faces))])
    save_index(tmp_path / "index", embeddings, faces, images)
    for sub in ("thumbs", "crops"):
        (tmp_path / sub).mkdir()
    return tmp_path


class FakeEmbedder:
    """Stands in for paintmatch.faces.FaceEmbedder.

    The "face" found in an image comes from its top-left pixel: the red channel picks
    embedding_for(red), and a dark green channel means "no face found".
    """

    instances: list["FakeEmbedder"] = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        FakeEmbedder.instances.append(self)

    def faces(self, rgb: np.ndarray) -> list[Face]:
        red, green = int(rgb[0, 0, 0]), int(rgb[0, 0, 1])
        if green < 128:
            return []
        h, w = rgb.shape[:2]
        bbox = np.array([w * 0.25, h * 0.2, w * 0.75, h * 0.7], dtype=np.float32)
        return [Face(bbox=bbox, det_score=0.9, embedding=embedding_for(red))]


def photo_bytes(red: int = 0, face: bool = True, size: tuple[int, int] = (200, 160), fmt: str = "PNG") -> bytes:
    """An image whose top-left pixel tells FakeEmbedder which face to "find" (PNG is lossless)."""
    import io

    img = Image.new("RGB", size, (red, 255 if face else 0, 0))
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()
