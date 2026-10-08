"""Face detection + embedding, shared by the indexer and the web backend.

Both sides MUST go through this module so that paintings and selfies are
decoded, detected and embedded identically.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

EMBEDDING_DIM = 512

# insightface 0.7.3 calls a scikit-image API that is deprecated but still works.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.")


@dataclass
class Face:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in pixels of the image that was searched
    det_score: float
    embedding: np.ndarray  # (512,) float32, L2-normalized

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def area(self) -> float:
        return self.width * float(self.bbox[3] - self.bbox[1])


def load_image(source: str | BinaryIO, max_side: int | None = None) -> np.ndarray:
    """Decode an image to an RGB uint8 array, respecting EXIF rotation.

    If max_side is given, the image is downscaled so its longest side is at
    most max_side. For JPEGs this happens during decode, which is much faster
    than decoding a full-size scan and resizing afterwards.
    """
    with Image.open(source) as im:
        if max_side:
            im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(im)


def face_crop(rgb: np.ndarray, bbox: np.ndarray, size: int, margin: float = 0.3) -> Image.Image:
    """Square crop around a face, padded by `margin` of the face size on each side."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, w, h)
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    box = tuple(int(round(v)) for v in (left, top, left + side, top + side))
    crop = Image.fromarray(rgb).crop(box)
    return crop.resize((size, size), Image.Resampling.LANCZOS)


class FaceEmbedder:
    """InsightFace detector (SCRFD) + ArcFace recognizer.

    Note: the pretrained `buffalo_l` weights are licensed for non-commercial
    research use only. Swap the model before any commercial use.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        det_thresh: float = 0.5,
        providers: list[str] | None = None,
        model_root: str | None = None,
    ):
        from insightface.app import FaceAnalysis

        # Weights are downloaded to <model_root>/models/<model_name> on first use.
        self._app = FaceAnalysis(
            name=model_name,
            root=model_root or os.environ.get("PAINTMATCH_MODEL_ROOT", "~/.insightface"),
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)

    def faces(self, rgb: np.ndarray) -> list[Face]:
        """Detect and embed every face in an RGB image, largest face first."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # insightface expects OpenCV BGR
        faces = [
            Face(
                bbox=f.bbox.astype(np.float32),
                det_score=float(f.det_score),
                embedding=f.normed_embedding.astype(np.float32),
            )
            for f in self._app.get(bgr)
        ]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces
