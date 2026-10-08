"""DINOv2 "appearance" embeddings of face crops: overall look (hair, colors, pose,
lighting), as opposed to ArcFace's identity features. Evaluation-only for now; it moves
into src/paintmatch if a blended score wins the match-quality review.

DINOv2 is Apache-2.0 licensed, so unlike the InsightFace weights it's usable commercially.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ID = "onnx-community/dinov2-small"
REVISION = "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
DIM = 384

# From the model's preprocessor_config.json (BitImageProcessor).
RESIZE_SHORTEST_EDGE = 256
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AppearanceEmbedder:
    def __init__(self, providers: list[str] | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO_ID, "onnx/model.onnx", revision=REVISION)
        self._session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        scale = RESIZE_SHORTEST_EDGE / min(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC
        )
        left = (image.width - CROP_SIZE) // 2
        top = (image.height - CROP_SIZE) // 2
        image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
        return pixels.transpose(2, 0, 1)  # HWC -> CHW

    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """L2-normalized (n, 384) embeddings: the CLS token of DINOv2's last layer."""
        out = []
        for start in range(0, len(images), batch_size):
            batch = np.stack([self.preprocess(im) for im in images[start : start + batch_size]])
            hidden = self._session.run(["last_hidden_state"], {"pixel_values": batch})[0]
            out.append(hidden[:, 0])
        emb = np.concatenate(out) if out else np.empty((0, DIM), np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def embed_crops(embedder: AppearanceEmbedder, crop_paths: list[Path], cache_file: Path) -> np.ndarray:
    """Embed painting face crops, reusing a cache keyed by crop filename."""
    cached: dict[str, np.ndarray] = {}
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as z:
            cached = dict(zip(z["names"].tolist(), z["embeddings"]))

    missing = [p for p in crop_paths if p.name not in cached]
    if missing:
        from tqdm import tqdm

        for start in tqdm(range(0, len(missing), 256), desc="dinov2 crops", unit="batch"):
            chunk = missing[start : start + 256]
            images = []
            for p in chunk:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            for p, e in zip(chunk, embedder.embed(images)):
                cached[p.name] = e
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(cached)
        tmp = cache_file.with_name(f".{cache_file.name}.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, names=np.array(names), embeddings=np.stack([cached[n] for n in names]))
        tmp.replace(cache_file)

    return np.stack([cached[p.name] for p in crop_paths])
