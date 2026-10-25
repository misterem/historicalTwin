"""DINOv2 appearance embeddings of face crops: the model production used before DINOv3.

Kept so review_matches.py can compare against it. A drop-in for
paintmatch.appearance.AppearanceEmbedder (same embed() interface, same 384-dim CLS token).
Paintings are embedded from the indexer's face crops, cached per crop file:

    uv run evaluation/dinov2.py          # ~15 min for every indexed face on a laptop CPU

DINOv2 is Apache-2.0 licensed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from paintmatch.appearance import embed_crops
from paintmatch.index import FaceIndex, crop_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "onnx-community/dinov2-small"
REVISION = "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
MODEL_ID = f"{REPO_ID}@{REVISION}"
DIM = 384

# From the model's preprocessor_config.json (BitImageProcessor).
RESIZE_SHORTEST_EDGE = 256
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cache_path(data_dir: Path) -> Path:
    return data_dir / "cache" / "appearance-dinov2-small.npz"


class DINOv2Embedder:
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


def painting_embeddings(embedder: DINOv2Embedder, data_dir: Path, index: FaceIndex) -> np.ndarray:
    """DINOv2 embeddings of every indexed face, row-aligned with the index."""
    crop_paths = [
        data_dir / "crops" / crop_name(image_id, int(face_idx))
        for image_id, face_idx in zip(index.faces["image_id"].tolist(), index.faces["face_idx"].tolist())
    ]
    return embed_crops(embedder, crop_paths, cache_path(data_dir), desc="dinov2 crops")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = p.parse_args()
    index = FaceIndex(args.data_dir / "index")
    embeddings = painting_embeddings(DINOv2Embedder(), args.data_dir, index)
    print(f"{len(embeddings)} DINOv2 embeddings -> {cache_path(args.data_dir)}")


if __name__ == "__main__":
    main()
