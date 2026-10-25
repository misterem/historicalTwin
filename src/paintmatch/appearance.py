"""DINOv3 "appearance" embeddings of face crops: overall look (hair, colors, pose,
lighting), as opposed to ArcFace's identity features.

Matching blends the two (see FaceIndex.search). In a blind review of 28 portraits,
rows using DINOv2 appearance were preferred ~2:1 over ArcFace alone, while ArcFace still
won when the actual person had been painted. A second blind review of the same photos
then preferred DINOv3 over DINOv2 (18 vs 8 picks, data/reviews/dinov3/), and DINOv3-small
is also faster on CPU (~38 vs ~65 ms per face). The previous DINOv2 model lives on in
evaluation/dinov2.py for comparisons. Paintings and selfies are both embedded from a
face_crop(..., size=CROP_SIZE) with the default margin, so they're framed alike.

The weights are under Meta's DINOv3 License, not Apache-2.0 like DINOv2's: royalty-free and
commercial use is allowed, but with use restrictions (trade controls, military uses). The
facebook/ repo is gated; the onnx-community export used here isn't.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ID = "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX"
REVISION = "48988dfe73065df8d6f5ccc0edc7c8bcf307de41"
# Stored next to painting embeddings, so a model change can't silently mix incompatible vectors.
MODEL_ID = f"{REPO_ID}@{REVISION}"
DIM = 384
# Per-crop embedding cache under <data-dir>/cache/. Named for the model, so switching models
# never reuses another model's vectors.
CACHE_FILE = "appearance-dinov3-vits16.npz"

# From the model's preprocessor_config.json (DINOv3ViTImageProcessor): a plain bilinear
# resize to 224x224, no center crop. Face crops are already square, so nothing is cut off.
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def model_dir() -> Path:
    """Where the model is downloaded to: under HF_HOME, which the Dockerfiles bake into the image."""
    from huggingface_hub import constants

    return Path(constants.HF_HOME) / "paintmatch" / f"dinov3-vits16-{REVISION[:12]}"


class AppearanceEmbedder:
    def __init__(self, providers: list[str] | None = None):
        import onnxruntime as ort

        # The weights are in an external-data file next to model.onnx. onnxruntime resolves
        # symlinks and refuses external data outside the model's folder, which rules out the HF
        # cache (its files are symlinks into a shared blob store), so the files are downloaded
        # as real files into a folder of our own. Once there, nothing touches the network, so
        # this works with HF_HUB_OFFLINE=1.
        folder = model_dir()
        files = [folder / "onnx" / "model.onnx", folder / "onnx" / "model.onnx_data"]
        if not all(f.exists() for f in files):
            from huggingface_hub import snapshot_download

            snapshot_download(REPO_ID, revision=REVISION, local_dir=folder,
                              allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data", "LICENSE.md"])
        self._session = ort.InferenceSession(str(files[0]), providers=providers or ["CPUExecutionProvider"])

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
        return pixels.transpose(2, 0, 1)  # HWC -> CHW

    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """L2-normalized (n, 384) embeddings: the CLS token of DINOv3's last layer, after its
        norm (the model's pooler_output)."""
        out = []
        for start in range(0, len(images), batch_size):
            batch = np.stack([self.preprocess(im) for im in images[start : start + batch_size]])
            out.append(self._session.run(["pooler_output"], {"pixel_values": batch})[0])
        emb = np.concatenate(out) if out else np.empty((0, DIM), np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def embed_crops(embedder, crop_paths: list[Path], cache_file: Path, desc: str = "dinov3 crops") -> np.ndarray:
    """Embed painting face crops, reusing a cache keyed by crop filename.

    `embedder` is anything with AppearanceEmbedder's embed(); give each model its own cache_file.
    """
    cached: dict[str, np.ndarray] = {}
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as z:
            cached = dict(zip(z["names"].tolist(), z["embeddings"]))

    missing = [p for p in crop_paths if p.name not in cached]
    if missing:
        from tqdm import tqdm

        for start in tqdm(range(0, len(missing), 256), desc=desc, unit="batch"):
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
