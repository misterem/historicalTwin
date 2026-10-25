"""Prototype face features for matching: a tighter DINOv3 crop and a skin-tone measure.

Builds both for every face in the index and caches them, so review_matches.py can try them
as extra blend terms:

    uv run evaluation/face_features.py                 # ~15 min, resumable from the cache
    uv run evaluation/face_features.py --margin 0.1

Skin tone is the Individual Typology Angle (ITA) used in dermatology, measured on a patch of
cheeks and nose inside the detector's box: ITA = atan2(L* - 50, b*) in degrees. Higher is
lighter. Paintings and photos are measured the same way, so compare them on equal terms only
after checking their distributions (varnish yellows paintings, which lowers ITA).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paintmatch.appearance import CROP_SIZE, DIM, MODEL_ID, AppearanceEmbedder
from paintmatch.faces import face_crop, load_image
from paintmatch.index import FaceIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
TIGHT_MARGIN = 0.1  # face_crop's default (the crops in data/crops) is 0.3
# Cheeks and nose, as fractions of the detector box: below the eyes, above the mouth.
SKIN_BOX = (0.2, 0.45, 0.8, 0.72)
# Below this chroma (sqrt(a*^2 + b*^2)) the face is effectively monochrome: no skin-tone reading.
MIN_CHROMA = 5.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> CIELAB (D65)."""
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def skin_lab(rgb: np.ndarray, bbox) -> np.ndarray:
    """Median L*a*b* of the cheek/nose patch, ignoring the darkest and brightest 15% of pixels
    (nostrils, shadows, specular highlights)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    bx1, by1, bx2, by2 = SKIN_BOX
    H, W = rgb.shape[:2]
    patch = rgb[max(int(y1 + by1 * h), 0): min(int(y1 + by2 * h), H), max(int(x1 + bx1 * w), 0): min(int(x1 + bx2 * w), W)]
    if patch.size == 0:
        return np.full(3, np.nan, np.float32)
    lab = srgb_to_lab(patch.reshape(-1, 3))
    lo, hi = np.percentile(lab[:, 0], [15, 85])
    keep = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    return np.median(keep if len(keep) else lab, axis=0).astype(np.float32)


def ita(lab: np.ndarray) -> np.ndarray:
    """Individual Typology Angle in degrees; NaN for monochrome faces (grayscale photos, grisaille)."""
    value = np.degrees(np.arctan2(lab[..., 0] - 50, lab[..., 2]))
    return np.where(np.hypot(lab[..., 1], lab[..., 2]) < MIN_CHROMA, np.nan, value)


def cache_path(data_dir: Path, margin: float) -> Path:
    return data_dir / "cache" / f"face-features-m{margin:g}.npz"


def load_features(data_dir: Path, index: FaceIndex, margin: float = TIGHT_MARGIN) -> dict[str, np.ndarray]:
    """Cached features, row-aligned with the index. Raises if the cache is stale or missing."""
    with np.load(cache_path(data_dir, margin)) as z:
        keys = [f"{i}_{j}" for i, j in zip(index.faces["image_id"].tolist(), index.faces["face_idx"].tolist())]
        if z["keys"].tolist() != keys or str(z["model"]) != MODEL_ID:
            raise ValueError("face feature cache doesn't match the index; re-run evaluation/face_features.py")
        return {"tight": z["tight"], "lab": z["lab"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--margin", type=float, default=TIGHT_MARGIN, help="crop margin for the tight DINOv3 crop")
    p.add_argument("--max-side", type=int, default=1280, help="must match build_index.py's --max-side")
    args = p.parse_args()

    index = FaceIndex(args.data_dir / "index")
    faces = index.faces
    by_image: dict[str, list[int]] = {}
    for row, path in enumerate(faces["image_path"].tolist()):
        by_image.setdefault(path, []).append(row)

    embedder = AppearanceEmbedder()
    tight = np.zeros((len(index), DIM), np.float32)
    lab = np.zeros((len(index), 3), np.float32)

    def load(path: str):
        rgb = load_image(args.data_dir / "raw" / path, max_side=args.max_side)
        H, W = rgb.shape[:2]
        out = []
        for row in by_image[path]:
            bbox = np.array([faces["x1"][row] * W, faces["y1"][row] * H, faces["x2"][row] * W, faces["y2"][row] * H])
            out.append((row, face_crop(rgb, bbox, CROP_SIZE, margin=args.margin), skin_lab(rgb, bbox)))
        return out

    pending: list[tuple[int, object]] = []

    def flush():
        rows = [r for r, _ in pending]
        tight[rows] = embedder.embed([c for _, c in pending])
        pending.clear()

    with ThreadPoolExecutor(6) as pool:
        for items in tqdm(pool.map(load, by_image), total=len(by_image), unit="painting"):
            for row, crop, skin in items:
                lab[row] = skin
                pending.append((row, crop))
            if len(pending) >= 128:
                flush()
    if pending:
        flush()

    out = cache_path(args.data_dir, args.margin)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array([f"{i}_{j}" for i, j in zip(faces["image_id"].tolist(), faces["face_idx"].tolist())])
    np.savez(out, tight=tight, lab=lab, keys=keys, model=np.array(MODEL_ID), margin=np.array(args.margin))
    print(f"wrote {out}: {len(index)} faces, ITA median {np.nanmedian(ita(lab)):.1f}")


if __name__ == "__main__":
    main()
