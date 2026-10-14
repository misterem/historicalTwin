"""Embed every indexed painting face with DINOv2 for appearance-blended matching.

Reads the face crops written by build_index.py (data/crops/<image_id>_<face_idx>.jpg)
and writes data/index/appearance.npz, row-aligned with index.npz. The backend blends
these with ArcFace scores (see FaceIndex.search); without the file it uses ArcFace only.

Re-run after build_index.py adds or changes faces. Embeddings are cached per crop file,
so a re-run only embeds new crops.

    uv run indexing/build_appearance.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paintmatch.appearance import AppearanceEmbedder, embed_crops
from paintmatch.index import APPEARANCE_FILE, crop_name, load_index, save_appearance

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--cache", type=Path, help="per-crop embedding cache (default: <data-dir>/cache/...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = args.data_dir / "index"
    _, faces, _ = load_index(index_dir)
    image_ids = faces["image_id"].tolist()
    face_idxs = faces["face_idx"].tolist()
    crop_paths = [args.data_dir / "crops" / crop_name(i, f) for i, f in zip(image_ids, face_idxs)]
    if missing := [p for p in crop_paths if not p.exists()]:
        raise SystemExit(f"{len(missing)} face crops are missing (e.g. {missing[0].name}): re-run build_index.py")

    cache = args.cache or args.data_dir / "cache" / "appearance-dinov2-small.npz"
    embeddings = embed_crops(AppearanceEmbedder(), crop_paths, cache)
    save_appearance(index_dir, embeddings, image_ids, face_idxs)
    print(f"{len(embeddings)} appearance embeddings -> {index_dir / APPEARANCE_FILE}")


if __name__ == "__main__":
    main()
