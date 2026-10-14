"""Build a side-by-side page for judging match quality.

Runs every photo in a folder through several matching variants and writes a single
HTML page. For each photo you pick the variant whose matches look best. Variant names
stay hidden and rows are shuffled until you click "Reveal", so the comparison is blind.

Variants:
  arcface         ArcFace identity similarity, detector input 640 (what the app ships)
  arcface-det320  the same with detector input 320 (~1.7x faster API)
  dinov2          DINOv2 appearance similarity of the face crop (hair, colors, pose)
  blend           alpha * arcface + (1 - alpha) * dinov2, each z-scored per photo
  blend-tight     the blend with DINOv2 on a tighter face crop (less hair, background, clothes)
  blend-skin      the blend plus --skin-weight * skin-tone closeness (ITA, see face_features.py)
  blend-tight-skin  both

The last three need the feature cache from evaluation/face_features.py.

The page embeds small crops of your photos' faces and lives under data/ (gitignored).
Don't publish it.

    uv run evaluation/review_matches.py ~/Pictures/selfies
    uv run evaluation/review_matches.py ~/Pictures/selfies --variants arcface,blend --alpha 0.6
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from paintmatch.appearance import CROP_SIZE, AppearanceEmbedder, embed_crops
from paintmatch.faces import FaceEmbedder, face_crop, load_image
from paintmatch.index import DEFAULT_ARCFACE_WEIGHT, FaceIndex, blend_scores, crop_name, thumb_name, zscore

from face_features import TIGHT_MARGIN, ita, load_features, skin_lab

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "report_template.html"
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VARIANTS = {
    "arcface": "ArcFace, detector 640 (current)",
    "arcface-det320": "ArcFace, detector 320",
    "dinov2": "DINOv2 appearance",
    "blend": "Blend: ArcFace + DINOv2 (production)",
    "blend-tight": "Blend, tight DINOv2 crop",
    "blend-skin": "Blend + skin tone",
    "blend-tight-skin": "Blend, tight DINOv2 crop + skin tone",
}
DEFAULT_SKIN_WEIGHT = 0.5


def skin_closeness(query_ita: float, painting_ita: np.ndarray) -> np.ndarray:
    """z-scored closeness of skin tone; 0 (neutral) where either side has no reading."""
    if np.isnan(query_ita):
        return np.zeros(len(painting_ita), np.float32)
    closeness = -np.abs(painting_ita - query_ita)
    return np.nan_to_num(zscore(np.where(np.isnan(closeness), np.nanmean(closeness), closeness)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photos", type=Path, help="folder of selfies / photos of people")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of: {', '.join(VARIANTS)}")
    p.add_argument("--alpha", type=float, default=DEFAULT_ARCFACE_WEIGHT, help="ArcFace weight in the blend (0-1)")
    p.add_argument("--skin-weight", type=float, default=DEFAULT_SKIN_WEIGHT, help="skin-tone weight in the *-skin variants")
    p.add_argument("--k", type=int, default=5, help="matches shown per variant")
    p.add_argument("--out", type=Path, help="output folder (default: data/reviews/<photos>-<timestamp>)")
    return p.parse_args()


def jpeg_data_uri(image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if unknown := set(variants) - set(VARIANTS):
        sys.exit(f"unknown variants: {', '.join(sorted(unknown))}")
    needs_dinov2 = any(v.startswith(("dinov2", "blend")) for v in variants)
    needs_features = any(v.startswith("blend-") for v in variants)

    photos = sorted(p for p in args.photos.rglob("*") if p.suffix.lower() in PHOTO_SUFFIXES)
    if not photos:
        sys.exit(f"no .jpg/.png/.webp photos found in {args.photos}")

    stamp = time.strftime("%Y%m%d-%H%M")
    out_dir = args.out or args.data_dir / "reviews" / f"{args.photos.resolve().name}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = FaceIndex(args.data_dir / "index")
    crops_dir, thumbs_dir = args.data_dir / "crops", args.data_dir / "thumbs"
    print(f"{len(photos)} photos, {len(index)} painting faces, variants: {', '.join(variants)}")

    detector = FaceEmbedder(det_size=640)
    detector_320 = FaceEmbedder(det_size=320) if "arcface-det320" in variants else None
    appearance = painting_appearance = None
    if needs_dinov2:
        appearance = AppearanceEmbedder()
        crop_paths = [
            crops_dir / crop_name(image_id, int(face_idx))
            for image_id, face_idx in zip(index.faces["image_id"].tolist(), index.faces["face_idx"].tolist())
        ]
        painting_appearance = (
            index.appearance if index.has_appearance
            else embed_crops(appearance, crop_paths, args.data_dir / "cache" / "appearance-dinov2-small.npz")
        )

    features = load_features(args.data_dir, index) if needs_features else None
    painting_ita = ita(features["lab"]) if features else None

    def to_entry(match: dict) -> dict:
        caption = " · ".join(str(match[k]) for k in ("title", "artist", "year") if match.get(k))
        return {
            "crop": os.path.relpath(crops_dir / crop_name(match["image_id"], match["face_idx"]), out_dir),
            "thumb": os.path.relpath(thumbs_dir / thumb_name(match["image_id"]), out_dir),
            "score": round(match["score"], 3),
            "caption": caption,
        }

    entries = []
    for photo in photos:
        entry: dict = {"name": photo.name, "face": None, "error": None, "results": {}}
        entries.append(entry)
        try:
            rgb = load_image(photo, max_side=1280)
        except Exception as e:
            entry["error"] = f"couldn't read image ({e.__class__.__name__})"
            continue
        faces = detector.faces(rgb)
        if not faces:
            entry["error"] = "no face found"
            continue
        face = faces[0]
        entry["face"] = jpeg_data_uri(face_crop(rgb, face.bbox, 160))

        arcface = index.embeddings @ face.embedding
        dino = None
        if needs_dinov2:
            dino = painting_appearance @ appearance.embed([face_crop(rgb, face.bbox, CROP_SIZE)])[0]
        if needs_features:
            tight = features["tight"] @ appearance.embed([face_crop(rgb, face.bbox, CROP_SIZE, margin=TIGHT_MARGIN)])[0]
            skin = skin_closeness(float(ita(skin_lab(rgb, face.bbox))), painting_ita)

        for variant in variants:
            if variant == "arcface":
                scores = arcface
            elif variant == "arcface-det320":
                faces_320 = detector_320.faces(rgb)
                if not faces_320:
                    entry["results"][variant] = None  # detector missed the face at this size
                    continue
                scores = index.embeddings @ faces_320[0].embedding
            elif variant == "dinov2":
                scores = dino
            else:
                scores = blend_scores(arcface, tight if "tight" in variant else dino, args.alpha)
                if variant.endswith("-skin"):
                    scores = scores + args.skin_weight * skin
            entry["results"][variant] = [to_entry(m) for m in index.rank(scores, k=args.k)]
        print(f"  {photo.name}: ok")

    data = {
        "id": out_dir.name,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "alpha": args.alpha,
        "paintingFaces": len(index),
        "variants": [{"id": v, "label": VARIANTS[v] + (f" (α={args.alpha})" if v.startswith("blend") else "")
                     + (f" (skin {args.skin_weight})" if v.endswith("-skin") else "")} for v in variants],
        "photos": entries,
    }
    payload = json.dumps(data).replace("</", "<\\/")
    report = out_dir / "report.html"
    report.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8")

    n_ok = sum(1 for e in entries if not e["error"])
    print(f"\n{n_ok}/{len(entries)} photos had a face -> {report}")


if __name__ == "__main__":
    main()
