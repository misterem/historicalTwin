"""Build the face index for the portrait dataset.

For every painting: decode -> detect faces -> embed each face with ArcFace ->
write embeddings + metadata, a web-sized thumbnail, and a crop per face.
See src/paintmatch/index.py for the output layout.

The run is resumable: progress is checkpointed, and re-running skips
paintings that are already indexed (paintings that failed are retried).

    uv run indexing/build_index.py --limit 200   # quick smoke test
    uv run indexing/build_index.py               # full dataset
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from paintmatch.faces import EMBEDDING_DIM, FaceEmbedder, face_crop, load_image
from paintmatch.index import INDEX_FILE, columns_to_rows, crop_name, load_index, save_index, thumb_name

REPO_ROOT = Path(__file__).resolve().parents[1]

# The dataset is our own and trusted; some museum scans exceed PIL's default
# decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="mixitymax/wikiart-portraits", help="Hugging Face dataset repo")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--limit", type=int, help="only index the first N images (for testing)")
    p.add_argument("--skip-download", action="store_true", help="use images already in <data-dir>/raw")
    p.add_argument("--download-workers", type=int, default=8)
    p.add_argument("--load-workers", type=int, default=4, help="threads decoding images ahead of the model")
    p.add_argument("--max-side", type=int, default=1280, help="downscale paintings to this before detection")
    p.add_argument("--det-size", type=int, default=640, help="detector input resolution")
    p.add_argument("--det-thresh", type=float, default=0.5, help="minimum face detection confidence")
    p.add_argument("--min-face-px", type=int, default=40, help="drop faces narrower than this (in decoded px)")
    p.add_argument("--max-faces", type=int, default=4, help="keep at most this many faces per painting, largest first")
    p.add_argument("--thumb-size", type=int, default=512)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="onnxruntime providers, e.g. CUDAExecutionProvider on a cloud GPU",
    )
    p.add_argument("--checkpoint-every", type=int, default=500, help="save progress every N paintings")
    p.add_argument("--fresh", action="store_true", help="ignore any existing index and start over")
    return p.parse_args()


def list_images(args: argparse.Namespace, raw_dir: Path) -> list[str]:
    """Repo-relative paths (images/<shard>/<id>.jpg) of the paintings to index."""
    if args.skip_download:
        paths = sorted(p.relative_to(raw_dir).as_posix() for p in (raw_dir / "images").rglob("*.jpg"))
    else:
        repo_files = HfApi().list_repo_files(args.repo_id, repo_type="dataset")
        paths = sorted(f for f in repo_files if f.startswith("images/") and f.lower().endswith(".jpg"))
    return paths[: args.limit] if args.limit else paths


def download(args: argparse.Namespace, raw_dir: Path, paths: list[str]) -> list[str]:
    """Download missing images; returns the paths that are available locally."""

    def fetch(path: str) -> str | None:
        try:
            # Files already present in local_dir are skipped by huggingface_hub.
            hf_hub_download(args.repo_id, path, repo_type="dataset", local_dir=raw_dir)
            return path
        except Exception as e:  # keep going; a re-run will retry
            tqdm.write(f"download failed: {path}: {e}")
            return None

    fetched = thread_map(fetch, paths, max_workers=args.download_workers, desc="download", unit="img")
    ok = [p for p in fetched if p is not None]
    if len(ok) < len(paths):
        print(f"{len(paths) - len(ok)} downloads failed; re-run to retry (`hf auth login` raises rate limits)")
    return ok


def prefetch(fn, items, workers: int, depth: int):
    """Yield (item, result, error) in order, running fn on up to `depth` items ahead in threads."""
    items = iter(items)
    with ThreadPoolExecutor(workers) as pool:
        queue = deque((item, pool.submit(fn, item)) for item in islice(items, depth))
        while queue:
            item, future = queue.popleft()
            for nxt in islice(items, 1):
                queue.append((nxt, pool.submit(fn, nxt)))
            try:
                result = future.result()
            except Exception as e:
                yield item, None, e
                continue
            yield item, result, None


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    raw_dir, index_dir = data_dir / "raw", data_dir / "index"
    thumbs_dir, crops_dir = data_dir / "thumbs", data_dir / "crops"
    for d in (raw_dir, index_dir, thumbs_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths = list_images(args, raw_dir)
    if not args.skip_download:
        paths = download(args, raw_dir, paths)
    print(f"{len(paths)} paintings to consider")

    # Resume from a previous run. Paintings that errored are dropped so they get retried.
    embeddings: list[np.ndarray] = []
    face_rows: list[dict] = []
    image_rows: list[dict] = []
    if not args.fresh and (index_dir / INDEX_FILE).exists():
        prev_emb, prev_faces, prev_images = load_index(index_dir)
        image_rows = [r for r in columns_to_rows(prev_images) if not r["error"]]
        face_rows = columns_to_rows(prev_faces)
        embeddings = list(prev_emb)
        print(f"resuming: {len(image_rows)} paintings / {len(face_rows)} faces already indexed")
    done = {r["image_id"] for r in image_rows}
    todo = [p for p in paths if Path(p).stem not in done]

    def checkpoint() -> None:
        emb = np.stack(embeddings) if embeddings else np.empty((0, EMBEDDING_DIM), np.float32)
        save_index(index_dir, emb, face_rows, image_rows)

    if not todo:
        print("nothing to do")
        checkpoint()
        return

    embedder = FaceEmbedder(det_size=args.det_size, det_thresh=args.det_thresh, providers=args.providers)

    def load(path: str) -> np.ndarray:
        # Runs in worker threads: decode, and write the web thumbnail while we have the pixels.
        rgb = load_image(raw_dir / path, max_side=args.max_side)
        thumb_path = thumbs_dir / thumb_name(Path(path).stem)
        if not thumb_path.exists():
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((args.thumb_size, args.thumb_size), Image.Resampling.LANCZOS)
            thumb.save(thumb_path, quality=85, optimize=True)
        return rgb

    started = time.time()
    n_errors = n_no_face = 0
    for i, (path, rgb, error) in enumerate(
        tqdm(prefetch(load, todo, args.load_workers, depth=4 * args.load_workers), total=len(todo), desc="index", unit="img"),
        start=1,
    ):
        image_id = Path(path).stem
        n_faces = 0
        try:
            if error is not None:
                raise error
            h, w = rgb.shape[:2]
            faces = [f for f in embedder.faces(rgb) if f.width >= args.min_face_px][: args.max_faces]
            for face_idx, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.tolist()
                face_crop(rgb, face.bbox, args.crop_size).save(crops_dir / crop_name(image_id, face_idx), quality=90)
                embeddings.append(face.embedding)
                face_rows.append({
                    "image_id": image_id,
                    "image_path": path,
                    "face_idx": face_idx,
                    "x1": max(x1 / w, 0.0),
                    "y1": max(y1 / h, 0.0),
                    "x2": min(x2 / w, 1.0),
                    "y2": min(y2 / h, 1.0),
                    "det_score": face.det_score,
                    "face_px": face.width,
                })
            n_faces = len(faces)
            n_no_face += n_faces == 0
            image_rows.append({"image_id": image_id, "image_path": path, "n_faces": n_faces, "error": ""})
        except Exception as e:
            n_errors += 1
            tqdm.write(f"failed: {path}: {e!r}")
            # Roll back any faces appended for this painting before the failure.
            while face_rows and face_rows[-1]["image_id"] == image_id:
                face_rows.pop()
                embeddings.pop()
            image_rows.append({"image_id": image_id, "image_path": path, "n_faces": 0, "error": repr(e)})

        if i % args.checkpoint_every == 0:
            checkpoint()

    checkpoint()
    elapsed = time.time() - started
    print(
        f"\nindexed {len(todo)} paintings in {elapsed / 60:.1f} min "
        f"({len(todo) / max(elapsed, 1e-9):.1f} img/s)\n"
        f"  no face found: {n_no_face}\n"
        f"  errors:        {n_errors}\n"
        f"  index total:   {len(face_rows)} faces from "
        f"{sum(r['n_faces'] > 0 for r in image_rows)} paintings -> {index_dir}"
    )


if __name__ == "__main__":
    main()
