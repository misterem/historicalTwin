"""Attach artist / title / year to the indexed paintings.

The portrait dataset (from the GANGogh WikiArt scrape) has no metadata: files are
named by the MD5 of their contents. Most of its paintings also appear in the ArtGAN
WikiArt dataset (Kaggle: steubk/wikiart), whose classes.csv lists each image's
artist, title slug, size, and perceptual hash (imagehash.phash). We phash our images
and accept the nearest CSV row only if it differs by at most --max-bits of 64 bits
and has the same aspect ratio. In testing, matches at <= 4 bits were always the
same painting and matches at >= 6 bits never were. Unmatched paintings simply get
no metadata, which is better than a wrong attribution.

Writes <data-dir>/index/paintings.json, which the backend merges into results.
Re-run it after build_index.py downloads new images; it doesn't touch the face index.

    uv run indexing/build_metadata.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import thread_map

from paintmatch.index import PAINTINGS_FILE

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES_CSV_URL = "https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart/classes.csv"

# The dataset is our own and trusted; some museum scans exceed PIL's default limit.
Image.MAX_IMAGE_PIXELS = None

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "da", "de", "del", "della", "der", "des", "di", "du",
               "for", "from", "in", "la", "le", "of", "on", "or", "ter", "the", "to", "van", "von", "with"}
ROMAN_NUMERAL = re.compile(r"^(?=[ivxl]+$)x{0,3}(ix|iv|v?i{0,3})$")
# "...-1794", "...-1889-1" (duplicate counter), "...-c-1606" (circa)
TRAILING_YEAR = re.compile(r"-(c-)?(1[0-9]{3}|20[0-2][0-9])(-\d{1,2})?$")
# WikiArt disambiguates same-titled works with "(1)" or a trailing "-2"
DUPLICATE_MARKER = re.compile(r"\(\d+\)$")
TRAILING_COUNTER = re.compile(r"(?:^|-)([a-z]+)-\d{1,2}$")
# 8-bit popcount lookup table for Hamming distances between 64-bit hashes.
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--max-bits", type=int, default=4, help="max differing phash bits to accept a match")
    p.add_argument("--max-aspect-diff", type=float, default=0.03, help="max relative aspect-ratio difference")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def smart_case(words: list[str]) -> str:
    out = []
    for i, w in enumerate(words):
        if ROMAN_NUMERAL.match(w):  # "charles-i", "louis-xiv"; also uppercases the pronoun "i"
            out.append(w.upper())
        elif i > 0 and w in SMALL_WORDS:
            out.append(w)
        else:  # also capitalize each part of hyphenated names: "pierre-auguste" -> "Pierre-Auguste"
            out.append("-".join(part[:1].upper() + part[1:] for part in w.split("-")))
    return " ".join(out)


def parse_slug(slug: str) -> tuple[str | None, str | None]:
    """'portrait-of-catherine-ii-of-russia-1794' -> ('Portrait of Catherine II of Russia', '1794')."""
    slug = DUPLICATE_MARKER.sub("", slug)
    year = None
    if m := TRAILING_YEAR.search(slug):
        year = f"c. {m.group(2)}" if m.group(1) else m.group(2)
        slug = slug[: m.start()]
    # Drop "female-portrait-2" style counters, but keep "portrait-at-the-age-of-10".
    if (m := TRAILING_COUNTER.search(slug)) and m.group(1) not in SMALL_WORDS:
        slug = slug[: m.end(1)]
    if slug.startswith("not_detected"):  # WikiArt placeholder for untitled works
        return None, year
    return smart_case([w for w in slug.split("-") if w]), year


def load_reference(data_dir: Path) -> list[dict]:
    path = data_dir / "kaggle-artgan" / "classes.csv"
    if not path.exists():
        print(f"downloading {CLASSES_CSV_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CLASSES_CSV_URL) as resp:
            data = resp.read()
        if data[:2] == b"PK":  # Kaggle serves single files zipped
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                data = zf.read("classes.csv")
        path.write_bytes(data)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fingerprint(path: Path) -> tuple[int, float] | None:
    try:
        with Image.open(path) as im:
            aspect = im.width / im.height
            im.draft("RGB", (512, 512))  # phash works on a 32x32 thumbnail; no need for full-size decode
            return int(str(imagehash.phash(im)), 16), aspect
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    rows = load_reference(args.data_dir)
    ref_hashes = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    ref_aspects = np.array([int(r["width"]) / int(r["height"]) for r in rows])
    ref_bytes = ref_hashes.view(np.uint8).reshape(-1, 8)
    print(f"{len(rows)} reference paintings")

    images = sorted((args.data_dir / "raw" / "images").rglob("*.jpg"))
    prints = thread_map(fingerprint, images, max_workers=args.workers, desc="phash", unit="img", chunksize=32)

    paintings: dict[str, dict] = {}
    n_unreadable = n_far = n_aspect = 0
    for path, fp in zip(images, prints):
        if fp is None:
            n_unreadable += 1
            continue
        h, aspect = fp
        query = np.array([h], dtype=np.uint64).view(np.uint8)
        dist = POPCOUNT[ref_bytes ^ query].sum(axis=1)
        best = int(dist.argmin())
        if dist[best] > args.max_bits:
            n_far += 1
            continue
        if abs(aspect - ref_aspects[best]) / ref_aspects[best] > args.max_aspect_diff:
            n_aspect += 1
            continue
        ref = rows[best]
        title, year = parse_slug(ref["description"])
        paintings[path.stem] = {
            "artist": smart_case(ref["artist"].split()),
            "title": title,
            "year": year,
            "source": ref["filename"],
            "phash_bits": int(dist[best]),
        }

    out = args.data_dir / "index" / PAINTINGS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(paintings, ensure_ascii=False, indent=0))
    os.replace(tmp, out)

    n = len(images)
    print(
        f"\nmatched {len(paintings)}/{n} paintings ({len(paintings) / max(n, 1):.1%}) -> {out}\n"
        f"  no close match:     {n_far}\n"
        f"  aspect mismatch:    {n_aspect}\n"
        f"  unreadable:         {n_unreadable}"
    )


if __name__ == "__main__":
    main()
