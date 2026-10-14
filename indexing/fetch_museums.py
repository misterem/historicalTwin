"""Fetch public-domain portrait paintings from museum open-access collections.

Sources (all public domain or CC0, chosen by a sampling study of face-detection rates):
  met   The Met: European Paintings, The American Wing and the Robert Lehman Collection,
        paintings and portrait miniatures tagged or titled as portraits
        (MetObjects.csv + the collection API)
  npg   Smithsonian National Portrait Gallery: every CC0 painting
  saam  Smithsonian American Art Museum: CC0 paintings tagged "Portraits"

Images go to <data-dir>/raw/museums/<image_id>.jpg, where build_index.py picks them up.
<data-dir>/raw/museums/manifest.json records each object's status and its artist, title,
year, museum, link and credit line; build_metadata.py turns those into captions.

Paintings that are already in the WikiArt portrait set are skipped: each download is
phashed and compared with the WikiArt images, using the same <= 4 bits + same-aspect test
as build_metadata.py.

The run is resumable. The catalogs (~650 MB) are downloaded on the first run. The Met's API
returns 403s under parallel load, so its requests are made one per second.

    uv run indexing/fetch_museums.py --limit 20    # a few objects per source
    uv run indexing/fetch_museums.py               # everything (~2,700 objects, ~30 min)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from build_metadata import MUSEUM_MANIFEST, POPCOUNT, fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "historicalTwin/1.0 (+https://github.com/misterem/historicalTwin)"
MET_CSV_URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"
MET_OBJECT_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
MET_DEPARTMENTS = {"European Paintings", "The American Wing", "The Robert Lehman Collection"}
SI_DUMP_URL = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/{unit}/{shard:02x}.txt"
SI_MUSEUMS = {"npg": "National Portrait Gallery", "saam": "Smithsonian American Art Museum"}
SI_IMAGE_URL = "https://ids.si.edu/ids/deliveryService?id={}&max={}"
# Statuses that are final; anything else ("error ...") is retried on the next run.
DONE = {"ok", "duplicate", "no image"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--sources", default="met,npg,saam", help="comma-separated subset of: met, npg, saam")
    p.add_argument("--limit", type=int, help="only the first N objects per source (for testing)")
    p.add_argument("--max-side", type=int, default=1280, help="requested image size where the API allows it")
    p.add_argument("--max-bits", type=int, default=4, help="phash distance at or below which a painting is a duplicate")
    p.add_argument("--max-aspect-diff", type=float, default=0.03)
    p.add_argument("--workers", type=int, default=4, help="parallel image downloads (Smithsonian)")
    return p.parse_args()


def safe_id(text: str) -> str:
    """File-name-safe image id: letters, digits and dashes only (crop names add '_<face>')."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def fetch(url: str, timeout: int = 60) -> bytes:
    # Met image URLs can contain spaces.
    url = urllib.parse.quote(url, safe=":/?=&%#+,;@")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def download_to(url: str, path: Path) -> None:
    if path.exists():
        return
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(fetch(url, timeout=300))
    os.replace(tmp, path)


# --- The Met ---------------------------------------------------------------------------

def met_selected(row: dict) -> bool:
    if row["Is Public Domain"] != "True" or row["Department"] not in MET_DEPARTMENTS:
        return False
    # The American Wing leaves Classification empty, so the object name counts too.
    if not (row["Classification"].lower().startswith(("paintings", "miniatures"))
            or row["Object Name"].lower().startswith("painting")):
        return False
    tags = row["Tags"].split("|")
    return "Portraits" in tags or "portrait" in row["Title"].lower() or "portrait" in row["Object Name"].lower()


def met_candidate(row: dict) -> dict:
    return {
        "image_id": f"met-{row['Object ID']}",
        "object_id": row["Object ID"],
        "artist": row["Artist Display Name"].split("|")[0].strip() or None,
        "title": row["Title"].strip() or None,
        "year": row["Object Date"].strip() or None,
        "museum": "The Metropolitan Museum of Art",
        "url": row["Link Resource"],
        "credit": row["Credit Line"].strip() or None,
        "license": "CC0",
    }


def met_candidates(catalog_dir: Path) -> list[dict]:
    path = catalog_dir / "met" / "MetObjects.csv"
    if not path.exists():
        print("downloading the Met's catalog (~300 MB)")
        path.parent.mkdir(parents=True, exist_ok=True)
        download_to(MET_CSV_URL, path)
    csv.field_size_limit(10**8)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [met_candidate(row) for row in csv.DictReader(f) if met_selected(row)]


def met_image_url(candidate: dict) -> str | None:
    time.sleep(1.0)  # one request per second keeps the API from returning 403s
    obj = json.loads(fetch(MET_OBJECT_API.format(candidate["object_id"])))
    return obj.get("primaryImageSmall") or obj.get("primaryImage") or None


# --- Smithsonian -----------------------------------------------------------------------

def _first(freetext: dict, field: str, label: str | None = None) -> str | None:
    for item in freetext.get(field, []):
        if label is None or item.get("label") == label:
            return item.get("content", "").strip() or None
    return None


def si_candidate(record: dict, unit: str, max_side: int) -> dict | None:
    """A candidate from one line of the Smithsonian dump, or None if it doesn't qualify."""
    if record.get("type") != "edanmdm":
        return None
    content = record.get("content", {})
    structured = content.get("indexedStructured", {})
    if not {"Paintings", "Miniatures"} & set(structured.get("object_type", [])):
        return None
    if unit == "saam" and "Portraits" not in structured.get("topic", []):
        return None
    meta = content.get("descriptiveNonRepeating", {})
    media = [m for m in meta.get("online_media", {}).get("media", [])
             if m.get("type") == "Images" and m.get("usage", {}).get("access") == "CC0" and m.get("idsId")]
    if not media:
        return None
    freetext = content.get("freetext", {})
    artist = _first(freetext, "name", "Artist")
    return {
        "image_id": safe_id(f"si-{meta.get('record_ID', record['id'])}"),
        "artist": artist.split(",")[0].strip() if artist else None,  # "Gilbert Stuart, 3 Dec 1755 - 9 Jul 1828"
        "title": meta.get("title", {}).get("content"),
        "year": _first(freetext, "date"),
        "museum": SI_MUSEUMS[unit],
        "url": meta.get("record_link"),
        "credit": _first(freetext, "creditLine"),
        "license": "CC0",
        "image_url": SI_IMAGE_URL.format(media[0]["idsId"], max_side),
    }


def si_candidates(catalog_dir: Path, unit: str, max_side: int) -> list[dict]:
    unit_dir = catalog_dir / "smithsonian" / unit
    unit_dir.mkdir(parents=True, exist_ok=True)
    missing = [s for s in range(256) if not (unit_dir / f"{s:02x}.txt").exists()]
    if missing:
        thread_map(lambda s: download_to(SI_DUMP_URL.format(unit=unit, shard=s), unit_dir / f"{s:02x}.txt"),
                   missing, max_workers=16, desc=f"{unit} catalog", unit="file")
    out = []
    for shard in sorted(unit_dir.glob("*.txt")):
        with open(shard, encoding="utf-8") as f:
            out.extend(c for line in f if (c := si_candidate(json.loads(line), unit, max_side)))
    return out


# --- Duplicates ------------------------------------------------------------------------

class Fingerprints:
    """Perceptual hashes of the paintings we already have, to skip second copies."""

    def __init__(self, ids: list[str], hashes: np.ndarray, aspects: np.ndarray):
        self.ids, self.hashes, self.aspects = list(ids), np.asarray(hashes, np.uint64), np.asarray(aspects, float)

    def add(self, image_id: str, fp: tuple[int, float]) -> None:
        self.ids.append(image_id)
        self.hashes = np.append(self.hashes, np.uint64(fp[0]))
        self.aspects = np.append(self.aspects, fp[1])

    def find(self, fp: tuple[int, float], max_bits: int, max_aspect_diff: float) -> str | None:
        """Id of a painting within max_bits and the same aspect ratio, if any."""
        if not len(self.hashes):
            return None
        query = np.array([fp[0]], np.uint64).view(np.uint8)
        dist = POPCOUNT[self.hashes.view(np.uint8).reshape(-1, 8) ^ query].sum(axis=1)
        for i in np.flatnonzero(dist <= max_bits):
            if abs(fp[1] - self.aspects[i]) / self.aspects[i] <= max_aspect_diff:
                return self.ids[i]
        return None


def wikiart_fingerprints(data_dir: Path) -> Fingerprints:
    """Fingerprints of the WikiArt images, cached in data/cache."""
    cache = data_dir / "cache" / "phash-wikiart.npz"
    images = sorted((data_dir / "raw" / "images").rglob("*.jpg"))
    if cache.exists():
        with np.load(cache) as z:
            if len(z["ids"]) == len(images):
                return Fingerprints(z["ids"].tolist(), z["hashes"], z["aspects"])
    prints = thread_map(fingerprint, images, max_workers=8, desc="phash WikiArt", unit="img", chunksize=32)
    # Unreadable images keep a placeholder row so the cache's length still matches the folder.
    ids = [p.stem for p in images]
    hashes = np.array([fp[0] if fp else 0 for fp in prints], np.uint64)
    aspects = np.array([fp[1] if fp else np.inf for fp in prints])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), hashes=hashes, aspects=aspects)
    return Fingerprints(ids, hashes, aspects)


# --- Main ------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if unknown := set(sources) - {"met", *SI_MUSEUMS}:
        raise SystemExit(f"unknown sources: {', '.join(sorted(unknown))}")
    catalog_dir = args.data_dir / "sources"
    manifest_path = args.data_dir / MUSEUM_MANIFEST
    out_dir = manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def save_manifest() -> None:
        tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        os.replace(tmp, manifest_path)

    candidates: list[dict] = []
    for source in sources:
        found = met_candidates(catalog_dir) if source == "met" else si_candidates(catalog_dir, source, args.max_side)
        found = found[: args.limit] if args.limit else found
        print(f"{source}: {len(found)} candidates")
        candidates.extend(found)

    def done(c: dict) -> bool:
        entry = manifest.get(c["image_id"])
        return bool(entry) and entry["status"] in DONE and (entry["status"] != "ok" or (out_dir / f"{c['image_id']}.jpg").exists())

    todo = [c for c in candidates if not done(c)]
    print(f"{len(candidates) - len(todo)} already fetched, {len(todo)} to fetch")
    if not todo:
        return

    known = wikiart_fingerprints(args.data_dir)
    # Museum paintings fetched so far also count, so two museums' copies of a painting aren't both kept.
    for image_id, entry in manifest.items():
        if entry["status"] == "ok" and entry.get("phash"):
            known.add(image_id, (int(entry["phash"], 16), entry["aspect"]))

    def fetch_one(c: dict) -> dict:
        entry = {k: v for k, v in c.items() if k not in ("object_id", "image_url")}
        path = out_dir / f"{c['image_id']}.jpg"
        try:
            url = c.get("image_url") or met_image_url(c)
            if not url:
                return entry | {"status": "no image"}
            data = fetch(url, timeout=120)
            with Image.open(io.BytesIO(data)) as im:
                im.load()  # fail here, not in the indexer, on a truncated or non-image response
                rgb = im.convert("RGB")
            tmp = path.with_name(f".{path.name}.tmp")
            rgb.save(tmp, "JPEG", quality=92)
            os.replace(tmp, path)
            return entry | {"status": "ok"}
        except Exception as e:
            return entry | {"status": f"error {e.__class__.__name__}: {e}"[:200]}

    met_todo = [c for c in todo if c["image_id"].startswith("met-")]
    si_todo = [c for c in todo if not c["image_id"].startswith("met-")]
    counts: dict[str, int] = {}

    def record(entry: dict) -> None:
        path = out_dir / f"{entry['image_id']}.jpg"
        if entry["status"] == "ok":
            fp = fingerprint(path)
            dup = fp and known.find(fp, args.max_bits, args.max_aspect_diff)
            if dup:
                path.unlink()
                entry |= {"status": "duplicate", "duplicate_of": dup}
            elif fp:
                entry |= {"phash": f"{fp[0]:016x}", "aspect": round(fp[1], 4)}
                known.add(entry["image_id"], fp)
        status = entry["status"].split(" ")[0] if entry["status"].startswith("error") else entry["status"]
        counts[status] = counts.get(status, 0) + 1
        manifest[entry["image_id"]] = entry
        if sum(counts.values()) % 50 == 0:
            save_manifest()

    with ThreadPoolExecutor(args.workers) as pool, tqdm(total=len(todo), desc="fetch", unit="obj") as bar:
        # Smithsonian downloads run in the pool while the Met's rate-limited requests run here.
        futures = [pool.submit(fetch_one, c) for c in si_todo]
        for c in met_todo:
            record(fetch_one(c))
            bar.update()
            for f in [f for f in futures if f.done()]:
                record(f.result())
                futures.remove(f)
                bar.update()
        for f in futures:
            record(f.result())
            bar.update()
    save_manifest()

    print(f"\n{counts}")
    n_ok = sum(e["status"] == "ok" for e in manifest.values())
    print(f"{n_ok} museum paintings in {out_dir}; next: build_index.py, build_metadata.py, build_appearance.py")


if __name__ == "__main__":
    main()
