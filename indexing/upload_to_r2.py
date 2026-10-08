"""Upload the painting thumbnails and face crops to a Cloudflare R2 bucket.

The backend then links to them via PAINTMATCH_IMAGE_BASE_URL instead of serving
them itself. Credentials are read from the environment; create an R2 API token
with "Object Read & Write" on the bucket:

    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
    R2_ENDPOINT_URL   optional; overrides the R2 endpoint (any S3-compatible store works)

    uv run indexing/upload_to_r2.py --dry-run
    uv run indexing/upload_to_r2.py

Files already in the bucket with the same size are skipped, so re-running after
indexing more paintings only uploads what's new.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from tqdm.contrib.concurrent import thread_map

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("thumbs", "crops")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="only report what would be uploaded")
    p.add_argument("--force", action="store_true", help="re-upload files that already exist")
    p.add_argument(
        "--cache-control",
        default="public, max-age=2592000",
        help="Cache-Control header for the CDN and browsers (default: 30 days)",
    )
    return p.parse_args()


def make_client(workers: int):
    missing = [v for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET") if not os.environ.get(v)]
    if os.environ.get("R2_ENDPOINT_URL") and "R2_ACCOUNT_ID" in missing:
        missing.remove("R2_ACCOUNT_ID")
    if missing:
        sys.exit(f"missing environment variables: {', '.join(missing)}")
    endpoint = os.environ.get("R2_ENDPOINT_URL") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(max_pool_connections=workers, retries={"max_attempts": 5, "mode": "adaptive"}),
    )


def remote_sizes(client, bucket: str, prefix: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def main() -> None:
    args = parse_args()
    client = make_client(args.workers)
    bucket = os.environ["R2_BUCKET"]

    todo: list[tuple[Path, str]] = []
    for folder in FOLDERS:
        local = sorted((args.data_dir / folder).glob("*.jpg"))
        remote = {} if args.force else remote_sizes(client, bucket, f"{folder}/")
        pending = [(p, f"{folder}/{p.name}") for p in local if remote.get(f"{folder}/{p.name}") != p.stat().st_size]
        print(f"{folder}: {len(local)} local, {len(remote)} in bucket, {len(pending)} to upload")
        todo += pending

    if args.dry_run or not todo:
        return

    def upload(item: tuple[Path, str]) -> str | None:
        path, key = item
        try:
            client.upload_file(
                str(path), bucket, key, ExtraArgs={"ContentType": "image/jpeg", "CacheControl": args.cache_control}
            )
            return None
        except Exception as e:
            return f"{key}: {e}"

    errors = [e for e in thread_map(upload, todo, max_workers=args.workers, desc="upload", unit="file") if e]
    for e in errors[:20]:
        print("failed:", e)
    print(f"uploaded {len(todo) - len(errors)}/{len(todo)} files" + (" (re-run to retry failures)" if errors else ""))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
