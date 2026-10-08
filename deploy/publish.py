"""Publish the app's backend.

  assets    Upload the face index, painting metadata, thumbnails and face crops from
            data/ to a Hugging Face dataset. Re-running uploads only what changed.
            Requires `hf auth login` with a token that has write access.
  cloudrun  Deploy the API to Google Cloud Run. Cloud Build builds deploy/Dockerfile on
            Google's servers, pinned to the assets dataset's current commit so every
            build is reproducible. Requires the gcloud CLI, logged in, with a project set.

    uv run deploy/publish.py assets
    uv run deploy/publish.py cloudrun
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = Path(__file__).resolve().parent / "Dockerfile"
# What Cloud Build needs to build the API image (see deploy/Dockerfile).
BUILD_FILES = ["pyproject.toml", "uv.lock", ".python-version", "src", "backend"]
SITE_URL = "https://twin.maxaltman.com"
# Google APIs the deploy uses: Cloud Run, plus Cloud Build and Artifact Registry for --source builds.
GCP_SERVICES = ["run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com"]
INDEX_FILES = ["index/index.npz", "index/paintings.json"]
IMAGE_DIRS = ["thumbs", "crops"]
# Images are uploaded as one uncompressed tar per folder (images/thumbs.tar, images/crops.tar)
# rather than ~31k separate files: Hugging Face allows at most 10,000 files per directory, and
# downloading tens of thousands of small files at build time is slow and rate-limited.
# deploy/Dockerfile unpacks them to thumbs/<id>.jpg and crops/<id>_<n>.jpg.
ARCHIVE_DIR = "images"
# Remote paths from older layouts (loose or sharded images) that a new upload replaces.
LOOSE_IMAGE = re.compile(r"^(thumbs|crops)/")

ASSETS_README = """---
license: other
pretty_name: Historical Twin assets
---

# Historical Twin assets

Build-time assets for the matching API behind [Historical Twin]({site_url}): the face index
(`index/index.npz`), painting metadata (`index/paintings.json`), painting thumbnails
(`images/thumbs.tar`) and face crops (`images/crops.tar`). Images are packed into tar
archives because the image folders hold about 31,000 files.

Generated from [mixitymax/wikiart-portraits](https://huggingface.co/datasets/mixitymax/wikiart-portraits)
by https://github.com/misterem/historicalTwin. Paintings come from WikiArt. Face embeddings
were computed with InsightFace `buffalo_l` (non-commercial research use only), and painting
metadata comes from the ArtGAN WikiArt dataset (non-commercial research use only).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["assets", "cloudrun"])
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--assets-repo", default="mixitymax/historicaltwin-assets")
    p.add_argument("--allowed-origins", default=SITE_URL,
                   help="comma-separated site origins allowed to call the API (CORS)")
    p.add_argument("--service", default="historicaltwin-api", help="Cloud Run service name")
    # us-central1 is in Cloud Run's lowest price tier, where the free tier applies.
    p.add_argument("--region", default="us-central1")
    p.add_argument("--project", help="Google Cloud project (default: gcloud's configured project)")
    # Caps how far Cloud Run can scale, and therefore the worst-case bill.
    p.add_argument("--max-instances", type=int, default=3)
    return p.parse_args()


def gcloud_args(args: argparse.Namespace) -> list[str]:
    return ["--quiet"] + (["--project", args.project] if args.project else [])


def run_deploy_command(args: argparse.Namespace, source: Path) -> list[str]:
    # A custom delimiter ("^|^") lets the CORS value itself contain commas.
    env = f"^|^PAINTMATCH_ALLOWED_ORIGINS={args.allowed_origins}"
    return [
        "gcloud", "run", "deploy", args.service,
        "--source", str(source),
        "--region", args.region,
        "--allow-unauthenticated",   # a public API, called from the browser
        "--memory", "2Gi",           # ~0.6 GB in use; headroom for concurrent requests
        "--cpu", "1",
        "--cpu-boost",               # extra CPU while starting, to shorten cold starts
        "--concurrency", "20",       # matches are quick, and image requests are cheap
        "--min-instances", "0",      # scale to zero when idle: free, at the cost of cold starts
        "--max-instances", str(args.max_instances),
        "--timeout", "60",
        "--port", "8080",
        "--set-env-vars", env,
        *gcloud_args(args),
    ]


def check_assets(data_dir: Path) -> None:
    """Refuse to publish an index whose images are missing."""
    from paintmatch.index import crop_name, load_index, thumb_name

    _, faces, images = load_index(data_dir / "index")
    face_keys = list(zip(faces["image_id"].tolist(), faces["face_idx"].tolist()))
    missing_crops = [k for k in face_keys if not (data_dir / "crops" / crop_name(*k)).exists()]
    missing_thumbs = {i for i, _ in face_keys if not (data_dir / "thumbs" / thumb_name(i)).exists()}
    n_errors = sum(1 for e in images["error"].tolist() if e)
    print(f"index: {len(face_keys)} faces from {len({i for i, _ in face_keys})} paintings "
          f"({len(images['image_id'])} paintings seen, {n_errors} with errors)")
    if not (data_dir / "index" / "paintings.json").exists():
        sys.exit("data/index/paintings.json is missing: run indexing/build_metadata.py first")
    if missing_crops or missing_thumbs:
        sys.exit(f"missing {len(missing_crops)} crops and {len(missing_thumbs)} thumbnails: re-run build_index.py")


def _reset_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Fixed metadata, so the same images always produce a byte-identical archive."""
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def write_archive(src_dir: Path, archive: Path) -> int:
    """Pack src_dir/*.jpg into an uncompressed tar (JPEGs don't compress) as <folder>/<file>."""
    files = sorted(src_dir.glob("*.jpg"))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        for f in files:
            tar.add(f, arcname=f"{src_dir.name}/{f.name}", filter=_reset_tar_info)
    return len(files)


def stage_assets(data_dir: Path, stage_dir: Path) -> dict[str, int]:
    """Lay out the repo contents in stage_dir: index files (hard links) plus one tar per
    image folder. Returns the number of images in each archive."""
    if stage_dir.exists():  # rebuild, but keep the upload's resume cache
        for child in stage_dir.iterdir():
            if child.name != ".cache":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    for rel in INDEX_FILES:
        (stage_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(data_dir / rel, stage_dir / rel)
    return {folder: write_archive(data_dir / folder, stage_dir / ARCHIVE_DIR / f"{folder}.tar")
            for folder in IMAGE_DIRS}


def publish_assets(api: HfApi, args: argparse.Namespace) -> None:
    check_assets(args.data_dir)
    api.create_repo(args.assets_repo, repo_type="dataset", exist_ok=True)
    # Remove loose images from older layouts; the archives replace them.
    remote = api.list_repo_files(args.assets_repo, repo_type="dataset")
    for folder in IMAGE_DIRS:
        if any(LOOSE_IMAGE.match(f) and f.startswith(folder + "/") for f in remote):
            print(f"removing loose {folder}/ from the dataset")
            api.delete_folder(folder, repo_id=args.assets_repo, repo_type="dataset",
                              commit_message=f"Remove loose {folder}/ (replaced by images/{folder}.tar)")
    api.upload_file(
        repo_id=args.assets_repo,
        repo_type="dataset",
        path_or_fileobj=ASSETS_README.format(site_url=SITE_URL).encode(),
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )
    # Staged inside data/ so hard links work (same filesystem) and the upload's resume
    # cache, kept in the staged folder, survives between runs.
    stage_dir = args.data_dir / ".publish-assets"
    counts = stage_assets(args.data_dir, stage_dir)
    print("staged archives: " + ", ".join(f"{k}.tar ({n} images)" for k, n in counts.items()))
    # Re-running resumes an interrupted upload; unchanged archives upload almost nothing.
    api.upload_folder(
        repo_id=args.assets_repo,
        repo_type="dataset",
        folder_path=stage_dir,
        ignore_patterns=[".cache/**"],
        commit_message="Update face index and painting images",
    )
    print(f"assets: https://huggingface.co/datasets/{args.assets_repo} @ {api.dataset_info(args.assets_repo).sha}")


def stage_build_context(stage: Path, assets_repo: str, assets_revision: str) -> None:
    """Copy what Cloud Build needs into stage, with the Dockerfile pinned to the assets commit."""
    for name in BUILD_FILES:
        src = REPO_ROOT / name
        if src.is_dir():
            shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, stage / name)
    dockerfile = DOCKERFILE.read_text()
    dockerfile = dockerfile.replace("__ASSETS_REPO__", assets_repo).replace("__ASSETS_REVISION__", assets_revision)
    (stage / "Dockerfile").write_text(dockerfile)


def publish_cloudrun(args: argparse.Namespace) -> None:
    if not shutil.which("gcloud"):
        sys.exit("gcloud not found: install it with `brew install --cask gcloud-cli`, then `gcloud auth login`")
    revision = HfApi().dataset_info(args.assets_repo).sha
    print(f"assets: {args.assets_repo} @ {revision[:7]}")
    subprocess.run(["gcloud", "services", "enable", *GCP_SERVICES, *gcloud_args(args)], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        stage_build_context(stage, args.assets_repo, revision)
        print("building on Cloud Build and deploying (about 10 minutes the first time)...")
        subprocess.run(run_deploy_command(args, stage), check=True)
    url = subprocess.run(
        ["gcloud", "run", "services", "describe", args.service, "--region", args.region,
         "--format", "value(status.url)", *gcloud_args(args)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"api: {url}")


def main() -> None:
    args = parse_args()
    if args.command == "assets":
        api = HfApi()
        print(f"logged in to Hugging Face as {api.whoami()['name']}")
        publish_assets(api, args)
    else:
        publish_cloudrun(args)


if __name__ == "__main__":
    main()
