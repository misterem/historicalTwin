"""Publish the app's backend to Hugging Face (free hosting).

  assets  Upload the face index, painting metadata, thumbnails and face crops from
          data/ to a Hugging Face dataset. Re-running uploads only what changed.
  space   Push the API to a Hugging Face Space (Docker). Its Dockerfile is pinned to the
          assets dataset's current commit, so every Space build is reproducible. The
          Space builds on Hugging Face's servers after the push.

Requires `hf auth login` with a token that has write access.

    uv run deploy/publish.py assets
    uv run deploy/publish.py space
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_TEMPLATE = Path(__file__).resolve().parent / "huggingface"
# Code the Space needs to build the API image (see deploy/huggingface/Dockerfile).
SPACE_FILES = ["pyproject.toml", "uv.lock", ".python-version", "src", "backend"]
INDEX_FILES = ["index/index.npz", "index/paintings.json"]
IMAGE_DIRS = ["thumbs", "crops"]
# Hugging Face repos allow at most 10,000 files per directory, so images are uploaded into
# subfolders named by the first two hex characters of the image ID (256 folders of ~60 files).
# The Space's Dockerfile flattens them back to thumbs/<id>.jpg and crops/<id>_<n>.jpg.
FLAT_IMAGE = re.compile(r"^(thumbs|crops)/[^/]+\.jpg$")

ASSETS_README = """---
license: other
pretty_name: Historical Twin assets
---

# Historical Twin assets

Build-time assets for the [Historical Twin API Space]({space_url}): the face index
(`index/index.npz`), painting metadata (`index/paintings.json`), painting thumbnails
(`thumbs/`) and face crops (`crops/`). Images are split into subfolders by the first two
characters of their ID, because Hugging Face allows at most 10,000 files per folder.

Generated from [mixitymax/wikiart-portraits](https://huggingface.co/datasets/mixitymax/wikiart-portraits)
by https://github.com/misterem/historicalTwin. Paintings come from WikiArt. Face embeddings
were computed with InsightFace `buffalo_l` (non-commercial research use only), and painting
metadata comes from the ArtGAN WikiArt dataset (non-commercial research use only).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["assets", "space"])
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--assets-repo", default="mixitymax/historicaltwin-assets")
    p.add_argument("--space", default="mixitymax/historicaltwin-api")
    p.add_argument(
        "--allowed-origins",
        default="https://twin.maxaltman.com",
        help="comma-separated site origins allowed to call the API (CORS)",
    )
    return p.parse_args()


def space_url(space: str) -> str:
    return f"https://huggingface.co/spaces/{space}"


def space_api_url(space: str) -> str:
    return "https://" + space.replace("/", "-").replace("_", "-").replace(".", "-").lower() + ".hf.space"


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


def shard(filename: str) -> str:
    return f"{filename[:2]}/{filename}"


def stage_assets(data_dir: Path, stage_dir: Path) -> int:
    """Lay out the repo contents in stage_dir using hard links (no copying). Returns the file count."""
    if stage_dir.exists():  # rebuild the links, but keep the upload's resume cache
        for child in stage_dir.iterdir():
            if child.name != ".cache":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    n = 0
    for rel in INDEX_FILES:
        (stage_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(data_dir / rel, stage_dir / rel)
        n += 1
    for folder in IMAGE_DIRS:
        for src in (data_dir / folder).glob("*.jpg"):
            dst = stage_dir / folder / shard(src.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            n += 1
    return n


def publish_assets(api: HfApi, args: argparse.Namespace) -> None:
    check_assets(args.data_dir)
    api.create_repo(args.assets_repo, repo_type="dataset", exist_ok=True)
    # Remove images from an older flat layout before uploading the sharded one.
    remote = api.list_repo_files(args.assets_repo, repo_type="dataset")
    for folder in IMAGE_DIRS:
        if any(FLAT_IMAGE.match(f) and f.startswith(folder + "/") for f in remote):
            print(f"removing flat {folder}/ from the dataset")
            api.delete_folder(folder, repo_id=args.assets_repo, repo_type="dataset",
                              commit_message=f"Remove flat {folder}/ (too many files per folder)")
    api.upload_file(
        repo_id=args.assets_repo,
        repo_type="dataset",
        path_or_fileobj=ASSETS_README.format(space_url=space_url(args.space)).encode(),
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )
    # Staged inside data/ so hard links work (same filesystem) and the upload's resume
    # cache, kept in the staged folder, survives between runs.
    stage_dir = args.data_dir / ".publish-assets"
    print(f"staged {stage_assets(args.data_dir, stage_dir)} files")
    # Large folders are committed in batches; re-running resumes an interrupted upload.
    api.upload_folder(
        repo_id=args.assets_repo,
        repo_type="dataset",
        folder_path=stage_dir,
        ignore_patterns=[".cache/**"],
        commit_message="Update face index and painting images",
    )
    print(f"assets: https://huggingface.co/datasets/{args.assets_repo} @ {api.dataset_info(args.assets_repo).sha}")


def publish_space(api: HfApi, args: argparse.Namespace) -> None:
    revision = api.dataset_info(args.assets_repo).sha
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in SPACE_FILES:
            src = REPO_ROOT / name
            if src.is_dir():
                shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, stage / name)
        dockerfile = (SPACE_TEMPLATE / "Dockerfile").read_text()
        dockerfile = dockerfile.replace("__ASSETS_REPO__", args.assets_repo).replace("__ASSETS_REVISION__", revision)
        (stage / "Dockerfile").write_text(dockerfile)
        shutil.copy2(SPACE_TEMPLATE / "README.md", stage / "README.md")

        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_variable(args.space, "PAINTMATCH_ALLOWED_ORIGINS", args.allowed_origins,
                               description="Site origins allowed to call the API (CORS)")
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=stage,
            delete_patterns=["src/**", "backend/**"],  # remove files deleted from the repo since last push
            commit_message=f"Deploy API (assets @ {revision[:7]})",
        )
    print(f"space: {space_url(args.space)} (building; takes a few minutes)")
    print(f"api:   {space_api_url(args.space)}")


def main() -> None:
    args = parse_args()
    api = HfApi()
    print(f"logged in to Hugging Face as {api.whoami()['name']}")
    if args.command == "assets":
        publish_assets(api, args)
    else:
        publish_space(api, args)


if __name__ == "__main__":
    main()
