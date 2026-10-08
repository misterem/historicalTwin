# paintmatch

Take a selfie, find the portrait painting that looks most like you.

How it works: every face in the
[wikiart-portraits](https://huggingface.co/datasets/mixitymax/wikiart-portraits) dataset is
detected (SCRFD) and embedded (ArcFace) once, offline. At request time the selfie gets the same
treatment and the paintings are ranked by cosine similarity. Nothing is trained yet; see
`training/`.

```
src/paintmatch/   shared code: faces.py (decode/detect/embed), index.py (index files + search)
indexing/         build_index.py: dataset -> data/index, data/thumbs, data/crops
backend/          FastAPI /match endpoint + a dev test page
frontend/         the real web app (not started)
training/         optional fine-tuning (not started)
data/             generated, gitignored
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 and the dependencies:

```bash
uv sync
```

## Tests

```bash
uv run pytest
```

The tests use a tiny synthetic index and a fake face model, so they run in about a second
with no downloads. CI runs them on pull requests that touch Python code.

## Build the index

```bash
uv run indexing/build_index.py --limit 200   # smoke test
uv run indexing/build_index.py               # all 15k paintings
```

- On an M-series Mac's CPU it runs at about 6 paintings/s, so the full set takes roughly
  45 minutes plus download time. It checkpoints, so you can Ctrl-C and re-run to resume.
- To use a cloud GPU instead, `uv pip install onnxruntime-gpu` (after removing `onnxruntime`)
  and pass `--providers CUDAExecutionProvider CPUExecutionProvider`.
- Run `hf auth login` first if downloads get rate-limited.

## Add painting metadata

The dataset only has hashed filenames. This step looks each painting up in the
[ArtGAN WikiArt](https://www.kaggle.com/datasets/steubk/wikiart) dataset by perceptual hash
and records its artist, title and year. About 71% of paintings match; the rest get no
caption rather than a wrong one.

```bash
uv run indexing/build_metadata.py   # writes data/index/paintings.json
```

Run it after `build_index.py` has downloaded the images, and restart the API afterwards.

## Review match quality

Put 20–30 photos of different people in a folder, then build a side-by-side review page:

```bash
uv run evaluation/review_matches.py ~/Pictures/selfies
```

For each photo the page shows the top matches from four variants: ArcFace with detector
input 640 (what the app ships) and 320, DINOv2 appearance similarity, and a blend of ArcFace
and DINOv2. Variant names are hidden and rows shuffled until you click "Reveal", so you
judge them blind.

- The page is written to `data/reviews/<folder>-<timestamp>/report.html` and loads the
  painting images from `data/`. To view it, serve `data/` locally and open the report URL:

  ```bash
  python3 -m http.server 8790 --bind 127.0.0.1 --directory data
  ```

- The report embeds crops of your photos' faces, so keep it local.
- The first run embeds every painting face with DINOv2, which takes about 10 minutes. The
  result is cached in `data/reviews/_cache/`.
- `--variants arcface,blend` picks a subset, and `--alpha` sets ArcFace's weight in the
  blend (default 0.7).

## Run the backend

```bash
uv run uvicorn backend.app:app --reload
```

Open http://localhost:8000 and upload a selfie. The test page shrinks the photo to 800px in
the browser before uploading it.

## Deploy

The live app is free to host:

- **Website:** Netlify serves the static frontend at `twin.maxaltman.com`, building it from
  this repo.
- **API and images:** a Hugging Face Space (Docker, free CPU tier) runs the matching API and
  also serves the painting thumbnails and face crops.
- **Assets:** a Hugging Face dataset holds the face index, painting metadata and images. The
  Space downloads it when it builds.

### 1. Publish the assets and the API

Log in with a Hugging Face token that has write access (`hf auth login`), then run:

```bash
uv run deploy/publish.py assets   # index + ~30k images -> HF dataset (resumable)
uv run deploy/publish.py space    # code -> HF Space, pinned to the current assets commit
```

The Space builds on Hugging Face's servers in a few minutes. Re-run `space` after code
changes, and `assets` then `space` after re-indexing. The defaults (`mixitymax/...` repos,
CORS origin `https://twin.maxaltman.com`) can be changed with flags; see `--help`.

A free Space sleeps after 48 hours without visitors, so the first request after that takes
a minute or so while it restarts.

### 2. Website on Netlify

Create a new site from this GitHub repo with **base directory `frontend`**. Build settings
come from `frontend/netlify.toml`, including the API URL. Then add `twin.maxaltman.com`
under Domain management. With Netlify DNS, the record is created automatically.

### Alternative: your own container host

The root `Dockerfile` builds an API image that leaves the images out, to pair with a CDN:

1. **Images:** upload them to Cloudflare R2:
   ```bash
   export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
   uv run indexing/upload_to_r2.py
   ```
2. **API image:** build it. `data/index` must exist.
   ```bash
   docker build --platform linux/amd64 -t paintmatch-api .
   ```
   Deploy it with 1–2 GB of RAM.

| env var | purpose |
|---|---|
| `PAINTMATCH_IMAGE_BASE_URL` | public URL of the image bucket (required in that image) |
| `PAINTMATCH_ALLOWED_ORIGINS` | comma-separated site origins allowed by CORS |
| `PAINTMATCH_SELFIE_DET_SIZE` | `640` (default) or `320`: about 1.7x faster, scores shift slightly |
| `PORT` | set by the host; defaults to 8080 |

## Licensing note

InsightFace's pretrained `buffalo_l` weights are licensed for **non-commercial research only**.
Swap in a permissively licensed embedder before any commercial use. The ArtGAN WikiArt dataset
used for painting metadata is likewise provided for non-commercial research only.
