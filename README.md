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

## Run the backend

```bash
uv run uvicorn backend.app:app --reload
```

Open http://localhost:8000 and upload a selfie. The test page shrinks the photo to 800px in
the browser before uploading it.

## Deploy

1. **Images → Cloudflare R2.** Create a bucket and make it public: either connect a custom
   domain (recommended, since it goes through Cloudflare's CDN cache) or enable the `r2.dev`
   URL (rate-limited, fine for testing). Then create an R2 API token with Object Read & Write
   on that bucket and upload:

   ```bash
   export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
   uv run indexing/upload_to_r2.py --dry-run
   uv run indexing/upload_to_r2.py
   ```

   It skips files that are already in the bucket, so re-run it after re-indexing.

2. **API → container.** `data/index` must exist before you build. The model weights and the
   index are baked into the image (no downloads at startup), but the thumbnails and crops
   are not.

   ```bash
   docker build --platform linux/amd64 -t paintmatch-api .
   ```

   Deploy it to Cloud Run, Fly.io or Render with 1–2 GB RAM, and keep one instance always
   running to avoid cold starts.

   | env var | purpose |
   |---|---|
   | `PAINTMATCH_IMAGE_BASE_URL` | public URL of the R2 bucket, e.g. `https://images.yourdomain.com` (required in the container) |
   | `PAINTMATCH_ALLOWED_ORIGINS` | comma-separated frontend origins allowed by CORS |
   | `PAINTMATCH_SELFIE_DET_SIZE` | `640` (default) or `320`: about 1.7x faster, scores shift slightly |
   | `PORT` | set by the host; defaults to 8080 |

## Licensing note

InsightFace's pretrained `buffalo_l` weights are licensed for **non-commercial research only**.
Swap in a permissively licensed embedder before any commercial use. The ArtGAN WikiArt dataset
used for painting metadata is likewise provided for non-commercial research only.
