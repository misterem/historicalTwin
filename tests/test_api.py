import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

from paintmatch.index import PAINTINGS_FILE

from conftest import FakeEmbedder, photo_bytes

ENV_VARS = ("PAINTMATCH_IMAGE_BASE_URL", "PAINTMATCH_ALLOWED_ORIGINS", "PAINTMATCH_SELFIE_DET_SIZE")


def load_app(monkeypatch, data_dir, **env):
    """Import backend.app fresh, since its configuration is read from env vars at import time."""
    monkeypatch.setenv("PAINTMATCH_DATA_DIR", str(data_dir))
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    sys.modules.pop("backend.app", None)
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "FaceEmbedder", FakeEmbedder)
    return module


@pytest.fixture
def client(monkeypatch, data_dir):
    with TestClient(load_app(monkeypatch, data_dir).app) as c:
        yield c


def post_photo(client, data: bytes, k: int | None = None):
    url = "/match" if k is None else f"/match?k={k}"
    return client.post(url, files={"file": ("selfie.png", data, "image/png")})


def test_health(client):
    assert client.get("/health").json() == {"ok": True, "faces_indexed": 4}


def test_match_returns_best_painting_first(client):
    res = post_photo(client, photo_bytes(red=2))
    assert res.status_code == 200
    body = res.json()
    top = body["matches"][0]
    assert top["image_id"] == "bbb"
    assert top["score"] == pytest.approx(1.0, abs=1e-5)
    assert top["thumb_url"] == "/thumbs/bbb.jpg"
    assert top["crop_url"] == "/crops/bbb_0.jpg"
    assert body["selfie_face"] == pytest.approx({"x1": 0.25, "y1": 0.2, "x2": 0.75, "y2": 0.7}, abs=0.01)


def test_match_k_is_clamped(client):
    assert len(post_photo(client, photo_bytes(red=0), k=100).json()["matches"]) == 3  # one per painting
    assert len(post_photo(client, photo_bytes(red=0), k=0).json()["matches"]) == 1


def test_painting_images_are_served_with_cache_headers(client, data_dir):
    (data_dir / "crops" / "bbb_0.jpg").write_bytes(photo_bytes(red=2, fmt="JPEG"))
    res = client.get("/crops/bbb_0.jpg")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "public, max-age=604800"


def test_no_face_is_422(client):
    res = post_photo(client, photo_bytes(face=False))
    assert res.status_code == 422
    assert res.json()["detail"] == "No face found in the photo"


def test_unreadable_image_is_400(client):
    assert post_photo(client, b"definitely not an image").status_code == 400


def test_oversized_upload_is_413(client):
    assert post_photo(client, b"\0" * (15 * 1024 * 1024 + 1)).status_code == 413


def test_selfies_are_never_written_to_disk(client, data_dir):
    before = sorted(p for p in data_dir.rglob("*"))
    assert post_photo(client, photo_bytes(red=1)).status_code == 200
    assert sorted(p for p in data_dir.rglob("*")) == before


def test_painting_metadata_in_response(monkeypatch, data_dir):
    (data_dir / "index" / PAINTINGS_FILE).write_text(json.dumps(
        {"bbb": {"artist": "Ilya Repin", "title": "Ukranian Girl", "year": "1875"}}))
    with TestClient(load_app(monkeypatch, data_dir).app) as client:
        top = post_photo(client, photo_bytes(red=2)).json()["matches"][0]
    assert (top["artist"], top["title"], top["year"]) == ("Ilya Repin", "Ukranian Girl", "1875")


def test_image_base_url_makes_absolute_urls_and_skips_local_images(monkeypatch, data_dir):
    (data_dir / "thumbs").rmdir()  # the Docker image ships without thumbnails
    (data_dir / "crops").rmdir()
    module = load_app(monkeypatch, data_dir, PAINTMATCH_IMAGE_BASE_URL="https://img.example.com/")
    with TestClient(module.app) as client:
        top = post_photo(client, photo_bytes(red=2)).json()["matches"][0]
    assert top["thumb_url"] == "https://img.example.com/thumbs/bbb.jpg"
    assert top["crop_url"] == "https://img.example.com/crops/bbb_0.jpg"


def test_missing_images_without_base_url_fails_clearly(monkeypatch, data_dir):
    (data_dir / "thumbs").rmdir()
    with pytest.raises(RuntimeError, match="PAINTMATCH_IMAGE_BASE_URL"):
        load_app(monkeypatch, data_dir)


def test_cors_allows_only_configured_origins(monkeypatch, data_dir):
    module = load_app(monkeypatch, data_dir, PAINTMATCH_ALLOWED_ORIGINS="https://twin.example")
    with TestClient(module.app) as client:
        def preflight(origin):
            return client.options("/match", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})

        assert preflight("https://twin.example").headers.get("access-control-allow-origin") == "https://twin.example"
        assert "access-control-allow-origin" not in preflight("https://evil.example").headers


def test_selfie_detector_size_from_env(monkeypatch, data_dir):
    FakeEmbedder.instances.clear()
    module = load_app(monkeypatch, data_dir, PAINTMATCH_SELFIE_DET_SIZE="320")
    with TestClient(module.app):
        pass
    assert FakeEmbedder.instances[-1].kwargs["det_size"] == 320
