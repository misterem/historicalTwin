import sys

import pytest

import build_index
from paintmatch.index import load_index

from conftest import FakeEmbedder, photo_bytes


@pytest.fixture
def raw_dir(tmp_path):
    images = tmp_path / "raw" / "images" / "0"
    images.mkdir(parents=True)
    (images / "face1.jpg").write_bytes(photo_bytes(red=10, fmt="JPEG"))
    (images / "face2.jpg").write_bytes(photo_bytes(red=20, fmt="JPEG"))
    (images / "noface.jpg").write_bytes(photo_bytes(face=False, fmt="JPEG"))
    (images / "broken.jpg").write_bytes(b"not a jpeg")
    return images


def run_indexer(monkeypatch, data_dir, *extra_args):
    monkeypatch.setattr(build_index, "FaceEmbedder", FakeEmbedder)
    monkeypatch.setattr(sys, "argv", ["build_index.py", "--skip-download", "--data-dir", str(data_dir),
                                      "--load-workers", "2", *extra_args])
    build_index.main()
    return load_index(data_dir / "index")


def test_indexes_faces_and_records_failures(monkeypatch, tmp_path, raw_dir):
    embeddings, faces, images = run_indexer(monkeypatch, tmp_path)
    assert sorted(faces["image_id"].tolist()) == ["face1", "face2"]
    assert len(embeddings) == 2
    by_id = {r: (n, e) for r, n, e in zip(images["image_id"].tolist(), images["n_faces"].tolist(), images["error"].tolist())}
    assert by_id["noface"] == (0, "")
    assert by_id["broken"][0] == 0 and "UnidentifiedImageError" in by_id["broken"][1]
    assert sorted(p.name for p in (tmp_path / "thumbs").iterdir()) == ["face1.jpg", "face2.jpg", "noface.jpg"]
    assert sorted(p.name for p in (tmp_path / "crops").iterdir()) == ["face1_0.jpg", "face2_0.jpg"]
    assert all(0 <= v <= 1 for v in faces["x2"].tolist())  # boxes are normalized


def test_resume_skips_done_and_retries_failures(monkeypatch, tmp_path, raw_dir):
    run_indexer(monkeypatch, tmp_path)
    (raw_dir / "broken.jpg").write_bytes(photo_bytes(red=30, fmt="JPEG"))  # "fixed" download
    embeddings, faces, images = run_indexer(monkeypatch, tmp_path)
    assert sorted(faces["image_id"].tolist()) == ["broken", "face1", "face2"]
    assert len(embeddings) == 3
    assert sorted(images["image_id"].tolist()) == ["broken", "face1", "face2", "noface"]  # no duplicate rows
    assert images["error"].tolist() == ["", "", "", ""]


def test_fresh_rebuilds_from_scratch(monkeypatch, tmp_path, raw_dir):
    run_indexer(monkeypatch, tmp_path)
    (raw_dir / "face2.jpg").unlink()
    _, faces, images = run_indexer(monkeypatch, tmp_path, "--fresh")
    assert sorted(faces["image_id"].tolist()) == ["face1"]
    assert "face2" not in images["image_id"].tolist()


def test_prefetch_keeps_order_and_surfaces_errors():
    def work(n):
        if n == 3:
            raise ValueError("boom")
        return n * 10

    results = list(build_index.prefetch(work, range(6), workers=3, depth=2))
    assert [item for item, _, _ in results] == list(range(6))
    assert [r for _, r, e in results if e is None] == [0, 10, 20, 40, 50]
    assert isinstance(results[3][2], ValueError)
