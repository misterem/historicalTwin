import sys

import build_appearance
from paintmatch.appearance import CACHE_FILE
from paintmatch.index import FaceIndex, crop_name, load_index

from conftest import FakeAppearanceEmbedder, appearance_for, photo_bytes


def test_builds_aligned_appearance_embeddings(monkeypatch, data_dir):
    _, faces, _ = load_index(data_dir / "index")
    keys = list(zip(faces["image_id"].tolist(), faces["face_idx"].tolist()))
    for row, key in enumerate(keys):  # crop for face row i "looks like" appearance_for(10 + i)
        (data_dir / "crops" / crop_name(*key)).write_bytes(photo_bytes(blue=10 + row))
    monkeypatch.setattr(build_appearance, "AppearanceEmbedder", FakeAppearanceEmbedder)
    monkeypatch.setattr(sys, "argv", ["build_appearance.py", "--data-dir", str(data_dir)])
    build_appearance.main()

    index = FaceIndex(data_dir / "index")
    assert index.has_appearance
    for row in range(len(keys)):
        assert (index.appearance[row] @ appearance_for(10 + row)) > 0.999
    assert (data_dir / "cache" / CACHE_FILE).exists()


def test_refuses_when_crops_are_missing(monkeypatch, data_dir):
    import pytest

    monkeypatch.setattr(sys, "argv", ["build_appearance.py", "--data-dir", str(data_dir)])
    with pytest.raises(SystemExit, match="face crops are missing"):
        build_appearance.main()
