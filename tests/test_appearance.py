import huggingface_hub
import numpy as np
import onnxruntime
import pytest
from huggingface_hub import constants
from PIL import Image

from paintmatch import appearance
from paintmatch.appearance import CROP_SIZE, AppearanceEmbedder, model_dir


class FakeSession:
    def __init__(self, path, providers=None):
        self.path = path


@pytest.fixture
def hf_home(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    return tmp_path


def test_baked_model_loads_without_downloading(monkeypatch, hf_home):
    # The Docker images bake the model in and run with HF_HUB_OFFLINE=1, so startup must not
    # touch the Hub once the files are there.
    for name in ("model.onnx", "model.onnx_data"):
        (model_dir() / "onnx").mkdir(parents=True, exist_ok=True)
        (model_dir() / "onnx" / name).write_bytes(b"")

    def no_download(*args, **kwargs):
        raise AssertionError("downloaded a model that was already there")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_download)
    embedder = AppearanceEmbedder()
    assert embedder._session.path == str(model_dir() / "onnx" / "model.onnx")
    assert model_dir().is_relative_to(hf_home)


def test_missing_model_is_downloaded_as_real_files(monkeypatch, hf_home):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: calls.append(kw))
    AppearanceEmbedder()
    # local_dir gives real files: onnxruntime rejects external weights behind the HF cache's symlinks.
    assert calls and calls[0]["local_dir"] == model_dir() and calls[0]["revision"] == appearance.REVISION


def test_preprocess_matches_the_model_input():
    pixels = AppearanceEmbedder.preprocess(Image.new("RGB", (300, 300), (124, 116, 104)))
    assert pixels.shape == (3, CROP_SIZE, CROP_SIZE)
    # ImageNet mean gray normalizes to about zero.
    assert np.abs(pixels).max() < 0.01
