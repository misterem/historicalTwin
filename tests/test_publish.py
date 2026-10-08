import hashlib
import tarfile

from publish import LOOSE_IMAGE, stage_assets


def make_data(tmp_path):
    data = tmp_path / "data"
    (data / "index").mkdir(parents=True)
    (data / "index" / "index.npz").write_bytes(b"npz")
    (data / "index" / "paintings.json").write_text("{}")
    for folder, names in {"thumbs": ["cd34.jpg", "ab12.jpg"], "crops": ["ab12_0.jpg", "ab12_1.jpg"]}.items():
        (data / folder).mkdir()
        for name in names:
            (data / folder / name).write_bytes(name.encode())
    return data


def test_stage_assets_packs_images_into_archives(tmp_path):
    data = make_data(tmp_path)
    stage = data / ".publish-assets"
    assert stage_assets(data, stage) == {"thumbs": 2, "crops": 2}
    staged = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    assert staged == ["images/crops.tar", "images/thumbs.tar", "index/index.npz", "index/paintings.json"]
    # Index files are hard links, not copies.
    assert (stage / "index/index.npz").stat().st_ino == (data / "index/index.npz").stat().st_ino
    # Archives unpack to the flat paths the app serves, sorted, with the original bytes.
    with tarfile.open(stage / "images/thumbs.tar") as tar:
        assert tar.getnames() == ["thumbs/ab12.jpg", "thumbs/cd34.jpg"]
        assert tar.extractfile("thumbs/cd34.jpg").read() == b"cd34.jpg"


def test_archives_are_reproducible(tmp_path):
    data = make_data(tmp_path)
    stage = data / ".publish-assets"
    stage_assets(data, stage)
    first = hashlib.sha256((stage / "images/crops.tar").read_bytes()).hexdigest()
    (data / "crops" / "ab12_0.jpg").touch()  # newer mtime, same content
    stage_assets(data, stage)
    assert hashlib.sha256((stage / "images/crops.tar").read_bytes()).hexdigest() == first


def test_restaging_keeps_upload_cache_and_drops_stale_files(tmp_path):
    data = make_data(tmp_path)
    stage = data / ".publish-assets"
    stage_assets(data, stage)
    (stage / ".cache").mkdir()
    (stage / ".cache" / "resume.json").write_text("{}")
    (data / "thumbs" / "cd34.jpg").unlink()
    assert stage_assets(data, stage)["thumbs"] == 1
    assert (stage / ".cache" / "resume.json").exists()


def test_loose_image_detection():
    assert LOOSE_IMAGE.match("crops/ab12_0.jpg")
    assert LOOSE_IMAGE.match("thumbs/ab/ab12.jpg")
    assert not LOOSE_IMAGE.match("images/crops.tar")
    assert not LOOSE_IMAGE.match("index/index.npz")


def test_build_context_pins_assets_commit(tmp_path):
    from publish import BUILD_FILES, stage_build_context

    stage_build_context(tmp_path, "someone/assets", "abc123")
    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "ARG ASSETS_REPO=someone/assets" in dockerfile
    assert "ARG ASSETS_REVISION=abc123" in dockerfile
    assert "__ASSETS" not in dockerfile
    for name in BUILD_FILES:
        assert (tmp_path / name).exists()
    assert not list(tmp_path.rglob("__pycache__"))


def test_cloud_run_command(tmp_path):
    import argparse

    from publish import run_deploy_command

    args = argparse.Namespace(service="svc", region="us-central1", project="proj-1", max_instances=3,
                              allowed_origins="https://a.example,https://b.example")
    cmd = run_deploy_command(args, tmp_path)
    assert cmd[:4] == ["gcloud", "run", "deploy", "svc"]
    assert cmd[cmd.index("--source") + 1] == str(tmp_path)
    assert cmd[cmd.index("--max-instances") + 1] == "3"
    assert cmd[cmd.index("--min-instances") + 1] == "0"
    assert "--allow-unauthenticated" in cmd
    # Custom delimiter, so the comma-separated origins stay one value.
    assert cmd[cmd.index("--set-env-vars") + 1] == "^|^PAINTMATCH_ALLOWED_ORIGINS=https://a.example,https://b.example"
    assert cmd[cmd.index("--project") + 1] == "proj-1"
