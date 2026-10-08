from publish import FLAT_IMAGE, shard, stage_assets


def make_data(tmp_path):
    data = tmp_path / "data"
    (data / "index").mkdir(parents=True)
    (data / "index" / "index.npz").write_bytes(b"npz")
    (data / "index" / "paintings.json").write_text("{}")
    for folder, names in {"thumbs": ["ab12.jpg", "cd34.jpg"], "crops": ["ab12_0.jpg", "ab12_1.jpg"]}.items():
        (data / folder).mkdir()
        for name in names:
            (data / folder / name).write_bytes(name.encode())
    return data


def test_stage_assets_shards_images_by_id_prefix(tmp_path):
    data = make_data(tmp_path)
    stage = data / ".publish-assets"
    assert stage_assets(data, stage) == 6
    staged = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    assert staged == ["crops/ab/ab12_0.jpg", "crops/ab/ab12_1.jpg", "index/index.npz",
                      "index/paintings.json", "thumbs/ab/ab12.jpg", "thumbs/cd/cd34.jpg"]
    # Hard links, not copies.
    assert (stage / "thumbs/ab/ab12.jpg").stat().st_ino == (data / "thumbs/ab12.jpg").stat().st_ino


def test_restaging_keeps_upload_cache_and_drops_stale_files(tmp_path):
    data = make_data(tmp_path)
    stage = data / ".publish-assets"
    stage_assets(data, stage)
    (stage / ".cache").mkdir()
    (stage / ".cache" / "resume.json").write_text("{}")
    (data / "thumbs" / "cd34.jpg").unlink()
    stage_assets(data, stage)
    assert (stage / ".cache" / "resume.json").exists()
    assert not (stage / "thumbs" / "cd").exists()


def test_flat_layout_detection():
    assert shard("ab12_0.jpg") == "ab/ab12_0.jpg"
    assert FLAT_IMAGE.match("crops/ab12_0.jpg")
    assert not FLAT_IMAGE.match("crops/ab/ab12_0.jpg")
    assert not FLAT_IMAGE.match("index/index.npz")
