import json

import numpy as np
import pytest

from paintmatch.index import (
    INDEX_FILE,
    PAINTINGS_FILE,
    FaceIndex,
    columns_to_rows,
    load_index,
    rows_to_columns,
    save_index,
    FACE_FIELDS,
)

from conftest import appearance_for, embedding_for, face_row


def test_save_load_round_trip(data_dir):
    embeddings, faces, images = load_index(data_dir / "index")
    assert embeddings.shape == (4, 512)
    assert faces["image_id"].tolist() == ["aaa", "aaa", "bbb", "ccc"]
    assert faces["face_idx"].tolist() == [0, 1, 0, 0]
    assert images["n_faces"].tolist() == [2, 1, 1]
    assert images["error"].tolist() == ["", "", ""]
    np.testing.assert_allclose(embeddings[2], embedding_for(2))


def test_rows_columns_round_trip():
    rows = [face_row("aaa"), face_row("bbb", face_idx=1)]
    back = columns_to_rows(rows_to_columns(rows, FACE_FIELDS))
    assert [r["image_id"] for r in back] == ["aaa", "bbb"]
    assert back[1]["face_idx"] == 1 and isinstance(back[1]["face_idx"], int)
    assert back[0]["det_score"] == pytest.approx(0.8)


def test_save_is_atomic_and_overwrites(data_dir):
    index_dir = data_dir / "index"
    save_index(index_dir, np.stack([embedding_for(9)]), [face_row("zzz")],
               [{"image_id": "zzz", "image_path": "images/0/zzz.jpg", "n_faces": 1, "error": ""}])
    assert sorted(p.name for p in index_dir.iterdir()) == [INDEX_FILE]  # no leftover temp file
    assert load_index(index_dir)[1]["image_id"].tolist() == ["zzz"]


def test_save_rejects_misaligned_rows(tmp_path):
    with pytest.raises(AssertionError):
        save_index(tmp_path, np.stack([embedding_for(0)]), [], [])


def test_search_finds_exact_match_first(data_dir):
    index = FaceIndex(data_dir / "index")
    results = index.search(embedding_for(2), k=2)
    assert results[0]["image_id"] == "bbb"
    assert results[0]["face_id"] == 2
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    json.dumps(results)  # plain Python types only, so the API can serialize them


def test_one_result_per_painting_by_default(data_dir):
    index = FaceIndex(data_dir / "index")
    assert [r["image_id"] for r in index.search(embedding_for(1), k=10)].count("aaa") == 1
    all_faces = index.search(embedding_for(1), k=10, one_per_image=False)
    assert [r["image_id"] for r in all_faces].count("aaa") == 2


def test_min_det_score_filters_faces(data_dir):
    index = FaceIndex(data_dir / "index")
    results = index.search(embedding_for(3), k=10, min_det_score=0.5)
    assert "ccc" not in [r["image_id"] for r in results]


def test_search_and_rank_agree(data_dir):
    index = FaceIndex(data_dir / "index")
    query = embedding_for(0)
    scores = index.embeddings @ query
    assert index.search(query, k=3) == index.rank(scores, k=3, extra_scores={"arcface_score": scores})


def test_painting_metadata_merged_when_present(data_dir):
    (data_dir / "index" / PAINTINGS_FILE).write_text(json.dumps({
        "aaa": {"artist": "Rembrandt", "title": "Flora", "year": "1634", "source": "x.jpg", "phash_bits": 0},
        "bbb": {"artist": "Ilya Repin", "title": None, "year": None, "source": "y.jpg", "phash_bits": 2},
    }))
    index = FaceIndex(data_dir / "index")
    by_id = {r["image_id"]: r for r in index.search(embedding_for(0), k=10)}
    assert {k: by_id["aaa"][k] for k in ("artist", "title", "year")} == {"artist": "Rembrandt", "title": "Flora", "year": "1634"}
    assert by_id["bbb"]["artist"] == "Ilya Repin" and "title" not in by_id["bbb"] and "year" not in by_id["bbb"]
    assert "artist" not in by_id["ccc"]
    assert "source" not in by_id["aaa"] and "phash_bits" not in by_id["aaa"]  # internal fields stay internal


def test_no_metadata_file_is_fine(data_dir):
    index = FaceIndex(data_dir / "index")
    assert index.paintings == {}
    assert "artist" not in index.search(embedding_for(0), k=1)[0]


def test_blend_follows_the_arcface_weight(data_dir_with_appearance):
    index = FaceIndex(data_dir_with_appearance / "index")
    assert index.has_appearance
    # Identity points at row 0 ("aaa"); appearance_for(1) points at row 2 ("bbb").
    def top(weight):
        return index.search(embedding_for(0), k=1, appearance_query=appearance_for(1), arcface_weight=weight)[0]
    assert top(1.0)["image_id"] == "aaa"
    assert top(0.0)["image_id"] == "bbb"
    assert top(0.3)["image_id"] == "bbb"  # the production default leans on appearance
    result = top(0.3)
    assert result["appearance_score"] == pytest.approx(1.0, abs=1e-5)
    assert -1 <= result["arcface_score"] <= 1


def test_no_appearance_query_means_arcface_only(data_dir_with_appearance):
    index = FaceIndex(data_dir_with_appearance / "index")
    result = index.search(embedding_for(2), k=1)[0]
    assert result["image_id"] == "bbb" and "appearance_score" not in result


def test_stale_appearance_file_is_rejected(data_dir):
    from paintmatch.index import save_appearance

    # Rows in the wrong order, as if built for a different index.
    save_appearance(data_dir / "index", np.stack([appearance_for(i) for i in range(4)]),
                    ["bbb", "aaa", "aaa", "ccc"], [0, 0, 1, 0])
    with pytest.raises(ValueError, match="doesn't match"):
        FaceIndex(data_dir / "index")


def test_appearance_from_another_model_is_rejected(data_dir):
    from paintmatch.index import APPEARANCE_FILE

    _, faces, _ = load_index(data_dir / "index")
    np.savez(data_dir / "index" / APPEARANCE_FILE, embeddings=np.stack([appearance_for(i) for i in range(4)]),
             image_id=faces["image_id"], face_idx=faces["face_idx"], model=np.array("some/other-model@abc"))
    with pytest.raises(ValueError, match="built with"):
        FaceIndex(data_dir / "index")
