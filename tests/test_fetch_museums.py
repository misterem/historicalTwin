import json

import numpy as np

from build_metadata import MUSEUM_MANIFEST, museum_metadata
from fetch_museums import Fingerprints, met_candidate, met_selected, safe_id, si_candidate


def met_row(**overrides):
    row = {
        "Object ID": "436532", "Is Public Domain": "True", "Department": "European Paintings",
        "Classification": "Paintings", "Object Name": "Painting", "Title": "Portrait of a Woman",
        "Tags": "Women|Portraits", "Artist Display Name": "Frans Hals|Workshop", "Object Date": "ca. 1650",
        "Link Resource": "http://www.metmuseum.org/art/collection/search/436532", "Credit Line": "Bequest, 1913",
    }
    return row | overrides


def test_met_selects_public_domain_painted_portraits():
    assert met_selected(met_row())
    assert met_selected(met_row(Tags="", Title="Mrs. Smith", **{"Object Name": "Portrait"}))
    assert not met_selected(met_row(**{"Is Public Domain": "False"}))
    assert not met_selected(met_row(Department="Asian Art"))
    assert not met_selected(met_row(Classification="Prints", **{"Object Name": "Print"}))
    # The American Wing has no Classification; its object names say what the work is.
    assert met_selected(met_row(Department="The American Wing", Classification="", **{"Object Name": "Painting, miniature"}))
    assert not met_selected(met_row(Department="The American Wing", Classification="", **{"Object Name": "Bust"}))
    assert not met_selected(met_row(Tags="Landscapes", Title="View of Delft"))


def test_met_candidate_fields():
    c = met_candidate(met_row())
    assert c["image_id"] == "met-436532"
    assert c["artist"] == "Frans Hals"  # first of several constituents
    assert (c["title"], c["year"], c["license"]) == ("Portrait of a Woman", "ca. 1650", "CC0")


def si_record(object_types=("Paintings",), topics=("Portraits",), access="CC0"):
    return {
        "id": "ld1-1", "type": "edanmdm",
        "content": {
            "freetext": {
                "name": [{"label": "Sitter", "content": "Thomas Jonathan Jackson"},
                         {"label": "Artist", "content": "John Adams Elder, 3 Feb 1833 - 24 Feb 1895"}],
                "date": [{"label": "Date", "content": "1876"}],
                "creditLine": [{"label": "Credit Line", "content": "National Portrait Gallery"}],
            },
            "indexedStructured": {"object_type": list(object_types), "topic": list(topics)},
            "descriptiveNonRepeating": {
                "record_ID": "npg_NPG.2019.24", "record_link": "https://npg.si.edu/object/npg_NPG.2019.24",
                "title": {"content": "Thomas Jonathan Jackson"},
                "online_media": {"media": [{"type": "Images", "idsId": "NPG-NPG_2019_24", "usage": {"access": access}}]},
            },
        },
    }


def test_si_candidate_from_dump_record():
    c = si_candidate(si_record(), "npg", max_side=1280)
    assert c["image_id"] == "si-npg-NPG-2019-24"
    assert c["artist"] == "John Adams Elder"  # life dates dropped; the sitter isn't the artist
    assert (c["title"], c["year"], c["museum"]) == ("Thomas Jonathan Jackson", "1876", "National Portrait Gallery")
    assert c["image_url"].endswith("id=NPG-NPG_2019_24&max=1280")


def test_si_candidate_rules():
    assert si_candidate(si_record(object_types=("Prints",)), "npg", 1280) is None
    assert si_candidate(si_record(access="Usage conditions apply"), "npg", 1280) is None
    # American Art Museum paintings need the Portraits tag; every NPG painting is a portrait.
    assert si_candidate(si_record(topics=("Landscapes",)), "saam", 1280) is None
    assert si_candidate(si_record(topics=("Landscapes",)), "npg", 1280) is not None


def test_safe_id_has_no_underscores():
    assert safe_id("si-npg_S_NPG.65.103") == "si-npg-S-NPG-65-103"


def test_fingerprints_find_duplicates():
    known = Fingerprints(["aaa"], np.array([0b1111], np.uint64), np.array([0.75]))
    assert known.find((0b0111, 0.75), max_bits=4, max_aspect_diff=0.03) == "aaa"  # 1 bit off
    assert known.find((0b0111, 1.33), max_bits=4, max_aspect_diff=0.03) is None  # other shape
    assert known.find((0b1111 << 40, 0.75), max_bits=4, max_aspect_diff=0.03) is None  # far
    known.add("met-1", (1 << 60, 1.0))
    assert known.find((1 << 60, 1.0), max_bits=4, max_aspect_diff=0.03) == "met-1"


def test_museum_metadata_comes_from_the_manifest(tmp_path):
    assert museum_metadata(tmp_path) == {}
    path = tmp_path / MUSEUM_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "met-1": {"status": "ok", "artist": "Frans Hals", "title": "Portrait", "year": "1650",
                  "museum": "The Metropolitan Museum of Art", "url": "http://x", "credit": "Gift", "license": "CC0",
                  "phash": "00ff", "aspect": 0.75},
        "met-2": {"status": "duplicate", "duplicate_of": "abc"},
        "met-3": {"status": "error HTTPError: 403"},
    }))
    meta = museum_metadata(tmp_path)
    assert list(meta) == ["met-1"]
    assert meta["met-1"]["artist"] == "Frans Hals" and "phash" not in meta["met-1"]
    assert meta["met-1"]["url"] == "https://x"  # upgraded from the Met's http links
