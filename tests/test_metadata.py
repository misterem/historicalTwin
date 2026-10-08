import pytest

from build_metadata import parse_slug, smart_case


@pytest.mark.parametrize(
    "slug, expected",
    [
        ("portrait-of-catherine-ii-of-russia-1794", ("Portrait of Catherine II of Russia", "1794")),
        ("louis-xiv-of-france-1701", ("Louis XIV of France", "1701")),
        ("portrait-of-charles-i-on-horseback", ("Portrait of Charles I on Horseback", None)),
        ("the-death-of-the-virgin-c-1606", ("The Death of the Virgin", "c. 1606")),
        ("self-portrait-1889-1", ("Self Portrait", "1889")),
        ("daughter-of-jacob-meyer-1881(1)", ("Daughter of Jacob Meyer", "1881")),
        ("portrait-of-simonetta-vespucci(1)", ("Portrait of Simonetta Vespucci", None)),
        ("female-portrait-2", ("Female Portrait", None)),
        ("women-17", ("Women", None)),
        ("portrait-of-a-girl-at-the-age-of-10", ("Portrait of a Girl at the Age of 10", None)),
        ("not_detected_242418", (None, None)),
        ("the-milkmaid", ("The Milkmaid", None)),
    ],
)
def test_parse_slug(slug, expected):
    assert parse_slug(slug) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("vincent van gogh", "Vincent van Gogh"),
        ("leonardo da vinci", "Leonardo da Vinci"),
        ("pierre-auguste renoir", "Pierre-Auguste Renoir"),
        ("rembrandt", "Rembrandt"),
    ],
)
def test_smart_case_artist_names(name, expected):
    assert smart_case(name.split()) == expected
