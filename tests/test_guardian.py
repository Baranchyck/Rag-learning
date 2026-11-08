import pytest

from rag_learning.ingest.guardian import clean_text, parse_headline


@pytest.mark.parametrize(
    ("headline", "artist", "album"),
    [
        ("La Roux: Old Flames review – a breezy twist", "La Roux", "Old Flames"),
        (
            "Stravinsky: The Firebird album review – Rattle revels",
            "Stravinsky",
            "The Firebird",
        ),
        (
            "Strauss: An Alpine Symphony album review – mystic ",
            "Strauss",
            "An Alpine Symphony",
        ),
        (
            "Enrico Rava Fearless Five with Joe Lovano: Corners review – daring",
            "Enrico Rava Fearless Five with Joe Lovano",
            "Corners",
        ),
        ("Jazz album of the month – Julian Siegel Quartet: Vista", "Julian Siegel Quartet", "Vista"),
        ("Messiaen: Catalogue d'Oiseaux – Aimard's birds soar", "Messiaen", "Catalogue d'Oiseaux"),
        ("Mø review – plenty of swagger", "Mø", None),
        ("Some interview that is not a review", None, None),
    ],
)
def test_parse_headline(headline, artist, album):
    assert parse_headline(headline) == (artist, album)


def test_clean_text_unescapes_and_strips_junk():
    raw = "Soulful R&amp;B\u00a0vibes. Listen on Apple Music (above) or Spotify"
    assert clean_text(raw) == "Soulful R&B vibes."
