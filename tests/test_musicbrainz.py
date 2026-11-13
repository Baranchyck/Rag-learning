import json
from pathlib import Path

import pytest

from rag_learning.ingest.musicbrainz import (
    RateLimiter,
    build_query,
    choose_candidate,
    load_done,
    normalize_name,
    to_match,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def ok_computer() -> dict:
    return json.loads((FIXTURES / "mb_ok_computer.json").read_text(encoding="utf-8"))


def test_build_query_escapes_quotes_and_backslash():
    q = build_query("AC\\DC", 'The "Black" Album')
    assert q == 'releasegroup:"The \\"Black\\" Album" AND artist:"AC\\\\DC"'


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("The National", "National"),
        ("Sigur Rós", "sigur ros"),
        ("Simon & Garfunkel", "Simon and Garfunkel"),
    ],
)
def test_normalize_name(a, b):
    assert normalize_name(a) == normalize_name(b)


def test_choose_real_response(ok_computer):
    g = choose_candidate(ok_computer, "Radiohead", "OK Computer")
    assert g["id"] == "b1392450-e666-3926-a536-22c65f834433"


def test_choose_rejects_other_artist(ok_computer):
    assert choose_candidate(ok_computer, "Coldplay", "OK Computer") is None


def test_choose_prefers_exact_title_over_score():
    def group(title, score, gid):
        return {
            "id": gid,
            "title": title,
            "score": score,
            "primary-type": "Album",
            "artist-credit": [{"name": "Radiohead"}],
        }

    data = {
        "release-groups": [
            group("OK Computer OKNOTOK 1997 2017", 100, "reissue"),
            group("OK Computer", 95, "original"),
        ]
    }
    assert choose_candidate(data, "Radiohead", "OK Computer")["id"] == "original"


def test_to_match_real_response(ok_computer):
    m = to_match("music/2017/ok", ok_computer["release-groups"][0])
    assert m.mbid == "b1392450-e666-3926-a536-22c65f834433"
    assert m.first_release_date == "1997-05-21"
    assert m.tags[:2] == ["alternative rock", "art rock"]


def test_rate_limiter_sleeps_only_remaining_time():
    t = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        t[0] += s

    rl = RateLimiter(1.0, clock=lambda: t[0], sleep=sleep)
    rl.wait()
    t[0] += 0.3
    rl.wait()
    assert slept == [pytest.approx(0.7)]


def test_load_done_skips_junk(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        '{"guardian_id": "a"}\n'
        '{"created": "raw api response"}\n'
        '{"guardian_id": "c", "mb',
        encoding="utf-8",
    )
    assert load_done(p) == {"a"}
