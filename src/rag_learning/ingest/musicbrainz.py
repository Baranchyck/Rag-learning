import argparse
import json
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://musicbrainz.org/ws/2/release-group/"
USER_AGENT = "rag-learning/0.1 ( https://github.com/Baranchyck )"
MIN_INTERVAL = 1.0
SEARCH_LIMIT = 5
TOP_TAGS = 5

NON_WORD_RE = re.compile(r"[\W_]+")
LEADING_THE_RE = re.compile(r"^the\s+")


class MBMatch(BaseModel):
    guardian_id: str
    mbid: str | None = None
    mb_title: str | None = None
    mb_artist: str | None = None
    first_release_date: str | None = None
    primary_type: str | None = None
    tags: list[str] = []
    score: int | None = None


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retry = Retry(
        total=5,
        backoff_factor=1,
        # MusicBrainz при перевищенні ліміту відповідає 503, а не 429
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class RateLimiter:
    def __init__(
        self,
        min_interval: float = MIN_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_query(artist: str, album: str) -> str:
    return f'releasegroup:"{_escape(album)}" AND artist:"{_escape(artist)}"'


def search(
    session: requests.Session, limiter: RateLimiter, artist: str, album: str
) -> dict:
    limiter.wait()
    resp = session.get(
        API_URL,
        params={
            "query": build_query(artist, album),
            "limit": SEARCH_LIMIT,
            "fmt": "json",
        },
        timeout=(5, 30),
    )
    resp.raise_for_status()
    return resp.json()


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().strip().replace("&", " and ")
    s = LEADING_THE_RE.sub("", s)
    normalized = NON_WORD_RE.sub("", s)
    return normalized or name.casefold().strip()


def credit_names(group: dict) -> set[str]:
    credits = group.get("artist-credit", [])
    names = {normalize_name(c["name"]) for c in credits}
    full = "".join(c["name"] + c.get("joinphrase", "") for c in credits)
    if full:
        names.add(normalize_name(full))
    return names


def choose_candidate(
    data: dict, artist: str, album: str, min_score: int = 90
) -> dict | None:
    artist_norm = normalize_name(artist)
    album_norm = normalize_name(album)

    passed = [
        g
        for g in data.get("release-groups", [])
        if g.get("score", 0) >= min_score and artist_norm in credit_names(g)
    ]
    if not passed:
        return None

    def rank(g: dict) -> tuple[bool, bool, int]:
        return (
            normalize_name(g.get("title", "")) == album_norm,
            g.get("primary-type") == "Album",
            g.get("score", 0),
        )

    return max(passed, key=rank)


def to_match(guardian_id: str, group: dict | None) -> MBMatch:
    if group is None:
        return MBMatch(guardian_id=guardian_id)
    tags = sorted(group.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
    credits = group.get("artist-credit", [])
    return MBMatch(
        guardian_id=guardian_id,
        mbid=group["id"],
        mb_title=group.get("title"),
        mb_artist="".join(c["name"] + c.get("joinphrase", "") for c in credits) or None,
        first_release_date=group.get("first-release-date") or None,
        primary_type=group.get("primary-type"),
        tags=[t["name"] for t in tags[:TOP_TAGS]],
        score=group.get("score"),
    )


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("guardian_id"), str):
                done.add(obj["guardian_id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Знайти альбоми з рецензій Guardian у MusicBrainz"
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("data/processed/guardian_reviews.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/musicbrainz_matches.jsonl")
    )
    parser.add_argument(
        "--limit", type=int, help="обробити не більше N нових рецензій (для перевірки)"
    )
    parser.add_argument("--min-score", type=int, default=90)
    args = parser.parse_args()

    done = load_done(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"вже оброблено раніше: {len(done)}")

    matched = not_found = skipped = errors = 0
    limiter = RateLimiter()
    try:
        with (
            make_session() as session,
            args.inp.open(encoding="utf-8") as src,
            args.out.open("a", encoding="utf-8") as fh,
        ):
            for line in src:
                if args.limit is not None and matched + not_found >= args.limit:
                    break
                review = json.loads(line)
                gid = review["guardian_id"]
                if gid in done:
                    continue
                artist, album = review.get("artist"), review.get("album")
                if not artist or not album:
                    skipped += 1
                    continue

                try:
                    data = search(session, limiter, artist, album)
                except requests.RequestException as e:
                    print(f"помилка {gid}: {e}")
                    errors += 1
                    continue

                group = choose_candidate(data, artist, album, args.min_score)
                fh.write(to_match(gid, group).model_dump_json() + "\n")
                fh.flush()
                if group:
                    matched += 1
                else:
                    not_found += 1

                total = matched + not_found
                if total % 50 == 0:
                    print(f"{total}: знайдено {matched}, не знайдено {not_found}")
    except KeyboardInterrupt:
        print("\nперервано — наступний запуск продовжить з цього місця")

    print(
        f"знайдено {matched}, не знайдено {not_found}, "
        f"пропущено (немає artist/album) {skipped}, помилок {errors} → {args.out}"
    )


if __name__ == "__main__":
    main()
