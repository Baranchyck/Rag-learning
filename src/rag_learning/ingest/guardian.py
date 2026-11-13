import argparse
import html
import re
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import requests
from pydantic import BaseModel, Field, ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://content.guardianapis.com/search"
PAGE_SIZE = 200  # максимум, який дозволяє API
FIELDS = "headline,trailText,byline,starRating,bodyText"

PREFIX_RE = re.compile(
    r"^[\w ]*album of the (?:week|month)(?:\s+review)?\s*[–:-]\s*", re.IGNORECASE
)
TITLE_RE = re.compile(
    r"^(?P<artist>[^:]+?):\s*(?P<album>.+?)\s+(?:(?:album|soundtrack)\s+)?review\b",
    re.IGNORECASE,
)
NO_REVIEW_RE = re.compile(r"^(?P<artist>[^:]+?):\s*(?P<album>.+?)(?:\s+[–-]\s+|\s*$)")
ARTIST_ONLY_RE = re.compile(
    r"^(?P<artist>[^:–-]+?)\s+(?:(?:album|soundtrack)\s+)?review\s*[–-]", re.IGNORECASE
)
DIGEST_RE = re.compile(r"^Home listening\b", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
OTHER_PICKS_RE = re.compile(
    r"\s*(?:Also out this (?:week|month)|This (?:week|month)[’']s other picks)\b.*$",
    re.DOTALL,
)
JUNK_RE = re.compile(r"\s*Listen on (Apple Music|Spotify)[^.]*$")
WS_RE = re.compile(r"\s+")


class Review(BaseModel):
    guardian_id: str
    url: str
    published_at: datetime
    headline: str
    artist: str | None = None
    album: str | None = None
    author: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    trail: str | None = None
    body: str


def parse_headline(headline: str) -> tuple[str | None, str | None]:
    h = PREFIX_RE.sub("", headline.strip())
    for rx in (TITLE_RE, NO_REVIEW_RE):  # від суворішого до м'якшого
        if m := rx.match(h):
            return m["artist"].strip(), m["album"].strip()
    if m := ARTIST_ONLY_RE.match(h):
        return m["artist"].strip(), None
    return None, None


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = JUNK_RE.sub("", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def to_review(item: dict) -> Review:
    f = item.get("fields", {})
    headline = clean_text(f.get("headline", item["webTitle"]))
    artist, album = parse_headline(headline)
    return Review(
        guardian_id=item["id"],
        url=item["webUrl"],
        published_at=item["webPublicationDate"],
        headline=headline,
        artist=artist,
        album=album,
        author=f.get("byline"),
        rating=f.get("starRating"),
        trail=clean_text(f.get("trailText", "")) or None,
        body=clean_text(OTHER_PICKS_RE.sub("", f.get("bodyText", ""))),
    )


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,  # паузи 1, 2, 4, 8... секунд між повторами
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_items(
    session: requests.Session,
    api_key: str,
    from_date: str | None,
    max_pages: int | None = None,
    delay: float = 1.0,
) -> Iterator[dict]:
    page = 1
    while True:
        params = {
            "api-key": api_key,
            "tag": "tone/albumreview",
            "show-fields": FIELDS,
            "page-size": PAGE_SIZE,
            "page": page,
            "order-by": "oldest",
        }
        if from_date:
            params["from-date"] = from_date

        resp = session.get(API_URL, params=params, timeout=(5, 30))
        resp.raise_for_status()
        data = resp.json()["response"]

        print(f"сторінка {data['currentPage']}/{data['pages']}")
        yield from data["results"]

        if page >= data["pages"] or (max_pages and page >= max_pages):
            break
        page += 1
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Завантажити рецензії на альбоми з Guardian API"
    )
    parser.add_argument(
        "--from-date", default="2018-01-01", help="YYYY-MM-DD; 'all' — весь архів"
    )
    parser.add_argument(
        "--max-pages", type=int, help="обмежити кількість сторінок (для перевірки)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/guardian_reviews.jsonl")
    )
    args = parser.parse_args()

    from rag_learning.config import GUARDIAN_API_KEY

    from_date = None if args.from_date == "all" else args.from_date
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    saved = skipped = 0
    with make_session() as session, args.out.open("w", encoding="utf-8") as fh:
        for item in fetch_items(session, GUARDIAN_API_KEY, from_date, args.max_pages):
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            if DIGEST_RE.match(item.get("fields", {}).get("headline", "")):
                skipped += 1
                continue
            try:
                review = to_review(item)
            except ValidationError as e:
                print(f"пропущено {item['id']}: {e.error_count()} помилок валідації")
                skipped += 1
                continue
            if not review.body:
                skipped += 1
                continue
            fh.write(review.model_dump_json() + "\n")
            saved += 1

    print(f"збережено {saved}, пропущено {skipped} → {args.out}")


if __name__ == "__main__":
    main()
