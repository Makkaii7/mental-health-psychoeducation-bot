"""
Download public psychoeducation content into ``data/rag_corpus/`` for the RAG pipeline.

Sources are either US-federal (public domain under 17 U.S.C. § 105) or WHO content
(typically CC BY-NC-SA 3.0 IGO for fact sheets / Q&As). Each saved file carries a
metadata header: title, source URL, date accessed, and a license/terms note.

Run from project root:
    python scripts/download_rag_corpus.py

Re-runs overwrite existing files. Failures are logged and skipped — the script keeps
going so a single blocked URL does not abort the whole corpus build.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # bootstrap on first run
    print("[setup] Installing beautifulsoup4...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "beautifulsoup4"])
    from bs4 import BeautifulSoup  # noqa: E402


USER_AGENT = (
    "Mozilla/5.0 (compatible; PsychoeducationBotCorpus/1.0; "
    "+educational research; contact=student@khalifa.example)"
)
REQUEST_TIMEOUT = 30
INTER_REQUEST_DELAY_S = 1.0  # be polite; staggered hits across hosts

MIN_WORDS = 300
MAX_WORDS = 3000

NIMH_LICENSE = "U.S. federal government work — public domain (17 U.S.C. § 105)."
WHO_LICENSE = (
    "© World Health Organization. Fact sheets and Q&A content are typically "
    "licensed CC BY-NC-SA 3.0 IGO. Verify the source page before redistribution."
)
HHS_LICENSE = "U.S. federal government work — public domain (17 U.S.C. § 105)."
CDC_LICENSE = "U.S. federal government work — public domain (17 U.S.C. § 105)."


# (filename, url, title, license_note)
SOURCES: list[tuple[str, str, str, str]] = [
    # ── NIMH topic pages ──────────────────────────────────────────────
    (
        "nimh_depression.txt",
        "https://www.nimh.nih.gov/health/topics/depression",
        "Depression — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_anxiety_disorders.txt",
        "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
        "Anxiety Disorders — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_caring_for_mental_health.txt",
        "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health",
        "Caring for Your Mental Health — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_psychotherapies.txt",
        "https://www.nimh.nih.gov/health/topics/psychotherapies",
        "Psychotherapies — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_stress.txt",
        "https://www.nimh.nih.gov/health/publications/stress",
        "I'm So Stressed Out! Fact Sheet — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_ptsd.txt",
        "https://www.nimh.nih.gov/health/topics/post-traumatic-stress-disorder-ptsd",
        "Post-Traumatic Stress Disorder (PTSD) — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_ocd.txt",
        "https://www.nimh.nih.gov/health/topics/obsessive-compulsive-disorder-ocd",
        "Obsessive-Compulsive Disorder (OCD) — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    (
        "nimh_bipolar.txt",
        "https://www.nimh.nih.gov/health/topics/bipolar-disorder",
        "Bipolar Disorder — National Institute of Mental Health (NIMH)",
        NIMH_LICENSE,
    ),
    # ── WHO fact sheets / Q&As ───────────────────────────────────────
    (
        "who_mental_health.txt",
        "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
        "Mental health: strengthening our response — World Health Organization",
        WHO_LICENSE,
    ),
    (
        "who_depressive_disorder.txt",
        "https://www.who.int/news-room/fact-sheets/detail/depression",
        "Depressive disorder (depression) — World Health Organization",
        WHO_LICENSE,
    ),
    (
        "who_anxiety_disorders.txt",
        "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders",
        "Anxiety disorders — World Health Organization",
        WHO_LICENSE,
    ),
    (
        "who_stress.txt",
        "https://www.who.int/news-room/questions-and-answers/item/stress",
        "Stress — Questions and Answers — World Health Organization",
        WHO_LICENSE,
    ),
    # ── CDC (current /mental-health/ paths; old /mentalhealth/ URLs 404) ─
    (
        "cdc_about_mental_health.txt",
        "https://www.cdc.gov/mental-health/about/index.html",
        "About Mental Health — Centers for Disease Control and Prevention",
        CDC_LICENSE,
    ),
    (
        "cdc_living_with_mental_health.txt",
        "https://www.cdc.gov/mental-health/living-with/index.html",
        "Living with a Mental Health Condition — Centers for Disease Control and Prevention",
        CDC_LICENSE,
    ),
    # ── MedlinePlus (NIH / NLM, public domain) ───────────────────────
    (
        "medlineplus_depression.txt",
        "https://medlineplus.gov/depression.html",
        "Depression — MedlinePlus, U.S. National Library of Medicine (NIH)",
        NIMH_LICENSE,
    ),
    (
        "medlineplus_anxiety.txt",
        "https://medlineplus.gov/anxiety.html",
        "Anxiety — MedlinePlus, U.S. National Library of Medicine (NIH)",
        NIMH_LICENSE,
    ),
    (
        "medlineplus_stress.txt",
        "https://medlineplus.gov/stress.html",
        "Stress — MedlinePlus, U.S. National Library of Medicine (NIH)",
        NIMH_LICENSE,
    ),
]


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _pick_main_container(soup: BeautifulSoup):
    """Heuristic: prefer <main>, then <article>, then a known content div."""
    for finder in (
        lambda: soup.find("main"),
        lambda: soup.find("article"),
        lambda: soup.find("div", attrs={"role": "main"}),
        lambda: soup.find("div", id=re.compile(r"(?i)^(main|content|primary)")),
        lambda: soup.find("div", class_=re.compile(r"(?i)(content|article|body|main)")),
    ):
        node = finder()
        if node is not None:
            return node
    return soup


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Tag-name-based noise removal (these are unambiguous layout regions).
    for tag_name in [
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "iframe",
        "svg",
        "button",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Token-level boilerplate match: only strip if a class/id token *equals*
    # or *starts with* a known boilerplate term. Substring matching falls over
    # on classes like "sidebar-false" that just happen to mention "sidebar".
    NOISE_TOKENS = {
        "cookie", "cookies", "cookie-banner", "cookie-consent",
        "breadcrumb", "breadcrumbs",
        "share", "social", "social-share", "share-buttons",
        "skip", "skip-link", "skip-to-content",
        "back-to-top",
        "subscribe", "newsletter",
        "search-form", "search-bar",
        "usa-banner", "official-banner",
        "related-content", "related-links",
    }

    def _is_noise_attr(value) -> bool:
        if not value:
            return False
        tokens = value if isinstance(value, list) else str(value).split()
        for tok in tokens:
            tok_l = tok.lower()
            if tok_l in NOISE_TOKENS:
                return True
            for prefix in NOISE_TOKENS:
                if tok_l.startswith(prefix + "-") or tok_l.startswith(prefix + "_"):
                    return True
        return False

    for tag in list(soup.find_all(True)):
        # Skip tags already detached by an ancestor's decompose() (bs4 sets .attrs=None)
        if tag.attrs is None or tag.parent is None:
            continue
        if _is_noise_attr(tag.get("class")) or _is_noise_attr(tag.get("id")):
            tag.decompose()

    main = _pick_main_container(soup)

    parts: list[str] = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(separator=" ", strip=True)
        if not text:
            continue
        # Whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()
        if el.name in ("h1", "h2", "h3", "h4"):
            if len(text) < 3 or len(text) > 200:
                continue
            parts.append(f"\n## {text}\n")
        else:
            if len(text) < 25:  # drop one-liners that are usually nav fragments
                continue
            # Drop boilerplate-y junk
            if re.search(r"(?i)^(skip to|share this|related (links|content|topics)|"
                         r"last reviewed|page last (updated|reviewed)|content source)",
                         text):
                continue
            parts.append(text)

    cleaned = "\n\n".join(parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def truncate_words(body: str, max_words: int) -> tuple[str, int]:
    words = body.split()
    if len(words) <= max_words:
        return body, len(words)
    truncated = " ".join(words[:max_words])
    return truncated + "\n\n[... content truncated to fit corpus word budget ...]", max_words


def write_doc(out_path: Path, url: str, title: str, license_note: str, body: str) -> int:
    today = dt.date.today().isoformat()
    word_count = len(body.split())
    header_lines = [
        f"Title: {title}",
        f"Source URL: {url}",
        f"Date accessed: {today}",
        f"License / terms: {license_note}",
        f"Word count: {word_count}",
        "-" * 72,
        "",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header_lines) + body + "\n", encoding="utf-8")
    return word_count


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data" / "rag_corpus"

    successes: list[tuple[str, int, str]] = []
    failures: list[tuple[str, str, str]] = []

    print(f"[corpus] Writing to: {out_dir}")
    print(f"[corpus] Attempting {len(SOURCES)} sources\n")

    for filename, url, title, license_note in SOURCES:
        try:
            html = fetch_html(url)
            body = extract_text(html)
            word_count = len(body.split())
            if word_count < MIN_WORDS:
                failures.append((filename, url, f"too short ({word_count} words after cleaning)"))
                print(f"  [SKIP] {filename}: too short ({word_count} words)")
                time.sleep(INTER_REQUEST_DELAY_S)
                continue
            body, word_count = truncate_words(body, MAX_WORDS)
            write_doc(out_dir / filename, url, title, license_note, body)
            successes.append((filename, word_count, url))
            print(f"  [OK]   {filename}  ({word_count} words)")
        except requests.HTTPError as e:
            failures.append((filename, url, f"HTTP {e.response.status_code}"))
            print(f"  [FAIL] {filename}: HTTP {e.response.status_code}")
        except requests.RequestException as e:
            failures.append((filename, url, f"network: {type(e).__name__}: {e}"))
            print(f"  [FAIL] {filename}: network error - {type(e).__name__}")
        except Exception as e:
            failures.append((filename, url, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {filename}: {type(e).__name__}: {e}")
        finally:
            time.sleep(INTER_REQUEST_DELAY_S)

    print("\n" + "=" * 72)
    print(f"SUMMARY: {len(successes)} succeeded / {len(failures)} failed "
          f"(target: >= 8 documents, 10-15 ideal)")
    print("=" * 72)
    if successes:
        print("\nSaved files:")
        for fn, wc, _ in successes:
            print(f"  {fn:42s}  {wc:>5d} words")
    if failures:
        print("\nFailed sources:")
        for fn, url, reason in failures:
            print(f"  {fn:42s}  {reason}")
            print(f"    {url}")

    if len(successes) < 8:
        print(
            f"\n[warning] only {len(successes)} successful docs (< 8 target). "
            "Consider adding more URLs or checking blocked sources."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
