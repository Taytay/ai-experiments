#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["semanticscholar>=0.10"]
# ///
"""
Semantic Scholar helper for the `research-papers` Claude Code skill.

Run with `uv run s2_cli.py <command> ...` (uv installs the `semanticscholar` package on first run).
Works without credentials (shared public pool); set SEMANTIC_SCHOLAR_API_KEY for a dedicated
1 request/second allowance.

Commands
  search      Relevance-ranked paper search with citation counts (or --sort citations|date for
              bulk mode, which supports boolean query syntax and larger result sets).
  paper       Details for one paper by arXiv id/URL, DOI, Semantic Scholar id or URL; includes TLDR.
  citations   Papers that cite the given paper (who built on it).
  references  Papers the given paper cites (its background).
  recommend   Semantic Scholar's recommended papers for the given paper.

Every result prints the arXiv id when one exists, so it can be passed straight to
`arxiv_cli.py fetch` for the TeX source.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper
from semanticscholar.SemanticScholarException import (
    BadQueryParametersException,
    ObjectNotFoundException,
    SemanticScholarException,
)

MIN_INTERVAL = 1.0  # seconds between the end of one S2 request and the start of the next, machine-wide
_LOCK_PATH = Path(tempfile.gettempdir()) / "research-papers-s2.lock"
_STAMP_PATH = Path(tempfile.gettempdir()) / "research-papers-s2.last-request"

# Fields the search endpoint allows (tldr is only available from get_paper).
SEARCH_FIELDS = [
    "paperId", "title", "year", "publicationDate", "venue", "publicationVenue", "authors",
    "citationCount", "influentialCitationCount", "referenceCount", "externalIds", "openAccessPdf",
    "isOpenAccess", "abstract", "publicationTypes", "url",
]
PAPER_FIELDS = SEARCH_FIELDS + ["tldr", "fieldsOfStudy", "s2FieldsOfStudy", "journal"]
# Fields allowed on the nested paper of a citation/reference entry.
NESTED_FIELDS = [
    "paperId", "title", "year", "publicationDate", "venue", "authors", "citationCount",
    "influentialCitationCount", "externalIds", "openAccessPdf", "abstract", "url",
]

_ARXIV_RE = re.compile(r"(?:(?:[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})|(?:\d{4}\.\d{4,5}))(?:v\d+)?", re.IGNORECASE)


# --------------------------------------------------------------------------- utils
def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def note(msg: str) -> None:
    print(f"note: {msg}", file=sys.stderr)


if os.name == "nt":
    import msvcrt

    def _lock(fh) -> None:
        # msvcrt.locking(LK_LOCK) retries for ~10 s then raises; loop until we actually hold it.
        waited = False
        while True:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                if not waited:
                    note("waiting for another Semantic Scholar request (global one-at-a-time lock)")
                    waited = True

    def _unlock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def s2_slot():
    """Exclusive machine-wide slot: one Semantic Scholar call at a time, MIN_INTERVAL apart.

    Every `uv run` is a separate process, so this takes an OS file lock, waits until MIN_INTERVAL
    has passed since the previous call finished, runs the call while holding the lock, and
    records the finish time. Parallel invocations queue instead of bursting.
    """
    _LOCK_PATH.touch(exist_ok=True)
    with open(_LOCK_PATH, "r+b") as fh:
        _lock(fh)
        try:
            try:
                last = float(_STAMP_PATH.read_text().strip() or 0)
            except (OSError, ValueError):
                last = 0.0
            wait = MIN_INTERVAL - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            yield
        finally:
            try:
                _STAMP_PATH.write_text(f"{time.time():.3f}")
            except OSError:
                pass
            _unlock(fh)


def client() -> SemanticScholar:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None
    return SemanticScholar(timeout=60, api_key=key, retry=True)


def normalize_paper_id(raw: str) -> str:
    """Map an arXiv id/URL, DOI, S2 URL or S2 id to the identifier syntax the API expects."""
    s = raw.strip()
    if "semanticscholar.org/paper/" in s:
        return urllib.parse.urlparse(s).path.rstrip("/").rsplit("/", 1)[-1]
    if "arxiv.org/" in s or s.lower().startswith("arxiv:"):
        path = urllib.parse.urlparse(s).path if "://" in s else s.split(":", 1)[1]
        m = _ARXIV_RE.search(re.sub(r"\.pdf$", "", path))
        if not m:
            die(f"could not find an arXiv id in {raw!r}")
        return "ArXiv:" + re.sub(r"v\d+$", "", m.group(0))
    if s.lower().startswith("doi:") or s.startswith("10."):
        return "DOI:" + s.split(":", 1)[-1] if s.lower().startswith("doi:") else "DOI:" + s
    if s.lower().startswith(("corpusid:", "mag:", "acl:", "pmid:", "pmcid:", "url:")):
        return s
    if _ARXIV_RE.fullmatch(s):
        return "ArXiv:" + re.sub(r"v\d+$", "", s)
    return s  # assume a raw Semantic Scholar paperId (40 hex chars)


def paper_to_entry(p: Paper) -> dict:
    # `Paper` properties raise AttributeError for any field the API response did not include
    # (e.g. nested citation/reference papers carry fewer fields), so read everything defensively.
    def g(name: str, default=None):
        return getattr(p, name, default)

    ext = g("externalIds") or {}
    oa = g("openAccessPdf") or {}
    venue = g("venue") or ""
    pub_venue = g("publicationVenue")
    if not venue and pub_venue:
        venue = pub_venue.get("name", "") if isinstance(pub_venue, dict) else str(pub_venue)
    tldr_obj = g("tldr")
    tldr = ""
    if tldr_obj:
        tldr = tldr_obj.get("text", "") if isinstance(tldr_obj, dict) else (getattr(tldr_obj, "text", "") or "")
    paper_id = g("paperId") or ""
    return {
        "s2_id": paper_id,
        "title": " ".join((g("title") or "").split()),
        "year": g("year"),
        "date": g("publicationDate") or "",
        "venue": venue,
        "authors": [a.name for a in (g("authors") or [])],
        "citations": g("citationCount") or 0,
        "influential_citations": g("influentialCitationCount") or 0,
        "references": g("referenceCount") or 0,
        "arxiv": ext.get("ArXiv", ""),
        "doi": ext.get("DOI", ""),
        "open_access_pdf": oa.get("url", "") if isinstance(oa, dict) else "",
        "publication_types": list(g("publicationTypes") or []),
        "tldr": tldr or "",
        "abstract": " ".join((g("abstract") or "").split()),
        "url": g("url") or f"https://www.semanticscholar.org/paper/{paper_id}",
    }


# --------------------------------------------------------------------------- formatting
def fmt_authors(authors: list[str], limit: int = 5) -> str:
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + f" (+{len(authors) - limit} more)"


def truncate(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return text
    return text[: n - 1].rsplit(" ", 1)[0] + "..."


def print_entries(entries: list[dict], fmt: str, abstract_len: int, start: int = 0) -> None:
    if fmt == "json":
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return
    if fmt == "table":
        print(f"{'#':>3}  {'cites':>6} {'year':<5} {'arXiv':<14} {'venue':<18} title")
        for i, e in enumerate(entries, start + 1):
            print(f"{i:>3}  {e['citations']:>6} {str(e['year'] or ''):<5} {e['arxiv']:<14} "
                  f"{truncate(e['venue'], 18):<18} {truncate(e['title'], 80)}")
        return
    for i, e in enumerate(entries, start + 1):
        ids = []
        if e["arxiv"]:
            ids.append(f"arXiv `{e['arxiv']}`")
        if e["doi"]:
            ids.append(f"doi {e['doi']}")
        print(f"{i}. **{e['title']}** ({e['year'] or 'n.d.'})")
        print(f"   {e['citations']} citations ({e['influential_citations']} influential)"
              + (f" - {e['venue']}" if e["venue"] else "")
              + (" - " + ", ".join(ids) if ids else ""))
        print(f"   {fmt_authors(e['authors'])}")
        if e["tldr"]:
            print(f"   TLDR: {e['tldr']}")
        if abstract_len != 0 and e["abstract"]:
            print(f"   {truncate(e['abstract'], abstract_len)}")
        print(f"   {e['url']}" + (f"  |  PDF: {e['open_access_pdf']}" if e["open_access_pdf"] else ""))
        print()


# --------------------------------------------------------------------------- commands
def run(fn):
    """Run one Semantic Scholar operation inside the global slot and translate errors.

    `fn` must fully materialize its results (PaginatedResults fetch pages lazily), so that every
    HTTP request happens while the lock is held.
    """
    try:
        with s2_slot():
            return fn()
    except ObjectNotFoundException as e:
        die(f"not found: {e}")
    except BadQueryParametersException as e:
        die(f"bad query: {e}")
    except ConnectionRefusedError as e:
        die(f"Semantic Scholar rate limit (HTTP 429) persisted through retries: {e}. "
            "Set SEMANTIC_SCHOLAR_API_KEY for a dedicated allowance, or wait a minute.")
    except SemanticScholarException as e:
        die(f"Semantic Scholar API: {e.__class__.__name__}: {e}")
    except Exception as e:  # httpx transport errors, timeouts
        die(f"request failed: {e.__class__.__name__}: {e}")


def take(paginated, n: int) -> list:
    """Pull up to n items from a PaginatedResults without fetching more pages than needed."""
    out = []
    for item in paginated:
        out.append(item)
        if len(out) >= n:
            break
    return out


def cmd_search(args: argparse.Namespace) -> None:
    n = max(1, min(args.max, 1000 if args.sort != "relevance" else 100))
    bulk = args.sort != "relevance"
    sort = {"citations": "citationCount:desc", "date": "publicationDate:desc", "relevance": None}[args.sort]
    kwargs = dict(
        query=" ".join(args.query),
        year=args.year,
        venue=args.venue or None,
        fields_of_study=args.fos or None,
        open_access_pdf=True if args.open_access else None,
        min_citation_count=args.min_citations,
        publication_types=args.type or None,
        fields=SEARCH_FIELDS,
        limit=min(n, 100),
        bulk=bulk,
        sort=sort,
    )
    def go():
        r = client().search_paper(**kwargs)
        return r, take(r, n)

    results, items = run(go)
    entries = [paper_to_entry(p) for p in items]
    if args.format != "json":
        mode = f"bulk, sorted by {args.sort}" if bulk else "relevance-ranked"
        filters = ", ".join(f for f in [
            f"year={args.year}" if args.year else "",
            f"venue={args.venue}" if args.venue else "",
            f"fields_of_study={args.fos}" if args.fos else "",
            "open-access only" if args.open_access else "",
            f"min_citations={args.min_citations}" if args.min_citations else "",
            f"type={args.type}" if args.type else "",
        ] if f)
        print(f"query: {kwargs['query']}  ({mode}{'; ' + filters if filters else ''})")
        print(f"showing {len(entries)} of {results.total} matches\n")
    print_entries(entries, args.format, args.abstract_len)


def cmd_paper(args: argparse.Namespace) -> None:
    ids = [normalize_paper_id(x) for x in args.ids]
    if len(ids) == 1:
        papers = run(lambda: [client().get_paper(ids[0], fields=PAPER_FIELDS)])
    else:
        papers = run(lambda: client().get_papers(ids, fields=PAPER_FIELDS))
    print_entries([paper_to_entry(p) for p in papers], args.format, args.abstract_len)


def _linked(args: argparse.Namespace, getter_name: str, attr: str) -> None:
    pid = normalize_paper_id(args.id)
    n = max(1, min(args.max, 1000))
    def go():
        r = getattr(client(), getter_name)(pid, fields=NESTED_FIELDS, limit=min(n, 100))
        return r, take(r, n)

    results, items = run(go)
    entries = [paper_to_entry(getattr(it, attr)) for it in items]
    if args.sort == "citations":
        entries.sort(key=lambda e: e["citations"], reverse=True)
    elif args.sort == "date":
        entries.sort(key=lambda e: (e["date"] or str(e["year"] or "")), reverse=True)
    if args.format != "json":
        kind = getter_name.replace("get_paper_", "")
        total = f" of {results.total}" if results.total else ""
        print(f"{kind} of {pid}: showing {len(entries)}{total} (API order is newest first; "
              f"--sort applies to the fetched page, raise -n to widen it)\n")
    print_entries(entries, args.format, args.abstract_len)


def cmd_citations(args: argparse.Namespace) -> None:
    _linked(args, "get_paper_citations", "paper")


def cmd_references(args: argparse.Namespace) -> None:
    _linked(args, "get_paper_references", "paper")


def cmd_recommend(args: argparse.Namespace) -> None:
    pid = normalize_paper_id(args.id)
    papers = run(lambda: client().get_recommended_papers(
        pid, fields=NESTED_FIELDS, limit=max(1, min(args.max, 500)), pool_from=args.pool))
    if args.format != "json":
        print(f"recommendations for {pid} (pool: {args.pool}): {len(papers)}\n")
    print_entries([paper_to_entry(p) for p in papers], args.format, args.abstract_len)


# --------------------------------------------------------------------------- main
def add_output_flags(p: argparse.ArgumentParser, abstract_default: int) -> None:
    p.add_argument("-f", "--format", choices=["md", "table", "json"], default="md")
    p.add_argument("--abstract-len", type=int, default=abstract_default,
                   help="chars of abstract to show; 0 hides, -1 shows all")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="s2_cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search papers")
    s.add_argument("query", nargs="+", help="free-text query (bulk mode also accepts boolean syntax)")
    s.add_argument("--year", help="a year or range, e.g. 2024 or 2022-2025 or 2023-")
    s.add_argument("--venue", action="append", help="venue name, repeatable (e.g. NeurIPS, ICLR, ACL)")
    s.add_argument("--fos", action="append", help="field of study, repeatable (default none; e.g. 'Computer Science')")
    s.add_argument("--type", action="append", help="publication type, repeatable (JournalArticle, Conference, Review, ...)")
    s.add_argument("--open-access", action="store_true", help="only papers with an open-access PDF")
    s.add_argument("--min-citations", type=int, help="minimum citation count")
    s.add_argument("-n", "--max", type=int, default=10, help="results to return (relevance: max 100; bulk: max 1000)")
    s.add_argument("--sort", choices=["relevance", "citations", "date"], default="relevance",
                   help="citations/date switch to bulk mode (no relevance ranking, boolean query syntax)")
    add_output_flags(s, 300)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("paper", help="details (incl. TLDR) for one or more papers")
    d.add_argument("ids", nargs="+", help="arXiv id/URL, DOI, Semantic Scholar id/URL, CorpusId:...")
    add_output_flags(d, -1)
    d.set_defaults(func=cmd_paper)

    for name, fn, helptext in (
        ("citations", cmd_citations, "papers citing the given paper"),
        ("references", cmd_references, "papers the given paper cites"),
    ):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("id")
        c.add_argument("-n", "--max", type=int, default=25)
        c.add_argument("--sort", choices=["api", "citations", "date"], default="citations",
                       help="client-side ordering of the fetched page (default: most cited first)")
        add_output_flags(c, 0)
        c.set_defaults(func=fn)

    r = sub.add_parser("recommend", help="Semantic Scholar recommendations for a paper")
    r.add_argument("id")
    r.add_argument("-n", "--max", type=int, default=10)
    r.add_argument("--pool", choices=["recent", "all-cs"], default="recent")
    add_output_flags(r, 200)
    r.set_defaults(func=cmd_recommend)

    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args.func(args)


if __name__ == "__main__":
    main()
