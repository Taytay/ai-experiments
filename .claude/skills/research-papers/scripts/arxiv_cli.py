#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["arxiv>=2.1"]
# ///
"""
arXiv helper for the `research-papers` Claude Code skill.

Run with `uv run arxiv_cli.py <command> ...` (uv installs the `arxiv` package on first run).

Commands
  search   Search arXiv (fielded query builder or raw Lucene-style query).
  info     Fetch metadata for one or more arXiv ids / URLs.
  fetch    Download a paper's TeX source (and optionally PDF) into references/papers/<id>/,
           unpack it, locate the entrypoint, and write a flattened single .tex file.
  flatten  (Re)build the flattened .tex for an already-downloaded paper.
  list     List papers already present in references/papers/.

All commands accept --dest to change the papers directory (default: ./references/papers).

Every request goes to export.arxiv.org, the host arXiv sets aside for programmatic access (its
bulk-data guidelines ask tools to use it and leave arxiv.org to people). The API is used first, via
the `arxiv` package, one attempt per command (a 429 penalises the IP until a quiet period passes, so
retrying only extends it). When throttled, `fetch` and `info` fall back to the paper's abs page;
`search` has no fallback, since arxiv.org/search is disallowed for non-browser clients.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import html as htmllib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import arxiv

USER_AGENT = "research-papers-skill/1.0 (Claude Code skill; contact: local user)"
DEFAULT_DEST = Path("references") / "papers"

# New-style ids: 2601.07372 (optionally v2). Old-style: hep-th/9901001, math.GT/0309136.
_ID_RE = re.compile(
    r"(?P<base>(?:[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})|(?:\d{4}\.\d{4,5}))(?P<ver>v\d+)?",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- utils
class FetchError(RuntimeError):
    pass


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def note(msg: str) -> None:
    print(f"note: {msg}", file=sys.stderr)


def normalize_id(raw: str, keep_version: bool = False) -> str:
    """Accept an arXiv id or any arxiv.org URL and return the bare id."""
    s = raw.strip()
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.IGNORECASE)
    if "://" in s:
        s = urllib.parse.urlparse(s).path
        s = re.sub(r"\.pdf$", "", s)
    m = _ID_RE.search(s)
    if not m:
        die(f"could not find an arXiv id in {raw!r}")
    base = m.group("base")
    ver = m.group("ver") or ""
    return base + (ver if keep_version else "")


def split_version(id_with_ver: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)(v\d+)?$", id_with_ver)
    return m.group(1), (m.group(2) or "")


def id_to_dirname(base_id: str) -> str:
    # Old-style ids contain '/', which cannot be a directory name.
    return base_id.replace("/", "_")


EXPORT = "https://export.arxiv.org"  # the host arXiv sets aside for programmatic access; arxiv.org itself is for people
MIN_INTERVAL = 3.0  # seconds between the end of one arXiv request and the start of the next, machine-wide
_LOCK_PATH = Path(tempfile.gettempdir()) / "research-papers-arxiv.lock"
_STAMP_PATH = Path(tempfile.gettempdir()) / "research-papers-arxiv.last-request"

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
                    note("waiting for another arXiv request (global one-at-a-time lock)")
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
def arxiv_slot():
    """Hold the machine-wide arXiv slot for the duration of one request (or one API call).

    Every `uv run` is a separate process, so nothing in-process can stop two invocations from
    hitting arXiv at the same instant. This takes an exclusive OS file lock, waits until
    MIN_INTERVAL has passed since the previous request *finished*, runs the request while still
    holding the lock, and records the finish time. Parallel invocations therefore queue up and
    come out spaced >= MIN_INTERVAL apart, which is what arXiv asks for.
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


def http_get(url: str, timeout: int = 90, retries: int = 4) -> bytes:
    """Plain GET with backoff, used for arxiv.org (source, abs page). One request at a time, machine-wide."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with arxiv_slot(), urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            # 406 shows up intermittently from arxiv.org when throttled; treat it as transient.
            if e.code in (406, 429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 5 * 2**attempt
                note(f"HTTP {e.code} from {urllib.parse.urlparse(url).netloc}; retrying in {wait}s")
                time.sleep(wait)
                continue
            raise FetchError(f"HTTP {e.code} fetching {url}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(5 * 2**attempt)
                continue
    raise FetchError(f"failed to fetch {url}: {last}")


def strip_tags(s: str) -> str:
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", "", s)).split())


# --------------------------------------------------------------------------- arXiv API (via `arxiv` package)
def client(page_size: int) -> arxiv.Client:
    # arxiv.Client asks the API for `page_size` results on every request, so size it to the call
    # instead of always pulling 200 (slower responses, more load, more likely to be throttled).
    # num_retries=0: once export.arxiv.org answers 429 "Rate exceeded." the IP stays penalised until a
    # quiet period passes and every further request (even 30 s apart) restarts it, so retrying only hurts.
    return arxiv.Client(page_size=max(1, min(page_size, 200)), delay_seconds=3.0, num_retries=0)


def result_to_entry(r: arxiv.Result) -> dict:
    full_id = r.get_short_id()
    base_id, ver = split_version(full_id)
    return {
        "id": base_id,
        "version": ver,
        "id_versioned": full_id,
        "title": " ".join(r.title.split()),
        "authors": [a.name for a in r.authors],
        "published": r.published.date().isoformat(),
        "updated": r.updated.date().isoformat(),
        "primary_category": r.primary_category or "",
        "categories": list(r.categories),
        "comment": " ".join((r.comment or "").split()),
        "journal_ref": " ".join((r.journal_ref or "").split()),
        "doi": r.doi or "",
        "abstract": " ".join(r.summary.split()),
        "abs_url": f"https://arxiv.org/abs/{base_id}",
        "pdf_url": r.pdf_url or f"https://arxiv.org/pdf/{full_id}",
        "src_url": f"{EXPORT}/src/{full_id}",
        "metadata_source": "api",
    }


def api_search(
    query: str = "",
    id_list: list[str] | None = None,
    max_results: int = 10,
    offset: int = 0,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> list[dict]:
    """Run one arXiv API query; raises FetchError when the API is unavailable/throttled."""
    search = arxiv.Search(
        query=query,
        id_list=id_list or [],
        max_results=offset + max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    # Hold the machine-wide slot for the whole call: the client may page or retry internally
    # (spaced 3 s by its own delay), and no other process may talk to arXiv meanwhile.
    try:
        with arxiv_slot():
            return [result_to_entry(r) for r in client(offset + max_results).results(search, offset=offset)]
    except arxiv.ArxivError as e:
        raise FetchError(f"arXiv API: {e}") from e
    except Exception as e:  # requests connection errors, timeouts, feed parse failures
        raise FetchError(f"arXiv API request failed: {e.__class__.__name__}: {e}") from e


# --------------------------------------------------------------------------- fallbacks on export.arxiv.org
def metadata_from_abs_page(id_v: str) -> dict:
    """Fallback when the API is throttled: scrape the citation_* meta tags of the abs page."""
    page = http_get(f"{EXPORT}/abs/{id_v}").decode("utf-8", errors="ignore")

    def metas(name: str) -> list[str]:
        return [htmllib.unescape(m) for m in re.findall(rf'<meta\s+name="{name}"\s+content="([^"]*)"', page)]

    def first(name: str) -> str:
        v = metas(name)
        return v[0] if v else ""

    base_id, ver = split_version(id_v)
    if not ver:
        # No version requested: the abs page lists every version as [vN]; take the latest.
        versions = [int(v) for v in re.findall(r"\[v(\d+)\]", page)]
        ver = f"v{max(versions)}" if versions else ""
    authors = [
        " ".join(reversed([p.strip() for p in a.split(",", 1)])) if "," in a else a
        for a in metas("citation_author")
    ]
    date = first("citation_date").replace("/", "-")
    cat_m = re.search(r'class="primary-subject">[^<(]*\(([^)]+)\)', page)
    cats = re.findall(r'class="tablecell subjects">(.*?)</td>', page, re.DOTALL)
    cat_codes = re.findall(r"\(([a-z\-]+(?:\.[A-Za-z]{2})?)\)", cats[0]) if cats else []
    comment_m = re.search(r'class="tablecell comments[^"]*">(.*?)</td>', page, re.DOTALL)
    return {
        "id": base_id,
        "version": ver,
        "id_versioned": base_id + ver,
        "title": " ".join(first("citation_title").split()),
        "authors": authors,
        "published": date,
        "updated": date,
        "primary_category": cat_m.group(1) if cat_m else (cat_codes[0] if cat_codes else ""),
        "categories": cat_codes,
        "comment": strip_tags(comment_m.group(1)) if comment_m else "",
        "journal_ref": "",
        "doi": first("citation_doi"),
        "abstract": " ".join(first("citation_abstract").split()),
        "abs_url": f"https://arxiv.org/abs/{base_id}",
        "pdf_url": first("citation_pdf_url") or f"https://arxiv.org/pdf/{base_id + ver}",
        "src_url": f"{EXPORT}/src/{base_id + ver}",
        "metadata_source": "abs-page",
    }




# --------------------------------------------------------------------------- query builder
_TOKEN_RE = re.compile(r'"([^"]+)"|(\S+)')
_OPS = ("AND", "OR", "ANDNOT")


def field_terms(field: str, text: str) -> list[str]:
    """Turn free text into `field:term` clauses; quoted phrases stay phrases."""
    out = []
    for phrase, word in _TOKEN_RE.findall(text):
        if phrase:
            out.append(f'{field}:"{phrase}"')
        elif word.upper() in _OPS:
            out.append(word.upper())
        else:
            out.append(f"{field}:{word}")
    return out


def join_terms(terms: list[str]) -> str:
    """AND together clauses unless the user already wrote explicit operators."""
    out: list[str] = []
    for t in terms:
        if out and t not in _OPS and out[-1] not in _OPS:
            out.append("AND")
        out.append(t)
    return " ".join(out)


def build_search_query(args: argparse.Namespace) -> str:
    if args.raw:
        return args.raw
    groups: list[str] = []
    if args.query:
        groups.append(join_terms(field_terms("all", " ".join(args.query))))
    for field, values in (("ti", args.title), ("abs", args.abstract), ("au", args.author)):
        for v in values or []:
            groups.append(join_terms(field_terms(field, v)))
    if args.category:
        groups.append(" OR ".join(f"cat:{c}" for c in args.category))
    if args.since or args.until:
        lo = (args.since or "1991-01-01").replace("-", "") + "0000"
        hi = (args.until or "2099-12-31").replace("-", "") + "2359"
        groups.append(f"submittedDate:[{lo} TO {hi}]")
    if not groups:
        die("nothing to search for: give a query, --title/--abstract/--author, --category, or --raw")
    if len(groups) == 1:
        return groups[0]
    return " AND ".join(f"({g})" for g in groups)


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
        print(f"{'#':>3}  {'arXiv id':<14} {'date':<10} {'cat':<9} title")
        for i, e in enumerate(entries, start + 1):
            print(f"{i:>3}  {e['id']:<14} {e['published']:<10} {e['primary_category']:<9} {truncate(e['title'], 90)}")
        return
    # markdown (default)
    for i, e in enumerate(entries, start + 1):
        ver = f" ({e['version']})" if e["version"] and e["version"] != "v1" else ""
        updated = f", updated {e['updated']}" if e["updated"] != e["published"] else ""
        print(f"{i}. **{e['title']}**")
        print(f"   `{e['id']}`{ver} - {e['primary_category']} - submitted {e['published']}{updated}")
        print(f"   {fmt_authors(e['authors'])}")
        if e["comment"]:
            print(f"   Comment: {truncate(e['comment'], 160)}")
        if e["journal_ref"]:
            print(f"   Journal: {truncate(e['journal_ref'], 160)}")
        if abstract_len != 0:
            print(f"   {truncate(e['abstract'], abstract_len)}")
        print(f"   {e['abs_url']}")
        print()


# --------------------------------------------------------------------------- commands
def cmd_search(args: argparse.Namespace) -> None:
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "date": arxiv.SortCriterion.SubmittedDate,
        "updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    query = build_search_query(args)
    n = min(args.max, 200)
    header = f"search_query: {query}"
    try:
        entries = api_search(
            query=query,
            max_results=n,
            offset=args.start,
            sort_by=sort_map[args.sort],
            sort_order=arxiv.SortOrder.Ascending if args.ascending else arxiv.SortOrder.Descending,
        )
        footer = f"showing {args.start + 1}-{args.start + len(entries)} (API)"
    except FetchError as e:
        raise FetchError(
            f"{e}. The arXiv API is rate-limiting this IP. The penalty clears only after a quiet period and every request "
            "(even spaced 30 s apart) extends it, so do NOT retry now. Search the same papers with "
            "s2_cli.py search or openalex_cli.py search instead; `fetch` and `info` still work via the abs page."
        ) from e
    if args.format != "json":
        print(header)
        print(footer + "\n")
    print_entries(entries, args.format, args.abstract_len, start=args.start)


def fetch_metadata(id_v: str) -> dict:
    try:
        entries = api_search(id_list=[id_v], max_results=1)
        if entries:
            return entries[0]
        raise FetchError(f"no arXiv record found for {id_v}")
    except FetchError as e:
        note(f"{e}; falling back to the abs page")
        meta = metadata_from_abs_page(id_v)
        if not meta["title"]:
            raise FetchError(f"could not get metadata for {id_v} from the API or the abs page") from e
        return meta


def cmd_info(args: argparse.Namespace) -> None:
    ids = [normalize_id(x, keep_version=True) for x in args.ids]
    try:
        entries = api_search(id_list=ids, max_results=len(ids))
        if not entries:
            raise FetchError(f"no arXiv record found for {', '.join(ids)}")
    except FetchError as e:
        note(f"{e}; falling back to the abs page(s)")
        entries = []
        for k, id_v in enumerate(ids):
            if k:
                time.sleep(3)
            entries.append(metadata_from_abs_page(id_v))
    print_entries(entries, args.format, args.abstract_len)


def find_entrypoints(src_dir: Path) -> list[Path]:
    candidates = []
    for f in sorted(src_dir.rglob("*.tex")):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"^\s*\\documentclass", txt, re.MULTILINE):
            candidates.append(f)
    preferred = ("main", "paper", "ms", "manuscript", "arxiv", "root")

    def rank(p: Path):
        stem = p.stem.lower()
        return (0 if stem in preferred else 1, len(p.parts), stem)

    return sorted(candidates, key=rank)


_COMMENT_ENV_RE = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.DOTALL)
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
_INPUT_BARE_RE = re.compile(r"\\input\s+([^\s{}\\%]+)")


def strip_comments(tex: str) -> str:
    tex = _COMMENT_ENV_RE.sub("", tex)
    out_lines = []
    for line in tex.splitlines():
        # remove an unescaped % and everything after it
        i = 0
        while True:
            j = line.find("%", i)
            if j == -1:
                break
            if j > 0 and line[j - 1] == "\\":
                i = j + 1
                continue
            line = line[:j]
            break
        out_lines.append(line.rstrip())
    text = "\n".join(out_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def resolve_input(base_dir: Path, root_dir: Path, name: str) -> Path | None:
    name = name.strip()
    for d in (base_dir, root_dir):
        for cand in (d / name, d / f"{name}.tex"):
            if cand.is_file():
                return cand
    return None


def flatten_tex(entry: Path, root_dir: Path, seen: set[Path] | None = None) -> str:
    seen = seen if seen is not None else set()
    entry = entry.resolve()
    if entry in seen:
        return f"% [flatten] skipped already-included file {entry.name}\n"
    seen.add(entry)
    tex = strip_comments(entry.read_text(encoding="utf-8", errors="ignore"))

    def repl(m: re.Match) -> str:
        target = resolve_input(entry.parent, root_dir, m.group(1))
        if target is None:
            return f"% [flatten] could not resolve {m.group(0)}"
        try:
            rel = target.resolve().relative_to(root_dir.resolve())
        except ValueError:
            rel = target
        return f"\n% ===== begin {rel} =====\n{flatten_tex(target, root_dir, seen)}\n% ===== end {rel} =====\n"

    tex = _INPUT_RE.sub(repl, tex)
    tex = _INPUT_BARE_RE.sub(repl, tex)
    return tex


def write_flat(paper_dir: Path, src_dir: Path, entrypoints: list[Path]) -> Path | None:
    if not entrypoints:
        return None
    flat = flatten_tex(entrypoints[0], src_dir)
    # Append any .bbl (compiled bibliography) so references are readable too.
    for bbl in sorted(src_dir.rglob("*.bbl")):
        flat += f"\n% ===== begin {bbl.relative_to(src_dir)} =====\n"
        flat += strip_comments(bbl.read_text(encoding="utf-8", errors="ignore"))
    out = paper_dir / "paper_flat.tex"
    out.write_text(flat, encoding="utf-8")
    return out


def unpack_source(blob: bytes, paper_dir: Path, src_dir: Path, base_id: str) -> tuple[str, Path]:
    """Detect what /src returned and unpack it. Returns (kind, saved_archive_path)."""
    dirname = id_to_dirname(base_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    if blob.startswith(b"%PDF"):
        p = paper_dir / f"{dirname}.pdf"
        p.write_bytes(blob)
        return "pdf-only", p
    if blob[:2] == b"\x1f\x8b":
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                src_dir.mkdir(parents=True, exist_ok=True)
                tf.extractall(src_dir, filter="data")
            p = paper_dir / f"{dirname}.tar.gz"
            p.write_bytes(blob)
            return "tar", p
        except tarfile.ReadError:  # a gzip that is not a tar; filter errors (path traversal) propagate
            data = gzip.decompress(blob)
            src_dir.mkdir(parents=True, exist_ok=True)
            name = "main.tex" if b"\\documentclass" in data else "source.txt"
            (src_dir / name).write_bytes(data)
            p = paper_dir / f"{dirname}.gz"
            p.write_bytes(blob)
            return "single-file", p
    if blob[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            src_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(src_dir)
        p = paper_dir / f"{dirname}.zip"
        p.write_bytes(blob)
        return "zip", p
    p = paper_dir / f"{dirname}.src.bin"
    p.write_bytes(blob)
    return "unknown", p


def cmd_fetch(args: argparse.Namespace) -> None:
    id_v = normalize_id(args.id, keep_version=True)
    base_id, _ = split_version(id_v)
    dest_root = Path(args.dest)
    paper_dir = dest_root / id_to_dirname(base_id)
    src_dir = paper_dir / "src"
    paper_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"id": base_id, "requested": id_v, "dir": str(paper_dir), "steps": []}

    meta_path = paper_dir / "meta.json"
    if meta_path.exists() and not args.force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result["steps"].append("metadata: cached")
    else:
        meta = fetch_metadata(id_v)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        result["steps"].append(f"metadata: downloaded ({meta.get('metadata_source', 'api')})")
        time.sleep(1)
    result["title"] = meta["title"]
    result["version"] = meta["version"]
    result["meta"] = str(meta_path)

    kind = None
    pdf_path = paper_dir / f"{id_to_dirname(base_id)}.pdf"
    have_src = src_dir.exists() and any(src_dir.iterdir())
    have_pdf_only = pdf_path.exists() and not have_src
    if (have_src or have_pdf_only) and not args.force:
        kind = "cached" if have_src else "pdf-only"
        result["steps"].append("source: cached")
    elif not args.no_src:
        blob = http_get(f"{EXPORT}/src/{id_v}")
        kind, saved = unpack_source(blob, paper_dir, src_dir, base_id)
        result["archive"] = str(saved)
        result["steps"].append(f"source: downloaded ({kind}, {len(blob) // 1024} KiB)")
    result["source_kind"] = kind

    if src_dir.exists():
        entrypoints = find_entrypoints(src_dir)
        result["entrypoints"] = [str(p) for p in entrypoints]
        result["tex_files"] = len(list(src_dir.rglob("*.tex")))
        if not args.no_flatten:
            flat = write_flat(paper_dir, src_dir, entrypoints)
            if flat:
                result["flat"] = str(flat)
                result["flat_lines"] = sum(1 for _ in flat.open(encoding="utf-8"))
                result["steps"].append("flatten: wrote paper_flat.tex")
            else:
                result["steps"].append("flatten: no entrypoint with \\documentclass found")
        figs = [p for p in src_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".eps")]
        result["figures"] = len(figs)

    if args.pdf or kind == "pdf-only":
        if not pdf_path.exists() or (args.force and kind != "pdf-only"):
            time.sleep(1)
            pdf_path.write_bytes(http_get(f"{EXPORT}/pdf/{id_v}"))
            result["steps"].append("pdf: downloaded")
        else:
            result["steps"].append("pdf: present")
        result["pdf"] = str(pdf_path)

    if kind == "pdf-only":
        result["note"] = "arXiv has no TeX source for this paper; read the PDF instead."
    elif kind == "unknown":
        result["note"] = "unrecognised source format; inspect the saved archive manually."

    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_flatten(args: argparse.Namespace) -> None:
    base_id, _ = split_version(normalize_id(args.id))
    paper_dir = Path(args.dest) / id_to_dirname(base_id)
    src_dir = paper_dir / "src"
    if not src_dir.is_dir():
        die(f"no unpacked source at {src_dir}; run `fetch` first")
    entrypoints = find_entrypoints(src_dir)
    if args.entry:
        entrypoints = [Path(args.entry)] + entrypoints
    flat = write_flat(paper_dir, src_dir, entrypoints)
    if not flat:
        die("no entrypoint found; pass --entry path/to/main.tex")
    print(json.dumps({"flat": str(flat), "entry": str(entrypoints[0]),
                      "lines": sum(1 for _ in flat.open(encoding="utf-8"))}, indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    dest_root = Path(args.dest)
    if not dest_root.is_dir():
        print(f"(no papers directory at {dest_root})")
        return
    rows = []
    for d in sorted(dest_root.iterdir()):
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append((meta["id"], meta["published"], meta["primary_category"], meta["title"],
                     (d / "paper_flat.tex").exists(), (d / "summary.md").exists()))
    if not rows:
        print(f"(no downloaded papers in {dest_root})")
        return
    for id_, date, cat, title, flat, summ in rows:
        flags = ("flat " if flat else "     ") + ("summary" if summ else "")
        print(f"{id_:<14} {date:<10} {cat:<9} {flags:<13} {title}")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="arxiv_cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=str(DEFAULT_DEST), help="papers directory (default: references/papers)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search arXiv")
    s.add_argument("query", nargs="*", help="free-text terms searched in all fields (quote phrases)")
    s.add_argument("-t", "--title", action="append", help="terms that must appear in the title (repeatable)")
    s.add_argument("-a", "--abstract", action="append", help="terms that must appear in the abstract (repeatable)")
    s.add_argument("-au", "--author", action="append", help="author name, e.g. 'Vaswani' or 'del_maestro' (repeatable)")
    s.add_argument("-c", "--category", action="append", help="arXiv category, e.g. cs.CL, cs.LG (repeatable, OR-ed)")
    s.add_argument("--since", help="submitted on/after YYYY-MM-DD")
    s.add_argument("--until", help="submitted on/before YYYY-MM-DD")
    s.add_argument("--raw", help="raw arXiv search_query string; overrides all other query flags")
    s.add_argument("-n", "--max", type=int, default=10, help="results to return (default 10, max 200)")
    s.add_argument("--start", type=int, default=0, help="offset for paging")
    s.add_argument("--sort", choices=["relevance", "date", "updated"], default="relevance")
    s.add_argument("--ascending", action="store_true", help="oldest first (default is descending)")
    s.add_argument("-f", "--format", choices=["md", "table", "json"], default="md")
    s.add_argument("--abstract-len", type=int, default=350, help="chars of abstract to show; 0 hides, -1 shows all")
    s.set_defaults(func=cmd_search)

    i = sub.add_parser("info", help="metadata for given arXiv ids or URLs")
    i.add_argument("ids", nargs="+")
    i.add_argument("-f", "--format", choices=["md", "table", "json"], default="md")
    i.add_argument("--abstract-len", type=int, default=-1)
    i.set_defaults(func=cmd_info)

    f = sub.add_parser("fetch", help="download source (+ optional PDF) into references/papers/<id>/")
    f.add_argument("id", help="arXiv id or URL, e.g. 2601.07372 or https://arxiv.org/abs/2601.07372v2")
    f.add_argument("--pdf", action="store_true", help="also download the PDF")
    f.add_argument("--no-src", action="store_true", help="skip the TeX source download")
    f.add_argument("--no-flatten", action="store_true", help="do not write paper_flat.tex")
    f.add_argument("--force", action="store_true", help="re-download even if cached")
    f.set_defaults(func=cmd_fetch)

    fl = sub.add_parser("flatten", help="rebuild paper_flat.tex for a downloaded paper")
    fl.add_argument("id")
    fl.add_argument("--entry", help="explicit entrypoint .tex if auto-detection picks the wrong file")
    fl.set_defaults(func=cmd_flatten)

    l = sub.add_parser("list", help="list downloaded papers")
    l.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    try:
        args.func(args)
    except FetchError as e:
        die(str(e))


if __name__ == "__main__":
    main()
