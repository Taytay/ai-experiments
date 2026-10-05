#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyalex>=0.18", "requests>=2.31", "pymupdf4llm>=0.0.17"]
# ///
"""
OpenAlex helper for the `research-papers` Claude Code skill.

Run with `uv run openalex_cli.py <command> ...` (uv installs `pyalex` on first run).

API key: OPENALEX_API_KEY env var, else the file ~/.config/openalex/api_key. Search works
without a key (small shared budget); content downloads (PDF / Grobid TEI XML) require one.
The free key gives $1/day: a search costs $0.001, a list/filter call $0.0001, and each
PDF or XML download $0.01. `status` shows what is left.

Commands
  search     Search works (title/abstract/full text) with venue, OA and content-availability
             filters. Prints OpenAlex id, DOI, arXiv id when known, citations, FWCI.
  work       Details for one or more works by OpenAlex id, DOI, or arXiv id/URL.
  citations  Works that cite the given work (OpenAlex's citation data is thin for CS;
             prefer s2_cli.py citations).
  fetch      Download a work's Grobid TEI XML (default) and/or PDF into references/papers/<W-id>/
             and write a readable paper.md from the TEI.
  status     Show the key's remaining daily budget.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pyalex
import requests
from pyalex import Works

CONTENT_BASE = "https://content.openalex.org"
API_BASE = "https://api.openalex.org"
USER_AGENT = "research-papers-skill/1.0 (Claude Code skill)"
DEFAULT_DEST = Path("references") / "papers"
KEY_FILE = Path.home() / ".config" / "openalex" / "api_key"
TEI_NS = "{http://www.tei-c.org/ns/1.0}"

_ARXIV_RE = re.compile(r"(?:(?:[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})|(?:\d{4}\.\d{4,5}))(?:v\d+)?", re.IGNORECASE)


# --------------------------------------------------------------------------- utils
def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def note(msg: str) -> None:
    print(f"note: {msg}", file=sys.stderr)


def api_key() -> str | None:
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if key:
        return key
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def configure() -> str | None:
    key = api_key()
    pyalex.config.api_key = key
    pyalex.config.email = os.environ.get("OPENALEX_EMAIL") or None
    pyalex.config.user_agent = USER_AGENT
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 502, 503, 504]
    return key


def session(key: str | None) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    if key:
        s.headers["Authorization"] = f"Bearer {key}"
    return s


def budget_line(headers) -> str:
    rem = headers.get("x-ratelimit-remaining-usd")
    lim = headers.get("x-ratelimit-limit-usd")
    cost = headers.get("x-ratelimit-cost-usd")
    reset = headers.get("x-ratelimit-reset")
    if rem is None:
        return "budget: n/a (no key)"
    out = f"budget: ${rem} of ${lim}/day remaining"
    if cost:
        out += f" (this call ${cost})"
    if reset:
        try:
            out += f", resets in {int(reset) // 3600}h{(int(reset) % 3600) // 60:02d}m"
        except ValueError:
            pass
    return out


def w_id(work: dict) -> str:
    return (work.get("id") or "").rsplit("/", 1)[-1]


def arxiv_id_of(work: dict) -> str:
    """Derive an arXiv id from the DOI (10.48550/arXiv.X) or any arxiv.org location URL."""
    doi = (work.get("doi") or (work.get("ids") or {}).get("doi") or "").lower()
    m = re.search(r"10\.48550/arxiv\.(.+)$", doi)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    for loc in work.get("locations") or [work.get("primary_location") or {}, work.get("best_oa_location") or {}]:
        for url in ((loc or {}).get("landing_page_url"), (loc or {}).get("pdf_url")):
            if url and "arxiv.org/" in url:
                m = _ARXIV_RE.search(url.rsplit("/", 1)[-1].replace(".pdf", ""))
                if m:
                    return re.sub(r"v\d+$", "", m.group(0))
    return ""


def normalize_work_id(raw: str) -> str:
    """OpenAlex id, DOI, or arXiv id/URL -> an identifier pyalex's Works()[...] accepts."""
    s = raw.strip()
    if re.fullmatch(r"W\d+", s, re.IGNORECASE):
        return s.upper()
    if "openalex.org/" in s:
        return s.rsplit("/", 1)[-1].upper()
    if "arxiv.org/" in s or s.lower().startswith("arxiv:"):
        path = s.split("arxiv.org/", 1)[-1] if "arxiv.org/" in s else s.split(":", 1)[1]
        m = _ARXIV_RE.search(re.sub(r"\.pdf$", "", path))
        if not m:
            die(f"could not find an arXiv id in {raw!r}")
        return "doi:10.48550/arXiv." + re.sub(r"v\d+$", "", m.group(0))
    if _ARXIV_RE.fullmatch(s):
        return "doi:10.48550/arXiv." + re.sub(r"v\d+$", "", s)
    if s.lower().startswith("doi:"):
        return "doi:" + s.split(":", 1)[1]
    if s.startswith("10.") or "doi.org/" in s:
        return "doi:" + s.split("doi.org/", 1)[-1]
    return s


WORK_FIELDS = [
    "id", "doi", "ids", "title", "publication_year", "publication_date", "type", "cited_by_count",
    "fwci", "citation_normalized_percentile", "referenced_works_count", "authorships",
    "primary_location", "best_oa_location", "open_access", "has_content", "topics", "language",
    "abstract_inverted_index",
]


def work_to_entry(w: dict) -> dict:
    pl = w.get("primary_location") or {}
    src = pl.get("source") or {}
    bo = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    hc = w.get("has_content") or {}
    pct = w.get("citation_normalized_percentile") or {}
    topics = [t.get("display_name") for t in (w.get("topics") or [])[:1] if t.get("display_name")]
    authors = []
    for a in w.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    venue = src.get("display_name") or ""
    if not venue:
        # e.g. ACL Anthology papers have no OpenAlex source; show where the paper lives instead
        landing = pl.get("landing_page_url") or bo.get("landing_page_url") or ""
        host = re.sub(r"^https?://(www\.)?", "", landing).split("/", 1)[0]
        venue = host if host and "doi.org" not in host else ""
    return {
        "openalex_id": w_id(w),
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "arxiv": arxiv_id_of(w),
        "title": " ".join((w.get("title") or "").split()),
        "year": w.get("publication_year"),
        "date": w.get("publication_date") or "",
        "type": w.get("type") or "",
        "venue": venue,
        "authors": authors,
        "citations": w.get("cited_by_count") or 0,
        "fwci": w.get("fwci"),
        "citation_percentile": pct.get("value"),
        "references": w.get("referenced_works_count") or 0,
        "oa_status": oa.get("oa_status") or "",
        "oa_pdf": bo.get("pdf_url") or oa.get("oa_url") or "",
        "has_pdf": bool(hc.get("pdf")),
        "has_xml": bool(hc.get("grobid_xml")),
        "topics": topics,
        "abstract": " ".join(invert(w.get("abstract_inverted_index")).split()),
        "url": w.get("id") or "",
    }


def invert(index: dict | None) -> str:
    if not index:
        return ""
    words: dict[int, str] = {}
    for word, positions in index.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words))


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
        print(f"{'#':>3}  {'cites':>6} {'fwci':>5} {'year':<5} {'OpenAlex':<12} {'arXiv':<12} {'content':<7} {'venue':<18} title")
        for i, e in enumerate(entries, start + 1):
            content = ("pdf" if e["has_pdf"] else "   ") + ("xml" if e["has_xml"] else "   ")
            fwci = f"{e['fwci']:.1f}" if isinstance(e["fwci"], (int, float)) else ""
            print(f"{i:>3}  {e['citations']:>6} {fwci:>5} {str(e['year'] or ''):<5} {e['openalex_id']:<12} {e['arxiv']:<12} "
                  f"{content:<7} {truncate(e['venue'], 18):<18} {truncate(e['title'], 70)}")
        return
    for i, e in enumerate(entries, start + 1):
        ids = [f"`{e['openalex_id']}`"]
        if e["arxiv"]:
            ids.append(f"arXiv `{e['arxiv']}`")
        if e["doi"]:
            ids.append(f"doi {e['doi']}")
        content = [k for k, v in (("pdf", e["has_pdf"]), ("xml", e["has_xml"])) if v]
        fwci = f", FWCI {e['fwci']:.1f}" if isinstance(e["fwci"], (int, float)) else ""
        print(f"{i}. **{e['title']}** ({e['year'] or 'n.d.'})")
        print(f"   {e['citations']} citations{fwci} - {e['type']}"
              + (f" - {e['venue']}" if e["venue"] else "")
              + f" - OA: {e['oa_status'] or 'closed'}"
              + (f" - content: {'+'.join(content)}" if content else ""))
        print(f"   {', '.join(ids)}")
        print(f"   {fmt_authors(e['authors'])}")
        if e["topics"]:
            print(f"   Topics: {'; '.join(e['topics'])}")
        if abstract_len != 0 and e["abstract"]:
            print(f"   {truncate(e['abstract'], abstract_len)}")
        print(f"   {e['url']}" + (f"  |  PDF: {e['oa_pdf']}" if e["oa_pdf"] else ""))
        print()


# --------------------------------------------------------------------------- commands
def parse_year(y: str | None) -> str | None:
    if not y:
        return None
    y = y.strip()
    if re.fullmatch(r"\d{4}", y):
        return y
    if re.fullmatch(r"\d{4}-\d{4}", y):
        return y
    if re.fullmatch(r"\d{4}-", y):
        return f">{int(y[:4]) - 1}"
    if re.fullmatch(r"-\d{4}", y):
        return f"<{int(y[1:]) + 1}"
    die(f"bad --year {y!r}; use 2024, 2020-2024, 2022-, or -2019")


def cmd_search(args: argparse.Namespace) -> None:
    configure()
    query = " ".join(args.query)
    q = Works()
    filters: dict = {}
    if args.title_abstract:
        filters["title_and_abstract"] = {"search": query}
    else:
        q = q.search(query)
    if (y := parse_year(args.year)):
        filters["publication_year"] = y
    if args.type:
        filters["type"] = "|".join(args.type)
    if args.oa:
        filters["is_oa"] = True
    if args.with_content:
        filters["has_content"] = {"grobid_xml": True}
    if args.min_citations:
        filters["cited_by_count"] = f">{args.min_citations - 1}"
    if args.language:
        filters["language"] = args.language
    if filters:
        q = q.filter(**filters)
    if args.sort == "citations":
        q = q.sort(cited_by_count="desc")
    elif args.sort == "date":
        q = q.sort(publication_date="desc")
    q = q.select(WORK_FIELDS)
    n = max(1, min(args.max, 100))
    try:
        results = q.get(per_page=n, page=args.page)
    except Exception as e:
        die(f"OpenAlex API: {e.__class__.__name__}: {e}")
    meta = getattr(results, "meta", {}) or {}
    entries = [work_to_entry(w) for w in results]
    if args.format != "json":
        mode = "title+abstract match" if args.title_abstract else "title/abstract/full-text match"
        print(f"query: {query}  ({mode}, sort={args.sort}"
              + (f", filters={filters}" if filters else "") + ")")
        print(f"showing {(args.page - 1) * n + 1}-{(args.page - 1) * n + len(entries)} of {meta.get('count')} matches\n")
    print_entries(entries, args.format, args.abstract_len, start=(args.page - 1) * n)


def fetch_work(raw: str) -> dict:
    ident = normalize_work_id(raw)
    try:
        return Works()[ident]
    except Exception as e:
        die(f"could not fetch {raw!r} ({ident}): {e.__class__.__name__}: {e}")


def cmd_work(args: argparse.Namespace) -> None:
    configure()
    entries = [work_to_entry(fetch_work(x)) for x in args.ids]
    print_entries(entries, args.format, args.abstract_len)


def cmd_citations(args: argparse.Namespace) -> None:
    configure()
    w = fetch_work(args.id)
    wid = w_id(w)
    q = Works().filter(cites=wid).sort(cited_by_count="desc").select(WORK_FIELDS)
    n = max(1, min(args.max, 100))
    results = q.get(per_page=n)
    meta = getattr(results, "meta", {}) or {}
    entries = [work_to_entry(x) for x in results]
    if args.format != "json":
        print(f"works citing {wid} ({' '.join((w.get('title') or '').split())[:60]}): showing {len(entries)} of {meta.get('count')}")
        print("(OpenAlex undercounts citations of CS preprints; cross-check with s2_cli.py citations)\n")
    print_entries(entries, args.format, args.abstract_len)


# --------------------------------------------------------------------------- TEI -> markdown
def tei_text(el) -> str:
    """Flatten an element to text: formulas as $...$, refs kept as their visible text."""
    parts: list[str] = []

    def walk(e):
        tag = e.tag.replace(TEI_NS, "")
        if tag == "formula":
            parts.append(" $" + " ".join("".join(e.itertext()).split()) + "$ ")
            if e.tail:
                parts.append(e.tail)
            return
        if e.text:
            parts.append(e.text)
        for c in e:
            walk(c)
        if tag == "s":
            parts.append(" ")  # Grobid sentence elements carry no trailing space
        if e.tail:
            parts.append(e.tail)

    walk(el)
    return " ".join("".join(parts).split())


def head_level(head, depth: int) -> int:
    """Markdown heading level from Grobid's numbering (n="2.1" -> 3), else from div nesting."""
    n = (head.get("n") or "").strip().rstrip(".")
    if re.fullmatch(r"[A-Z\d]+(\.\d+)*", n):
        return min(1 + len(n.split(".")), 6)
    return min(2 + depth, 6)


def tei_to_markdown(xml_bytes: bytes, meta: dict) -> str:
    root = ET.fromstring(xml_bytes)
    out: list[str] = []
    title = meta.get("title") or ""
    t = root.find(f".//{TEI_NS}titleStmt/{TEI_NS}title")
    if t is not None and tei_text(t):
        title = tei_text(t)
    out.append(f"# {title}\n")
    if meta.get("authors"):
        out.append("Authors: " + ", ".join(meta["authors"]) + "\n")
    line = []
    if meta.get("venue"):
        line.append(meta["venue"])
    if meta.get("year"):
        line.append(str(meta["year"]))
    if meta.get("doi"):
        line.append(f"doi:{meta['doi']}")
    if meta.get("openalex_id"):
        line.append(f"OpenAlex {meta['openalex_id']}")
    if line:
        out.append(" - ".join(line) + "\n")
    out.append("_Converted from OpenAlex's Grobid TEI XML; equations are lossy plain text, tables may be flattened._\n")

    abstract = root.find(f".//{TEI_NS}abstract")
    if abstract is not None and tei_text(abstract):
        out.append("## Abstract\n")
        for p in abstract.iter(f"{TEI_NS}p"):
            if tei_text(p):
                out.append(tei_text(p) + "\n")
        if not any(True for _ in abstract.iter(f"{TEI_NS}p")):
            out.append(tei_text(abstract) + "\n")

    body = root.find(f".//{TEI_NS}body")
    if body is not None:
        def emit_div(div, depth):
            for child in div:
                tag = child.tag.replace(TEI_NS, "")
                if tag == "head":
                    txt = tei_text(child)
                    if txt:
                        n = (child.get("n") or "").strip()
                        label = f"{n} {txt}" if n and not txt.startswith(n) else txt
                        out.append("\n" + "#" * head_level(child, depth) + " " + label + "\n")
                elif tag == "p":
                    txt = tei_text(child)
                    if txt:
                        out.append(txt + "\n")
                elif tag == "formula":
                    out.append("$$ " + tei_text(child).strip("$ ") + " $$\n")
                elif tag == "figure":
                    head = child.find(f"{TEI_NS}head")
                    desc = child.find(f"{TEI_NS}figDesc")
                    label = tei_text(head) if head is not None else "Figure"
                    cap = tei_text(desc) if desc is not None else ""
                    is_table = child.get("type") == "table"
                    if is_table:
                        tbl = child.find(f"{TEI_NS}table")
                        rows = []
                        if tbl is not None:
                            for row in tbl.iter(f"{TEI_NS}row"):
                                rows.append(" | ".join(tei_text(c) for c in row.iter(f"{TEI_NS}cell")))
                        out.append(f"\n**{label}.** {cap}\n")
                        if rows:
                            out.append("\n".join("| " + r + " |" for r in rows) + "\n")
                    else:
                        out.append(f"\n**{label}.** {cap}\n")
                elif tag == "div":
                    emit_div(child, depth + 1)
                elif tag == "note":
                    txt = tei_text(child)
                    if txt:
                        out.append(f"> {txt}\n")
                elif tag == "list":
                    for item in child.iter(f"{TEI_NS}item"):
                        txt = tei_text(item)
                        if txt:
                            out.append(f"- {txt}\n")
                else:
                    txt = tei_text(child)
                    if txt:
                        out.append(txt + "\n")

        emit_div(body, 0)

    bibl = root.find(f".//{TEI_NS}listBibl")
    if bibl is not None:
        out.append("\n## References\n")
        for i, b in enumerate(bibl.iter(f"{TEI_NS}biblStruct"), 1):
            bt = b.find(f".//{TEI_NS}analytic/{TEI_NS}title")
            if bt is None or not tei_text(bt):
                bt = b.find(f".//{TEI_NS}monogr/{TEI_NS}title")
            authors = []
            for a in b.iter(f"{TEI_NS}author"):
                s = a.find(f"{TEI_NS}persName/{TEI_NS}surname")
                if s is not None and tei_text(s):
                    authors.append(tei_text(s))
            date = b.find(f".//{TEI_NS}date")
            year = date.get("when", "")[:4] if date is not None else ""
            venue = b.find(f".//{TEI_NS}monogr/{TEI_NS}title")
            ids = [f"{i.get('type')}:{tei_text(i)}" for i in b.iter(f"{TEI_NS}idno") if i.get("type") in ("DOI", "arXiv")]
            line = f"{i}. " + (tei_text(bt) if bt is not None else "(untitled)")
            if authors:
                line += " - " + ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
            if year:
                line += f" ({year})"
            if venue is not None and bt is not None and venue is not bt and tei_text(venue):
                line += f". {tei_text(venue)}"
            if ids:
                line += " " + " ".join(ids)
            out.append(line + "\n")
    return "\n".join(out)


def pdf_to_markdown(pdf_path: Path, entry: dict) -> str:
    """Fallback when there is no Grobid TEI: layout-aware text from the PDF via pymupdf4llm."""
    import pymupdf4llm

    body = pymupdf4llm.to_markdown(str(pdf_path))
    head = [f"# {entry.get('title') or pdf_path.stem}\n"]
    if entry.get("authors"):
        head.append("Authors: " + ", ".join(entry["authors"]) + "\n")
    line = [x for x in (entry.get("venue"), str(entry.get("year") or ""), f"doi:{entry['doi']}" if entry.get("doi") else "",
                        f"OpenAlex {entry['openalex_id']}" if entry.get("openalex_id") else "") if x]
    if line:
        head.append(" - ".join(line) + "\n")
    head.append("_Converted from the PDF with pymupdf4llm (no Grobid TEI available); headings and tables are heuristic._\n")
    return "\n".join(head) + "\n" + body


def fallback_pdf_url(entry: dict) -> tuple[str, str]:
    """(url, reason-to-skip). Direct open-access PDF outside OpenAlex; arXiv is left to arxiv_cli.py."""
    url = entry.get("oa_pdf") or ""
    if not url:
        return "", "no open-access URL known to OpenAlex"
    if "arxiv.org/" in url:
        return "", f"open-access copy is arXiv {entry.get('arxiv') or ''}: use arxiv_cli.py fetch (TeX source, rate-limited properly)"
    return url, ""


def download_fallback_pdf(url: str, dest: Path) -> tuple[bool, str]:
    """GET a publisher/repository PDF (no key needed). Returns (ok, message)."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"}, timeout=120, allow_redirects=True)
    except requests.RequestException as e:
        return False, f"{e.__class__.__name__}: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    if not r.content.startswith(b"%PDF"):
        return False, f"not a PDF ({r.headers.get('content-type', '?')}; probably a landing page)"
    dest.write_bytes(r.content)
    return True, f"downloaded ({len(r.content) // 1024} KiB) from {url}"


def write_markdown(paper_dir: Path, entry: dict, result: dict, xml_path: Path | None, pdf_path: Path | None) -> None:
    """paper.md from TEI when available, else from the PDF. Records what it used."""
    md_path = paper_dir / "paper.md"
    md = None
    if xml_path and xml_path.exists():
        try:
            md = tei_to_markdown(xml_path.read_bytes(), entry)
            result["markdown_source"] = "tei"
        except ET.ParseError as e:
            result["steps"].append(f"markdown: TEI parse failed ({e}); trying the PDF")
    if md is None and pdf_path and pdf_path.exists():
        try:
            md = pdf_to_markdown(pdf_path, entry)
            result["markdown_source"] = "pdf"
        except Exception as e:  # pymupdf can fail on odd PDFs
            result["steps"].append(f"markdown: PDF conversion failed ({e.__class__.__name__}: {e}); read the PDF directly")
    if md is None:
        return
    md_path.write_text(md, encoding="utf-8")
    result["markdown"] = str(md_path)
    result["markdown_lines"] = md.count("\n") + 1
    result["markdown_words"] = len(md.split())
    result["steps"].append(f"markdown: wrote paper.md from {result['markdown_source']}")


def cmd_fetch(args: argparse.Namespace) -> None:
    key = configure()
    w = fetch_work(args.id)
    entry = work_to_entry(w)
    wid = entry["openalex_id"]
    paper_dir = Path(args.dest) / wid
    paper_dir.mkdir(parents=True, exist_ok=True)
    meta_path = paper_dir / "meta.json"
    meta_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    result: dict = {"openalex_id": wid, "title": entry["title"], "dir": str(paper_dir), "meta": str(meta_path), "steps": ["metadata: saved"]}
    if entry["arxiv"]:
        result["arxiv"] = entry["arxiv"]
        result["note"] = (f"this is arXiv {entry['arxiv']}: `arxiv_cli.py fetch {entry['arxiv']}` gives the TeX source, "
                          "which reads better than Grobid XML")
    want_xml = args.xml or not args.pdf
    want_pdf = args.pdf
    xml_path = paper_dir / f"{wid}.tei.xml"
    pdf_path = paper_dir / f"{wid}.pdf"
    s = session(key)
    last_headers = {}

    def content_get(ext: str) -> requests.Response:
        nonlocal last_headers
        if not key:
            die("content downloads need an OpenAlex API key (OPENALEX_API_KEY or ~/.config/openalex/api_key)")
        r = s.get(f"{CONTENT_BASE}/works/{wid}.{ext}", timeout=120)
        last_headers = r.headers
        return r

    # 1. Grobid TEI from OpenAlex (best readable source after arXiv TeX)
    if want_xml:
        if xml_path.exists() and not args.force:
            result["steps"].append("xml: cached")
        elif not entry["has_xml"]:
            result["steps"].append("xml: not available from OpenAlex for this work")
        else:
            r = content_get("grobid-xml")
            if r.status_code != 200:
                result["steps"].append(f"xml: HTTP {r.status_code} {r.text[:120]}")
            else:
                data = r.content
                if data[:2] == b"\x1f\x8b":
                    import gzip
                    data = gzip.decompress(data)
                xml_path.write_bytes(data)
                result["steps"].append(f"xml: downloaded ({len(data) // 1024} KiB)")
        if xml_path.exists():
            result["xml"] = str(xml_path)

    # 2. PDF from OpenAlex when asked for, or when there is no TEI and OpenAlex has the PDF
    need_pdf_for_text = want_xml and not xml_path.exists()
    if want_pdf or need_pdf_for_text:
        if pdf_path.exists() and not args.force:
            result["steps"].append("pdf: cached")
        elif entry["has_pdf"]:
            r = content_get("pdf")
            if r.status_code != 200:
                result["steps"].append(f"pdf: HTTP {r.status_code} {r.text[:120]}")
            else:
                pdf_path.write_bytes(r.content)
                result["steps"].append(f"pdf: downloaded ({len(r.content) // 1024} KiB) from OpenAlex")
                result["pdf_source"] = "openalex"
        else:
            # 3. Free fallback: the open-access copy OpenAlex knows about (publisher / repository)
            url, why_not = fallback_pdf_url(entry)
            if not url:
                result["steps"].append(f"pdf: not available from OpenAlex; fallback skipped: {why_not}")
            else:
                ok, msg = download_fallback_pdf(url, pdf_path)
                result["steps"].append(f"pdf: fallback {msg}" if ok else f"pdf: fallback from {url} failed: {msg}")
                if ok:
                    result["pdf_source"] = url
        if pdf_path.exists():
            result["pdf"] = str(pdf_path)

    # 4. paper.md from whatever we have
    write_markdown(paper_dir, entry, result, xml_path if want_xml else None, pdf_path)
    if "markdown" not in result:
        if entry["arxiv"]:
            result["unavailable"] = f"no OpenAlex full text; this paper is on arXiv, run: arxiv_cli.py fetch {entry['arxiv']}"
        else:
            result["unavailable"] = ("no full text: OpenAlex has neither TEI nor PDF for this work and the open-access copy "
                                     "could not be downloaded; try the oa_url in a browser or ask the user for the PDF")
        if entry["oa_pdf"]:
            result["oa_url"] = entry["oa_pdf"]
    if last_headers:
        result["budget"] = budget_line(last_headers)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_status(args: argparse.Namespace) -> None:
    key = configure()
    if not key:
        print("no API key found (OPENALEX_API_KEY or ~/.config/openalex/api_key); search works on the shared pool, content downloads will fail")
        return
    r = session(key).get(f"{API_BASE}/works/W2741809807", params={"select": "id"}, timeout=30)  # singleton lookups are free
    if r.status_code != 200:
        die(f"key check failed: HTTP {r.status_code} {r.text[:200]}")
    print("key: ok (" + ("env OPENALEX_API_KEY" if os.environ.get("OPENALEX_API_KEY") else str(KEY_FILE)) + ")")
    print(budget_line(r.headers))
    print(f"requests remaining today: {r.headers.get('x-ratelimit-remaining')} of {r.headers.get('x-ratelimit-limit')}")


# --------------------------------------------------------------------------- main
def add_output_flags(p: argparse.ArgumentParser, abstract_default: int) -> None:
    p.add_argument("-f", "--format", choices=["md", "table", "json"], default="md")
    p.add_argument("--abstract-len", type=int, default=abstract_default, help="chars of abstract; 0 hides, -1 full")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="openalex_cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=str(DEFAULT_DEST), help="papers directory (default: references/papers)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search works")
    s.add_argument("query", nargs="+")
    s.add_argument("--title-abstract", action="store_true", help="match title+abstract only (default also matches indexed full text)")
    s.add_argument("--year", help="2024, 2020-2024, 2022- (since), -2019 (until)")
    s.add_argument("--type", action="append", help="article, preprint, conference-paper, review, book-chapter... (repeatable, OR-ed)")
    s.add_argument("--oa", action="store_true", help="open-access only")
    s.add_argument("--with-content", action="store_true", help="only works whose TEI XML full text can be fetched")
    s.add_argument("--min-citations", type=int)
    s.add_argument("--language", help="ISO code, e.g. en")
    s.add_argument("--sort", choices=["relevance", "citations", "date"], default="relevance",
                   help="citations/date sort the *matches* (full-text matches include noise; add --title-abstract)")
    s.add_argument("-n", "--max", type=int, default=10, help="results per page (max 100)")
    s.add_argument("--page", type=int, default=1)
    add_output_flags(s, 300)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("work", help="details for one or more works")
    d.add_argument("ids", nargs="+", help="OpenAlex id (W...), DOI, arXiv id/URL, or openalex.org URL")
    add_output_flags(d, -1)
    d.set_defaults(func=cmd_work)

    c = sub.add_parser("citations", help="works citing the given work")
    c.add_argument("id")
    c.add_argument("-n", "--max", type=int, default=25)
    add_output_flags(c, 0)
    c.set_defaults(func=cmd_citations)

    f = sub.add_parser("fetch", help="download TEI XML (default) and/or PDF into references/papers/<W-id>/")
    f.add_argument("id")
    f.add_argument("--xml", action="store_true", help="download the Grobid TEI XML and write paper.md (default when --pdf is not given)")
    f.add_argument("--pdf", action="store_true", help="download the PDF")
    f.add_argument("--force", action="store_true", help="re-download even if present")
    f.set_defaults(func=cmd_fetch)

    st = sub.add_parser("status", help="check the API key and remaining daily budget")
    st.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args.func(args)


if __name__ == "__main__":
    main()
