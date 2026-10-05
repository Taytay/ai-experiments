#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pytest",
#     "arxiv>=2.1",
#     "semanticscholar>=0.10",
#     "pyalex>=0.18",
#     "requests>=2.31",
#     "pymupdf4llm>=0.0.17",
# ]
# ///
"""
Tests for the research-papers skill scripts (arxiv_cli.py, s2_cli.py, openalex_cli.py).

Run unit tests only (no network):
    ./test_scripts.py
    # or: uv run test_scripts.py

Run all tests including integration (network; mind the arXiv / Semantic Scholar rate limits):
    ./test_scripts.py --integration
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SKILL_DIR = TESTS_DIR.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"
ARXIV_CLI = SCRIPTS_DIR / "arxiv_cli.py"
S2_CLI = SCRIPTS_DIR / "s2_cli.py"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ax():
    return load(ARXIV_CLI)


@pytest.fixture(scope="module")
def s2():
    return load(S2_CLI)


def run_script(script: Path, *args: str, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", str(script), *args],
        capture_output=True, text=True, cwd=cwd, timeout=timeout, encoding="utf-8", errors="replace",
    )


# ============================================================
# Skill structure
# ============================================================

def test_skill_layout():
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert ARXIV_CLI.is_file() and S2_CLI.is_file()
    assert (SKILL_DIR / "references" / "arxiv-query-syntax.md").is_file()
    assert (SKILL_DIR / "references" / "semantic-scholar.md").is_file()


def test_skill_frontmatter():
    head = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    assert "name: research-papers" in head
    assert "description:" in head


def test_scripts_do_not_shadow_their_packages():
    # `arxiv.py` would make `import arxiv` import itself; keep the names distinct.
    assert ARXIV_CLI.name != "arxiv.py"
    assert S2_CLI.name not in ("semanticscholar.py",)


@pytest.mark.parametrize("script", [ARXIV_CLI, S2_CLI])
def test_help_runs(script):
    r = run_script(script, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


# ============================================================
# arxiv_cli.py: id normalization
# ============================================================

@pytest.mark.parametrize(
    "raw, base, versioned",
    [
        ("2601.07372", "2601.07372", "2601.07372"),
        ("2601.07372v2", "2601.07372", "2601.07372v2"),
        ("arxiv:2601.07372", "2601.07372", "2601.07372"),
        ("https://www.arxiv.org/abs/2601.07372", "2601.07372", "2601.07372"),
        ("https://arxiv.org/pdf/2601.07372v1.pdf", "2601.07372", "2601.07372v1"),
        ("https://arxiv.org/html/2601.07372v3", "2601.07372", "2601.07372v3"),
        ("https://arxiv.org/src/1706.03762v7", "1706.03762", "1706.03762v7"),
        ("hep-th/9901001", "hep-th/9901001", "hep-th/9901001"),
        ("https://arxiv.org/abs/math.GT/0309136v2", "math.GT/0309136", "math.GT/0309136v2"),
    ],
)
def test_normalize_id(ax, raw, base, versioned):
    assert ax.normalize_id(raw) == base
    assert ax.normalize_id(raw, keep_version=True) == versioned


def test_normalize_id_rejects_garbage(ax):
    with pytest.raises(SystemExit):
        ax.normalize_id("not a paper")


def test_old_style_id_dirname(ax):
    assert ax.id_to_dirname("hep-th/9901001") == "hep-th_9901001"


# ============================================================
# arxiv_cli.py: query builder
# ============================================================

def _ns(**kw):
    import argparse

    d = dict(query=[], title=None, abstract=None, author=None, category=None, since=None, until=None, raw=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_query_plain_words_are_anded(ax):
    assert ax.build_search_query(_ns(query=["curriculum", "learning"])) == "all:curriculum AND all:learning"


def test_query_phrases_operators_categories_dates(ax):
    q = ax.build_search_query(_ns(query=['"mixture of experts"', "OR", "MoE"], category=["cs.LG", "cs.CL"], since="2025-01-01"))
    assert q == '(all:"mixture of experts" OR all:MoE) AND (cat:cs.LG OR cat:cs.CL) AND (submittedDate:[202501010000 TO 209912312359])'


def test_query_fielded(ax):
    assert ax.build_search_query(_ns(title=['"attention is all you need"'], author=["Vaswani"])) == '(ti:"attention is all you need") AND (au:Vaswani)'


def test_query_raw_wins(ax):
    assert ax.build_search_query(_ns(query=["ignored"], raw="ti:foo ANDNOT abs:bar")) == "ti:foo ANDNOT abs:bar"


def test_query_empty_is_an_error(ax):
    with pytest.raises(SystemExit):
        ax.build_search_query(_ns())


# ============================================================
# arxiv_cli.py: TeX handling
# ============================================================

def test_strip_comments(ax):
    tex = "a % comment\nb \\% kept % dropped\n\\begin{comment}\nhidden\n\\end{comment}\nc\n"
    out = ax.strip_comments(tex)
    assert out == "a\nb \\% kept\n\nc\n"


def test_flatten_inlines_inputs_and_bbl(ax, tmp_path):
    src = tmp_path / "src"
    (src / "sections").mkdir(parents=True)
    (src / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{sections/intro}\n\\include{sections/method.tex}\n"
        "\\input{missing}\n\\end{document}\n", encoding="utf-8")
    (src / "sections" / "intro.tex").write_text("\\section{Intro} % c\nHello\n", encoding="utf-8")
    (src / "sections" / "method.tex").write_text("\\section{Method}\n\\input{intro}\n", encoding="utf-8")
    (src / "refs.bbl").write_text("\\bibitem{x} X.\n", encoding="utf-8")

    entry = ax.find_entrypoints(src)
    assert [p.name for p in entry] == ["main.tex"]
    flat = ax.write_flat(tmp_path, src, entry)
    text = flat.read_text(encoding="utf-8")
    assert "\\section{Intro}" in text and "\\section{Method}" in text
    assert "% c" not in text
    assert "could not resolve \\input{missing}" in text
    assert "skipped already-included file intro.tex" in text  # method.tex re-includes intro
    assert "\\bibitem{x}" in text


def test_find_entrypoints_prefers_conventional_names(ax, tmp_path):
    for name in ("appendix.tex", "ms.tex", "zz.tex"):
        (tmp_path / name).write_text("\\documentclass{article}", encoding="utf-8")
    (tmp_path / "notes.tex").write_text("no documentclass here", encoding="utf-8")
    assert [p.name for p in ax.find_entrypoints(tmp_path)] == ["ms.tex", "appendix.tex", "zz.tex"]


def test_unpack_source_detects_formats(ax, tmp_path):
    # tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"\\documentclass{article}"
        info = tarfile.TarInfo("main.tex")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    kind, saved = ax.unpack_source(buf.getvalue(), tmp_path / "t", tmp_path / "t" / "src", "1234.56789")
    assert kind == "tar" and saved.name == "1234.56789.tar.gz"
    assert (tmp_path / "t" / "src" / "main.tex").read_bytes() == data

    # single gzipped .tex
    kind, saved = ax.unpack_source(gzip.compress(b"\\documentclass{article} x"), tmp_path / "g", tmp_path / "g" / "src", "1234.56789")
    assert kind == "single-file" and (tmp_path / "g" / "src" / "main.tex").exists()

    # pdf only
    kind, saved = ax.unpack_source(b"%PDF-1.7 ...", tmp_path / "p", tmp_path / "p" / "src", "1234.56789")
    assert kind == "pdf-only" and saved.suffix == ".pdf"


def test_tar_path_traversal_is_blocked(ax, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("../escape.tex")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.TarError):
        ax.unpack_source(buf.getvalue(), tmp_path / "t", tmp_path / "t" / "src", "1234.56789")
    # nothing escaped src/, and the blob was not silently reinterpreted as a single gzipped file
    assert not list(tmp_path.rglob("escape.tex"))
    assert not (tmp_path / "t" / "src" / "source.txt").exists()


# ============================================================
# Global rate-limit lock
# ============================================================

def _slot_timings(mod, slot_name: str, workers: int, hold: float) -> list[tuple[float, float]]:
    slot = getattr(mod, slot_name)

    def one(_):
        with slot():
            start = time.time()
            time.sleep(hold)
            return start, time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sorted(ex.map(one, range(workers)))


@pytest.mark.parametrize("script_fixture, slot, interval", [("ax", "arxiv_slot", 3.0), ("s2", "s2_slot", 1.0)])
def test_slot_serializes_and_spaces_requests(request, tmp_path, monkeypatch, script_fixture, slot, interval):
    mod = request.getfixturevalue(script_fixture)
    monkeypatch.setattr(mod, "_LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(mod, "_STAMP_PATH", tmp_path / "stamp")
    timings = _slot_timings(mod, slot, workers=3, hold=0.2)
    for (_, prev_end), (next_start, _) in zip(timings, timings[1:]):
        gap = next_start - prev_end
        assert gap >= interval - 0.05, f"requests only {gap:.2f}s apart (need {interval}s)"


def test_slot_tolerates_missing_or_corrupt_stamp(ax, tmp_path, monkeypatch):
    monkeypatch.setattr(ax, "_LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(ax, "_STAMP_PATH", tmp_path / "stamp")
    (tmp_path / "stamp").write_text("garbage")
    t = time.time()
    with ax.arxiv_slot():
        pass
    assert time.time() - t < 1.0
    float((tmp_path / "stamp").read_text())  # rewritten as a valid timestamp


# ============================================================
# s2_cli.py: identifier normalization and entry mapping
# ============================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2211.17192", "ArXiv:2211.17192"),
        ("2211.17192v3", "ArXiv:2211.17192"),
        ("arXiv:2211.17192", "ArXiv:2211.17192"),
        ("https://arxiv.org/abs/2211.17192v2", "ArXiv:2211.17192"),
        ("https://arxiv.org/pdf/2211.17192.pdf", "ArXiv:2211.17192"),
        ("10.48550/arXiv.2211.17192", "DOI:10.48550/arXiv.2211.17192"),
        ("doi:10.1000/xyz", "DOI:10.1000/xyz"),
        ("CorpusId:12345", "CorpusId:12345"),
        ("https://www.semanticscholar.org/paper/Some-Title/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
         "204e3073870fae3d05bcbc2f6a8e263d9b72e776"),
        ("204e3073870fae3d05bcbc2f6a8e263d9b72e776", "204e3073870fae3d05bcbc2f6a8e263d9b72e776"),
    ],
)
def test_s2_normalize_paper_id(s2, raw, expected):
    assert s2.normalize_paper_id(raw) == expected


def test_s2_paper_to_entry_handles_sparse_records(s2):
    from semanticscholar.Paper import Paper

    p = Paper({"paperId": "abc", "title": "  A   title ", "externalIds": {"ArXiv": "2211.17192"},
               "authors": [{"name": "A. Author"}], "citationCount": 5})
    e = s2.paper_to_entry(p)
    assert e["title"] == "A title" and e["arxiv"] == "2211.17192" and e["citations"] == 5
    assert e["authors"] == ["A. Author"] and e["influential_citations"] == 0 and e["tldr"] == ""
    assert e["url"].endswith("/abc")


# ============================================================
# openalex_cli.py: identifiers, work mapping, TEI conversion
# ============================================================

OPENALEX_CLI = SCRIPTS_DIR / "openalex_cli.py"


@pytest.fixture(scope="module")
def oa():
    return load(OPENALEX_CLI)


def test_openalex_help_runs():
    r = run_script(OPENALEX_CLI, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("W4402683901", "W4402683901"),
        ("w4402683901", "W4402683901"),
        ("https://openalex.org/W4402683901", "W4402683901"),
        ("2211.17192", "doi:10.48550/arXiv.2211.17192"),
        ("2211.17192v3", "doi:10.48550/arXiv.2211.17192"),
        ("arXiv:2211.17192", "doi:10.48550/arXiv.2211.17192"),
        ("https://arxiv.org/abs/2211.17192v2", "doi:10.48550/arXiv.2211.17192"),
        ("https://arxiv.org/pdf/2211.17192.pdf", "doi:10.48550/arXiv.2211.17192"),
        ("10.18653/v1/2024.findings-acl.456", "doi:10.18653/v1/2024.findings-acl.456"),
        ("https://doi.org/10.18653/v1/2024.findings-acl.456", "doi:10.18653/v1/2024.findings-acl.456"),
        ("doi:10.1000/xyz", "doi:10.1000/xyz"),
    ],
)
def test_openalex_normalize_work_id(oa, raw, expected):
    assert oa.normalize_work_id(raw) == expected


@pytest.mark.parametrize("y, expected", [("2024", "2024"), ("2020-2024", "2020-2024"), ("2022-", ">2021"), ("-2019", "<2020")])
def test_openalex_parse_year(oa, y, expected):
    assert oa.parse_year(y) == expected


def test_openalex_parse_year_rejects_garbage(oa):
    with pytest.raises(SystemExit):
        oa.parse_year("last year")


def test_openalex_arxiv_id_from_doi_and_locations(oa):
    assert oa.arxiv_id_of({"doi": "https://doi.org/10.48550/arXiv.2211.17192"}) == "2211.17192"
    assert oa.arxiv_id_of({"ids": {"doi": "https://doi.org/10.48550/arxiv.2211.17192v2"}}) == "2211.17192"
    w = {"doi": "https://doi.org/10.1000/other", "locations": [{"landing_page_url": "https://arxiv.org/abs/1706.03762v7"}]}
    assert oa.arxiv_id_of(w) == "1706.03762"
    assert oa.arxiv_id_of({"doi": "https://doi.org/10.1000/other"}) == ""


def test_openalex_invert_abstract(oa):
    idx = {"decoding": [1], "Speculative": [0], "works.": [2]}
    assert oa.invert(idx) == "Speculative decoding works."
    assert oa.invert(None) == ""


def test_openalex_work_to_entry_sparse_and_venue_fallback(oa):
    w = {
        "id": "https://openalex.org/W1", "doi": "https://doi.org/10.18653/v1/x", "title": " A  title ",
        "cited_by_count": 7, "has_content": {"pdf": True, "grobid_xml": False},
        "primary_location": {"source": None, "landing_page_url": "https://aclanthology.org/2024.x"},
        "authorships": [{"author": {"display_name": "A. Author"}}, {"author": None}],
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
    }
    e = oa.work_to_entry(w)
    assert e["openalex_id"] == "W1" and e["doi"] == "10.18653/v1/x" and e["title"] == "A title"
    assert e["venue"] == "aclanthology.org" and e["authors"] == ["A. Author"]
    assert e["has_pdf"] is True and e["has_xml"] is False and e["abstract"] == "Hello world"
    assert e["citations"] == 7 and e["fwci"] is None and e["arxiv"] == ""


TEI_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
 <teiHeader><fileDesc><titleStmt><title>A Sample Paper</title></titleStmt>
  <profileDesc><abstract><p><s>First sentence.</s><s>Second sentence.</s></p></abstract></profileDesc></fileDesc></teiHeader>
 <text><body>
  <div><head n="1">Introduction</head><p><s>Intro text with a ref <ref type="bibr">[1]</ref>.</s></p></div>
  <div><head n="2.1">Method</head><p>We define <formula>x = y + 1</formula> here.</p>
   <formula xml:id="f1">E = mc^2 (1)</formula>
   <figure><head>Figure 1</head><figDesc>A diagram.</figDesc></figure>
   <figure type="table"><head>Table 1</head><figDesc>Results.</figDesc>
    <table><row><cell>a</cell><cell>b</cell></row><row><cell>1</cell><cell>2</cell></row></table></figure>
  </div>
 </body><back><div><listBibl>
  <biblStruct><analytic><title>Cited Work</title><author><persName><surname>Smith</surname></persName></author>
   <idno type="DOI">10.1/abc</idno></analytic><monogr><title>Some Venue</title><imprint><date when="2020-01-01"/></imprint></monogr></biblStruct>
 </listBibl></div></back></text></TEI>"""


def test_openalex_tei_to_markdown(oa):
    md = oa.tei_to_markdown(TEI_SAMPLE.encode(), {"title": "ignored", "authors": ["A. Author"], "year": 2024, "doi": "10.1/x", "openalex_id": "W1", "venue": "Venue"})
    assert md.startswith("# A Sample Paper")
    assert "Authors: A. Author" in md
    assert "First sentence. Second sentence." in md          # sentence re-spacing
    assert "\n## 1 Introduction\n" in md and "\n### 2.1 Method\n" in md  # heading levels from n=
    assert "Intro text with a ref [1]." in md
    assert "$x = y + 1$" in md and "$$ E = mc^2 (1) $$" in md
    assert "**Figure 1.** A diagram." in md
    assert "**Table 1.** Results." in md and "| a | b |" in md and "| 1 | 2 |" in md
    assert "## References" in md and "1. Cited Work - Smith (2020). Some Venue DOI:10.1/abc" in md


def test_openalex_fallback_pdf_url_rules(oa):
    url, why = oa.fallback_pdf_url({"oa_pdf": "https://aclanthology.org/2023.x.pdf", "arxiv": ""})
    assert url == "https://aclanthology.org/2023.x.pdf" and why == ""
    url, why = oa.fallback_pdf_url({"oa_pdf": "https://arxiv.org/pdf/2507.02620", "arxiv": "2507.02620"})
    assert url == "" and "arxiv_cli.py" in why           # arXiv is left to the rate-limited arXiv script
    url, why = oa.fallback_pdf_url({"oa_pdf": ""})
    assert url == "" and "no open-access" in why


def test_openalex_pdf_to_markdown_on_generated_pdf(oa, tmp_path):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Abstract", fontsize=16)
    page.insert_text((72, 110), "Speculative decoding drafts tokens and verifies them in parallel.", fontsize=11)
    pdf = tmp_path / "t.pdf"
    doc.save(str(pdf))
    md = oa.pdf_to_markdown(pdf, {"title": "T", "authors": ["A"], "year": 2024, "doi": "10.1/x", "openalex_id": "W1", "venue": "V"})
    assert md.startswith("# T\n") and "Authors: A" in md and "pymupdf4llm" in md
    assert "Abstract" in md and "drafts tokens and verifies them in parallel" in md


def test_openalex_download_fallback_rejects_non_pdf(oa, tmp_path, monkeypatch):
    class R:
        status_code = 200
        content = b"<html>landing page</html>"
        headers = {"content-type": "text/html"}

    monkeypatch.setattr(oa.requests, "get", lambda *a, **k: R())
    ok, msg = oa.download_fallback_pdf("https://example.org/x", tmp_path / "x.pdf")
    assert ok is False and "not a PDF" in msg and not (tmp_path / "x.pdf").exists()


def test_openalex_api_key_lookup_order(oa, tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "KEY_FILE", tmp_path / "api_key")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    assert oa.api_key() is None
    (tmp_path / "api_key").write_text("  filekey\n")
    assert oa.api_key() == "filekey"
    monkeypatch.setenv("OPENALEX_API_KEY", "envkey")
    assert oa.api_key() == "envkey"


# ============================================================
# Integration (network)
# ============================================================

@pytest.mark.integration
def test_arxiv_fetch_transformer_paper(tmp_path):
    r = run_script(ARXIV_CLI, "--dest", str(tmp_path), "fetch", "https://arxiv.org/abs/1706.03762v7")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["title"] == "Attention Is All You Need"
    assert out["source_kind"] == "tar"
    assert Path(out["flat"]).is_file() and out["flat_lines"] > 500
    assert (tmp_path / "1706.03762" / "meta.json").is_file()


@pytest.mark.integration
def test_s2_search_returns_ranked_results():
    r = run_script(S2_CLI, "search", "speculative", "decoding", "-n", "3", "-f", "json")
    assert r.returncode == 0, r.stderr
    entries = json.loads(r.stdout)
    assert len(entries) == 3
    assert all("title" in e and "citations" in e for e in entries)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
