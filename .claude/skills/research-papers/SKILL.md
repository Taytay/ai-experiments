---
name: research-papers
description: Find CS/ML papers via Semantic Scholar, OpenAlex and arXiv, and read them in full (arXiv TeX source, or OpenAlex full text for other open-access papers). Use when the user shares an arXiv URL/id or DOI, asks to find/search for papers on a topic, wants a reading list, "what's the latest on X", "who built on this paper", or asks to read, summarize, or compare a paper. Papers are downloaded into the current workspace's references/papers/ folder and summaries are written there too.
allowed-tools: Bash(uv run *)
---

# Research papers

Three bundled scripts, all run with `uv run` (no install step; uv fetches a Python if none is
present and installs the script's declared package on first run):

| script | backed by | use it for |
|---|---|---|
| `${CLAUDE_SKILL_DIR}/scripts/s2_cli.py` | Semantic Scholar Graph API (`semanticscholar` package) | **finding and ranking papers**: relevance search with citation counts, venues, TLDRs; who cites / what a paper cites; recommendations |
| `${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py` | arXiv API (`arxiv` package) + arxiv.org | **reading arXiv papers**: download TeX source, flatten, list what is downloaded; also arXiv-only fielded search (date windows, categories, newest preprints) |
| `${CLAUDE_SKILL_DIR}/scripts/openalex_cli.py` | OpenAlex API (`pyalex`) + content.openalex.org | **reading open-access papers that are not on arXiv** (Grobid full text to `paper.md`); venue / OA / type filters; field-weighted impact (FWCI) |

```bash
uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" <command> [options]
uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" <command> [options]
uv run "${CLAUDE_SKILL_DIR}/scripts/openalex_cli.py" <command> [options]
```

Credentials: Semantic Scholar and OpenAlex search both work without a key. `SEMANTIC_SCHOLAR_API_KEY`
gives S2 a dedicated allowance. OpenAlex reads `OPENALEX_API_KEY` or `~/.config/openalex/api_key`;
its content downloads (full text) require the key and spend from a $1/day budget ($0.01 per
paper), so run `openalex_cli.py status` if in doubt and do not bulk-download.

Everything lands in `./references/papers/` relative to the **current working directory** (the user's
workspace). Pass `--dest <dir>` before the command name to change that. Never download into
`~/.cache` or the skill directory.

```
references/papers/
  <arxiv_id>/                 one folder per paper (old-style ids: hep-th/9901001 -> hep-th_9901001)
    meta.json                 title, authors, abstract, categories, dates, links (from the API)
    <arxiv_id>.tar.gz         the raw source archive as served by arXiv (kept for provenance)
    src/                      unpacked TeX source, figures, .bbl, etc.
    paper_flat.tex            all \input/\include files inlined, comments stripped, .bbl appended
    <arxiv_id>.pdf            only when --pdf was passed or arXiv has no TeX source
    summary.md                your summary of the paper (see "Write the summary")
  <W-id>/                     an OpenAlex work fetched with openalex_cli.py (non-arXiv papers)
    meta.json                 title, authors, venue, DOI, citations, FWCI, OA links
    <W-id>.tei.xml            Grobid full text as served by OpenAlex
    paper.md                  readable conversion of the TEI (sections, equations, refs)
    <W-id>.pdf                only when --pdf was passed
    summary.md
  INDEX.md                    one line per paper you have summarized
```

## Rate limits

The arXiv and Semantic Scholar scripts hold a **machine-wide exclusive lock** (an OS file lock
in the temp dir) for the whole duration of every request, and wait out a minimum interval
measured from the *end* of the previous request before starting the next: 3 s for arXiv, 1 s for
Semantic Scholar. This holds across separate `uv run` processes, so commands launched back to
back or even in parallel are serialized and spaced correctly rather than bursting. You still
should not launch many at once (they just queue and take longer), and never retry a 429 in a loop.
Both rules come from arXiv's API terms of use ("no more than one request every three seconds,
and limit requests to a single connection at a time", counted across all your machines). All arXiv
traffic, including source and PDF downloads, goes to `export.arxiv.org`, the host arXiv sets aside
for programmatic access.

**OpenAlex** has no rate problem (100 req/s) but a **daily budget**: with the free key, $1/day,
where a search costs $0.001 and each full-text or PDF download $0.01. `fetch` prints the
remaining budget after every download; stop and tell the user if it drops below about $0.10.

**Semantic Scholar**: the unauthenticated pool is shared and can return HTTP 429 under load;
the `semanticscholar` package retries with backoff. With `SEMANTIC_SCHOLAR_API_KEY` set you get a
dedicated 1 request/second.

**arXiv** asks for at most one request every 3 seconds and penalises the IP (HTTP 429 "Rate
exceeded." or 503) when a client goes faster. The lock above is what keeps this skill inside
that limit. Once penalised, the block clears only after a **quiet period** (typically tens of
minutes) and every further request, even one 30 s later, restarts the clock. So on a 429: do not
retry, do not probe "to see if it is back",
and do not loop. The script itself makes a single attempt per command.

If the API is blocked, `fetch` and `info` fall back automatically to the paper's abs page (noted
on stderr), and source/PDF downloads come from arxiv.org, so **reading a known paper keeps
working**. For `search`, use `s2_cli.py search` or `openalex_cli.py search` instead; they cover
the same papers except the newest few days of preprints, and there is no scrape fallback
(arxiv.org/search refuses non-browser clients and its robots.txt disallows it). Tell the user
the arXiv API is rate-limiting the machine and that it clears after a quiet period.

## Workflow A: find papers

Use when the user asks to find papers, wants a reading list, or asks what exists on a topic.
Start with Semantic Scholar; use arXiv search only for the cases listed in step 4.

1. **Frame the question** in one line: topic, subtopic, time window, whether they want seminal
   or recent work. If they only said "find papers on X", aim for a mix of both.
2. **Search Semantic Scholar**, 2-3 phrasings (synonyms, method names, benchmark names),
   run sequentially. Read [references/semantic-scholar.md](references/semantic-scholar.md) for
   filters and bulk-mode query syntax.

   ```bash
   # relevance-ranked, with citation counts and venues (best first pass)
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" search speculative decoding --year 2023- -n 15

   # the seminal papers: bulk mode sorted by citations, compact table
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" search "speculative decoding" --sort citations -n 15 -f table

   # restrict to venues / open-access / a citation floor
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" search "mixture of experts routing" --venue NeurIPS --venue ICLR --min-citations 20 -n 20

   # machine-readable
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" search "state space models" -n 50 -f json > /tmp/ssm.json
   ```

   Every result line carries the arXiv id when one exists; that id goes straight into
   `arxiv_cli.py fetch` (Workflow B). `-f table` is best for scanning; `--abstract-len -1`
   shows full abstracts.
3. **Expand from a seed paper** when the user has one (or once step 2 finds the key paper):

   ```bash
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" paper 2211.17192                      # details + TLDR
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" citations 2211.17192 -n 100 -f table  # who built on it (most cited first)
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" references 2211.17192 -n 50 -f table  # what it builds on
   uv run "${CLAUDE_SKILL_DIR}/scripts/s2_cli.py" recommend 2211.17192 -n 10            # S2's related-paper suggestions
   ```

   `paper` accepts arXiv ids/URLs, DOIs, and Semantic Scholar ids/URLs.
4. **Use arXiv search** (`arxiv_cli.py search`) when Semantic Scholar is the wrong tool: the
   newest preprints from the last days or weeks (S2 indexing lags), a precise submission date
   window, arXiv category filters, or fielded title/abstract/author queries. Syntax and category
   codes are in [references/arxiv-query-syntax.md](references/arxiv-query-syntax.md).

   ```bash
   uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" search "speculative decoding" -c cs.CL -c cs.LG --sort date -n 15
   uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" search -t '"mixture of experts"' --since 2025-06-01 -f table -n 25
   uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" search --raw '(ti:curriculum OR abs:curriculum) AND cat:cs.CL ANDNOT abs:vision' -n 20
   uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" info 2601.07372 https://arxiv.org/abs/1706.03762v7
   ```

   **Use OpenAlex search** (`openalex_cli.py search`) when you need venue/type/open-access
   filters, field-normalized impact (FWCI), or to know which non-arXiv papers have fetchable
   full text (`--with-content`). Its citation counts for CS are far below S2's (preprints and
   published versions are split, and preprint citations are mostly missed), so do not rank by
   them. Details in [references/openalex.md](references/openalex.md).

   ```bash
   uv run "${CLAUDE_SKILL_DIR}/scripts/openalex_cli.py" search speculative decoding --title-abstract --year 2023- -n 15 -f table
   uv run "${CLAUDE_SKILL_DIR}/scripts/openalex_cli.py" search "retrieval augmented generation" --type review --oa --with-content -n 10
   uv run "${CLAUDE_SKILL_DIR}/scripts/openalex_cli.py" work 10.18653/v1/2024.findings-acl.456 2211.17192
   ```

5. **Screen and rank.** Read the abstracts and drop off-topic hits. Prefer: directly on-topic
   > methodologically substantive > influential (citations, and especially *influential*
   citations, adjusted for age) > published at a real venue > recent. Bulk-mode results are
   unranked keyword matches and include noise (e.g. hardware papers for "speculative"), so
   filter them by eye. Anything whose venue is "arXiv.org" is a preprint; say so.
6. **Deduplicate** by arXiv id or DOI, then by normalized title.
7. **Report** a shortlist: for each paper one line of "why it matters for the question", year,
   venue, citation count, arXiv id and link. Say how you scoped the search (queries, filters,
   date window) and what you did not cover. Offer to fetch and read any of them (Workflow B).

## Workflow B: read a paper

Use when the user gives an arXiv URL/id, a DOI, or asks you to read, summarize, or explain a
paper. Pick the source by where the paper lives:

- **On arXiv** (any of `2601.07372`, `2601.07372v2`, `arxiv:2601.07372`,
  `https://arxiv.org/abs|pdf|html/...`, old-style `hep-th/9901001`): use `arxiv_cli.py fetch`,
  steps 1-3 below. The TeX source is exact.
- **Not on arXiv but open access** (a DOI, an ACL Anthology / PMC / journal paper): use
  `openalex_cli.py fetch <DOI or W-id>` instead of step 1. It writes
  `references/papers/<W-id>/paper.md`, trying in order: OpenAlex's Grobid full text (best), OpenAlex's
  PDF, then the open-access PDF URL OpenAlex knows about (free), converting PDFs with
  pymupdf4llm. `markdown_source` in the result says which one it used. Equations are lossy plain
  text; add `--pdf` when a table or figure matters. If the result has `unavailable`, no full
  text could be obtained: report the `oa_url` to the user or ask for the PDF. If it carries an
  `arxiv` field, the paper is on arXiv and the script stops so you use the arXiv path. Then
  continue at step 2 reading `paper.md` instead of `paper_flat.tex`.

1. **Fetch the TeX source** (preferred over the PDF: exact equations, tables, and captions):

   ```bash
   uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" fetch https://arxiv.org/abs/2601.07372
   ```

   It skips the download if the paper is already present (use `--force` to refresh). It prints
   JSON with `title`, `dir`, `entrypoints`, `flat` (path to `paper_flat.tex`), `flat_lines`, and
   `figures`. Check `source_kind`:
   - `tar` / `single-file` / `zip`: TeX source is unpacked in `src/`; read `paper_flat.tex`.
   - `pdf-only`: arXiv has no TeX for this paper. The PDF was saved; Read it instead.
   - `unknown`: inspect the saved archive in the paper dir manually.
   - If `flat` is missing (no `\documentclass` found), look in `src/` for the real entrypoint
     and run `flatten <id> --entry <path>`.

2. **Read the whole paper.** Read `paper_flat.tex` from start to end. If `flat_lines` exceeds
   what one Read returns, keep reading with `offset` until you reach the appended `.bbl`
   bibliography (or the end). Do not stop after the introduction; methods and appendix tables
   are where the caveats and headline numbers live. When a figure matters to the argument and
   `src/` contains it as PNG/JPG, Read the image. If the flattening left `% [flatten] could not
   resolve` markers, open the referenced file from `src/` directly.

3. **Optionally get the PDF** when rendering matters (complex tables, figures only in PDF):

   ```bash
   uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" fetch 2601.07372 --pdf
   ```

4. **Write the summary** to `references/papers/<arxiv_id>/summary.md`. Structure:

   ```markdown
   # <Title>

   - arXiv: <id> (<version>, <primary category>, submitted <date>) - https://arxiv.org/abs/<id>
   - Authors: ...
   - Venue/status: <from comment/journal_ref, or "preprint">
   - Source: references/papers/<id>/paper_flat.tex

   ## One-paragraph summary
   ## Problem and motivation
   ## Method (with the key equations / algorithm, in words and where needed in LaTeX)
   ## Experiments and results (main numbers, baselines, ablations; cite table/figure numbers)
   ## Limitations and open questions (the authors' plus your own)
   ## Relevance to this workspace   <- only if the workspace has a related project
   ## Key references worth following up
   ```

   For "Relevance to this workspace": if the current workspace is a codebase or project that
   the paper plausibly relates to, first read the relevant parts of the project (grep for the
   mechanism the paper discusses), then state concretely what could be applied, tried, or
   compared, and what would be needed. If the workspace is unrelated or empty, omit the section.

5. **Update the index.** Append one line to `references/papers/INDEX.md` (create it with a heading
   if missing):

   ```markdown
   - [<id>](<id>/summary.md) - <Title> (<year>) - <one-line takeaway>
   ```

6. **Report back** with the summary's one-paragraph version, the path to `summary.md`, and
   anything you could not read (missing figures, pdf-only source, unresolved includes).

## Workflow C: compare several papers

Fetch each paper (Workflow B steps 1-2, sequentially), write an individual
`summary.md` for each, then write `references/papers/compare_<tag>.md` with a comparison table
(problem, method, data, headline results, cost, limitations) followed by a narrative of where
they agree, disagree, and what remains open. Add it to `INDEX.md`.

## Housekeeping

```bash
uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" list            # arXiv papers already in references/papers
uv run "${CLAUDE_SKILL_DIR}/scripts/arxiv_cli.py" flatten <id>    # rebuild paper_flat.tex
uv run "${CLAUDE_SKILL_DIR}/scripts/openalex_cli.py" status       # OpenAlex key + remaining daily budget
```

OpenAlex fetches live in `references/papers/W*/`; `ls references/papers` shows both kinds.

Before fetching, run `list` if the user may already have the paper. Never invent citation
counts, venues, or results you did not read in the source; label anything taken only from the
abstract as such.
