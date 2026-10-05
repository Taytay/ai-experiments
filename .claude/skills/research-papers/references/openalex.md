# OpenAlex (for `openalex_cli.py`)

API docs: https://help.openalex.org/api/ (LLM quick reference: https://help.openalex.org/api/llm-quick-reference/)
Python client used: `pyalex` (query builder, nested filters, retries). Content downloads use
plain HTTPS against `content.openalex.org`.

## Key and budget

- Key lookup order: `OPENALEX_API_KEY` env var, then the file `~/.config/openalex/api_key`.
- Search and metadata work without a key on a small shared pool. **Content downloads need a key.**
- A free key gives **$1 per day**: search `$0.001`, filtered list `$0.0001`, singleton lookup
  free, **each PDF or XML download `$0.01`**. So roughly 100 full texts a day if you do nothing
  else. `openalex_cli.py status` prints what is left and when it resets; `fetch` prints the
  budget after every download.
- OpenAlex allows 100 requests/second, so there is no spacing lock in this script. The budget is
  the constraint, not the rate.

## Commands

| command | what it does |
|---|---|
| `search QUERY...` | works matching the query; relevance-ranked by default |
| `work ID...` | details for works by OpenAlex id (`W...`), DOI, arXiv id/URL, or openalex.org URL |
| `citations ID` | works citing ID, most cited first (thin for CS; see below) |
| `fetch ID [--xml] [--pdf]` | get full text into `references/papers/<W-id>/` and write `paper.md`: Grobid TEI XML from OpenAlex ($0.01) if available, else OpenAlex's PDF ($0.01), else the open-access PDF URL OpenAlex reports (free, publisher/repository); PDFs are converted with pymupdf4llm. arXiv copies are skipped in favour of `arxiv_cli.py fetch`. `--pdf` also keeps the PDF |
| `status` | key check and remaining daily budget |

Common flags: `-n/--max` (up to 100 per page, `--page N` to page), `-f md|table|json`,
`--abstract-len N` (0 hides, -1 full). `--dest DIR` before the command changes the papers dir.

### search filters

- `--title-abstract`: match title and abstract only. **Default `search=` also matches indexed
  full text**, which is why a query like "speculative decoding" returns 25k hits including papers
  that merely mention the phrase. Use `--title-abstract` whenever you sort by citations.
- `--year 2024`, `2020-2024`, `2022-` (since), `-2019` (until)
- `--type article|preprint|conference-paper|review|book-chapter|...` (repeatable, OR-ed)
- `--oa`: open-access only. `--with-content`: only works whose TEI XML can be fetched.
- `--min-citations N`, `--language en`
- `--sort relevance|citations|date`. Sorting by citations returns the most-cited *matches*, not
  the most relevant; with full-text matching that surfaces 1990s microprocessor papers for
  "speculative". Always combine `--sort citations` with `--title-abstract`, and screen by eye.

## What each result carries

- `openalex_id` (W-number), `doi`, and `arxiv` (derived from a `10.48550/arXiv.*` DOI or an
  arxiv.org location URL). The arXiv id feeds `arxiv_cli.py fetch`.
- `citations` (`cited_by_count`) and `fwci` (field-weighted citation impact: 1.0 = world average
  for that field and year; 10 = ten times the average). `citation_percentile` is in JSON output.
- `venue` from `primary_location.source`; when OpenAlex has no source record (ACL Anthology
  papers, for example) the landing-page host is shown instead.
- `oa_status` (gold/green/hybrid/bronze/closed) and `oa_pdf` (best open-access PDF URL, often
  a publisher or arXiv link you can fetch without a key).
- `has_pdf` / `has_xml`: whether `content.openalex.org` can serve the PDF / Grobid TEI XML.
  `has_fulltext` on the API means "indexed for search", which is *not* the same thing.
- `topics`: OpenAlex's primary topic label (noisy; treat as a hint).

## Where OpenAlex is weak for CS/ML

Measured on 2026-09-14 against Semantic Scholar:

| paper | OpenAlex cited_by | S2 citations | OpenAlex referenced_works |
|---|---|---|---|
| Leviathan et al., speculative decoding (arXiv 2211.17192) | 34 | 1988 | 0 |
| Xia et al., speculative decoding survey (ACL Findings 2024) | 43 | 324 | 0 |

- arXiv preprints and their published versions are separate records, and citations of the
  preprint are largely missed. Reference lists for CS papers are often empty.
- So: **rank and build citation graphs with Semantic Scholar**, not OpenAlex. Use OpenAlex's
  counts only as a secondary signal, and its `citations` command only when S2 is unavailable.
- Search relevance for CS queries is comparable to S2 on the first page.

## Where OpenAlex is strong

- **Full text for open-access papers that are not on arXiv**: `fetch` downloads Grobid TEI XML
  and converts it to `paper.md` (sections with numbering, paragraphs, equations as plain text,
  figure and table captions, flattened tables, numbered references with DOIs). For a
  "speculative decoding" query, ~84% of post-2019 matches had XML available. arXiv preprints
  themselves usually do *not* have OpenAlex content; use `arxiv_cli.py fetch` for those, the TeX
  is exact where Grobid output is lossy.
- Venue, publisher, institution, and open-access metadata; FWCI and percentiles for
  field-normalized impact; `--type` and `--oa` filters.
- Structured PDF when you need the rendered paper for an OA work with `has_pdf`.

## The official `openalex-official` CLI

`pip install openalex-official` provides `openalex download` (bulk metadata + PDF/XML by
filter, id list, or stdin, with checkpoints and concurrency) and `openalex status`. It is a bulk
harvester, not a search tool, and version 0.3.2 crashes on Windows (`loop.add_signal_handler`
is unsupported there). This skill therefore calls the content endpoint directly; reach for the
official CLI only for harvesting thousands of works on macOS/Linux.

## TEI conversion notes

`paper.md` is built from Grobid's TEI: `<div><head>` become headings (level from the `n`
numbering when present), `<p>` paragraphs, `<formula>` as `$...$` / `$$...$$` plain text (no
LaTeX), `<figure>` captions as bold labels, `type="table"` figures as pipe tables, `<listBibl>`
as a numbered reference list with surnames, year, venue, and DOI/arXiv ids when Grobid found
them. Sentence boundaries are re-spaced. If a section looks garbled, open the `.tei.xml` or
fetch the `--pdf`.
