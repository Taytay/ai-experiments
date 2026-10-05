# Semantic Scholar (for `s2_cli.py`)

Graph API docs: https://api.semanticscholar.org/api-docs/graph
Python client used: `semanticscholar` (community package, typed `Paper` objects).
Key (optional, free): request at https://www.semanticscholar.org/product/api#api-key-form and
export `SEMANTIC_SCHOLAR_API_KEY`. Without it you share the public pool (works, but may 429
under load); with it you get a dedicated 1 request/second.

## Commands

| command | what it returns |
|---|---|
| `search QUERY...` | papers matching the query; relevance-ranked by default |
| `paper ID...` | one or more papers with TLDR, venue, citation counts, external ids, OA PDF |
| `citations ID` | papers citing ID (API order newest first; script re-sorts the fetched page by citations) |
| `references ID` | papers ID cites |
| `recommend ID` | Semantic Scholar's related-paper recommendations (`--pool recent` or `all-cs`) |

Common flags: `-n/--max`, `-f md|table|json`, `--abstract-len N` (0 hides, -1 full).

## Identifiers accepted by `paper`, `citations`, `references`, `recommend`

- arXiv: `2211.17192`, `2211.17192v2`, `arXiv:2211.17192`, any `arxiv.org/abs|pdf|html/...` URL
- DOI: `10.48550/arXiv.2211.17192` or `DOI:...`
- Semantic Scholar: 40-hex `paperId`, or a `semanticscholar.org/paper/.../<id>` URL
- Others passed through verbatim: `CorpusId:123`, `MAG:...`, `ACL:...`, `PMID:...`, `PMCID:...`

## Search modes

**Relevance (default).** Free text; the API ranks by relevance. Max 100 results per call.
Filters: `--year 2024` / `--year 2022-2025` / `--year 2023-`, `--venue NAME` (repeatable),
`--fos "Computer Science"` (repeatable), `--type Conference|JournalArticle|Review|...`,
`--open-access`, `--min-citations N`.

**Bulk (`--sort citations` or `--sort date`).** No relevance ranking; results are keyword matches
sorted by the chosen field, up to 1000. Same filters apply. Bulk mode supports boolean query
syntax on the query string:

```
speculative decoding                      all terms must match
"speculative decoding"                    exact phrase
(speculative | lookahead) decoding        OR
speculative -hardware                     exclude
decod*                                    prefix
```

Bulk results include noise from unrelated fields that share a keyword; screen them.

## Fields worth knowing

- `citationCount` and `influentialCitationCount` (citations judged to build substantively on the
  paper; a better signal than raw count for "is this the paper to read").
- `externalIds.ArXiv` / `externalIds.DOI`: printed on every result; the arXiv id feeds
  `arxiv_cli.py fetch`.
- `venue` of `arXiv.org` means preprint only. A conference or journal name means S2 matched it to
  a published version.
- `tldr` is only returned by `paper`, not by `search`.
- `openAccessPdf.url` is a direct PDF link when S2 knows one.

## Etiquette

- Sequential calls only; the script spaces invocations 1 s apart via a temp-dir timestamp.
- The package retries 429s with exponential backoff (up to ~10 attempts), so a call can take a
  minute under load before failing. Do not wrap it in a retry loop.
- Search indexing lags arXiv by days; for this week's preprints use `arxiv_cli.py search`.
