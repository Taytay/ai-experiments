# arXiv search syntax (for `arxiv_cli.py search`)

The script builds a `search_query` for the arXiv API (`https://export.arxiv.org/api/query`,
Atom XML, no auth). Official manual: https://info.arxiv.org/help/api/user-manual.html

## How the flags map to the query

| Flag                       | Query fragment                          | Notes                                        |
|----------------------------|-----------------------------------------|----------------------------------------------|
| positional words           | `all:w1 AND all:w2 ...`                 | quoted `"phrase"` becomes `all:"phrase"`     |
| `-t/--title TEXT`          | `ti:...`                                | repeatable, each is ANDed                    |
| `-a/--abstract TEXT`       | `abs:...`                               | repeatable                                   |
| `-au/--author NAME`        | `au:...`                                | use `au:del_maestro` style for multi-word    |
| `-c/--category CAT`        | `cat:cs.CL OR cat:cs.LG`                | repeatable, OR-ed together                   |
| `--since / --until`        | `submittedDate:[YYYYMMDD0000 TO ...]`   | date of first submission                     |
| `--raw QUERY`              | used verbatim                           | overrides every other query flag             |

Groups from different flags are ANDed. Inside one flag's text you may write `AND`, `OR`,
`ANDNOT` yourself, e.g. `search "curriculum OR ordering" -t transformer`.

## Field prefixes (for `--raw`)

| Prefix | Field                              |
|--------|------------------------------------|
| `ti`   | title                              |
| `au`   | author                             |
| `abs`  | abstract                           |
| `co`   | comment                            |
| `jr`   | journal reference                  |
| `cat`  | subject category                   |
| `rn`   | report number                      |
| `id`   | arXiv id (prefer `info` instead)   |
| `all`  | all of the above                   |

Operators: `AND`, `OR`, `ANDNOT`. Group with parentheses. Phrase search with double quotes.
Date range: `submittedDate:[202401010000 TO 202412312359]` or `lastUpdatedDate:[...]`.

Examples:

```
--raw 'ti:"speculative decoding" AND cat:cs.CL'
--raw '(abs:"mixture of experts" OR abs:MoE) AND cat:cs.LG ANDNOT abs:vision'
--raw 'au:Vaswani AND ti:attention'
--raw 'cat:cs.AI AND submittedDate:[202506010000 TO 202512312359]'
```

## Sorting and paging

- `--sort relevance` (default), `date` (= submittedDate), `updated` (= lastUpdatedDate).
- Descending by default; `--ascending` flips it.
- `-n/--max` up to 200 per call (the API allows more but gets slow); `--start N` to page.
- `totalResults` is printed so you know how many matches exist beyond the page shown.

## Common CS categories

| Code     | Area                                        |
|----------|---------------------------------------------|
| cs.AI    | Artificial Intelligence                     |
| cs.CL    | Computation and Language (NLP)              |
| cs.CV    | Computer Vision                             |
| cs.LG    | Machine Learning                            |
| cs.NE    | Neural and Evolutionary Computing           |
| cs.IR    | Information Retrieval                       |
| cs.RO    | Robotics                                    |
| cs.HC    | Human-Computer Interaction                  |
| cs.SE    | Software Engineering                        |
| cs.PL    | Programming Languages                       |
| cs.DC    | Distributed, Parallel, and Cluster Computing|
| cs.AR    | Hardware Architecture                       |
| cs.PF    | Performance                                 |
| cs.DB    | Databases                                   |
| cs.DS    | Data Structures and Algorithms              |
| cs.CR    | Cryptography and Security                   |
| cs.CY    | Computers and Society                       |
| cs.GT    | Computer Science and Game Theory            |
| cs.SD    | Sound                                       |
| cs.MA    | Multiagent Systems                          |
| stat.ML  | Machine Learning (Statistics)               |
| eess.AS  | Audio and Speech Processing                 |
| math.OC  | Optimization and Control                    |

Full list: https://arxiv.org/category_taxonomy

## Etiquette

- One request per 3 seconds, one connection at a time (arXiv API terms of use). All traffic goes to
  `export.arxiv.org`, the programmatic-access host. HTTP 429 "Rate exceeded." means the IP is penalised until a quiet period passes and
  every further request extends it: do not retry or probe; use s2_cli.py or openalex_cli.py search.
- Relevance ranking is lexical. Run several phrasings and union the results rather than
  trusting one query.
- `Comment` often carries venue info ("Accepted at ICLR 2026"); `journal_ref` and `doi` are set
  only when the authors add them. Absence means "preprint as far as arXiv knows".
