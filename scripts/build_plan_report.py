"""Assemble the plan-history report (owner, 2026-10-03: "explaining every step in our entire plan so far ... link between each plan row,
both forward and back"; "You can also have them write separate html pages ... and then link between them").

Input: row fragments (one <article class="row" id="row-N" data-from="..." data-era="..." data-status="..."> per PLAN row) written by
subagents into SRC (rows_001_030.html ...). Output in DST: index.html (the arc of the project by era, every row in a table) and one page
per era (era-1.html ...), with:
  - "Led to" sections built from every row's data-from (the reverse links), so back and forward links always agree;
  - every href="#row-N" rewritten to the page that holds row N;
  - era navigation and previous / next links on every page; a shared stylesheet inlined in each page.
usage: uv run python scripts/build_plan_report.py <src dir> <dst dir>
"""
import html
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
CHAPTERS = [(1, 12, "Measuring and injecting facts"), (13, 23, "Other routes, scale and realism"), (24, 32, "Recipe tuning and graph probes"),
            (33, 48, "The REAL-6 categoriser"), (49, 60, "Jev review and the move to Modal"), (61, 76, "Novel inputs and real places"),
            (77, 90, "Open base, one-slot reader, final recipe"), (91, 104, "Realistic users and the first blind set"),
            (105, 120, "Prompt formats and serving"), (121, 139, "Building the system and product scores"),
            (140, 150, "Blind validation and payee kinds"), (151, 162, "Decision models and embeddings"), (163, 176, "The owner's real budget"),
            (177, 190, "Merchant knowledge and the crowd line"), (191, 200, "History-aware encoders"),
            (201, 208, "Paper lessons and a faster loop"), (209, 224, "The late-interaction decision model"),
            (225, 235, "Matched effort and better households"), (236, 239, "EmbeddingGemma 2")]

STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Literata:opsz,wght@7..72,500;7..72,650&display=swap">
<style>
:root{--paper:#F4F6F3;--panel:#FBFCFA;--ink:#1A2320;--muted:#5A6560;--rule:#D5DCD7;--accent:#17667F;--accent-soft:#E2EEF1;--led:#A8571C;--led-soft:#F6EADF;
 --done:#2F6B45;--todo:#8A6A12;--doing:#17667F;--dropped:#7A7F7C;--code:#EEF1ED;
 --display:"Literata",Georgia,"Times New Roman",serif;--body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--paper:#121815;--panel:#18201C;--ink:#E3EAE6;--muted:#9AA6A0;--rule:#2B3530;
 --accent:#6DB3C9;--accent-soft:#1B2F35;--led:#E39361;--led-soft:#33241A;--done:#7CC293;--todo:#D9B45A;--doing:#6DB3C9;--dropped:#8C928F;--code:#1D2622}}
:root[data-theme="dark"]{color-scheme:dark;--paper:#121815;--panel:#18201C;--ink:#E3EAE6;--muted:#9AA6A0;--rule:#2B3530;--accent:#6DB3C9;--accent-soft:#1B2F35;
 --led:#E39361;--led-soft:#33241A;--done:#7CC293;--todo:#D9B45A;--doing:#6DB3C9;--dropped:#8C928F;--code:#1D2622}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:15px/1.6 var(--body);margin:0}
.wrap{display:grid;grid-template-columns:minmax(0,15rem) minmax(0,1fr);gap:2.5rem;max-width:78rem;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
nav.side{position:sticky;top:calc(env(safe-area-inset-top,0px) + 16px);align-self:start;max-height:calc(100vh - 32px);overflow:auto;font-size:13px}
nav.side h2{font:600 11px/1.3 var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .5rem}
nav.side ol{list-style:none;margin:0 0 1.2rem;padding:0;display:grid;gap:2px}
nav.side a{display:block;padding:4px 8px;border-radius:6px;color:var(--ink);text-decoration:none}
nav.side a:hover{background:var(--accent-soft)}
nav.side a[aria-current="page"]{background:var(--accent-soft);color:var(--accent);font-weight:600}
nav.side .range{color:var(--muted);font:12px var(--mono);font-variant-numeric:tabular-nums}
nav.side details summary{cursor:pointer;color:var(--muted);font-size:12px;margin-bottom:.4rem}
main{min-width:0}
.pagehead{border-bottom:1px solid var(--rule);padding-bottom:1.2rem;margin-bottom:1.6rem}
.pagehead .kicker{font:600 11px/1.3 var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0}
.pagehead h1{font:650 clamp(26px,3.4vw,36px)/1.15 var(--display);margin:.35rem 0 .5rem;text-wrap:balance}
.pagehead p{max-width:65ch;color:var(--muted);margin:0}
article.row{border-top:1px solid var(--rule);padding-block:1.8rem;scroll-margin-top:16px}
article.row header{display:grid;gap:.3rem;margin-bottom:.8rem}
.eyebrow{font:500 12px/1.4 var(--mono);color:var(--muted);margin:0;font-variant-numeric:tabular-nums}
.eyebrow .qid{color:var(--accent)}
.status{display:inline-block;font:600 10px/1 var(--body);letter-spacing:.08em;text-transform:uppercase;padding:4px 7px;border-radius:999px;border:1px solid currentColor;margin-left:.5rem;vertical-align:2px}
.status.done{color:var(--done)}.status.todo{color:var(--todo)}.status.doing{color:var(--doing)}.status.dropped{color:var(--dropped)}
article.row h2{font:650 21px/1.25 var(--display);margin:0;text-wrap:balance}
article.row h3{font:600 11px/1.3 var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:1.1rem 0 .35rem}
article.row p,article.row li{max-width:68ch}
article.row ul{padding-left:1.1rem;margin:.2rem 0;display:grid;gap:.25rem}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
a.back{font-weight:500}
pre.prompt{font:12.5px/1.5 var(--mono);background:var(--code);border:1px solid var(--rule);border-radius:8px;padding:12px 14px;overflow-x:auto;white-space:pre;max-width:100%}
.note{font-size:13px;color:var(--muted);margin:.4rem 0 0}
.tablewrap{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;margin:.2rem 0}
th,td{text-align:left;padding:6px 12px 6px 0;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px}
section.led{margin-top:1.1rem;background:var(--led-soft);border-radius:8px;padding:10px 14px}
section.led h3{color:var(--led);margin-top:0}
section.led a{color:var(--led);font-weight:500}
section.led ul{list-style:none;padding:0}
.pager{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;border-top:1px solid var(--rule);margin-top:2rem;padding-top:1rem}
.eras{display:grid;grid-template-columns:repeat(auto-fill,minmax(16rem,1fr));gap:12px;margin:1rem 0 2rem}
.era{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;display:grid;gap:.35rem;align-content:start}
a.era{text-decoration:none;color:inherit}a.era:hover{border-color:var(--accent)}h2.sect{font:650 20px/1.3 var(--display);margin:0 0 .6rem}
.era .range{font:12px var(--mono);color:var(--muted)}
.era h2{font:650 18px/1.25 var(--display);margin:0}
.era p{margin:0;font-size:13.5px;color:var(--muted)}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend .b::before{background:var(--accent)}.legend .f::before{background:var(--led)}
@media (max-width:820px){.wrap{grid-template-columns:minmax(0,1fr);gap:1rem}nav.side{position:static;max-height:none}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
</style>"""


def main():
    arts = OrderedDict()
    for f in sorted(SRC.glob("rows_*.html")):
        text = f.read_text()
        for m in re.finditer(r'<article class="row"[^>]*\bid="row-(\d+)"[^>]*>.*?</article>', text, re.S):
            arts[int(m.group(1))] = m.group(0)
    arts = OrderedDict(sorted(arts.items()))
    attr = lambda a, k: (re.search(rf'\b{k}="([^"]*)"', a.split(">", 1)[0]) or [None, ""])[1]  # noqa: E731
    era_of = {n: attr(a, "data-era").strip() or "Other" for n, a in arts.items()}
    title_of = {n: re.sub(r"<[^>]+>", "", (re.search(r"<h2>(.*?)</h2>", a, re.S) or [None, f"Row {n}"])[1]).strip() for n, a in arts.items()}
    status_of = {n: attr(a, "data-status") or "done" for n, a in arts.items()}
    from_of = {n: [int(x) for x in re.findall(r"\d+", attr(a, "data-from")) if int(x) in arts and int(x) != n] for n, a in arts.items()}
    led = defaultdict(list)
    for n, fs in from_of.items():
        for f in fs:
            led[f].append(n)
    # pages: contiguous chapters of the plan, named by hand (subagents' era labels interleave, so automatic grouping cascades)
    merged = []
    for lo, hi, name in CHAPTERS:
        rows = [n for n in arts if lo <= n <= hi]
        if rows:
            merged.append(dict(era=name, eras=[name], rows=rows))
    page_of = {}
    for k, p in enumerate(merged, 1):
        p["file"] = f"era-{k}.html"
        for n in p["rows"]:
            page_of[n] = p["file"]

    def relink(s, here):
        return re.sub(r'href="#row-(\d+)"', lambda m: f'href="{"" if page_of.get(int(m.group(1))) == here else page_of.get(int(m.group(1)), "index.html")}#row-{m.group(1)}"', s)

    def side(current):
        items = "".join(f'<li><a href="{p["file"]}"{" aria-current=\"page\"" if p["file"] == current else ""}>{html.escape(" · ".join(p["eras"]))} '
                        f'<span class="range">{p["rows"][0]}–{p["rows"][-1]}</span></a></li>' for p in merged)
        return (f'<nav class="side" aria-label="Eras"><h2>Plan history</h2><ol><li><a href="index.html"{" aria-current=\"page\"" if current == "index.html" else ""}>'
                f'Overview and all rows</a></li></ol><h2>Eras</h2><ol>{items}</ol></nav>')

    def page(title, current, body):
        inner = f"<title>{html.escape(title)}</title>{STYLE}<div class=\"wrap\">{side(current)}<main>{body}</main></div>"
        if current == "index.html":  # the published page: the Artifact host adds the document skeleton
            return inner
        head, rest = inner.split("<div class=\"wrap\">", 1)  # the other pages are served as they are: a full document each
        return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, '
                f'viewport-fit=cover">{head}</head><body><div class="wrap">{rest}</body></html>')

    DST.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(merged):
        parts = []
        for n in p["rows"]:
            a = arts[n]
            st = status_of[n]
            a = re.sub(r'(<p class="eyebrow">.*?)(</p>)', rf'\1<span class="status {st}">{html.escape(st)}</span>\2', a, count=1, flags=re.S)
            if led.get(n):
                links = "".join(f'<li><a href="#row-{m}">Row {m}</a> · {html.escape(title_of[m])}</li>' for m in sorted(led[n]))
                a = a.replace("</article>", f'<section class="led"><h3>Led to</h3><ul>{links}</ul></section></article>')
            parts.append(relink(a, p["file"]))
        prev_ = merged[k - 1] if k else None
        next_ = merged[k + 1] if k + 1 < len(merged) else None
        pager = ('<div class="pager">' + (f'<a href="{prev_["file"]}">← {html.escape(" · ".join(prev_["eras"]))}</a>' if prev_ else '<a href="index.html">← Overview</a>')
                 + (f'<a href="{next_["file"]}">{html.escape(" · ".join(next_["eras"]))} →</a>' if next_ else '<a href="index.html">Overview →</a>') + "</div>")
        head = (f'<header class="pagehead"><p class="kicker">Rows {p["rows"][0]}–{p["rows"][-1]} · {len(p["rows"])} rows</p>'
                f'<h1>{html.escape(" · ".join(p["eras"]))}</h1><p>Each row: why it ran (links back), what was tried, an example prompt, results, '
                f'lessons, open questions, and the rows it led to.</p></header>')
        (DST / p["file"]).write_text(page(f"Plan history: {' · '.join(p['eras'])}", p["file"], head + "".join(parts) + pager))
    # index
    cards = "".join(f'<a class="era" href="{p["file"]}"><span class="range">Rows {p["rows"][0]}–{p["rows"][-1]} · {len(p["rows"])}</span>'
                    f'<h2>{html.escape(" · ".join(p["eras"]))}</h2><p>{html.escape("; ".join(title_of[n] for n in p["rows"][:3]))}…</p></a>' for p in merged)
    rows = "".join(f'<tr><td><a href="{page_of[n]}#row-{n}">{n}</a></td><td>{html.escape(title_of[n])}</td><td><span class="status {status_of[n]}">{status_of[n]}</span></td>'
                   f'<td>{html.escape(era_of[n])}</td><td>{" ".join(f"<a href=\"{page_of[f]}#row-{f}\">{f}</a>" for f in from_of[n])}</td>'
                   f'<td>{" ".join(f"<a href=\"{page_of[m]}#row-{m}\">{m}</a>" for m in sorted(led.get(n, [])))}</td></tr>' for n in arts)
    intro = ('<header class="pagehead"><p class="kicker">Research plan · ' + f"{len(arts)} rows" + '</p><h1>Teaching small models to file transactions</h1>'
             '<p>Every row of the plan in order: what prompted it, what it tried, how it did, and where it led. Rows link back to the rows that caused '
             'them and forward to the rows they caused.</p></header>'
             '<p class="legend"><span class="b">back-links: why a row ran</span><span class="f">led to: rows it caused</span></p>'
             f'<div class="eras">{cards}</div><h2 class="sect">Every row</h2><div class="tablewrap"><table><thead><tr><th>Row</th><th>Title</th>'
             f'<th>Status</th><th>Era</th><th>Because of</th><th>Led to</th></tr></thead><tbody>{rows}</tbody></table></div>')
    (DST / "index.html").write_text(page("Categoriser Plan History", "index.html", intro))
    print(f"{len(arts)} rows -> {len(merged)} era pages + index in {DST}; links: {sum(len(v) for v in from_of.values())}; missing rows: "
          f"{[n for n in range(1, max(arts) + 1) if n not in arts]}")


if __name__ == "__main__":
    main()
