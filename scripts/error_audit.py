"""Row 242 (owner, 2026-10-07: "a nice little private artifact where I can make notes and tell you whether it was a reasonable miss or
not"; "keep the auditor to a local file/server ... a harness that could be used to examine other budgets as well"): an error audit for
any budget whose items and reader scores sit in a real_budget_eval folder. It samples a reader's misses by stratum, serves them on a
page bound to localhost only, keeps the labeller's verdicts and notes beside the budget's private files (0600), and summarises them as
counts only. Nothing here prints a payee, category or amount; the page shows them only to the person at this machine.

  build    sample misses into <OUT>/audit/<AUDIT>/audit_set.json
  serve    the labelling page on http://127.0.0.1:<PORT>/ (open it in a browser on this machine; WSL forwards localhost to Windows)
  summary  verdict counts per stratum, and what they make of the reader's accuracy (aggregates only: safe for REPORT.md)
  demo     a synthetic budget folder (rational households read by decider v5, row 234) under data/interim/audit_demo, to try the harness

env: BUDGET (default: ~/.config/ynab/budget_id) or OUT (a budget folder; default ~/.local/share/ynab-real-eval/<BUDGET>),
  ITEMS (items_grp_sim2.json), READERS (name=score file, comma list; the first is the one audited; default decider v5 seeds 0 and 1),
  AUDIT (misses_v1), QUOTA (first_time=150,known_changed=100,known_other=50), SEED (0), PORT (8765)
usage: uv run python scripts/error_audit.py build|serve|summary
       OUT=data/interim/audit_demo READERS=decider=scores.jsonl uv run python scripts/error_audit.py build   (after `demo`)
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments.paths import ROOT  # noqa: E402

VERDICTS = [  # (key, label, hint): the reasons a miss happened; "fine" and "noise" mean the reader was not really wrong
    ("knowable", "Knowable", "name or history should have told it"),
    ("item", "Needs item data", "depends on what was bought (receipt, order)"),
    ("purpose", "Purpose or person", "depends on why or for whom; not in the data"),
    ("fine", "Acceptable", "the reader's answer is also right"),
    ("noise", "Label noise", "the filed category is wrong or arbitrary"),
    ("other", "Other", "say why in the note"),
]
STRATA = {"first_time": "first-time payee", "known_changed": "known payee, rule wrong", "known_other": "known payee, rule right"}


def budget_dir():
    if os.environ.get("OUT"):
        return Path(os.environ["OUT"])
    b = os.environ.get("BUDGET") or (Path.home() / ".config" / "ynab" / "budget_id").read_text().strip()
    return Path.home() / ".local" / "share" / "ynab-real-eval" / b


def audit_dir(out):
    d = out / "audit" / os.environ.get("AUDIT", "misses_v1")
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def private_write(p, text):
    p.write_text(text)
    os.chmod(p, 0o600)


def readers():
    spec = os.environ.get("READERS", "decider-v5-s0=scores_r231-dv5-s0_split_grp_sim2.jsonl,decider-v5-s1=scores_r231-dv5-s1_split_grp_sim2.jsonl")
    return [tuple(x.split("=", 1)) for x in spec.split(",")]


def softmax(v):
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def query_line(it):
    """the transaction being filed: the prompt's last 'Transaction:' line ("date | payee | $amount | weekday")"""
    m = re.findall(r"Transaction: (.*)", it.get("prompt") or it.get("prompt_split") or "")
    return m[-1] if m else ""


def history_text(it):
    """what the reader saw before the question: the split prompt without the category list (the options show that) and the question"""
    p = it.get("prompt_split") or it.get("prompt") or ""
    p = re.sub(r"\ACategories: .*?\n\n", "", p, flags=re.S)
    return p.rsplit("Transaction:", 1)[0].rstrip()


def stratum(it):
    if not it.get("payee_seen"):
        return "first_time"
    return "known_changed" if it.get("rule") != it["gold"] else "known_other"


def build():
    out = budget_dir()
    items = {it["id"]: it for it in json.loads((out / os.environ.get("ITEMS", "items_grp_sim2.json")).read_text())["items"] if it["answer"] >= 0}
    rs = readers()
    scores = {}
    for name, f in rs:
        scores[name] = {r["id"]: r["lp"] for r in map(json.loads, open(out / f))}
    ids = [i for i in items if all(i in scores[n] for n, _ in rs)]
    main = rs[0][0]
    misses = defaultdict(list)
    for i in ids:
        if int(np.argmax(scores[main][i])) != items[i]["answer"]:
            misses[stratum(items[i])].append(i)
    quota = dict((k, int(v)) for k, v in (x.split("=") for x in os.environ.get("QUOTA", "first_time=150,known_changed=100,known_other=50").split(",")))
    rng = random.Random(int(os.environ.get("SEED", "0")))
    chosen = []
    for s in STRATA:
        pool = sorted(misses[s])
        chosen += [(s, i) for i in rng.sample(pool, min(quota.get(s, 0), len(pool)))]
    rng.shuffle(chosen)  # strata interleaved, so stopping early still leaves a mixed sample
    recs = []
    for s, i in chosen:
        it = items[i]
        opts = it["options"]
        recs.append(dict(id=i, stratum=s, date=it["date"], line=query_line(it), gold=it["gold"], rule=it.get("rule"),
                         first_use=it.get("first_use"), n_hist=it.get("n_hist"), history=history_text(it),
                         readers={n: [[opts[j], round(float(p), 3)] for j, p in sorted(enumerate(softmax(scores[n][i])), key=lambda x: -x[1])[:5]]
                                  for n, _ in rs}))
    base = {s: dict(scored=sum(stratum(items[i]) == s for i in ids), misses=len(misses[s]), sampled=sum(c[0] == s for c in chosen)) for s in STRATA}
    d = audit_dir(out)
    private_write(d / "audit_set.json", json.dumps(dict(readers=[n for n, _ in rs], audited=main, strata=base, items=recs)))
    print(f"{len(ids)} items scored by {len(rs)} readers; {main} misses {sum(len(v) for v in misses.values())}; sampled {len(recs)}")
    for s, b in base.items():
        print(f"  {STRATA[s]}: {b['scored']} items, {b['misses']} misses, {b['sampled']} sampled")
    print(f"written to the budget folder's audit/{d.name}/ (0600); next: serve")


def load_labels(d):
    lab = {}
    f = d / "labels.jsonl"
    if f.exists():
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                lab[r["id"]] = r  # the latest line for an id wins
    return lab


def serve():
    d = audit_dir(budget_dir())
    aset = json.loads((d / "audit_set.json").read_text())
    page = PAGE.replace("/*VERDICTS*/", json.dumps(VERDICTS)).replace("/*STRATA*/", json.dumps(STRATA))

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            b = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, page, "text/html")
            if self.path == "/api/set":
                return self._send(200, json.dumps(dict(aset, labels=load_labels(d))))
            self._send(404, "{}")

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, "{}")
            r = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            rec = {k: r.get(k) for k in ("id", "verdict", "note", "better")}
            with open(d / "labels.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            os.chmod(d / "labels.jsonl", 0o600)
            self._send(200, json.dumps(dict(ok=True)))

        def log_message(self, *a):  # no request log: paths only, but keep the console quiet
            pass

    port = int(os.environ.get("PORT", "8765"))
    print(f"audit {d.name}: {len(aset['items'])} misses; open http://localhost:{port}/ (Ctrl-C stops; verdicts are saved as you go)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def summary():
    d = audit_dir(budget_dir())
    aset = json.loads((d / "audit_set.json").read_text())
    lab = load_labels(d)
    by = defaultdict(Counter)
    for r in aset["items"]:
        v = (lab.get(r["id"]) or {}).get("verdict")
        if v:
            by[r["stratum"]][v] += 1
    keys = [k for k, _, _ in VERDICTS]
    print(f"audit {d.name} of {aset['audited']}: {sum(map(sum, (c.values() for c in by.values())))} of {len(aset['items'])} misses labelled")
    print("| stratum | items | misses | labelled | " + " | ".join(lbl for _, lbl, _ in VERDICTS) + " | reader right first | if acceptable + noise count as right |")
    print("|---" * (len(keys) + 6) + "|")
    tot = Counter()
    for s, name in STRATA.items():
        b, c = aset["strata"][s], by[s]
        n = sum(c.values())
        tot.update(dict(scored=b["scored"], misses=b["misses"], labelled=n)); tot.update(c)
        acc = 100 * (1 - b["misses"] / b["scored"]) if b["scored"] else float("nan")
        lift = 100 * b["misses"] / b["scored"] * (c["fine"] + c["noise"]) / n if n and b["scored"] else float("nan")
        f1 = lambda x: "–" if x != x else f"{x:.1f}"  # noqa: E731  (nan: an empty stratum)
        print(f"| {name} | {b['scored']} | {b['misses']} | {n} | " + " | ".join(f"{100 * c[k] / n:.0f}%" if n else "–" for k in keys)
              + f" | {f1(acc)} | {f1(acc + lift) if n else '–'} |")
    n = tot["labelled"]
    if n:
        acc = 100 * (1 - tot["misses"] / tot["scored"])
        lift = sum(100 * aset["strata"][s]["misses"] / tot["scored"] * (by[s]["fine"] + by[s]["noise"]) / max(1, sum(by[s].values())) for s in STRATA)
        print(f"| all | {tot['scored']} | {tot['misses']} | {n} | " + " | ".join(f"{100 * tot[k] / n:.0f}%" for k in keys) + f" | {acc:.1f} | {acc + lift:.1f} |")
    print("Each stratum's verdict shares are weighted by its own misses in the last column (strata are sampled by quota, not in proportion).")


def demo():
    """a synthetic budget folder with real reads: the rational households (row 234) and decider v5's scores of them"""
    os.environ.setdefault("RATIONAL_PAYEES", ""); os.environ.setdefault("RATIONAL_CATS", "")
    import rational_decider as RD  # sets BUDGET=rational and the reader's switches before real_budget_eval loads
    import rational_budgets as RG
    sf = RD.OUT / "scores__.jsonl"
    assert sf.exists(), f"{sf} missing: run scripts/chains/r234_rational.sh first"
    want = {r["id"] for r in map(json.loads, open(sf))}
    items = []
    for b in RG.budgets("bank"):
        RD.RB.OUT = RD.OUT / b["id"]
        items += [it for it in RD.RB.build_items(RD._with_account(b)) if it["id"] in want]
    out = ROOT / "data" / "interim" / "audit_demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(dict(budget="audit_demo", n=len(items), items=items)))
    (out / "scores.jsonl").write_text(sf.read_text())
    print(f"{len(items)} of {len(want)} scored items rebuilt into {out}; next: OUT={out.relative_to(ROOT)} ITEMS=items.json READERS=decider=scores.jsonl ... build")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Miss Audit</title>
<style>
/* a desk for one reader: the queue on the left, one miss at a time on the right; system faces only (nothing loads from the network) */
:root{--bg:#F3F5F2;--panel:#FCFDFB;--ink:#18211D;--muted:#5C6762;--rule:#D3DAD5;--accent:#1D6A80;--accent-soft:#E0EDF1;
 --gold:#8A5A12;--gold-soft:#F5ECDD;--ok:#2F6B45;--code:#ECF0EC;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#111714;--panel:#171F1B;--ink:#E2E9E5;--muted:#9AA5A0;--rule:#2A3430;
 --accent:#6DB3C9;--accent-soft:#1A2E34;--gold:#E0B064;--gold-soft:#302618;--ok:#7CC293;--code:#1C2521}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.55 var(--sans)}
.app{display:grid;grid-template-columns:minmax(0,17rem) minmax(0,1fr);height:100%}
aside{border-right:1px solid var(--rule);display:grid;grid-template-rows:auto auto 1fr;min-height:0}
.head{padding:16px 16px 10px;display:grid;gap:6px}
h1{font-size:17px;margin:0}
.bar{height:6px;background:var(--rule);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
.count{font:12px var(--mono);color:var(--muted)}
.filters{display:flex;flex-wrap:wrap;gap:6px;padding:0 16px 10px}
.filters button{font:12px var(--sans);border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:999px;padding:3px 9px;cursor:pointer}
.filters button[aria-pressed=true]{border-color:var(--accent);background:var(--accent-soft)}
ol.queue{list-style:none;margin:0;padding:0 8px 16px;overflow:auto}
ol.queue li button{width:100%;text-align:left;border:0;background:none;color:var(--ink);padding:6px 8px;border-radius:6px;cursor:pointer;display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;font:13px var(--sans)}
ol.queue li button span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
ol.queue li button[aria-current=true]{background:var(--accent-soft)}
.dot{width:9px;height:9px;border-radius:50%;border:1.5px solid var(--muted);margin-top:4px}.dot.done{background:var(--ok);border-color:var(--ok)}
main{overflow:auto;padding:22px clamp(16px,4vw,40px) 40px;min-width:0}
.card{max-width:52rem;display:grid;gap:16px}
.eyebrow{font:12px var(--mono);color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}
.pill{border:1px solid currentColor;border-radius:999px;padding:1px 8px;font:600 11px var(--sans);letter-spacing:.04em}
.line{font:600 19px/1.35 var(--mono);word-break:break-word}
.gold{background:var(--gold-soft);border-radius:8px;padding:10px 14px}.gold b{color:var(--gold)}
.readers{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:12px}
.reader{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 14px}
.reader h3{font:600 12px var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.reader table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
.reader td{padding:3px 0;border-bottom:1px solid var(--rule)}.reader td.p{text-align:right;color:var(--muted);width:4rem}
.reader tr.is-gold td:first-child{color:var(--gold);font-weight:600}
details{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:8px 14px}
summary{cursor:pointer;color:var(--muted);font-size:13px}
pre{font:12.5px/1.5 var(--mono);background:var(--code);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre-wrap;max-height:28rem;overflow-y:auto}
fieldset{border:0;padding:0;margin:0;display:grid;gap:8px}
legend{font-weight:600;margin-bottom:6px}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:8px}
.verdicts label{border:1px solid var(--rule);background:var(--panel);border-radius:8px;padding:8px 10px;cursor:pointer;display:grid;gap:2px}
.verdicts label:has(input:checked){border-color:var(--accent);background:var(--accent-soft)}
.verdicts input{position:absolute;opacity:0}.verdicts label:focus-within{outline:2px solid var(--accent)}
.verdicts b{font-size:14px}.verdicts small{color:var(--muted)}.verdicts kbd{font:11px var(--mono);color:var(--muted);float:right}
textarea,input[type=text]{width:100%;font:14px var(--sans);background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:8px;padding:8px 10px}
textarea{min-height:5rem;resize:vertical}
.nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.nav button{font:14px var(--sans);border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:8px;padding:7px 14px;cursor:pointer}
.nav button.primary{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.saved{font-size:12.5px;color:var(--ok)}
@media (max-width:760px){.app{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid var(--rule)}ol.queue{max-height:14rem}}
</style></head><body>
<div class="app">
 <aside>
  <div class="head"><h1>Miss audit</h1><div class="bar"><i id="prog"></i></div><div class="count" id="count"></div></div>
  <div class="filters" id="filters"></div>
  <ol class="queue" id="queue"></ol>
 </aside>
 <main><div class="card" id="card"></div></main>
</div>
<script>
const VERDICTS=/*VERDICTS*/, STRATA=/*STRATA*/;
let S, labels={}, cur=0, filt="all";
const $=id=>document.getElementById(id), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const view=()=>S.items.map((r,i)=>i).filter(i=>filt==="all"||(filt==="todo"?!labels[S.items[i].id]?.verdict:S.items[i].stratum===filt));
async function save(r){labels[r.id]=r;await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(r)});
 const m=$("saved");if(m)m.textContent="Saved";side()}
function side(){const n=S.items.filter(r=>labels[r.id]?.verdict).length;$("prog").style.width=(100*n/S.items.length)+"%";
 $("count").textContent=`${n} of ${S.items.length} labelled · auditing ${S.audited}`;
 $("filters").innerHTML=[["all","All"],["todo","Not yet labelled"],...Object.entries(STRATA)].map(([k,v])=>`<button aria-pressed="${filt===k}" data-f="${k}">${esc(v)}</button>`).join("");
 $("queue").innerHTML=view().map(i=>{const r=S.items[i];return `<li><button data-i="${i}" aria-current="${i===cur}"><span class="dot ${labels[r.id]?.verdict?"done":""}"></span><span>${esc(r.line.split(" | ")[1]||r.line)}</span></button></li>`}).join("")}
function show(i){cur=i;const r=S.items[i],L=labels[r.id]||{};
 const rd=Object.entries(r.readers).map(([n,top])=>`<div class="reader"><h3>${esc(n)}</h3><table>${top.map(([c,p])=>`<tr class="${c===r.gold?"is-gold":""}"><td>${esc(c)}</td><td class="p">${(100*p).toFixed(1)}%</td></tr>`).join("")}</table></div>`).join("");
 $("card").innerHTML=`<div class="eyebrow"><span class="pill">${esc(STRATA[r.stratum])}</span>${r.first_use?'<span class="pill">first use of this category</span>':""}<span>${r.n_hist} earlier transactions</span><span>${i+1} / ${S.items.length}</span></div>
  <div class="line">${esc(r.line)}</div>
  <div class="gold">Filed as <b>${esc(r.gold)}</b>${r.rule?` · YNAB's rule said ${esc(r.rule)}`:" · no rule (payee not seen)"}</div>
  <div class="readers">${rd}</div>
  <details><summary>What the reader saw before this transaction</summary><pre>${esc(r.history)}</pre></details>
  <fieldset><legend>Why did the reader miss?</legend><div class="verdicts">${VERDICTS.map(([k,l,h],j)=>`<label><input type="radio" name="v" value="${k}" ${L.verdict===k?"checked":""}><b>${esc(l)} <kbd>${j+1}</kbd></b><small>${esc(h)}</small></label>`).join("")}</div></fieldset>
  <label>Note<textarea id="note" placeholder="Anything that would have told it, or why it couldn't know">${esc(L.note)}</textarea></label>
  <label>Better answer, if neither the filed one nor the reader's<input type="text" id="better" value="${esc(L.better)}"></label>
  <div class="nav"><button id="prev">← Previous</button><button class="primary" id="next">Next →</button><span class="saved" id="saved"></span></div>`;
 const rec=()=>({id:r.id,verdict:document.querySelector("input[name=v]:checked")?.value||null,note:$("note").value,better:$("better").value});
 document.querySelectorAll("input[name=v]").forEach(x=>x.onchange=()=>save(rec()));
 let t;["note","better"].forEach(id=>$(id).oninput=()=>{$("saved").textContent="";clearTimeout(t);t=setTimeout(()=>save(rec()),600)});
 $("prev").onclick=()=>step(-1);$("next").onclick=()=>step(1);side();$("card").parentElement.scrollTop=0}
function step(d){const v=view(),k=v.indexOf(cur);const n=v[Math.min(v.length-1,Math.max(0,(k<0?0:k+d)))];if(n!==undefined)show(n)}
document.addEventListener("click",e=>{const b=e.target.closest("[data-i]");if(b)show(+b.dataset.i);const f=e.target.closest("[data-f]");if(f){filt=f.dataset.f;side()}});
document.addEventListener("keydown",e=>{if(e.target.matches("textarea,input[type=text]"))return;
 const n=+e.key;if(n>=1&&n<=VERDICTS.length){const x=document.querySelectorAll("input[name=v]")[n-1];x.checked=true;x.dispatchEvent(new Event("change"))}
 if(e.key==="j"||e.key==="ArrowRight")step(1);if(e.key==="k"||e.key==="ArrowLeft")step(-1)});
fetch("/api/set").then(r=>r.json()).then(d=>{S=d;labels=d.labels||{};const first=S.items.findIndex(r=>!labels[r.id]?.verdict);show(first<0?0:first)});
</script></body></html>"""

if __name__ == "__main__":
    dict(build=build, serve=serve, summary=summary, demo=demo)[sys.argv[1]]()
