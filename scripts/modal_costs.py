"""Modal cost per launch, job and PLAN row (owner, 2026-10-05: "record how much each job cost, and which jobs were which, from now on ...
so we have it all in our report").

Launches: reports/modal_launches.jsonl, written by scripts/modal_app.py at each launch (app id, kind, job list or reader, PLAN rows,
job tags) and as each job reports back (minutes). `backfill` adds launches from before the ledger by reading old launch logs (the app
URL, the job-list name, the "<tag>: N files ..., M min" lines). Costs: `modal billing report` per app (daily, in 31-day windows from
START), so a launch's cost is the bill of its app id, exactly; inside a launch, cost is split across jobs by their minutes (equally when
minutes are missing). Apps billed but in no launch record are listed as unattributed (before the ledger these were mostly the owner's-
budget scoring runs, which never logged their app ids).
Writes reports/modal_costs.md: totals, per PLAN row, per launch with its jobs, unattributed apps; `--rows 192-193` prints one row's
lines for a REPORT section.
usage: uv run python scripts/modal_costs.py backfill <log files...>   (once)
       uv run python scripts/modal_costs.py [--rows N[-M]]
env: START (2026-09-25)
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from ai_experiments.paths import ROOT

LEDGER, OUT = ROOT / "reports" / "modal_launches.jsonl", ROOT / "reports" / "modal_costs.md"
APP = "ai-experiments-training"
START = os.environ.get("START", "2026-09-25")


def load():
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()] if LEDGER.exists() else []


def rows_from(name):
    m = re.search(r"(?:^|[_-])r(\d+)(?:_(\d+)(?![a-z]))?", name)
    return (m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")) if m else ""


def backfill(paths):
    known = {r["app_id"] for r in load()}
    added = 0
    with open(LEDGER, "a") as f:
        for p in map(Path, paths):
            text = p.read_text(errors="replace")
            ids = re.findall(r"modal\.com/apps/[^/\s]+/[^/\s]+/(ap-[A-Za-z0-9]+)", text)
            if not ids or ids[0] in known:
                continue
            app = ids[0]
            jl = re.search(r"modal_jobs/([A-Za-z0-9_.-]+)\.json", text)
            what = jl.group(1) if jl else re.sub(r"[._]?(launch|out|log)$", "", p.stem.replace(".launch", ""))
            when = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds")
            done = re.findall(r"^(\S+): \d+ files to .*?, ([\d.]+) min", text, re.M)
            f.write(json.dumps(dict(utc=when, app_id=app, kind="jobs", what=what, rows=rows_from(what), gpu="", tags=[t for t, _ in done],
                                    source=f"backfill:{p.name}")) + "\n")
            for t, m in done:
                f.write(json.dumps(dict(utc=when, app_id=app, kind="job_done", tag=t, minutes=float(m), source="backfill")) + "\n")
            known.add(app)
            added += 1
    print(f"backfilled {added} launches")


def bills():
    """{app id: (cost, first day, last day)} for this project's app over START..now, from daily billing reports"""
    out = defaultdict(lambda: [0.0, None, None])
    a = dt.date.fromisoformat(START)
    today = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    while a < today:
        b = min(a + dt.timedelta(days=30), today)
        res = subprocess.run(["modal", "billing", "report", "--start", a.isoformat(), "--end", b.isoformat(), "--json"],
                             capture_output=True, text=True)
        for r in json.loads(res.stdout or "[]"):
            if r.get("description") != APP:
                continue
            o = out[r["object_id"]]
            o[0] += float(r["cost"])
            d = r["interval_start"][:10]
            o[1] = min(o[1] or d, d)
            o[2] = max(o[2] or d, d)
        a = b
    return out


def main():
    want = None
    if "--rows" in sys.argv:
        want = sys.argv[sys.argv.index("--rows") + 1]
    led = load()
    launches = [r for r in led if r["kind"] != "job_done"]
    minutes = defaultdict(dict)
    for r in led:
        if r["kind"] == "job_done":
            minutes[r["app_id"]][r["tag"]] = r["minutes"]
    bill = bills()
    total = sum(v[0] for v in bill.values())
    by_app = defaultdict(list)
    for r in launches:
        by_app[r["app_id"]].append(r)
    lines, per_row = [], defaultdict(float)
    for app, recs in sorted(by_app.items(), key=lambda kv: kv[1][0]["utc"]):
        cost = bill.get(app, [0.0])[0]
        rec = recs[0]
        tags = rec.get("tags") or list(minutes[app]) or [rec.get("what", "")]
        mins = {t: minutes[app].get(t) for t in tags}
        tot_min = sum(v for v in mins.values() if v)
        share = {t: (cost * (m / tot_min) if tot_min and m else cost / len(tags)) for t, m in mins.items()}
        rows = rec.get("rows", "") or rows_from(rec.get("what", "")) or "other (smoke tests, caches, probes)"
        per_row[rows] += cost
        lines.append((rec["utc"][:16].replace("T", " "), rows, rec.get("kind", ""), rec.get("what", ""), app, cost, tot_min, share, mins))
    attributed = sum(l[5] for l in lines)
    unattr = {a: v for a, v in bill.items() if a not in by_app}
    if want:
        for when, rows, kind, what, app, cost, tm, share, mins in lines:
            if rows == want:
                print(f"- {when} UTC, {what} ({kind}, {app}): ${cost:.2f}" + (f", {tm:.0f} job-minutes" if tm else ""))
                for t, c in share.items():
                    print(f"  - {t}: ${c:.2f}" + (f" ({mins[t]:.0f} min)" if mins.get(t) else ""))
        print(f"rows {want}: ${per_row.get(want, 0):.2f}")
        return
    md = [f"# Modal costs (app `{APP}`)", "",
          f"Generated by `scripts/modal_costs.py` on {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC from Modal's billing report "
          f"(complete days only) and `reports/modal_launches.jsonl`. A launch's cost is its app's bill; inside a launch it is split by job "
          f"minutes. Billed since {START}: **${total:.2f}**; attributed to a launch: ${attributed:.2f}; unattributed: "
          f"${sum(v[0] for v in unattr.values()):.2f} ({len(unattr)} apps, mostly owner's-budget scoring before 2026-10-05).", "",
          "## By PLAN row", "", "| rows | cost |", "|---|---|"]
    def rowkey(k):
        m = re.match(r"(\d+)", k)
        return int(m.group(1)) if m else 10 ** 6
    md += [f"| {k} | ${v:.2f} |" for k, v in sorted(per_row.items(), key=lambda kv: rowkey(kv[0]))]
    md += ["", "## By launch", "", "| launched (UTC) | rows | kind | job list / reader | app | cost | jobs (cost, minutes) |", "|---|---|---|---|---|---|---|"]
    for when, rows, kind, what, app, cost, tm, share, mins in lines:
        jobs = "; ".join(f"{t} ${c:.2f}" + (f" {mins[t]:.0f}m" if mins.get(t) else "") for t, c in share.items())
        md.append(f"| {when} | {rows} | {kind} | {what} | {app} | ${cost:.2f} | {jobs} |")
    md += ["", "## Unattributed apps (billed, no launch record)", "", "| app | first day | last day | cost |", "|---|---|---|---|"]
    md += [f"| {a} | {v[1]} | {v[2]} | ${v[0]:.2f} |" for a, v in sorted(unattr.items(), key=lambda kv: kv[1][1] or "")]
    OUT.write_text("\n".join(md) + "\n")
    print(f"billed ${total:.2f}; attributed ${attributed:.2f} over {len(lines)} launches; unattributed ${total - attributed:.2f} "
          f"({len(unattr)} apps) -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill(sys.argv[2:])
    else:
        main()
