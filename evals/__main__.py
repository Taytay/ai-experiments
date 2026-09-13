"""CLI for the experiment tracker.

  uv run python -m evals list [--exp NAME] [-n 20]
  uv run python -m evals show RUN_ID
  uv run python -m evals compare EXPERIMENT [--metric L3_induct_type_nonsense] [--condition lora]
  uv run python -m evals leaderboard            # writes evals/LEADERBOARD.md
  uv run python -m evals import-json PATH --exp NAME --model M [--commit SHA] [--note ...]
  uv run python -m evals export                 # rewrite evals/runs.jsonl
  uv run python -m evals rebuild                # recreate runs.db from runs.jsonl (after clone)
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from .tracker import DB_PATH, JSONL_PATH, ROOT, Run, connect, export_jsonl, import_jsonl


def _rows(con, sql, args=()):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def _fmt_time(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else ""


def _table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows]
    return "\n".join(out)


def cmd_list(con, a):
    sql = "SELECT run_id, experiment, model, status, git_commit, git_dirty, started_at, finished_at, note FROM runs"
    args = []
    if a.exp:
        sql += " WHERE experiment=?"; args.append(a.exp)
    sql += " ORDER BY started_at DESC LIMIT ?"; args.append(a.n)
    rows = _rows(con, sql, args)
    print(_table(["run_id", "experiment", "model", "status", "commit", "started", "min", "note"],
                 [[r["run_id"], r["experiment"], (r["model"] or "").split("/")[-1], r["status"],
                   (r["git_commit"] or "?")[:8] + ("*" if r["git_dirty"] else ""), _fmt_time(r["started_at"]),
                   round((r["finished_at"] - r["started_at"]) / 60, 1) if r["finished_at"] else "",
                   (r["note"] or "")[:40]] for r in rows]))


def cmd_show(con, a):
    r = _rows(con, "SELECT * FROM runs WHERE run_id LIKE ?", (a.run_id + "%",))
    if not r:
        sys.exit(f"no run matching {a.run_id}")
    r = r[0]
    for k in ("run_id", "experiment", "model", "status", "git_commit", "git_dirty", "git_branch", "script", "script_sha", "note", "error"):
        print(f"{k:12s} {r[k]}")
    print(f"{'started':12s} {_fmt_time(r['started_at'])}   finished {_fmt_time(r['finished_at'])}")
    print("config      ", json.dumps(json.loads(r["config"]), indent=None))
    env = json.loads(r["env"]); print("env         ", {k: env[k] for k in env if k != "argv"})
    mets = _rows(con, "SELECT condition, name, value, step FROM metrics WHERE run_id=? ORDER BY condition, name, step", (r["run_id"],))
    conds = sorted({m["condition"] for m in mets}); names = sorted({m["name"] for m in mets})
    grid = {(m["condition"], m["name"]): m["value"] for m in mets}
    print("\n" + _table(["metric", *conds], [[n, *[grid.get((c, n), "") for c in conds]] for n in names]))
    arts = _rows(con, "SELECT path, sha256 FROM artifacts WHERE run_id=?", (r["run_id"],))
    if arts:
        print("\nartifacts:", *[f"  {x['path']}  {(x['sha256'] or '')[:12]}" for x in arts], sep="\n")


def cmd_compare(con, a):
    """One row per (run, condition); columns = metrics. Filter by experiment, optional condition/metric."""
    sql = ("SELECT r.run_id, r.model, r.git_commit, r.git_dirty, r.config, m.condition, m.name, m.value "
           "FROM metrics m JOIN runs r ON r.run_id=m.run_id WHERE r.experiment=? AND r.status='finished'")
    args = [a.exp]
    if a.condition:
        sql += " AND m.condition=?"; args.append(a.condition)
    if a.metric:
        sql += " AND m.name LIKE ?"; args.append(a.metric + "%")
    rows = _rows(con, sql, args)
    if not rows:
        sys.exit("no finished runs / metrics match")
    keys = sorted({(r["run_id"], r["condition"]) for r in rows})
    names = sorted({r["name"] for r in rows})
    grid = {(r["run_id"], r["condition"], r["name"]): r["value"] for r in rows}
    meta = {r["run_id"]: r for r in rows}
    out = []
    for rid, cond in keys:
        m = meta[rid]; cfg = json.loads(m["config"])
        short = ",".join(f"{k}={v}" for k, v in cfg.items() if k in a.show_cfg) if a.show_cfg else ""
        out.append([rid[:15], (m["model"] or "").split("/")[-1], (m["git_commit"] or "?")[:7] + ("*" if m["git_dirty"] else ""),
                    cond, short, *[grid.get((rid, cond, n), "") for n in names]])
    print(_table(["run", "model", "commit", "condition", "config", *names], out))


def cmd_leaderboard(con, a):
    """Best finished run per (experiment, model, condition) for a headline metric, written to LEADERBOARD.md."""
    exps = [r["experiment"] for r in _rows(con, "SELECT DISTINCT experiment FROM runs WHERE status='finished' ORDER BY 1")]
    lines = [f"# Leaderboard\n\nGenerated {_fmt_time(time.time())} from `evals/runs.db`. One row per finished run+condition; all metrics.\n"]
    for exp in exps:
        rows = _rows(con, "SELECT r.run_id, r.model, r.git_commit, r.git_dirty, r.started_at, r.config, m.condition, m.name, m.value "
                          "FROM metrics m JOIN runs r ON r.run_id=m.run_id WHERE r.experiment=? AND r.status='finished'", (exp,))
        names = sorted({r["name"] for r in rows})
        keys = sorted({(r["started_at"], r["run_id"], r["condition"]) for r in rows})
        grid = {(r["run_id"], r["condition"], r["name"]): r["value"] for r in rows}
        meta = {r["run_id"]: r for r in rows}
        lines.append(f"\n## {exp}\n")
        lines.append(_table(["run", "model", "commit", "condition", *names],
                            [[rid[:15], (meta[rid]["model"] or "").split("/")[-1],
                              (meta[rid]["git_commit"] or "?")[:7] + ("*" if meta[rid]["git_dirty"] else ""), cond,
                              *[grid.get((rid, cond, n), "") for n in names]] for _, rid, cond in keys]))
        cfgs = {rid: json.loads(meta[rid]["config"]) for _, rid, _ in keys}
        lines.append("\nConfigs:\n" + "\n".join(f"- `{rid[:15]}`: `{json.dumps(c, sort_keys=True)}`" for rid, c in cfgs.items()))
    out = ROOT / "evals" / "LEADERBOARD.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(exps)} experiments)")


def cmd_import_json(con, a):
    """Backfill a results JSON of shape {condition: {metric: value}} as one finished run."""
    data = json.loads(Path(a.path).read_text())
    with Run(a.exp, model=a.model, config=json.loads(a.config) if a.config else {}, note=a.note or f"backfilled from {a.path}",
             script=a.script) as run:
        if a.commit:
            run.con.execute("UPDATE runs SET git_commit=?, git_dirty=0 WHERE run_id=?", (a.commit, run.run_id))
        for cond, mets in data.items():
            if isinstance(mets, dict):
                run.log(mets, condition=cond)
        run.artifact(a.path)
    print("imported as", run.run_id)


def main():
    p = argparse.ArgumentParser(prog="evals")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--exp"); s.add_argument("-n", type=int, default=20)
    s = sub.add_parser("show"); s.add_argument("run_id")
    s = sub.add_parser("compare"); s.add_argument("exp"); s.add_argument("--metric"); s.add_argument("--condition")
    s.add_argument("--show-cfg", nargs="*", default=["lr", "steps", "r", "epochs"])
    sub.add_parser("leaderboard")
    sub.add_parser("export")
    sub.add_parser("rebuild", help="rebuild runs.db from runs.jsonl")
    s = sub.add_parser("import-json"); s.add_argument("path"); s.add_argument("--exp", required=True); s.add_argument("--model")
    s.add_argument("--commit"); s.add_argument("--note"); s.add_argument("--config"); s.add_argument("--script")
    a = p.parse_args()
    con = connect(DB_PATH)
    {"list": cmd_list, "show": cmd_show, "compare": cmd_compare, "leaderboard": cmd_leaderboard,
     "import-json": cmd_import_json, "export": lambda c, a: (export_jsonl(c), print("exported")),
     "rebuild": lambda c, a: print(f"rebuilt {import_jsonl(c)} runs from {JSONL_PATH.name}")}[a.cmd](con, a)


if __name__ == "__main__":
    main()
