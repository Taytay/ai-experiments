"""Minimal experiment tracker with provenance.

One SQLite file (`evals/runs.db`) holds every run: who/what/when, the exact config,
the git commit (and whether the tree was dirty), environment, and all metrics.
A diff-friendly JSONL mirror (`evals/runs.jsonl`) is rewritten on every finish so
the history is also versioned in git and reviewable in a PR.

Storage is the stdlib `sqlite3` module. DoltLite (dolthub/doltlite) exposes the
same sqlite3_* API, so the DB can be swapped for a version-controlled one by
changing `connect()` once its Python binding loads on this machine.

Usage in an experiment:

    from evals.tracker import Run
    with Run("universe_ladder", model="Qwen/Qwen2.5-3B", config=dict(steps=600, lr=2e-4)) as run:
        ...
        run.log(dict(L1_recall=71.2, L3_induct=40.0), condition="lora")
        run.artifact("results/universe_Qwen2.5-3B.json")

CLI: `uv run python -m evals --help`
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "evals" / "runs.db"
JSONL_PATH = ROOT / "evals" / "runs.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    experiment  TEXT NOT NULL,
    model       TEXT,
    started_at  REAL NOT NULL,
    finished_at REAL,
    status      TEXT NOT NULL,            -- running | finished | failed
    git_commit  TEXT,
    git_dirty   INTEGER,
    git_branch  TEXT,
    script      TEXT,
    script_sha  TEXT,                     -- sha256 of the experiment file at run time
    config      TEXT NOT NULL,            -- JSON
    config_sha  TEXT NOT NULL,            -- sha256 of canonical config JSON (for grouping)
    env         TEXT NOT NULL,            -- JSON: python, torch, cuda, gpu, host, user
    note        TEXT,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    condition  TEXT NOT NULL DEFAULT '',  -- e.g. base / lora / lora_ctx
    name       TEXT NOT NULL,
    value      REAL,
    step       INTEGER,
    logged_at  REAL NOT NULL,
    PRIMARY KEY (run_id, condition, name, step)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    path      TEXT NOT NULL,
    sha256    TEXT,
    PRIMARY KEY (run_id, path)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment, started_at);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_state() -> dict:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "host": socket.gethostname(), "user": getpass.getuser(), "argv": sys.argv}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    from importlib.metadata import version
    for pkg in ("transformers", "peft", "sentence-transformers", "unsloth", "datasets", "trl"):
        try:
            env[pkg] = version(pkg)  # metadata only: does not import (unsloth import is slow + patches torch)
        except Exception:
            pass
    return env


def _sha256_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Run:
    """Context manager that records one experiment run."""

    def __init__(self, experiment: str, model: str | None = None, config: dict | None = None,
                 note: str | None = None, script: str | None = None, db: Path = DB_PATH):
        self.experiment, self.model, self.config = experiment, model, config or {}
        self.note = note
        self.script = script or (sys.argv[0] if sys.argv and sys.argv[0] else None)
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.db = db
        self.con: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "Run":
        self.con = connect(self.db)
        g = git_state()
        script_path = Path(self.script) if self.script else None
        self.con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, self.experiment, self.model, time.time(), None, "running",
             g["commit"], int(g["dirty"]), g["branch"],
             str(script_path.relative_to(ROOT)) if script_path and script_path.is_absolute() and ROOT in script_path.parents else self.script,
             _sha256_file(script_path) if script_path else None,
             _canon(self.config), hashlib.sha256(_canon(self.config).encode()).hexdigest()[:16],
             _canon(environment()), self.note, None))
        self.con.commit()
        print(f"[evals] run {self.run_id} experiment={self.experiment} model={self.model} "
              f"commit={(g['commit'] or '?')[:8]}{'*' if g['dirty'] else ''}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "failed" if exc_type else "finished"
        self.con.execute("UPDATE runs SET finished_at=?, status=?, error=? WHERE run_id=?",
                         (time.time(), status, repr(exc) if exc else None, self.run_id))
        self.con.commit()
        export_jsonl(self.con)
        self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.con.close()
        print(f"[evals] run {self.run_id} {status}", flush=True)
        return False

    # -- logging -------------------------------------------------------------
    def log(self, metrics: dict, condition: str = "", step: int | None = None):
        now = time.time()
        rows = [(self.run_id, condition, k, float(v), step, now)
                for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        self.con.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?)", rows)
        self.con.commit()

    def artifact(self, path: str | Path):
        p = Path(path)
        rel = str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p)
        self.con.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?)", (self.run_id, rel, _sha256_file(p)))
        self.con.commit()

    def set_config(self, **kv):
        self.config.update(kv)
        self.con.execute("UPDATE runs SET config=?, config_sha=? WHERE run_id=?",
                         (_canon(self.config), hashlib.sha256(_canon(self.config).encode()).hexdigest()[:16], self.run_id))
        self.con.commit()


# -- export ------------------------------------------------------------------
def export_jsonl(con: sqlite3.Connection, path: Path = JSONL_PATH):
    """Rewrite the git-friendly mirror: one line per run with nested metrics."""
    con.row_factory = sqlite3.Row
    runs = con.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
    with open(path, "w", encoding="utf-8") as f:
        for r in runs:
            d = dict(r)
            d["config"], d["env"] = json.loads(d["config"]), json.loads(d["env"])
            mets = {}
            for m in con.execute("SELECT condition, name, value, step FROM metrics WHERE run_id=? ORDER BY condition, name, step",
                                 (r["run_id"],)):
                mets.setdefault(m["condition"], {})[m["name"] if m["step"] is None else f"{m['name']}@{m['step']}"] = m["value"]
            d["metrics"] = mets
            d["artifacts"] = [dict(a) for a in con.execute("SELECT path, sha256 FROM artifacts WHERE run_id=?", (r["run_id"],))]
            f.write(json.dumps(d, sort_keys=True, default=str) + "\n")
    con.row_factory = None


def import_jsonl(con: sqlite3.Connection, path: Path = JSONL_PATH) -> int:
    """Rebuild runs/metrics/artifacts from the git-tracked mirror (idempotent)."""
    n = 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (d["run_id"], d["experiment"], d["model"], d["started_at"], d["finished_at"], d["status"],
                     d["git_commit"], d["git_dirty"], d["git_branch"], d["script"], d["script_sha"],
                     _canon(d["config"]), d["config_sha"], _canon(d["env"]), d["note"], d["error"]))
        for cond, mets in d["metrics"].items():
            for name, value in mets.items():
                base, _, step = name.partition("@")
                con.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?)",
                            (d["run_id"], cond, base, value, int(step) if step else None, d["finished_at"] or d["started_at"]))
        for a in d["artifacts"]:
            con.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?)", (d["run_id"], a["path"], a["sha256"]))
        n += 1
    con.commit()
    return n
