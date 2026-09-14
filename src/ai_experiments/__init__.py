"""ai_experiments: the synthetic datasets, eval item builders and the run tracker.

Installed editable by `uv sync` (see pyproject.toml), so scripts import it directly and no
longer edit sys.path. Repo-relative locations are in `ai_experiments.paths`.

Importing the package also makes stdout and stderr UTF-8. The dataset text contains characters
the Windows console codec (cp1252) cannot print; `PYTHONUTF8=1` does the same for the whole
process (`.claude/settings.json` sets it for agent sessions, `just doctor` checks it).
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    _enc = (getattr(_stream, "encoding", None) or "").lower().replace("-", "")
    if _stream is not None and _enc != "utf8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
del _stream, _enc
