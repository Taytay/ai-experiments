"""Check this machine and checkout against the working rules in CLAUDE.md.

  uv run python scripts/doctor.py          # or: just doctor
  uv run python scripts/doctor.py --gpu    # also allocate past VRAM to see whether the driver
                                           # spills to system RAM (the WDDM 3x-slowdown trap)

One line per check: OK, WARN (a machine setting worth changing, with the fix), FAIL (this
checkout cannot run experiments), SKIP. Exit code is 1 only if something FAILs.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = platform.system() == "Windows"
IS_WSL = "microsoft" in platform.release().lower()
SIDE = "Windows" if IS_WIN else "WSL2" if IS_WSL else platform.system()

rows: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    rows.append((status, name, detail))


def check_python() -> None:
    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    report("OK" if in_venv else "FAIL", "python",
           f"{platform.python_version()} at {sys.prefix}" + ("" if in_venv else "; run `uv run python`, not this interpreter"))


def check_package() -> None:
    try:
        from ai_experiments.paths import ROOT as pkg_root
    except ImportError as e:
        report("FAIL", "ai_experiments", f"not importable ({e}); run `uv sync`")
        return
    if pkg_root == ROOT:
        report("OK", "ai_experiments", f"editable install from {pkg_root}")
    else:
        report("FAIL", "ai_experiments", f"imports from {pkg_root}, expected {ROOT}; run `uv sync`")


def check_utf8() -> None:
    # Importing ai_experiments reconfigures stdout, so test the process settings, not the stream.
    if sys.flags.utf8_mode or os.environ.get("PYTHONIOENCODING", "").lower().startswith("utf"):
        report("OK", "utf-8", "UTF-8 mode on (PYTHONUTF8 or PYTHONIOENCODING set)")
    elif not IS_WIN:
        report("OK", "utf-8", f"locale gives {sys.getfilesystemencoding()}; UTF-8 mode not needed")
    else:
        report("WARN", "utf-8", "UTF-8 mode off; `setx PYTHONUTF8 1` and reopen the shell. Agent sessions get it from "
                                ".claude/settings.json, and importing ai_experiments fixes stdout either way")


def check_store_alias() -> None:
    if not IS_WIN:
        report("OK", "python alias", f"{SIDE}: no Store alias to worry about")
        return
    hits = [h for h in (shutil.which("python"), shutil.which("python3")) if h]
    stubs = [h for h in hits if "WindowsApps" in h]
    if stubs:
        report("WARN", "python alias", f"{stubs[0]} is the Microsoft Store stub; turn off python.exe and python3.exe under "
                                       "Settings > Apps > Advanced app settings > App execution aliases")
    elif hits:
        report("OK", "python alias", f"bare python is {hits[0]} (still prefer `uv run python`)")
    else:
        report("OK", "python alias", "no bare python on PATH; `uv run python` is the only python")


def check_tool(name: str, win_hint: str, linux_hint: str) -> None:
    p = shutil.which(name)
    report("OK" if p else "WARN", name,
           p or f"not on PATH; every `just` recipe needs it. Run `{win_hint if IS_WIN else linux_hint}` (installs uv and just, then setup and doctor).")


def check_poppler() -> None:
    have = [t for t in ("pdftotext", "pdftoppm") if shutil.which(t)]
    missing = [t for t in ("pdftotext", "pdftoppm") if not shutil.which(t)]
    if not missing:
        report("OK", "poppler", f"pdftotext and pdftoppm on PATH ({shutil.which('pdftotext')})")
        return
    report("WARN", "poppler",
           f"{', '.join(missing)} not on PATH. Optional: only research steps need it, when an agent "
           "reads a PDF under references/papers/ (the Read tool cannot open PDFs; `pdftotext -layout` "
           "makes the paper.txt it reads instead). Training and evaluation never use it. "
           "Install with `just pdf-tools`" + (f" (have {', '.join(have)})" if have else "") + ".")


def check_line_endings() -> None:
    if not (ROOT / ".gitattributes").exists():
        report("WARN", "line endings", "no .gitattributes; add `* text=auto eol=lf`")
        return
    out = subprocess.run(["git", "ls-files", "--eol"], cwd=ROOT, capture_output=True, text=True).stdout
    bad = [line for line in out.splitlines() if line.startswith(("i/crlf", "i/mixed"))]
    if bad:
        report("WARN", "line endings", f"{len(bad)} tracked files are CRLF in the index; run `git add --renormalize .`")
    else:
        report("OK", "line endings", ".gitattributes present, every tracked file is LF in the index")


def check_dvc_remote() -> None:
    local = ROOT / ".dvc" / "config.local"
    text = local.read_text() if local.exists() else ""
    url = next((line.split("=", 1)[1].strip() for line in text.splitlines() if line.strip().startswith("url")), None)
    if not url:
        report("FAIL", "dvc remote", "no url in .dvc/config.local; run `just setup` (or the command in .dvc/config)")
        return
    if not Path(url).is_dir():
        report("FAIL", "dvc remote", f"{url} is not a folder on this side")
        return
    adapters = ROOT / "models" / "adapters"
    n = len([p for p in adapters.iterdir() if p.is_dir()]) if adapters.is_dir() else 0
    report("OK" if n else "WARN", "dvc remote", f"{url}; {n} adapter dirs in models/adapters" + ("" if n else " (run `uv run dvc pull`)"))


def check_claude_settings() -> None:
    p = ROOT / ".claude" / "settings.json"
    ok = p.exists() and "PYTHONUTF8" in p.read_text()
    report("OK" if ok else "WARN", "claude settings", ".claude/settings.json sets PYTHONUTF8" if ok else ".claude/settings.json missing or lacks env.PYTHONUTF8")


def check_gpu(oom_test: bool) -> None:
    try:
        import torch
    except ImportError:
        report("FAIL", "gpu", "torch not installed; run `uv sync`")
        return
    if not torch.cuda.is_available():
        report("FAIL", "gpu", f"torch {torch.__version__} sees no CUDA device")
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    report("OK", "gpu", f"{name}, {total / 2**30:.1f} GiB, torch {torch.__version__} (CUDA {torch.version.cuda})")
    if not oom_test:
        report("SKIP", "sysmem fallback", "pass --gpu to test whether an allocation past VRAM succeeds (spills) or raises OOM")
        return
    try:
        t = torch.empty(int(total * 1.25), dtype=torch.uint8, device="cuda")
        del t
        torch.cuda.empty_cache()
        report("WARN", "sysmem fallback", "allocating 125% of VRAM succeeded, so the driver spills to system RAM and an OOM "
                                          "becomes a 3x-slower run. NVIDIA Control Panel > Manage 3D settings > "
                                          "CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback")
    except torch.cuda.OutOfMemoryError:
        report("OK", "sysmem fallback", "allocating past VRAM raises OutOfMemoryError; no silent spill")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="also run the over-allocation test on the GPU")
    args = ap.parse_args()

    print(f"side: {SIDE} ({platform.platform()}), checkout: {ROOT}")
    check_python()
    check_package()
    check_utf8()
    check_store_alias()
    check_tool("uv", "bootstrap.ps1", "./bootstrap.sh")
    check_tool("just", "bootstrap.ps1", "./bootstrap.sh")
    check_poppler()
    check_line_endings()
    check_dvc_remote()
    check_claude_settings()
    check_gpu(args.gpu)

    width = max(len(name) for _, name, _ in rows)
    for status, name, detail in rows:
        print(f"{status:4s} {name:{width}s}  {detail}")
    n_fail = sum(1 for s, _, _ in rows if s == "FAIL")
    n_warn = sum(1 for s, _, _ in rows if s == "WARN")
    print(f"{n_fail} FAIL, {n_warn} WARN")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
