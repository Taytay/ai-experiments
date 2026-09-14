# Repo tasks. `just` alone lists them. Install just with `winget install Casey.Just` (Windows)
# or `sudo apt install just` (Ubuntu). Everything here is also a plain uv/dvc command; the
# recipes only save remembering the sequence and the per-OS DVC path.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

dvc_win := 'D:\repos\dvc\ai-experiments'
dvc_wsl := '/mnt/d/repos/dvc/ai-experiments'

default:
    @just --list --unsorted

# Once per checkout: venv + editable package, DVC remote path for this OS, then the adapters.
setup: sync dvc-remote pull

# Create or refresh .venv and install ai_experiments editable.
sync:
    uv sync

# Write this OS's spelling of the D: folder to the gitignored .dvc/config.local.
[windows]
[doc("Write this OS's spelling of the D: folder to the gitignored .dvc/config.local")]
dvc-remote:
    if (Test-Path '{{dvc_win}}') { uv run dvc remote modify --local dstore url '{{dvc_win}}'; Write-Host "dvc remote dstore = {{dvc_win}}" } else { Write-Host "{{dvc_win}} does not exist; set it by hand: uv run dvc remote modify --local dstore url <folder>" }

[unix]
[doc("Write this OS's spelling of the D: folder to the gitignored .dvc/config.local")]
dvc-remote:
    if [ -d '{{dvc_wsl}}' ]; then uv run dvc remote modify --local dstore url '{{dvc_wsl}}' && echo "dvc remote dstore = {{dvc_wsl}}"; else echo "{{dvc_wsl}} does not exist; set it by hand: uv run dvc remote modify --local dstore url <folder>"; fi

# Fetch the adapters this commit expects (3.8 GB the first time).
pull:
    uv run dvc pull

# After training: re-hash models/adapters, push new blobs, stage the updated .dvc pointer.
push-models:
    uv run dvc add models/adapters
    uv run dvc push

# Check machine and checkout against CLAUDE.md; `--gpu` also tests the VRAM-to-RAM spill.
doctor *ARGS:
    uv run python scripts/doctor.py {{ARGS}}

# Per machine, optional: poppler (pdftotext, pdftoppm) so an agent can read PDFs under
# references/papers/ during research steps. Not needed to train or evaluate. Needs admin/sudo.
[windows]
[doc("Optional, per machine: install poppler so an agent can read PDFs in references/papers/ (needs admin)")]
pdf-tools:
    if (Get-Command pdftotext -ErrorAction SilentlyContinue) { Write-Host "poppler already installed: $((Get-Command pdftotext).Source)" } else { winget install --id oschwartz10612.Poppler -e --accept-source-agreements --accept-package-agreements }

[unix]
[doc("Optional, per machine: install poppler so an agent can read PDFs in references/papers/ (needs sudo)")]
pdf-tools:
    if command -v pdftotext >/dev/null && command -v pdftoppm >/dev/null; then echo "poppler already installed: $(command -v pdftotext)"; else sudo apt-get install -y poppler-utils; fi

# Byte-compile the package and scripts, then import the package.
check:
    uv run python -m compileall -q src scripts
    uv run python -c "import ai_experiments.icl_suite, ai_experiments.evals.tracker; print('imports ok')"

# Regenerate evals/LEADERBOARD.md from the tracker.
leaderboard:
    uv run evals leaderboard

# Quick end-to-end run of one curriculum arm (default C) with SMOKE=1: subsampled eval items,
# results in results/*_smoke.json, nothing recorded in the tracker.
[windows]
[doc("Quick end-to-end run of one curriculum arm (default C) with SMOKE=1; not recorded in the tracker")]
smoke ARM='C':
    $env:SMOKE = '1'; uv run python scripts/exp_curriculum.py {{ARM}}

[unix]
[doc("Quick end-to-end run of one curriculum arm (default C) with SMOKE=1; not recorded in the tracker")]
smoke ARM='C':
    SMOKE=1 uv run python scripts/exp_curriculum.py {{ARM}}
