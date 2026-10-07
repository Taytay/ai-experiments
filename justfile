# Repo tasks. `just` alone lists them. Install just with `winget install Casey.Just` (Windows)
# or `sudo apt install just` (Ubuntu). Everything here is also a plain uv/dvc command; the
# recipes only save remembering the sequence and the per-OS DVC path.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

dvc_win := 'D:\repos\dvc\ai-experiments'
dvc_wsl := '/mnt/d/repos/dvc/ai-experiments'

default:
    @just --list --unsorted

# Once per checkout: venv + editable package, DVC remote path for this OS. Adapters are pulled
# on demand with `just pull NAME`.
setup: sync dvc-remote

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

# Fetch named adapters from D:, e.g. `just pull curriculum_Qwen2.5-3B_C_lora`.
pull +NAMES:
    cd models/adapters; uv run dvc pull {{NAMES}}

# Fetch every adapter this commit tracks (hundreds of GB; usually you want `just pull NAME`).
pull-all:
    uv run dvc pull

# After training: hash each adapter and encoder dir present locally (encoders were once missed: 18 found only in modal_out, 2026-10-07), push new blobs to D:, stage the .dvc files.
[windows]
[doc("After training: hash each adapter dir present locally, push new blobs to D:, stage the .dvc files")]
push-models:
    uv run dvc add (Get-ChildItem models/adapters, models/encoders -Directory | ForEach-Object { "models/$($_.Parent.Name)/$($_.Name)" }); uv run dvc push

[unix]
[doc("After training: hash each adapter dir present locally, push new blobs to D:, stage the .dvc files")]
push-models:
    uv run dvc add $(find models/adapters models/encoders -mindepth 1 -maxdepth 1 -type d | sort)
    uv run dvc push

# Free local disk: push, then delete every local adapter dir and the DVC cache. The .dvc files
# stay, so `just pull NAME` brings any adapter back from D:.
[windows]
[doc("Free local disk: push, then delete local adapter dirs and the DVC cache (.dvc files stay)")]
drop-all:
    uv run dvc push; if ($LASTEXITCODE -ne 0) { exit 1 }; Get-ChildItem models/adapters -Directory | Remove-Item -Recurse -Force; if (Test-Path .dvc/cache) { Remove-Item .dvc/cache -Recurse -Force }

[unix]
[doc("Free local disk: push, then delete local adapter dirs and the DVC cache (.dvc files stay)")]
drop-all:
    uv run dvc push
    find models/adapters -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
    rm -rf .dvc/cache

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

# Byte-compile the package and scripts, import the package, and confirm the item generators
# still reproduce the frozen eval sets in data/processed/.
check:
    uv run python -m compileall -q src scripts
    uv run python -c "import ai_experiments.icl_suite, ai_experiments.evals.tracker; print('imports ok')"
    uv run python -m ai_experiments.items check

# Regenerate evals/LEADERBOARD.md from the tracker.
leaderboard:
    uv run evals leaderboard

# Quick end-to-end plumbing check of one curriculum arm (default C) with SMOKE=1: 2 training steps,
# 1/40 of the eval items, results in results/*_smoke.json, adapter in models/smoke/, nothing in the tracker.
[windows]
[doc("Plumbing check: curriculum arm (default C) with SMOKE=1, 2 steps, tiny eval; not recorded in the tracker")]
smoke ARM='C':
    $env:SMOKE = '1'; uv run python scripts/exp_curriculum.py {{ARM}}

[unix]
[doc("Plumbing check: curriculum arm (default C) with SMOKE=1, 2 steps, tiny eval; not recorded in the tracker")]
smoke ARM='C':
    SMOKE=1 uv run python scripts/exp_curriculum.py {{ARM}}
