# Fresh Windows machine -> ready to run `just`. Installs only the two tools the justfile
# cannot install for itself (uv, just), then hands off to `just setup` and `just doctor`.
# Idempotent: anything already on PATH is skipped. Everything else (venv, DVC remote path,
# adapters) is the justfile's job; paper tools like pdftotext are the doctor's WARN, not ours.
#
#   powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" }

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Step "uv already installed: $(uv --version)"
} else {
    Step "installing uv (winget install astral-sh.uv)"
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv not on PATH after install; open a new terminal and re-run" }
}

if (Get-Command just -ErrorAction SilentlyContinue) {
    Step "just already installed: $(just --version)"
} else {
    Step "installing just (winget install Casey.Just)"
    winget install --id Casey.Just -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
    if (-not (Get-Command just -ErrorAction SilentlyContinue)) { throw "just not on PATH after install; open a new terminal and re-run" }
}

Step "just setup   (uv sync, DVC remote path for this OS, dvc pull)"
just setup

Step "just doctor"
just doctor
