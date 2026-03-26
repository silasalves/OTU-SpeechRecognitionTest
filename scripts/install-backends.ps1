param(
    [string[]]$Extras = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv-python"

uv python install 3.11 --no-registry

$extraArgs = @()
foreach ($extra in $Extras) {
    $extraArgs += "--extra"
    $extraArgs += $extra
}

uv sync @extraArgs
