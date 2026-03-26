param(
    [string]$Language = "fr",
    [string]$Device = "auto",
    [string[]]$Engine = @(),
    [string[]]$Include = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv-python"
$env:XDG_CACHE_HOME = Join-Path $root ".cache"
$env:HF_HOME = Join-Path $env:XDG_CACHE_HOME "huggingface"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "hub"
$env:TORCH_HOME = Join-Path $env:XDG_CACHE_HOME "torch"
$env:WHISPER_CACHE_DIR = Join-Path $env:XDG_CACHE_HOME "whisper"
$env:TEMP = Join-Path $root ".tmp"
$env:TMP = $env:TEMP
$env:PYTHONPATH = $root
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$args = @(
    "run",
    "asr-bench",
    "--data-root", (Join-Path $root "data"),
    "--output-dir", (Join-Path $root "outputs"),
    "--language", $Language,
    "--device", $Device
)

foreach ($engineName in $Engine) {
    $args += "--engine"
    $args += $engineName
}

foreach ($modelName in $Include) {
    $args += "--include"
    $args += $modelName
}

uv @args
