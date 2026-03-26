param(
    [string]$Language = "fr",
    [string]$Device = "auto",
    [string[]]$Locale = @("fr-ca"),
    [string[]]$Engine = @(),
    [string[]]$Include = @(),
    [string]$ReferenceSample = "",
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv-python"
$env:XDG_CACHE_HOME = Join-Path $root ".cache"
$env:HF_HOME = Join-Path $env:XDG_CACHE_HOME "huggingface"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "hub"
$env:TORCH_HOME = Join-Path $env:XDG_CACHE_HOME "torch"
$env:TEMP = Join-Path $root ".tmp"
$env:TMP = $env:TEMP
$env:PYTHONPATH = $root
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$args = @(
    "run",
    "--no-sync",
    "python",
    "-m",
    "tts_bench.cli",
    "--data-root", (Join-Path $root "data"),
    "--output-dir", (Join-Path $root "outputs\\tts"),
    "--language", $Language,
    "--device", $Device
)

foreach ($localeName in $Locale) {
    $args += "--locale"
    $args += $localeName
}

foreach ($engineName in $Engine) {
    $args += "--engine"
    $args += $engineName
}

foreach ($modelName in $Include) {
    $args += "--include"
    $args += $modelName
}

if ($ReferenceSample) {
    $args += "--reference-sample"
    $args += $ReferenceSample
}

if ($Limit -gt 0) {
    $args += "--limit"
    $args += $Limit
}

uv @args
