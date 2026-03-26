param(
    [string[]]$Engine = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv-python"

$engineToExtra = @{
    "kokoro" = "kokoro"
    "chatterbox" = "chatterbox"
    "f5-tts" = "f5tts"
    "xtts-v2" = "xtts"
}

$selectedEngines = $Engine
if ($selectedEngines.Count -eq 0) {
    $selectedEngines = @("kokoro", "chatterbox", "f5-tts", "xtts-v2")
}

$extras = New-Object System.Collections.Generic.List[string]
foreach ($engineName in $selectedEngines) {
    if (-not $engineToExtra.ContainsKey($engineName)) {
        throw "Unknown TTS engine: $engineName"
    }
    $extra = $engineToExtra[$engineName]
    if (-not $extras.Contains($extra)) {
        $extras.Add($extra)
    }
}

uv python install 3.11 --no-registry

$extraArgs = @()
foreach ($extra in $extras) {
    $extraArgs += "--extra"
    $extraArgs += $extra
}

uv sync @extraArgs
