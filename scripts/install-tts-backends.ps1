param(
    [string[]]$Engine = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv-python"

$engineToExtra = @{
    "kokoro" = "kokoro"
    "kitten-tts" = "kittentts"
    "chatterbox" = "chatterbox"
    "f5-tts" = "f5tts"
    "voxcpm" = ""
    "qwen3-tts" = ""
    "dia2" = ""
    "glm-tts" = ""
    "orpheus-tts" = ""
    "xtts-v2" = "xtts"
}

$selectedEngines = $Engine
if ($selectedEngines.Count -eq 0) {
    $selectedEngines = @("kokoro", "kitten-tts", "chatterbox", "f5-tts", "voxcpm", "qwen3-tts", "dia2", "glm-tts", "orpheus-tts", "xtts-v2")
}

$extras = New-Object System.Collections.Generic.List[string]
foreach ($engineName in $selectedEngines) {
    if (-not $engineToExtra.ContainsKey($engineName)) {
        throw "Unknown TTS engine: $engineName"
    }
    $extra = $engineToExtra[$engineName]
    if ([string]::IsNullOrWhiteSpace($extra)) {
        continue
    }
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
