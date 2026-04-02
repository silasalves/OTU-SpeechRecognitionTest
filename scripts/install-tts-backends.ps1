param(
    [string[]]$Engine = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv-python"

function Get-VenvPythonPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvPath
    )

    $windowsPython = Join-Path $VenvPath "Scripts\python.exe"
    if (Test-Path -LiteralPath $windowsPython) {
        return $windowsPython
    }

    $posixPython = Join-Path $VenvPath "bin/python"
    if (Test-Path -LiteralPath $posixPython) {
        return $posixPython
    }

    throw "Could not find a Python executable under venv path: $VenvPath"
}

function Install-VibeVoice {
    $repoPath = Join-Path $root ".tmp\VibeVoice-official"
    $venvPath = Join-Path $root ".venvs\vibevoice"

    if (Test-Path -LiteralPath (Join-Path $repoPath ".git")) {
        git -C $repoPath pull --ff-only
    } elseif (-not (Test-Path -LiteralPath $repoPath)) {
        git clone https://github.com/microsoft/VibeVoice.git $repoPath
    } else {
        throw "Expected either a git repo or no path at $repoPath"
    }

    if (-not (Test-Path -LiteralPath $venvPath)) {
        uv venv $venvPath --python 3.11
    }

    $pythonPath = Get-VenvPythonPath -VenvPath $venvPath

    try {
        uv pip install --python $pythonPath --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
    } catch {
        Write-Warning "CUDA PyTorch install for VibeVoice failed; falling back to the default torch wheel in the dedicated venv."
        uv pip install --python $pythonPath torch torchvision torchaudio
    }

    $editableSpec = "$repoPath[streamingtts]"
    uv pip install --python $pythonPath -e $editableSpec
}

$engineToExtra = @{
    "kokoro" = "kokoro"
    "kitten-tts" = "kittentts"
    "chatterbox" = "chatterbox"
    "f5-tts" = "f5tts"
    "voxcpm" = ""
    "qwen3-tts" = ""
    "cosyvoice2" = ""
    "vibevoice" = ""
    "dia2" = ""
    "glm-tts" = ""
    "orpheus-tts" = ""
    "xtts-v2" = "xtts"
}

$selectedEngines = $Engine
if ($selectedEngines.Count -eq 0) {
    $selectedEngines = @("kokoro", "kitten-tts", "chatterbox", "f5-tts", "voxcpm", "qwen3-tts", "cosyvoice2", "vibevoice", "dia2", "glm-tts", "orpheus-tts", "xtts-v2")
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

if ($selectedEngines -contains "vibevoice") {
    Install-VibeVoice
}
