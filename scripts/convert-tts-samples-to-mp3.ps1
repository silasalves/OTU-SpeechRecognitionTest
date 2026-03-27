param(
    [string]$SamplesRoot = "",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $SamplesRoot) {
    $SamplesRoot = Join-Path $root "samples\tts"
}

if (-not (Test-Path -LiteralPath $SamplesRoot)) {
    throw "Samples directory not found: $SamplesRoot"
}

$ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffmpegPath = if ($ffmpegCommand) {
    $ffmpegCommand.Source
} else {
    $fallback = "C:\Program Files\Virtual Desktop Streamer\ffmpeg.exe"
    if (Test-Path -LiteralPath $fallback) {
        $fallback
    } else {
        throw "ffmpeg was not found on PATH and no fallback executable was found."
    }
}

$wavFiles = Get-ChildItem -Path $SamplesRoot -Recurse -Filter *.wav | Sort-Object FullName
foreach ($wavFile in $wavFiles) {
    $mp3Path = [System.IO.Path]::ChangeExtension($wavFile.FullName, ".mp3")
    if ((-not $Overwrite) -and (Test-Path -LiteralPath $mp3Path)) {
        continue
    }

    & $ffmpegPath -y -i $wavFile.FullName -codec:a libmp3lame -qscale:a 2 $mp3Path | Out-Null
}

Write-Host "Converted $(($wavFiles | Measure-Object).Count) WAV file(s) under $SamplesRoot"
