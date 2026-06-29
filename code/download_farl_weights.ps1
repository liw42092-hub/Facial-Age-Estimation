param(
    [ValidateSet("ep16", "ep64")]
    [string]$Version = "ep64",
    [string]$OutputDirectory = ".\pretrained"
)

$ErrorActionPreference = "Stop"
$fileName = "FaRL-Base-Patch16-LAIONFace20M-$Version.pth"
$url = "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/$fileName"
$outputDirectoryPath = [System.IO.Path]::GetFullPath($OutputDirectory)
$destination = Join-Path $outputDirectoryPath $fileName
$partial = "$destination.partial"

New-Item -ItemType Directory -Force -Path $outputDirectoryPath | Out-Null
if (Test-Path -LiteralPath $destination) {
    Write-Host "Weights already exist: $destination"
    exit 0
}

Write-Host "Downloading official FaRL weights with resume support..."
Write-Host "URL: $url"
& curl.exe -L -C - --retry 20 --retry-delay 5 --retry-all-errors --connect-timeout 30 -o $partial $url
if ($LASTEXITCODE -ne 0) {
    throw "Download interrupted. Run this command again to resume: powershell -ExecutionPolicy Bypass -File .\download_farl_weights.ps1 -Version $Version"
}

Move-Item -LiteralPath $partial -Destination $destination
Write-Host "Saved: $destination"
