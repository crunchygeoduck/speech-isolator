param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pyinstaller = Join-Path $projectRoot ".venv\Scripts\pyinstaller.exe"
$model = Join-Path $env:LOCALAPPDATA "DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3"
$clearVoiceModel = Join-Path $projectRoot "checkpoints\MossFormer2_SE_48K"
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment missing. Run: uv venv --python 3.11 .venv; uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt"
}
if (-not (Test-Path -LiteralPath (Join-Path $model "config.ini"))) {
    throw "DeepFilterNet3 model cache missing. Run one CLI processing job before packaging."
}
if (-not (Test-Path -LiteralPath (Join-Path $clearVoiceModel "last_best_checkpoint.pt"))) {
    throw "ClearVoice MossFormer2 checkpoint missing from checkpoints\MossFormer2_SE_48K."
}

$arguments = @(
    "--noconfirm",
    "--onedir",
    "--windowed",
    "--name", "FactorySpeechCleaner",
    "--distpath", (Join-Path $projectRoot "release-staging"),
    "--paths", $projectRoot,
    "--add-data", "$(Join-Path $projectRoot 'presets');presets",
    "--add-data", "$model;models\DeepFilterNet3",
    "--add-data", "$clearVoiceModel;checkpoints\MossFormer2_SE_48K",
    "--add-binary", "$ffmpeg;.",
    "--add-binary", "$ffprobe;.",
    "--collect-all", "df",
    "--collect-all", "deepfilterlib",
    "--collect-all", "clearvoice",
    "--hidden-import", "scipy.signal",
    (Join-Path $projectRoot "gui_main.py")
)
if ($Clean) {
    $arguments = @("--clean") + $arguments
}

& $pyinstaller @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$stagedRelease = Join-Path $projectRoot "release-staging\FactorySpeechCleaner"
Copy-Item -LiteralPath (Join-Path $projectRoot "PORTABLE_START_HERE.txt") -Destination (Join-Path $stagedRelease "START_HERE.txt") -Force
$installedRelease = Join-Path $projectRoot "dist\FactorySpeechCleaner"
New-Item -ItemType Directory -Path $installedRelease -Force | Out-Null
# Merge application files; never erase the user's output, comparisons, or logs
# when updating an existing portable installation.
Get-ChildItem -LiteralPath $stagedRelease | Copy-Item -Destination $installedRelease -Recurse -Force
$exe = Join-Path $installedRelease "FactorySpeechCleaner.exe"
Write-Host "Built: $exe"
