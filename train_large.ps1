<#
train_large.ps1

Convenience PowerShell wrapper to start a "large" training run (yolo11m-scale).
This script activates the repository venv (if present), installs requirements if
requested, and runs `train_yolov11.py` with conservative defaults suitable for
large-model training. Edit the parameters at the top to match your GPU VRAM.

Important: training a large model (yolo11m-scale) requires a powerful GPU
with a large amount of VRAM (>=24GB recommended). If you don't have that,
use a smaller model (yolov8n/yolov8s) or reduce batch/imgsz.
#>

param(
    [string]$Weights = $(if (Test-Path .\best.pt) { Resolve-Path .\best.pt } elseif (Test-Path .\yolo11m.pt) { Resolve-Path .\yolo11m.pt } else { 'yolov11m.pt' }),
    [int]$PretuneEpochs = 5,
    [int]$Epochs = 100,
    [int]$ImgSz = 1280,
    [int]$Batch = 8,
    [string]$Device = '0',
    [switch]$InstallDeps
)

Write-Host "Starting large training wrapper"

if ($InstallDeps) {
    Write-Host "Installing dependencies from requirements.txt (this may take a while)"
    python -m pip install -r .\requirements.txt
}

# Activate venv if present
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    Write-Host "Activating .venv..."
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "No .venv found. Make sure you run this inside a virtual environment or have dependencies installed globally."
}

$timestamp = Get-Date -UFormat %s
$runName = "levitezer_large_$timestamp"

Write-Host "Using weights: $Weights"
Write-Host "Pretune epochs: $PretuneEpochs, Main epochs: $Epochs, imgsz: $ImgSz, batch: $Batch, device: $Device"

# Command to run (prints before executing)
$cmd = "python .\train_yolov11.py --data data.yaml --weights `"$Weights`" --pretune-epochs $PretuneEpochs --epochs $Epochs --imgsz $ImgSz --batch $Batch --device $Device --project runs/train --name $runName"

Write-Host "Running training command:`n$cmd"
Invoke-Expression $cmd

Write-Host "Training finished (or exited). Check runs/train/$runName for outputs."
