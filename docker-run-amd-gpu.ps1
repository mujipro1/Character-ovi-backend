# PowerShell script to run Docker container with AMD GPU (ROCm) support on Windows
# This script sets up the necessary device mounts and environment variables for AMD GPU access

Write-Host "Starting Docker container with AMD GPU support..." -ForegroundColor Green

# Check if running in WSL2
$isWSL = $false
try {
    $wslCheck = wsl cat /proc/version 2>$null
    if ($wslCheck -match "Microsoft|WSL") {
        $isWSL = $true
        Write-Host "Detected WSL2 environment" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Not running in WSL2 or WSL not available" -ForegroundColor Yellow
}

# Build docker run command with device mounts
$dockerCmd = "docker run -it --rm"

# For WSL2, we need to access GPU through WSL
if ($isWSL) {
    Write-Host "GPU access through WSL2 requires proper setup..." -ForegroundColor Yellow
    Write-Host "Make sure AMD GPU drivers are installed in WSL2" -ForegroundColor Yellow
}

# Set ROCm environment variables
$dockerCmd += " -e ROCM_VISIBLE_DEVICES=0"
$dockerCmd += " -e HIP_VISIBLE_DEVICES=0"
$dockerCmd += " -e HSA_OVERRIDE_GFX_VERSION=10.3.0"

# Add security options for GPU access
$dockerCmd += " --security-opt seccomp=unconfined"

# Port mapping for API server
$dockerCmd += " -p 8000:8000"

# Mount current directory (optional - adjust as needed)
$currentDir = (Get-Location).Path
$dockerCmd += " -v ${currentDir}:/workspace"

# Image name (adjust if different)
$imageName = "character-ovi-backend"

# Check if image exists
$imageExists = docker images | Select-String -Pattern $imageName -Quiet
if (-not $imageExists) {
    Write-Host "Image $imageName not found. Building image first..." -ForegroundColor Yellow
    docker build -t $imageName -f DOCKERFILE .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build Docker image. Exiting." -ForegroundColor Red
        exit 1
    }
}

# Add image name
$dockerCmd += " $imageName"

Write-Host ""
Write-Host "Running command:" -ForegroundColor Cyan
Write-Host $dockerCmd
Write-Host ""

# Execute docker run
Invoke-Expression $dockerCmd

