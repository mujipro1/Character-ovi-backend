# PowerShell script to run existing Docker container with AMD GPU environment variables
# Use this if you've already built the image and don't want to rebuild

$imageName = "character-ovi-backend"

# Check if image exists
$imageExists = docker images | Select-String -Pattern $imageName -Quiet
if (-not $imageExists) {
    Write-Host "Image $imageName not found. Please build it first or adjust imageName variable." -ForegroundColor Red
    exit 1
}

Write-Host "Running container with AMD GPU environment variables..." -ForegroundColor Green

docker run -it --rm `
  -e ROCM_PATH=/opt/rocm `
  -e PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/app/Character-ovi-backend/ovi-env/bin `
  -e LD_LIBRARY_PATH=/opt/rocm/lib `
  -e ROCM_VISIBLE_DEVICES=0 `
  -e HIP_VISIBLE_DEVICES=0 `
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 `
  --security-opt seccomp=unconfined `
  -p 8000:8000 `
  $imageName

