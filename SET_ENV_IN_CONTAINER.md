# Setting Environment Variables in Running Container

If you've already built your Docker image and don't want to rebuild, you can set the environment variables directly when running the container.

## Quick Commands

### For Linux/WSL2:

```bash
docker run -it --rm \
  --device=/dev/dri \
  --device=/dev/kfd \
  -e ROCM_PATH=/opt/rocm \
  -e PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/app/Character-ovi-backend/ovi-env/bin \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e ROCM_VISIBLE_DEVICES=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  character-ovi-backend
```

### For Windows PowerShell:

```powershell
docker run -it --rm `
  -e ROCM_PATH=/opt/rocm `
  -e PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/app/Character-ovi-backend/ovi-env/bin `
  -e LD_LIBRARY_PATH=/opt/rocm/lib `
  -e ROCM_VISIBLE_DEVICES=0 `
  -e HIP_VISIBLE_DEVICES=0 `
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 `
  --security-opt seccomp=unconfined `
  -p 8000:8000 `
  character-ovi-backend
```

## Using Helper Scripts

### Linux/WSL2:
```bash
chmod +x docker-run-with-env.sh
./docker-run-with-env.sh
```

### Windows PowerShell:
```powershell
.\docker-run-with-env.ps1
```

## Setting Variables in Already Running Container

If your container is already running, you can set variables in the current shell session:

```bash
# Enter the running container
docker exec -it <container_id> bash

# Then set variables in your current shell:
export ROCM_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib
export ROCM_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Now run your server
cd /app/Character-ovi-backend
source ovi-env/bin/activate
python3 api_server.py --config ovi/configs/inference/inference_fusion.yaml
```

## Environment Variables Explained

- `ROCM_PATH=/opt/rocm` - Path to ROCm installation
- `PATH=...` - Adds ROCm binaries to PATH (includes your venv path)
- `LD_LIBRARY_PATH=/opt/rocm/lib` - Library path for ROCm
- `ROCM_VISIBLE_DEVICES=0` - Makes GPU 0 visible to ROCm
- `HIP_VISIBLE_DEVICES=0` - Makes GPU 0 visible to HIP
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` - GPU version override (adjust for your GPU)

## Adjust GPU Version

If `HSA_OVERRIDE_GFX_VERSION=10.3.0` doesn't work for your GPU, try:
- `10.3.0` - For RDNA2 (RX 6000 series)
- `11.0.0` - For RDNA3 (RX 7000 series)
- Or remove the variable if your GPU is properly detected

## Verify Variables Are Set

Once in the container, verify:
```bash
echo $ROCM_PATH
echo $ROCM_VISIBLE_DEVICES
rocm-smi  # Should show your GPU if properly configured
```

