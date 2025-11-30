# AMD GPU Setup Guide for Character OVI Backend

This guide explains how to run the Character OVI Backend Docker container with AMD GPU support using ROCm.

## Prerequisites

1. **AMD GPU** - Ensure you have an AMD GPU that supports ROCm
2. **Docker** - Docker Desktop or Docker Engine installed
3. **WSL2** (for Windows) - If running on Windows, you need WSL2 with AMD GPU drivers installed

## WSL2 Setup (Windows Users)

If you're running on Windows with WSL2, you need to:

1. Install AMD GPU drivers in WSL2:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install ROCm (follow AMD's official guide for your GPU model)
   # Check: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html
   ```

2. Verify GPU is accessible:
   ```bash
   # Check if GPU devices are available
   ls -la /dev/dri/
   # Should see renderD* files
   ```

3. Test ROCm installation:
   ```bash
   rocm-smi
   # Should show your AMD GPU
   ```

## Running with AMD GPU

### Option 1: Using the Helper Script (Linux/WSL2)

```bash
chmod +x docker-run-amd-gpu.sh
./docker-run-amd-gpu.sh
```

### Option 2: Using PowerShell Script (Windows)

```powershell
.\docker-run-amd-gpu.ps1
```

### Option 3: Manual Docker Run Command

For Linux/WSL2:
```bash
docker run -it --rm \
  --device=/dev/dri \
  --device=/dev/kfd \
  --group-add $(stat -c '%g' /dev/dri/renderD128) \
  -e ROCM_VISIBLE_DEVICES=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  character-ovi-backend
```

**Note:** Adjust the render node path (e.g., `renderD128`) based on your system. Check available render nodes with:
```bash
ls -la /dev/dri/renderD*
```

## Environment Variables

The following environment variables are set for AMD GPU support:

- `ROCM_VISIBLE_DEVICES=0` - Makes GPU 0 visible to ROCm
- `HIP_VISIBLE_DEVICES=0` - Makes GPU 0 visible to HIP
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` - Overrides GPU version (adjust based on your GPU)

## Troubleshooting

### GPU Not Detected

1. **Check GPU accessibility in host system:**
   ```bash
   ls -la /dev/dri/
   rocm-smi
   ```

2. **Verify Docker can access GPU:**
   ```bash
   docker run --rm --device=/dev/dri rocm/pytorch:latest rocm-smi
   ```

3. **Check permissions:**
   ```bash
   # Add your user to render group
   sudo usermod -a -G render $USER
   # Log out and log back in for changes to take effect
   ```

### CUDA/ROCm Not Available Error

If you see "CUDA/ROCm not available" in the logs:

1. The container will fall back to CPU mode (slow but functional)
2. Verify GPU devices are mounted: Check if `/dev/dri` is accessible in container
3. Check ROCm installation in the host system

### Performance Issues

- Ensure you're using the correct ROCm version for your GPU
- Check GPU utilization: `rocm-smi` in the container
- Verify environment variables are set correctly

## CPU Fallback Mode

If GPU is not available, the system will automatically fall back to CPU mode. This is much slower but will work for testing purposes. You'll see warnings like:

```
WARNING: CUDA/ROCm not available. Falling back to CPU mode (very slow).
WARNING: Running in CPU mode. This will be very slow.
```

## Building the Docker Image

```bash
docker build -t character-ovi-backend -f DOCKERFILE .
```

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm Docker Support](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker.html)
- [AMD GPU Compatibility](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

## Notes

- The Dockerfile uses `rocm/pytorch:latest` base image which includes ROCm support
- CPU fallback is automatically enabled if GPU is not detected
- All code changes support both GPU (ROCm/CUDA) and CPU modes

