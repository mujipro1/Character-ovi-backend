#!/bin/bash

# Script to run Docker container with AMD GPU (ROCm) support on Windows/WSL2
# This script sets up the necessary device mounts and environment variables for AMD GPU access

echo "Starting Docker container with AMD GPU support..."

# Check if running in WSL2
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo "Detected WSL2 environment"
    WSL2=true
else
    echo "Running in native Linux or other environment"
    WSL2=false
fi

# Find AMD GPU device files
GPU_DEVICES=$(find /dev -name "renderD*" -o -name "kfd" 2>/dev/null)

if [ -z "$GPU_DEVICES" ]; then
    echo "WARNING: No AMD GPU devices found. Container will run in CPU mode."
    echo "Make sure AMD GPU drivers are installed and accessible."
fi

# Build docker run command with device mounts
DOCKER_CMD="docker run -it --rm"

# Mount AMD GPU devices (if available)
if [ ! -z "$GPU_DEVICES" ]; then
    echo "Found AMD GPU devices. Mounting GPU devices..."
    # Mount /dev/dri for render nodes
    if [ -d "/dev/dri" ]; then
        DOCKER_CMD="$DOCKER_CMD --device=/dev/dri"
    fi
    # Mount kfd if available
    if [ -e "/dev/kfd" ]; then
        DOCKER_CMD="$DOCKER_CMD --device=/dev/kfd"
    fi
    # Mount render nodes individually if needed
    for dev in /dev/dri/renderD*; do
        if [ -e "$dev" ]; then
            DOCKER_CMD="$DOCKER_CMD --device=$dev"
        fi
    done
fi

# Set ROCm environment variables
DOCKER_CMD="$DOCKER_CMD -e ROCM_VISIBLE_DEVICES=0"
DOCKER_CMD="$DOCKER_CMD -e HIP_VISIBLE_DEVICES=0"
DOCKER_CMD="$DOCKER_CMD -e HSA_OVERRIDE_GFX_VERSION=10.3.0"

# Add security options for GPU access (needed on some systems)
DOCKER_CMD="$DOCKER_CMD --security-opt seccomp=unconfined"

# Set group for render node access
if [ -n "$(stat -c '%g' /dev/dri/renderD* 2>/dev/null | head -1)" ]; then
    RENDER_GID=$(stat -c '%g' /dev/dri/renderD* 2>/dev/null | head -1)
    DOCKER_CMD="$DOCKER_CMD --group-add $RENDER_GID"
fi

# Mount current directory (optional - adjust as needed)
DOCKER_CMD="$DOCKER_CMD -v \$(pwd):/workspace"

# Port mapping for API server
DOCKER_CMD="$DOCKER_CMD -p 8000:8000"

# Image name (adjust if different)
IMAGE_NAME="character-ovi-backend"

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "Image $IMAGE_NAME not found. Building image first..."
    docker build -t $IMAGE_NAME -f DOCKERFILE .
fi

# Add remaining arguments
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"

echo "Running command:"
echo "$DOCKER_CMD"
echo ""

# Execute docker run
eval $DOCKER_CMD

