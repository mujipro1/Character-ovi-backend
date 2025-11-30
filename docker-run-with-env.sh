#!/bin/bash

# Quick script to run existing Docker container with AMD GPU environment variables
# Use this if you've already built the image and don't want to rebuild

IMAGE_NAME="character-ovi-backend"

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "Image $IMAGE_NAME not found. Please build it first or adjust IMAGE_NAME variable."
    exit 1
fi

echo "Running container with AMD GPU environment variables..."

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
  $IMAGE_NAME

