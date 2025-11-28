import subprocess
import time
import os
import sys

CONTAINER_NAME = "ovi-backend"
IMAGE_NAME = "ovi-backend"
HOST_PORT = "8001"
CONTAINER_PORT = "8001"

def docker_running():
    """Check if Docker Desktop is running."""
    try:
        subprocess.check_output(["docker", "info"], stderr=subprocess.STDOUT)
        return True
    except:
        return False

def image_exists():
    """Check if Docker image exists."""
    result = subprocess.run(
        ["docker", "images", "-q", IMAGE_NAME],
        capture_output=True, text=True
    )
    return result.stdout.strip() != ""

def container_exists():
    """Check if container exists (running or stopped)."""
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "-f", f"name={CONTAINER_NAME}"],
        capture_output=True, text=True
    )
    return result.stdout.strip() != ""

def container_running():
    """Check if container is already running."""
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"],
        capture_output=True, text=True
    )
    return result.stdout.strip() != ""

def gpu_available():
    """Check if GPU support is available in Docker."""
    # First check if nvidia-smi is available on the host (NVIDIA GPU)
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # NVIDIA GPU detected on host, now check Docker support
            try:
                # Try to run a test container with GPU support
                test_result = subprocess.run(
                    ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.0-base", "nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return test_result.returncode == 0
            except:
                return False
    except:
        pass
    
    # Check for AMD GPU (ROCm) - check if rocm-smi exists
    try:
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # AMD GPU detected, but Docker GPU support for AMD is more complex
            # Check if Docker has device access configured
            # For now, we'll be conservative and return False unless explicitly configured
            # The user can manually enable if they have ROCm Docker setup
            return False
    except:
        pass
    
    # No GPU detected or Docker GPU support not configured
    return False

def build_image():
    """Build the Docker image from Dockerfile."""
    print("=" * 60)
    print("Building Docker image...")
    print("This may take several minutes on first run.")
    print("The image includes all model weights and dependencies.")
    print("After first build, restarts will be fast (no re-downloading).")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dockerfile_path = os.path.join(script_dir, "DOCKERFILE")
    
    if not os.path.exists(dockerfile_path):
        print(f"ERROR: Dockerfile not found at {dockerfile_path}")
        sys.exit(1)
    
    # Build the image
    build_cmd = [
        "docker", "build",
        "-f", dockerfile_path,
        "-t", IMAGE_NAME,
        script_dir
    ]
    
    print(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, text=True)
    
    if result.returncode != 0:
        print("ERROR: Failed to build Docker image")
        sys.exit(1)
    
    print("✓ Docker image built successfully!")
    print("=" * 60)

# ---------------------------------------------------
# 1. Ensure Docker Desktop is running
# ---------------------------------------------------
print("Checking Docker Desktop...")
if not docker_running():
    print("Starting Docker Desktop...")
    subprocess.Popen([
        r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(7)

# Wait for docker to become ready
print("Waiting for Docker to be ready...")
for i in range(30):
    if docker_running():
        print("✓ Docker is ready!")
        break
    time.sleep(1)
    if i == 29:
        print("ERROR: Docker did not start. Please start Docker Desktop manually.")
        sys.exit(1)

# ---------------------------------------------------
# 2. Check if Docker image exists, build if not
# ---------------------------------------------------
print("\nChecking Docker image...")
if not image_exists():
    print(f"Image '{IMAGE_NAME}' not found. Building from Dockerfile...")
    build_image()
else:
    print(f"✓ Image '{IMAGE_NAME}' already exists")

# ---------------------------------------------------
# 3. Behavior logic: run once, never conflicts
# ---------------------------------------------------
print("\nChecking container status...")

# If container is already running → show status and exit
if container_running():
    print(f"✓ Container '{CONTAINER_NAME}' is already running!")
    print(f"\nAPI Server is available at: http://localhost:{HOST_PORT}")
    print(f"API Documentation: http://localhost:{HOST_PORT}/docs")
    print("\nTo view logs: docker logs -f ovi-backend")
    print("To stop: docker stop ovi-backend")
    sys.exit(0)

# If container exists but is stopped → start it
if container_exists():
    print(f"Container '{CONTAINER_NAME}' exists but is stopped. Starting...")
    result = subprocess.run(["docker", "start", CONTAINER_NAME], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Container '{CONTAINER_NAME}' started successfully!")
        print(f"\nWaiting for API server to start...")
        time.sleep(5)
        print(f"\nAPI Server is available at: http://localhost:{HOST_PORT}")
        print(f"API Documentation: http://localhost:{HOST_PORT}/docs")
        print("\nTo view logs: docker logs -f ovi-backend")
        print("To stop: docker stop ovi-backend")
    else:
        print(f"ERROR: Failed to start container: {result.stderr}")
        sys.exit(1)
    sys.exit(0)

# If container does not exist → create + run it
print(f"Creating and starting container '{CONTAINER_NAME}'...")

# Check if GPU is available
print("Checking GPU support...")
has_gpu = gpu_available()

if has_gpu:
    print("✓ GPU support detected. Using GPU acceleration.")
    docker_cmd = [
        "docker", "run",
        "--gpus", "all",
        "-d",                                        # detached mode
        "--restart=always",                          # auto-start at reboot
        "-p", f"{HOST_PORT}:{CONTAINER_PORT}",
        "--name", CONTAINER_NAME,
        IMAGE_NAME
    ]
else:
    print("⚠ GPU support not available. Running in CPU mode (slower but will work).")
    docker_cmd = [
        "docker", "run",
        "-d",                                        # detached mode
        "--restart=always",                          # auto-start at reboot
        "-p", f"{HOST_PORT}:{CONTAINER_PORT}",
        "--name", CONTAINER_NAME,
        IMAGE_NAME
    ]

result = subprocess.run(docker_cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✓ Container '{CONTAINER_NAME}' created and started!")
    print(f"\nWaiting for API server to start (this may take a minute)...")
    print("Loading models and starting server...")
    time.sleep(10)
    print(f"\n✓ API Server should be available at: http://localhost:{HOST_PORT}")
    print(f"✓ API Documentation: http://localhost:{HOST_PORT}/docs")
    print("\nTo view logs: docker logs -f ovi-backend")
    print("To stop: docker stop ovi-backend")
    if not has_gpu:
        print("\n⚠ Running in CPU mode. Performance will be slower than GPU mode.")
    print("\nNote: First request may take longer as models are loaded into memory.")
else:
    print(f"ERROR: Failed to create container: {result.stderr}")
    
    # If it failed with GPU, try without GPU as fallback
    if "--gpus" in ' '.join(docker_cmd) and "gpu" in result.stderr.lower():
        print("\n⚠ GPU mode failed. Retrying in CPU mode...")
        docker_cmd_cpu = [
            "docker", "run",
            "-d",
            "--restart=always",
            "-p", f"{HOST_PORT}:{CONTAINER_PORT}",
            "--name", CONTAINER_NAME,
            IMAGE_NAME
        ]
        result_cpu = subprocess.run(docker_cmd_cpu, capture_output=True, text=True)
        
        if result_cpu.returncode == 0:
            print(f"✓ Container '{CONTAINER_NAME}' created and started in CPU mode!")
            print(f"\nWaiting for API server to start...")
            time.sleep(10)
            print(f"\n✓ API Server should be available at: http://localhost:{HOST_PORT}")
            print(f"✓ API Documentation: http://localhost:{HOST_PORT}/docs")
            print("\n⚠ Running in CPU mode. Performance will be slower than GPU mode.")
            print("\nTo view logs: docker logs -f ovi-backend")
            print("To stop: docker stop ovi-backend")
        else:
            print(f"ERROR: Failed to create container even in CPU mode: {result_cpu.stderr}")
            sys.exit(1)
    else:
        sys.exit(1)
