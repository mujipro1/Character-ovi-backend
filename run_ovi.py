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

def build_image():
    """Build the Docker image from Dockerfile."""
    print("=" * 60)
    print("Building Docker image...")
    print("This may take several minutes on first run.")
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
result = subprocess.run([
    "docker", "run",
    "--gpus", "all",
    "-d",                                        # detached mode
    "--restart=always",                          # auto-start at reboot
    "-p", f"{HOST_PORT}:{CONTAINER_PORT}",
    "--name", CONTAINER_NAME,
    IMAGE_NAME
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"✓ Container '{CONTAINER_NAME}' created and started!")
    print(f"\nWaiting for API server to start (this may take a minute)...")
    print("Loading models and starting server...")
    time.sleep(10)
    print(f"\n✓ API Server should be available at: http://localhost:{HOST_PORT}")
    print(f"✓ API Documentation: http://localhost:{HOST_PORT}/docs")
    print("\nTo view logs: docker logs -f ovi-backend")
    print("To stop: docker stop ovi-backend")
    print("\nNote: First request may take longer as models are loaded into memory.")
else:
    print(f"ERROR: Failed to create container: {result.stderr}")
    if "gpus" in result.stderr.lower():
        print("\nNote: If you don't have a GPU, remove '--gpus all' from the docker run command.")
    sys.exit(1)
