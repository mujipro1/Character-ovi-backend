import subprocess
import time
import os

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

# ---------------------------------------------------
# 1. Ensure Docker Desktop is running
# ---------------------------------------------------
if not docker_running():
    subprocess.Popen([
        r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(7)

# Wait for docker to become ready
for _ in range(15):
    if docker_running():
        break
    time.sleep(1)

# ---------------------------------------------------
# 2. Behavior logic: run once, never conflicts
# ---------------------------------------------------

# If container is already running → do nothing
if container_running():
    # Nothing else needed
    raise SystemExit(0)

# If container exists but is stopped → start it
if container_exists():
    subprocess.Popen([
        "docker", "start", CONTAINER_NAME
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raise SystemExit(0)

# If container does not exist → create + run it
subprocess.Popen([
    "docker", "run",
    "--gpus", "all",
    "-d",                                        # detached mode
    "--restart=always",                          # auto-start at reboot
    "-p", f"{HOST_PORT}:{CONTAINER_PORT}",
    "--name", CONTAINER_NAME,
    IMAGE_NAME
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
