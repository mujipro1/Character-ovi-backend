# Client Package - What to Include

## Essential Files to Give to Client

### 1. **Main Files (Required)**
- `run_ovi.py` - The main script to run everything
- `DOCKERFILE` - Docker build configuration
- `.dockerignore` - Excludes unnecessary files from Docker build

### 2. **Source Code Files (Required)**
- `api_server.py` - Main API server with all your custom modifications
- `modify_prompt.py` - Prompt enhancement module
- `system_prompt.py` - System prompt configuration
- `download_weights.py` - Script to download model weights
- `requirements.txt` - Python dependencies

### 3. **Project Structure (Required)**
- `ovi/` - Entire directory with all modules
  - This includes all the model code, configs, etc.
- `ovi/configs/` - Configuration files

### 4. **Optional Files**
- `README.md` - Documentation (if you have one)
- `LICENSE` - License file

## What NOT to Include
- `build/` directory
- `dist/` directory  
- `__pycache__/` directories
- `.git/` directory (unless client needs version control)
- `*.pyc`, `*.pyo` files
- IDE files (`.vscode/`, `.idea/`)

## How Client Should Use It

1. **Extract the package** to a folder
2. **Open terminal/command prompt** in that folder
3. **Run**: `python run_ovi.py`
4. **Wait** for Docker image to build (first time only, 10-20 minutes)
5. **Access API** at `http://localhost:8001/docs`

## Notes

- The `.dockerignore` file ensures only necessary files are copied into Docker
- All model weights will be downloaded automatically during Docker build
- The script handles GPU/CPU detection automatically
- After first build, restarts are fast (no re-downloading)

