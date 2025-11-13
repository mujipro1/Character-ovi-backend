## Installation

### Prerequisites
- NVIDIA GPU with matching drivers, CUDA toolkit, and cuDNN compatible with your PyTorch build.
- FFmpeg installed and accessible on your `PATH`.

### Steps
1. Clone the repository  
   ```bash
   git clone https://github.com/character-ai/ovi.git
   cd ovi
   ```
2. (Recommended) Create and activate a Python 3.10 environment  
   ```bash
   conda create -y -n ovi python=3.10
   conda activate ovi
   ```
3. Install Python dependencies  
   ```bash
   pip install -r requirements.txt
   ```
4. Download model checkpoints  
   ```bash
   python download_weights.py --save-path ./ckpts
   ```
5. Start the FastAPI server (example)  
   ```bash
   python api_server.py --config ovi/configs/inference/inference_fusion.yaml
   ```

