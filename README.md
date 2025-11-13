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

## API Endpoints

### `POST /generate_video`
- **Content-Type:** `multipart/form-data`
- **Fields**
  - `video_prompt` (string, required): Visual prompt that describes the scene or edits for the generated video.
  - `audio_prompt` (string, optional): Audio-specific instructions; automatically wrapped as an `Audio:` clause when sent to the model.
  - `video_length` (float, optional, default `5.0`): Target duration in seconds.
  - `reference` (file, optional): Reference image (`.png`, `.jpg`, etc.) or video (`.mp4`, `.mov`, etc.) used for conditioning or first-frame guidance.
- **Response:** MP4 file containing the generated video with audio.

### `POST /inpaint_video`
- **Content-Type:** `multipart/form-data`
- **Fields**
  - `video_prompt` (string, required): Visual instructions describing the desired inpainted result.
  - `audio_prompt` (string, optional): Complementary audio guidance for the regenerated clip.
  - `generated_video` (file, required): Existing video to modify.
  - `frame` (file, optional): Image frame with a clearly drawn bounding box (preferably in red) highlighting the region to modify. If omitted, the first frame of the provided video is sampled and used; edits will focus on the detected box if present, otherwise a generic inpainting instruction is applied.
- **Response:** MP4 file containing the regenerated, inpainted video. Bounding-box coordinates are detected automatically and injected into the prompt to keep edits localized.

