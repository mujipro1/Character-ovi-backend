# 🎥 OVI Powered Video Generation Backend

Original Work: https://www.github.com/character-ai/ovi

**OVI (Omni Video Intelligence)** is a **GPU-accelerated backend service** for **AI-driven video generation and video inpainting**, powered by deep learning models and exposed through a **FastAPI** interface.

This repository serves as the **backend engine** responsible for generating short, high-quality videos from text prompts, audio guidance, and optional visual references.

---

## 🧠 Overview

OVI enables:
- Text-to-video generation
- Audio-guided video synthesis
- Reference-conditioned video creation
- Localized video inpainting using bounding boxes

It is designed to be:
✔ Modular  
✔ Scalable  
✔ API-driven  
✔ GPU-optimized  

This backend can be integrated with **web apps, creative tools, or character-based AI systems**.

---

## ✨ Key Features

🎬 Text-to-Video generation  
🎧 Optional audio-guided synthesis  
🖼 Reference image / video conditioning  
✏ Video inpainting with bounding-box localization  
⚡ FastAPI-based REST interface  
🚀 NVIDIA CUDA acceleration  

---

## 🛠️ Tech Stack

### Backend
- Python 3.10
- FastAPI
- PyTorch
- CUDA + cuDNN

### Media Processing
- FFmpeg

### Model Handling
- Pretrained diffusion/video generation models
- YAML-based inference configuration

---

## 📁 Project Structure

```

📦 Character-ovi-backend
┣ 📂 api
┣ 📂 ckpts
┣ 📂 ovi
┃ ┗ 📂 configs
┃   ┗ 📂 inference
┣ 📜 api_server.py
┣ 📜 download_weights.py
┣ 📜 requirements.txt
┣ 📜 README.md

````

---

## 🚀 Installation

### 🔧 Prerequisites

- NVIDIA GPU
- Matching NVIDIA drivers
- CUDA Toolkit + cuDNN (compatible with PyTorch)
- FFmpeg installed and available in `PATH`
- Python 3.10

---

### 📥 Setup Steps

1. Clone the repository
   ```bash
   git clone https://github.com/mujipro1/Character-ovi-backend.git
   cd ovi
   ```

2. (Recommended) Create and activate a Python environment

   ```bash
   conda create -y -n ovi python=3.10
   conda activate ovi
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Download model checkpoints

   ```bash
   python download_weights.py --save-path ./ckpts
   ```

5. Start the FastAPI server

   ```bash
   python api_server.py --config ovi/configs/inference/inference_fusion.yaml
   ```

📌 **Note:**
Ensure `video_frame_height_width` in the config matches the active model.
For:

```yaml
model_name: "720x720_5s"
```

Use:

```yaml
video_frame_height_width: [720, 720]
```

(This is already set in the default config.)

---

## 🌐 API Documentation

### ▶ `POST /generate_video`

Generate a new video from prompts and optional references.

**Content-Type:** `multipart/form-data`

#### Parameters

| Field          | Type   | Required | Description                                     |
| -------------- | ------ | -------- | ----------------------------------------------- |
| `video_prompt` | string | ✅        | Visual description of the scene                 |
| `audio_prompt` | string | ❌        | Audio guidance (wrapped as `Audio:` internally) |
| `video_length` | float  | ❌        | Duration in seconds (default: `5.0`)            |
| `reference`    | file   | ❌        | Image or video used for conditioning            |

**Response:**
🎞 MP4 video with synchronized audio

---

### ▶ `POST /inpaint_video`

Modify or regenerate specific regions of an existing video.

**Content-Type:** `multipart/form-data`

#### Parameters

| Field             | Type   | Required | Description                                   |
| ----------------- | ------ | -------- | --------------------------------------------- |
| `video_prompt`    | string | ✅        | Instructions for inpainting                   |
| `audio_prompt`    | string | ❌        | Optional audio guidance                       |
| `generated_video` | file   | ✅        | Input video to modify                         |
| `frame`           | file   | ❌        | Image frame with bounding box (red preferred) |

📌 **Behavior**

* If `frame` is provided, bounding box is detected automatically.
* If omitted, the first frame of the video is sampled.
* Bounding-box coordinates are injected into the prompt to keep edits localized.

**Response:**
🎞 Regenerated MP4 video with inpainted content

---

## 🧩 Use Cases

* AI character animation
* Short cinematic content generation
* Video editing with localized modifications
* Creative AI pipelines
* Research & experimentation with video diffusion models

---

## 🚧 Limitations

⚠ High GPU VRAM usage
⚠ Short-duration video focus
⚠ Inference speed depends on hardware

---

## 🔮 Future Improvements

* Longer video generation
* Multi-character control
* Temporal consistency improvements
* Web frontend integration
* Real-time streaming inference

---

## 📄 License

This project is intended for **research and development purposes**.
Please ensure compliance with model licenses and dataset usage terms.

---

🎬 *Turning prompts into motion*

```
