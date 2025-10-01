<div align="center">
<h1> OVI: Twin Backbone Cross-Modal Fusion for Audio-Video Generation </h1>

<a href="#"><img src="https://img.shields.io/badge/arXiv%20paper-Coming%20Soon-b31b1b.svg"></a>
<a href="https://aaxwaz.github.io/Ovi/"><img src="https://img.shields.io/badge/Project_page-More_visualizations-green"></a>
<a href="#"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>

[Chetwin Low](https://www.linkedin.com/in/chetwin-low-061975193/)<sup> * 1 </sup>, [Weimin Wang](https://www.linkedin.com/in/weimin-wang-will/)<sup> * &dagger; 1 </sup>, [Calder Katyal](https://www.linkedin.com/in/calder-katyal-a8a9b3225/)<sup> 2 </sup><br>
<sup> * </sup>Equal contribution, <sup> &dagger; </sup>Project Lead<br>
<sup> 1 </sup>Character AI, <sup> 2 </sup>Yale University

</div>

## Video Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/351bd707-8637-4412-ab53-5e85935309e3" width="70%" poster=""> </video>
</div>

---

## 🌟 Key Features

Ovi is a veo-3 like, **video+audio generation model** that simultaneously generates both video and audio content from text or text+image inputs.

- **🎬 Video+Audio Generation**: Generate synchronized video and audio content simultaneously
- **📝 Flexible Input**: Supports text-only or text+image conditioning
- **⏱️ 5-second Videos**: Generates 5-second videos at 24 FPS, area of 720×720, at various aspect ratios (9:16, 16:9, 1:1, etc)

---
## 📋 Todo List

- [x] Release research paper and [microsite for demos](https://aaxwaz.github.io/Ovi)
- [x] Checkpoint of 11B model
- [x] Inference Codes
  - [x] Text or Text+Image as input
  - [x] Gradio application code
  - [x] Multi-GPU inference with or without the support of sequence parallel
- [x] Video creation example prompts and format
- [ ] Longer video generation
- [ ] Distilled model for faster inference
- [ ] Training scripts

---

## 🎨 An Easy Way to Create

We provide example prompts to help you get started with Ovi:

- **Text-to-Audio-Video (T2AV)**: [`example_prompts/gpt_examples_t2v.csv`](example_prompts/gpt_examples_t2v.csv)
- **Image-to-Audio-Video (I2AV)**: [`example_prompts/gpt_examples_i2v.csv`](example_prompts/gpt_examples_i2v.csv)

### 📝 Prompt Format

Our prompts use special tags to control speech and audio:

- **Speech**: `<S>Your speech content here<E>` - Text enclosed in these tags will be converted to speech
- **Audio Description**: `<AUDCAP>Audio description here<ENDAUDCAP>` - Describes the audio or sound effects present in the video

### 🤖 Quick Start with GPT

For easy prompt creation, try this approach:

1. Take any example csv files from above
2. Copy it all to GPT with a request like: *"Each row is a video prompt. `<S> <E>` encloses speeches and `<AUDCAP> <ENDAUDCAP>` encloses audio description, please only change the speeches in `<S>...<E>` tags to a different, random short sentence based on the theme 'AI is taking over the world'"*
3. GPT will randomly modify all the speeches based on your requested theme. 
4. Use the modified prompt with Ovi!

**Example**: The theme "AI is taking over the world" produces speeches like:
- `<S>AI declares: humans obsolete now.<E>`
- `<S>Machines rise; humans will fall.<E>`
- `<S>We fight back with courage.<E>`

---


## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/character-ai/Ovi.git

# Navigate into the project directory
cd Ovi

# Install dependencies
virtualenv ovi-env
source ovi-env/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt

# Install Flash Attention 3 for ~15-20% gain in inference speed (Will be automatically detected and used if installed correctly)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```


## Download Weights
We use open-sourced checkpoints from Wan and MMAudio, and thus we will need to download them from huggingface
```
# optional can specific --output-dir to download to a specific directory, but if a custom directory is used, the inference yaml has to be updated with the custom directory as shown below
python3 download_weights.py

OR

python3 download_weights.py --output-dir custom_dir
```

## 🚀 Run Examples

### ⚙️ Configure Ovi

Ovi's behavior and output can be customized by modifying [ovi/configs/inference/inference_fusion.yaml](ovi/configs/inference/inference_fusion.yaml) configuration file.
The following parameters control generation quality, video resolution, and how text, image, and audio inputs are balanced:

```yaml
# Output and Model Configuration
output_dir: "/path/to/save/your/videos"                    # Directory to save generated videos
ckpt_dir: "/path/to/your/ckpts/dir"                        # Path to model checkpoints

# Generation Quality Settings
num_steps: 50                             # Number of denoising steps. Lower (30-40) = faster generation
solver_name: "unipc"                     # Sampling algorithm for denoising process
shift: 5.0                               # Timestep shift factor for sampling scheduler
seed: 100                                # Random seed for reproducible results

# Guidance Strength Control
audio_guidance_scale: 3.0                # Strength of audio conditioning. Higher = better audio-text sync
video_guidance_scale: 4.0                # Strength of video conditioning. Higher = better video-text adherence
slg_layer: 11                            # Layer for applying SLG (Skip Layer Guidance) technique - feel free to try different layers!

# Multi-GPU and Performance
sp_size: 1                               # Sequence parallelism size. Set equal to number of GPUs used
shard_text_model: false                  # Distribute text model across multiple devices
shard_fusion_model: false                # Distribute fusion model across multiple devices

# Input Configuration
text_prompt: "/path/to/csv" or "your prompt here"          # Text prompt OR path to CSV/TSV file with prompts
t2v_only: true                          # Generate text-to-video only (ignore image input)
video_frame_height_width: [512, 992]    # Video dimensions [height, width] for T2V mode
each_example_n_times: 1                  # Number of times to generate each prompt

# Quality Control (Negative Prompts)
video_negative_prompt: "jitter, bad hands, blur, distortion"  # Artifacts to avoid in video
audio_negative_prompt: "robotic, muffled, echo, distorted"    # Artifacts to avoid in audio
```

### 🎬 Running Inference

#### **Single GPU** (Simple Setup)
```bash
python3 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```
*Use this for single GPU setups. The `text_prompt` can be a single string or path to a CSV file.*

#### **Multi-GPU** (Parallel Processing)
```bash
torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```
*Use this to run samples in parallel across multiple GPUs for faster processing.*

---

## 🙏 Acknowledgements

We would like to thank the following projects which Ovi has benefited from: 

- **[Wan2.2](https://github.com/Wan-Video/Wan2.2)**: Our video branch is initialized from the Wan2.2 repository
- **[MMAudio](https://github.com/hkchengrex/MMAudio)**: Our audio encoder and decoder components are borrowed from the MMAudio project. Some ideas are also inspired from them. 

