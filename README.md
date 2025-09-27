<div align="center">
<h1> OVI: Twin Backbone Cross-Modal Fusion for Audio-Video Generation </h1>

<a href="#"><img src="https://img.shields.io/badge/arXiv%20paper-Coming%20Soon-b31b1b.svg"></a>
<a href="https://aaxwaz.github.io/Ovi/"><img src="https://img.shields.io/badge/Project_page-More_visualizations-green"></a>
<a href="#"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>

[Chetwin Low](https://scholar.google.com/)<sup> * 1 </sup>, [Weimin Wang](https://scholar.google.com/)<sup> * &dagger; 1 </sup>, [Calder Katyal](https://scholar.google.com/)<sup> 2 </sup><br>
<sup> * </sup>Equal contribution, <sup> &dagger; </sup>Project lead<br>
<sup> 1 </sup>Character AI, <sup> 2 </sup>Yale University

</div>

<p align="center">
[Trailer video placeholder - coming soon]
<p>

---

## 🌟 Key Features

Ovi is a veo-3 like, **video+audio generation model** that simultaneously generates both video and audio content from text or text+image inputs.

- **🎬 Video+Audio Generation**: Generate synchronized video and audio content simultaneously
- **📝 Flexible Input**: Supports text-only or text+image conditioning
- **⏱️ 5-second Videos**: Generates 5-second videos at 24 FPS
- **📐 Multiple Aspect Ratios**: 720×720 total area with various aspect ratios (9:16, 16:9, 1:1)
- **🏗️ Twin Backbone Architecture**:
  - Video branch based on WAN 2.2 5B text-to-image-to-video model
  - Symmetric audio branch of equivalent size (5B parameters)
  - Cross-modal fusion model
  - **Total: 11B parameters**

---

## 📋 Todo List

- [x] Release research paper and microsite for demos
- [x] Checkpoint of 11B model 
- [x] Inference Codes
  - [x] Text or Text+Image as input
  - [x] Gradio application code
  - [x] Multi-GPU inference with or without the support of sequence parallel
- [x] A fun way of video creations, and example prompts. 
- [ ] Longer video generation
- [ ] Distilled model for faster inference
- [ ] Training scripts

---

## 🚀 Features
- Feature 1
- Feature 2
- Feature 3

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

## Run Examples
Inference parameters are controlled via a yaml file, for example `ovi/configs/inference/inference_fusion.yaml`
```
output_dir: "<path to save generated outputs>"  # default: ./outputs
ckpt_dir: "<path to model checkpoints>"         # default: ./ckpts

num_steps: "<number of sampling steps>"         # default: 50
solver_name: "<sampler/solver algorithm>"       # default: unipc
shift: "<timestep shift factor>"                # default: 5.0
sp_size: "<sequence parallel size>"            # default: 1

audio_guidance_scale: "<strength of audio conditioning>"  # default: 3.0
video_guidance_scale: "<strength of video conditioning>"  # default: 4.0

shard_text_model: "<whether to shard text model across devices>"    # default: False
shard_fusion_model: "<whether to shard fusion model across devices>" # default: False

video_negative_prompt: "<undesired artifacts to avoid in video>"    # default: common artifacts (jitter, bad hands, blur, etc.)
audio_negative_prompt: "<undesired artifacts to avoid in audio>"    # default: common artifacts (robotic, muffled, echo, etc.)

seed: "<random seed for reproducibility>"             # default: 100
aspect_ratio: "<video aspect ratio>, please put in double quotations"  # default: "9:16"  (choices: ["9:16", "16:9", "1:1"])

text_prompt: "<either raw text prompt or path to TSV/CSV with prompts>, TSV/CSV files should have two columns, "text_prompt" and "image_path""
t2v_only: "<generate only text-to-video (ignore first frame even if provided)>"             # default: True
slg_layer: "<which layer to apply SLG guidance>"                     # default: 9
```

### a. single process (for a single GPU, text_prompt can be a single string input or a csv file containing prompts and images)
```
python3 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```

### b. multiprocess (for multiple GPUs, where we can run samples in parallel)
```
torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```




