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
- **⏱️ 5-second Videos**: Generates 5-second videos at 24 FPS, area of 720×720, at various aspect ratios (9:16, 16:9, 1:1, etc)

---
## 📋 Todo List

- [x] Release research paper and microsite for demos
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
video_frame_height_width: "<T2V inference height & width>, will only work for T2V

each_example_n_times: number of times to run inference for each prompt # default: 1 

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




