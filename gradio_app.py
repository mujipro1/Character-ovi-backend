import gradio as gr
from ovi.ovi_fusion_engine import OviFusionEngine
import tempfile
from ovi.utils.io_utils import save_video

# Initialize OviFusionEngine
print("loading model...")
ovi_engine = OviFusionEngine()
print("loaded model")


def generate_video(
    text_prompt,
    image,
    video_frame_height,
    video_frame_width,
    seed,
    solver_name,
    sample_steps,
    shift,
    video_guidance_scale,
    audio_guidance_scale,
    slg_layer,
    video_negative_prompt,
    audio_negative_prompt,
):
    try:
        # image from gr.Image comes as numpy array or file; adapt as needed
        image_path = None
        if image is not None:
            image_path = image  # if ovi_engine expects path, you may need to save temp file
        
        generated_video, generated_audio = ovi_engine.generate(
            text_prompt=text_prompt,
            image_path=image_path,
            video_frame_height_width=[video_frame_height, video_frame_width],
            seed=seed,
            solver_name=solver_name,
            sample_steps=sample_steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
        )
        
        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = tmpfile.name
        save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
        
        return  output_path
    except Exception as e:
        print(f"Error during video generation: {e}")
        return None



# Build UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Ovi Joint Video Audio Generation Demo")

    with gr.Row():
        with gr.Column():
            text_prompt = gr.Textbox(label="Text Prompt", placeholder="Describe your video...")
            image = gr.Image(type="filepath", label="First Frame Image (optional)")

            video_height = gr.Number(minimum=128, maximum=1280, value=512, step=32, label="Video Height")
            video_width = gr.Number(minimum=128, maximum=1280, value=992, step=32, label="Video Width")

            seed = gr.Number(minimum=0, maximum=100000, value=100, label="Seed")
            solver_name = gr.Dropdown(
                choices=["unipc", "euler", "dpm++"], value="unipc", label="Solver Name"
            )
            sample_steps = gr.Number(
                value=50,
                label="Sample Steps",
                precision=0,
                minimum=20,
                maximum=100
            )
            shift = gr.Slider(minimum=0.0, maximum=20.0, value=5.0, step=1.0, label="Shift")
            video_guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=4.0, step=0.5, label="Video Guidance Scale")
            audio_guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, step=0.5, label="Audio Guidance Scale")
            slg_layer = gr.Number(minimum=-1, maximum=30, value=11, step=1, label="SLG Layer")
            video_negative_prompt = gr.Textbox(label="Video Negative Prompt", placeholder="Things to avoid in video")
            audio_negative_prompt = gr.Textbox(label="Audio Negative Prompt", placeholder="Things to avoid in audio")

            run_btn = gr.Button("Generate 🚀")

        with gr.Column():
            output_path = gr.Video(label="Generated Video")

    run_btn.click(
        fn=generate_video,
        inputs=[
            text_prompt, image, video_height, video_width, seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, video_negative_prompt, audio_negative_prompt,
        ],
        outputs=[output_path],
    )

if __name__ == "__main__":
    demo.launch(share=True)