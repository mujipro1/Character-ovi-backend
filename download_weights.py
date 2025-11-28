import os
import argparse
import logging
import time
from huggingface_hub import snapshot_download


NAME_TO_MODELS_MAP = {
    "720x720_5s": "model.safetensors",
    "960x960_5s": "model_960x960.safetensors",
    "960x960_10s": "model_960x960_10s.safetensors"
}

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def timed_download(repo_id: str, local_dir: str, allow_patterns: list):
    """Download files from HF repo and log time + destination."""
    logging.info(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    elapsed = time.time() - start_time
    logging.info(
        f"✅ Finished downloading {repo_id} "
        f"in {elapsed:.2f} seconds. Files saved at: {local_dir}"
    )

def main(output_dir: str, download_ovi_models: bool = False, ovi_model_names: list = None):
    # Wan2.2
    wan_dir = os.path.join(output_dir, "Wan2.2-TI2V-5B")
    timed_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=wan_dir,
        allow_patterns=[
            "google/*",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.2_VAE.pth"
        ]
    )

    # MMAudio
    mm_audio_dir = os.path.join(output_dir, "MMAudio")
    timed_download(
        repo_id="hkchengrex/MMAudio",
        local_dir=mm_audio_dir,
        allow_patterns=[
            "ext_weights/best_netG.pt",
            "ext_weights/v1-16.pth"
        ]
    )

    # Ovi FP8 model (the one actually used, not the ones that get deleted)
    ovi_dir = os.path.join(output_dir, "Ovi")
    os.makedirs(ovi_dir, exist_ok=True)
    timed_download(
        repo_id="rkfg/Ovi-fp8_quantized",
        local_dir=ovi_dir,
        allow_patterns=["model_fp8_e4m3fn.safetensors"]
    )

    # Only download the old Ovi models if explicitly requested (for backward compatibility)
    if download_ovi_models:
        if ovi_model_names is None:
            ovi_model_names = ["720x720_5s", "960x960_5s", "960x960_10s"]
        assert all(m in NAME_TO_MODELS_MAP for m in ovi_model_names), f"Invalid model names {ovi_model_names}. Valid options are: {list(NAME_TO_MODELS_MAP.keys())}"
        models = [NAME_TO_MODELS_MAP[m] for m in ovi_model_names]
        timed_download(
            repo_id="chetwinlow1/Ovi",
            local_dir=ovi_dir,
            allow_patterns=models
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ckpts",
        help="Base directory to save downloaded models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["720x720_5s", "960x960_5s", "960x960_10s"],
        help="Ovi model names to download (only used with --download-ovi-models)"
    )
    parser.add_argument(
        "--download-ovi-models",
        action="store_true",
        help="Download the old Ovi models (model.safetensors, model_960x960.safetensors, etc.) that are normally deleted"
    )
    args = parser.parse_args()
    main(args.output_dir, download_ovi_models=args.download_ovi_models, ovi_model_names=args.models)