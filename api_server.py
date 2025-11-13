import argparse
import logging
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf

from ovi.ovi_fusion_engine import NAME_TO_MODEL_SPECS_MAP, OviFusionEngine
from ovi.utils.io_utils import save_video


logger = logging.getLogger("api")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_FPS = 24
DEFAULT_SAMPLE_RATE = 16000


def _ensure_cuda_device(device_index: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run the OVI FP8 quantized model.")
    device_index = max(device_index, 0)
    torch.cuda.set_device(device_index)
    return device_index


class VideoGenerationService:
    def __init__(self, config: DictConfig, device_index: int = 0):
        self._lock = threading.Lock()
        self._base_config = config
        self._device_index = _ensure_cuda_device(device_index)
        self._fps = DEFAULT_FPS
        self._sample_rate = DEFAULT_SAMPLE_RATE

        model_name = config.get("model_name", "720x720_5s")
        if model_name in NAME_TO_MODEL_SPECS_MAP:
            if "10s" in model_name:
                self._segment_duration = 10.0
            else:
                self._segment_duration = 5.0
        else:
            self._segment_duration = 5.0

        output_dir = config.get("output_dir", "./outputs")
        self._output_dir = Path(output_dir) / "api"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        target_dtype = torch.bfloat16
        logger.info("Loading OviFusionEngine for API service...")
        self._engine = OviFusionEngine(config=config, device=self._device_index, target_dtype=target_dtype)
        logger.info("OviFusionEngine loaded successfully.")

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.reshape(1, -1)
        if audio.ndim == 2:
            if audio.shape[0] <= audio.shape[1]:
                return audio
            return audio.T
        return audio.reshape(1, -1)

    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    @staticmethod
    def _is_video(path: Path) -> bool:
        return path.suffix.lower() in VIDEO_EXTENSIONS

    def _extract_reference_frame(self, video_path: Path) -> Path:
        capture = cv2.VideoCapture(str(video_path))
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            raise RuntimeError("Unable to extract reference frame from provided video.")
        temp_file = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
        cv2.imwrite(str(temp_file), frame)
        return temp_file

    def _extract_first_frame(self, video_path: Path) -> Path:
        return self._extract_reference_frame(video_path)

    def _prepare_reference(self, reference_path: Optional[Path]) -> Tuple[Optional[Path], List[Path]]:
        if reference_path is None:
            return None, []
        cleanup: List[Path] = []
        if self._is_video(reference_path):
            frame_path = self._extract_reference_frame(reference_path)
            cleanup.append(frame_path)
            return frame_path, cleanup
        if self._is_image(reference_path):
            return reference_path, cleanup
        raise HTTPException(status_code=400, detail="Unsupported reference file type. Provide an image or video.")

    def _generate_segments(
        self,
        prompt: str,
        target_frames: int,
        reference_path: Optional[Path],
        base_seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        combined_video: Optional[np.ndarray] = None
        combined_audio: Optional[np.ndarray] = None
        total_frames = 0
        total_samples = 0
        segment_index = 0

        cfg = self._base_config
        video_hw = cfg.get("video_frame_height_width")
        solver_name = cfg.get("solver_name", "unipc")
        sample_steps = cfg.get("sample_steps", 50)
        shift = cfg.get("shift", 5.0)
        video_guidance_scale = cfg.get("video_guidance_scale", 4.0)
        audio_guidance_scale = cfg.get("audio_guidance_scale", 3.0)
        slg_layer = cfg.get("slg_layer", 11)
        video_negative_prompt = cfg.get("video_negative_prompt", "")
        audio_negative_prompt = cfg.get("audio_negative_prompt", "")

        while total_frames < target_frames:
            seed = base_seed + segment_index
            generated_video, generated_audio, _ = self._engine.generate(
                text_prompt=prompt,
                image_path=str(reference_path) if reference_path else None,
                video_frame_height_width=video_hw,
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

            if generated_video is None:
                raise RuntimeError("Video generation failed.")

            if combined_video is None:
                combined_video = generated_video
            else:
                combined_video = np.concatenate([combined_video, generated_video], axis=1)

            total_frames = combined_video.shape[1]

            if generated_audio is not None:
                normalized_audio = self._normalize_audio(generated_audio)
                if combined_audio is None:
                    combined_audio = normalized_audio
                else:
                    combined_audio = np.concatenate([combined_audio, normalized_audio], axis=1)
                total_samples = combined_audio.shape[1]
            segment_index += 1

            if segment_index > 16:
                raise RuntimeError("Requested duration is too long for automated tiling.")

        if combined_video.shape[1] > target_frames:
            combined_video = combined_video[:, :target_frames, :, :]

        if combined_audio is not None:
            target_samples = math.ceil((target_frames / self._fps) * self._sample_rate)
            if combined_audio.shape[1] > target_samples:
                combined_audio = combined_audio[:, :target_samples]
            elif combined_audio.shape[1] < target_samples:
                pad_width = target_samples - combined_audio.shape[1]
                combined_audio = np.pad(combined_audio, ((0, 0), (0, pad_width)), mode="edge")

            if combined_audio.shape[0] == 1:
                combined_audio = combined_audio.squeeze(0)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return combined_video, combined_audio

    def generate_video(
        self,
        prompt: str,
        video_length: float,
        reference_path: Optional[Path],
    ) -> Tuple[Path, List[Path]]:
        if video_length <= 0:
            raise HTTPException(status_code=400, detail="Video length must be greater than zero.")

        prepared_reference, cleanup_paths = self._prepare_reference(reference_path)
        target_frames = max(int(video_length * self._fps), self._fps)
        base_seed = self._base_config.get("seed", 100)

        with self._lock:
            combined_video, combined_audio = self._generate_segments(
                prompt=prompt,
                target_frames=target_frames,
                reference_path=prepared_reference,
                base_seed=base_seed,
            )

            stem = "".join(ch for ch in prompt[:24] if ch.isalnum() or ch in ("-", "_"))
            if not stem:
                stem = "video"
            unique_id = torch.randint(0, 10_000, (1,)).item()
            output_path = self._output_dir / f"{stem}_{os.getpid()}_{unique_id}.mp4"
            save_video(
                output_path=str(output_path),
                video_numpy=combined_video,
                audio_numpy=combined_audio,
                fps=self._fps,
                sample_rate=self._sample_rate,
            )

        return output_path, cleanup_paths

    def inpaint_video(
        self,
        prompt: str,
        source_video: Path,
        frame_path: Optional[Path],
    ) -> Tuple[Path, List[Path]]:
        temp_paths: List[Path] = []
        duration = self._determine_video_duration(source_video)

        reference = frame_path
        if reference is None:
            reference = self._extract_first_frame(source_video)
            temp_paths.append(reference)

        output_path, cleanup = self.generate_video(
            prompt=prompt,
            video_length=duration,
            reference_path=reference,
        )
        temp_paths.extend(cleanup)
        return output_path, temp_paths

    def _determine_video_duration(self, video_path: Path) -> float:
        with VideoFileClip(str(video_path)) as clip:
            return max(clip.duration, self._segment_duration)

    @staticmethod
    def safe_delete(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:
            logger.warning("Failed to delete %s: %s", path, exc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OVI FastAPI server")
    parser.add_argument("--config", type=str, default="ovi/configs/inference/inference_fusion.yaml", help="Path to inference configuration YAML.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server.")
    parser.add_argument("--device-index", type=int, default=0, help="GPU device index to use.")
    return parser.parse_args()


def create_app(config_path: str, device_index: int) -> FastAPI:
    config = OmegaConf.load(config_path)
    service = VideoGenerationService(config=config, device_index=device_index)
    app = FastAPI(title="OVI Video Generation API")

    @app.post("/generate_video")
    async def generate_video_endpoint(
        background_tasks: BackgroundTasks,
        prompt: str = Form(...),
        video_length: float = Form(5.0),
        reference: Optional[UploadFile] = File(None),
    ):
        temp_paths: List[Path] = []

        try:
            reference_path: Optional[Path] = None
            if reference is not None:
                reference_path = Path(tempfile.NamedTemporaryFile(suffix=Path(reference.filename or '').suffix or ".dat", delete=False).name)
                with reference_path.open("wb") as buffer:
                    buffer.write(await reference.read())
                temp_paths.append(reference_path)

            output_path, additional_cleanup = await run_in_threadpool(
                service.generate_video,
                prompt,
                float(video_length),
                reference_path,
            )
            temp_paths.extend(additional_cleanup)

        except HTTPException as exc:
            for path in temp_paths:
                service.safe_delete(path)
            raise exc
        except Exception as exc:
            for path in temp_paths:
                service.safe_delete(path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        unique_cleanup = {path for path in temp_paths if path is not None}
        for path in unique_cleanup:
            if path != output_path:
                background_tasks.add_task(service.safe_delete, path)
        background_tasks.add_task(service.safe_delete, output_path)
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=output_path.name,
            background=background_tasks,
        )

    @app.post("/inpaint_video")
    async def inpaint_video_endpoint(
        background_tasks: BackgroundTasks,
        prompt: str = Form(...),
        generated_video: UploadFile = File(...),
        frame: Optional[UploadFile] = File(None),
    ):
        temp_paths: List[Path] = []
        try:
            source_video_path = Path(tempfile.NamedTemporaryFile(suffix=Path(generated_video.filename or '').suffix or ".mp4", delete=False).name)
            with source_video_path.open("wb") as buffer:
                buffer.write(await generated_video.read())
            temp_paths.append(source_video_path)

            frame_path: Optional[Path] = None
            if frame is not None:
                frame_path = Path(tempfile.NamedTemporaryFile(suffix=Path(frame.filename or '').suffix or ".png", delete=False).name)
                with frame_path.open("wb") as buffer:
                    buffer.write(await frame.read())
                temp_paths.append(frame_path)

            output_path, additional_cleanup = await run_in_threadpool(
                service.inpaint_video,
                prompt,
                source_video_path,
                frame_path,
            )
            temp_paths.extend(additional_cleanup)
        except HTTPException as exc:
            for path in temp_paths:
                service.safe_delete(path)
            raise exc
        except Exception as exc:
            for path in temp_paths:
                service.safe_delete(path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        unique_cleanup = {path for path in temp_paths if path is not None}
        for path in unique_cleanup:
            if path != output_path:
                background_tasks.add_task(service.safe_delete, path)
        background_tasks.add_task(service.safe_delete, output_path)

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=output_path.name,
            background=background_tasks,
        )

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    app = create_app(config_path=args.config, device_index=args.device_index)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

