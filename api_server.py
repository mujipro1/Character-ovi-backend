import argparse
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

from modify_prompt import modify_prompt_
PROMPT_MODIFICATION_AVAILABLE = True


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_FPS = 24
DEFAULT_SAMPLE_RATE = 16000
INPAINT_FALLBACK_INSTRUCTION = (
    "Apply inpainting only inside the highlighted region of the provided frame and leave the rest of the video untouched."
)


def _ensure_cuda_device(device_index: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required. Please ensure PyTorch is installed with CUDA or ROCm support.")
    device_index = max(device_index, 0)
    torch.cuda.set_device(device_index)
    return device_index


def compose_generation_prompt(
    video_prompt: str,
    audio_prompt: Optional[str] = None,
    extra_instruction: Optional[str] = None,
) -> str:
    prompt_sections: List[str] = []
    if video_prompt:
        prompt_sections.append(video_prompt.strip())
    if extra_instruction:
        prompt_sections.append(extra_instruction.strip())
    composed = " ".join(section for section in prompt_sections if section)
    if audio_prompt and audio_prompt.strip():
        audio_section = audio_prompt.strip()
        if not audio_section.lower().startswith("audio:"):
            audio_section = f"Audio: {audio_section}"
        composed = f"{composed} {audio_section}".strip()
    if not composed:
        raise HTTPException(status_code=400, detail="At least one of video_prompt or audio_prompt must be provided.")
    return composed


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
        print("Loading OviFusionEngine for API service...")
        self._engine = OviFusionEngine(config=config, device=self._device_index, target_dtype=target_dtype)
        print("OviFusionEngine loaded successfully.")

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

    def _save_frame_from_numpy(self, video_numpy: np.ndarray, frame_index: int = -1) -> Path:
        """
        Extract a frame from a numpy video array and save it as a PNG file.
        
        Args:
            video_numpy: Video array with shape (C, F, H, W)
            frame_index: Index of frame to extract (default: -1 for last frame)
        
        Returns:
            Path to the saved frame file
        """
        # Extract the frame: (C, F, H, W) -> (C, H, W)
        frame = video_numpy[:, frame_index, :, :]
        
        # Reorder to (H, W, C)
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        else:
            frame = frame.squeeze(0)  # (1, H, W) -> (H, W)
            frame = np.stack([frame, frame, frame], axis=-1)  # Convert grayscale to RGB
        
        # Normalize to [0, 255] if needed
        if frame.max() <= 1.0:
            frame = np.clip(frame, -1, 1)
            frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB for cv2 (cv2 uses BGR)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save to temporary file
        temp_file = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
        cv2.imwrite(str(temp_file), frame)
        return temp_file

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

    def _detect_bounding_box(self, frame_path: Path) -> Optional[Tuple[int, int, int, int, int, int]]:
        image = cv2.imread(str(frame_path))
        if image is None:
            return None

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        coords = cv2.findNonZero(mask)
        if coords is None or coords.size == 0:
            return None

        x_min, y_min, box_width, box_height = cv2.boundingRect(coords)
        x_max = x_min + box_width
        y_max = y_min + box_height
        return x_min, y_min, x_max, y_max, width, height

    def _build_inpaint_instruction(self, frame_path: Path) -> str:
        detection = self._detect_bounding_box(frame_path)
        if detection is None:
            return INPAINT_FALLBACK_INSTRUCTION

        x_min, y_min, x_max, y_max, width, height = detection
        x_min_pct = (x_min / max(width, 1)) * 100
        x_max_pct = (x_max / max(width, 1)) * 100
        y_min_pct = (y_min / max(height, 1)) * 100
        y_max_pct = (y_max / max(height, 1)) * 100

        return (
            "Focus edits within the boxed region (horizontal "
            f"{x_min_pct:.1f}%–{x_max_pct:.1f}%, vertical {y_min_pct:.1f}%–{y_max_pct:.1f}%) "
            "and preserve everything outside the box exactly as in the reference frame."
        )

    def _generate_segments(
        self,
        video_prompt: str,
        audio_prompt: Optional[str],
        target_frames: int,
        reference_path: Optional[Path],
        base_seed: int,
        extra_instruction: Optional[str] = None,
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

        # Calculate number of clips needed and get enhanced prompts
        duration_seconds = target_frames / self._fps
        num_clips = max(1, math.ceil(duration_seconds / self._segment_duration))
        
        print(f"\n{'='*60}")
        print(f"VIDEO GENERATION REQUEST")
        print(f"{'='*60}")
        print(f"Original Video Prompt: {video_prompt}")
        if audio_prompt:
            print(f"Audio Prompt: {audio_prompt}")
        print(f"Target Duration: {duration_seconds:.2f} seconds")
        print(f"Number of clips needed: {num_clips}")
        print(f"Segment duration: {self._segment_duration} seconds")
        
        # Get enhanced prompts for each clip
        enhanced_prompts = [video_prompt] * num_clips  # Fallback to original prompt
        if PROMPT_MODIFICATION_AVAILABLE:
            try:
                print(f"\nEnhancing prompts using modify_prompt...")
                enhanced_response = modify_prompt_(video_prompt, duration_seconds, audio_prompt)
                if enhanced_response and isinstance(enhanced_response, str):
                    prompt_lines = [line.strip() for line in enhanced_response.strip().split("\n") if line.strip()]
                    if len(prompt_lines) >= num_clips:
                        enhanced_prompts = prompt_lines[:num_clips]
                    elif len(prompt_lines) > 0:
                        # If we got fewer prompts than needed, repeat the last one
                        enhanced_prompts = prompt_lines + [prompt_lines[-1]] * (num_clips - len(prompt_lines))
                    print(f"Successfully enhanced {num_clips} prompts for video generation")
                    print(f"\nENHANCED PROMPTS:")
                    for i, prompt in enumerate(enhanced_prompts, 1):
                        print(f"  Clip {i}: {prompt}")
            except Exception as e:
                print(f"WARNING: Failed to enhance prompts: {e}. Using original prompt for all segments.")
        else:
            print(f"Using original prompt for all {num_clips} segments (prompt enhancement not available)")
        
        print(f"{'='*60}\n")

        # Track temporary reference frames for cleanup
        temp_reference_frames: List[Path] = []
        current_reference = reference_path

        while total_frames < target_frames:
            seed = base_seed + segment_index
            
            # Use the enhanced prompt for this segment
            segment_video_prompt = enhanced_prompts[min(segment_index, len(enhanced_prompts) - 1)]
            composed_prompt = compose_generation_prompt(
                video_prompt=segment_video_prompt,
                audio_prompt=audio_prompt,
                extra_instruction=extra_instruction,
            )
            
            print(f"\nGenerating Segment {segment_index + 1}/{num_clips}")
            print(f"  Seed: {seed}")
            print(f"  Video Prompt: {segment_video_prompt}")
            if current_reference:
                print(f"  Reference frame: {current_reference}")
            if extra_instruction:
                print(f"  Extra Instruction: {extra_instruction}")
            print(f"  Composed Prompt: {composed_prompt}")
            
            generated_video, generated_audio, _ = self._engine.generate(
                text_prompt=composed_prompt,
                image_path=str(current_reference) if current_reference else None,
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
            
            print(f"  Segment {segment_index + 1} generated successfully")

            if generated_video is None:
                raise RuntimeError("Video generation failed.")

            # Extract last frame from this clip to use as reference for next clip
            # Only do this if there are more clips to generate
            if segment_index < num_clips - 1 and generated_video.shape[1] > 0:
                last_frame_path = self._save_frame_from_numpy(generated_video, frame_index=-1)
                temp_reference_frames.append(last_frame_path)
                current_reference = last_frame_path
                print(f"  Extracted last frame from segment {segment_index + 1} to use as reference for next segment")

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
        
        # Clean up temporary reference frames
        for temp_frame in temp_reference_frames:
            self.safe_delete(temp_frame)

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

        print(f"\nAll segments generated successfully!")
        print(f"Total frames: {combined_video.shape[1]}")
        if combined_audio is not None:
            print(f"Total audio samples: {combined_audio.shape[0] if combined_audio.ndim == 1 else combined_audio.shape[1]}")
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return combined_video, combined_audio

    def generate_video(
        self,
        video_prompt: str,
        audio_prompt: Optional[str],
        video_length: float,
        reference_path: Optional[Path],
        extra_instruction: Optional[str] = None,
    ) -> Tuple[Path, List[Path]]:
        if video_length <= 0:
            raise HTTPException(status_code=400, detail="Video length must be greater than zero.")

        print(f"\nStarting video generation...")
        print(f"Video length: {video_length} seconds")
        if reference_path:
            print(f"Reference file: {reference_path}")
        if extra_instruction:
            print(f"Inpainting instruction: {extra_instruction}")

        prepared_reference, cleanup_paths = self._prepare_reference(reference_path)
        target_frames = max(int(video_length * self._fps), self._fps)
        base_seed = self._base_config.get("seed", 100)

        with self._lock:
            combined_video, combined_audio = self._generate_segments(
                video_prompt=video_prompt,
                audio_prompt=audio_prompt,
                target_frames=target_frames,
                reference_path=prepared_reference,
                base_seed=base_seed,
                extra_instruction=extra_instruction,
            )

            stem_source = video_prompt or "video"
            stem = "".join(ch for ch in stem_source[:24] if ch.isalnum() or ch in ("-", "_"))
            if not stem:
                stem = "video"
            unique_id = torch.randint(0, 10_000, (1,)).item()
            output_path = self._output_dir / f"{stem}_{os.getpid()}_{unique_id}.mp4"
            
            print(f"\nSaving video to: {output_path}")
            save_video(
                output_path=str(output_path),
                video_numpy=combined_video,
                audio_numpy=combined_audio,
                fps=self._fps,
                sample_rate=self._sample_rate,
            )
            print(f"Video saved successfully!")

        return output_path, cleanup_paths

    def inpaint_video(
        self,
        video_prompt: str,
        audio_prompt: Optional[str],
        source_video: Path,
        frame_path: Optional[Path],
        original_prompt: Optional[str] = None,
        original_audio_prompt: Optional[str] = None,
        original_video_length: Optional[float] = None,
        frame_time: Optional[float] = None,
        reference_path: Optional[Path] = None,
    ) -> Tuple[Path, List[Path]]:
        print(f"\n{'='*60}")
        print(f"INPAINT VIDEO REQUEST")
        print(f"{'='*60}")
        print(f"Source video: {source_video}")
        print(f"Inpainting video prompt: {video_prompt}")
        if original_prompt:
            print(f"Original video prompt: {original_prompt}")
        if audio_prompt:
            print(f"Inpainting audio prompt: {audio_prompt}")
        if original_audio_prompt:
            print(f"Original audio prompt: {original_audio_prompt}")
        if original_video_length:
            print(f"Original video length: {original_video_length:.2f} seconds")
        if frame_time is not None:
            print(f"Frame time: {frame_time:.2f} seconds")
        if reference_path:
            print(f"Reference asset: {reference_path}")
        
        temp_paths: List[Path] = []
        duration = self._determine_video_duration(source_video)
        print(f"Source video duration: {duration:.2f} seconds")

        # Determine if we need segment-based regeneration
        use_segment_regeneration = (
            original_video_length is not None and 
            original_video_length > self._segment_duration and
            frame_time is not None
        )

        if use_segment_regeneration:
            # Multi-segment video: regenerate only the affected segment
            print(f"\nMulti-segment video detected. Using segment-based regeneration...")
            
            # Generate prompts based on original prompt and length (like in normal generation)
            if not original_prompt:
                raise HTTPException(
                    status_code=400, 
                    detail="original_prompt is required for multi-segment video inpainting"
                )
            
            num_clips = max(1, math.ceil(original_video_length / self._segment_duration))
            print(f"Original video had {num_clips} segment(s)")
            
            # Get enhanced prompts for each segment
            enhanced_prompts = [original_prompt] * num_clips
            if PROMPT_MODIFICATION_AVAILABLE:
                try:
                    print(f"Generating segment prompts from original prompt...")
                    enhanced_response = modify_prompt_(original_prompt, original_video_length, original_audio_prompt)
                    if enhanced_response and isinstance(enhanced_response, str):
                        prompt_lines = [line.strip() for line in enhanced_response.strip().split("\n") if line.strip()]
                        if len(prompt_lines) >= num_clips:
                            enhanced_prompts = prompt_lines[:num_clips]
                        elif len(prompt_lines) > 0:
                            enhanced_prompts = prompt_lines + [prompt_lines[-1]] * (num_clips - len(prompt_lines))
                    print(f"Generated {len(enhanced_prompts)} segment prompts")
                except Exception as e:
                    print(f"WARNING: Failed to generate segment prompts: {e}. Using original prompt for all segments.")
            
            # Determine which segment the frame is from
            segment_index = int(frame_time // self._segment_duration)
            segment_index = min(segment_index, num_clips - 1)  # Ensure within bounds
            segment_start_time = segment_index * self._segment_duration
            segment_end_time = min((segment_index + 1) * self._segment_duration, original_video_length)
            
            print(f"Frame is from segment {segment_index + 1}/{num_clips}")
            print(f"Segment time range: {segment_start_time:.2f}s - {segment_end_time:.2f}s")
            
            # Load the original video
            print("Loading original video...")
            original_video, original_audio = self._load_video_as_numpy(source_video)
            original_frames = original_video.shape[1]
            original_total_frames = int(original_video_length * self._fps)
            
            # Calculate frame ranges
            segment_start_frame = int(segment_start_time * self._fps)
            segment_end_frame = int(segment_end_time * self._fps)
            segment_end_frame = min(segment_end_frame, original_total_frames)
            
            # Get reference frame for regeneration
            # Priority: 1) provided reference_path, 2) provided frame_path, 3) last frame from previous segment, 4) extract from video
            if reference_path:
                # Use provided reference asset (image or video)
                reference_frame_path, ref_cleanup = self._prepare_reference(reference_path)
                temp_paths.extend(ref_cleanup)
                print(f"Using provided reference asset: {reference_path}")
            elif segment_index > 0:
                # Use last frame from previous segment
                prev_segment_end_frame = segment_start_frame
                if prev_segment_end_frame > 0:
                    reference_frame_path = self._save_frame_from_numpy(original_video, frame_index=prev_segment_end_frame - 1)
                    temp_paths.append(reference_frame_path)
                    print(f"Using last frame from previous segment (frame {prev_segment_end_frame - 1}) as reference")
                else:
                    # Fallback to provided frame or extract from video
                    reference_frame_path = frame_path if frame_path else self._extract_first_frame(source_video)
                    if not frame_path:
                        temp_paths.append(reference_frame_path)
                    print(f"Using provided frame as reference (first segment)")
            else:
                # First segment, use provided frame or extract from video
                reference_frame_path = frame_path if frame_path else self._extract_first_frame(source_video)
                if not frame_path:
                    temp_paths.append(reference_frame_path)
                print(f"Using provided frame as reference (first segment)")
            
            # Build inpainting instruction
            extra_instruction = self._build_inpaint_instruction(reference_frame_path)
            print(f"Inpainting instruction: {extra_instruction}")
            
            # Get the segment prompt and combine with inpainting prompt
            segment_prompt = enhanced_prompts[segment_index]
            combined_video_prompt_parts = [segment_prompt.strip(), video_prompt.strip()]
            if extra_instruction:
                combined_video_prompt_parts.append(extra_instruction.strip())
            combined_video_prompt = ". ".join(part for part in combined_video_prompt_parts if part)
            
            print(f"Segment prompt: {segment_prompt}")
            print(f"Combined video prompt: {combined_video_prompt}")
            
            # Use inpainting audio prompt if provided, otherwise use original audio prompt
            final_audio_prompt = audio_prompt if audio_prompt else original_audio_prompt
            
            # Generate only the target segment
            segment_duration = segment_end_time - segment_start_time
            print(f"\nRegenerating segment {segment_index + 1} ({segment_duration:.2f} seconds)...")
            
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
            base_seed = cfg.get("seed", 100)
            
            composed_prompt = compose_generation_prompt(
                video_prompt=combined_video_prompt,
                audio_prompt=final_audio_prompt,
                extra_instruction=None,  # Already included
            )
            
            with self._lock:
                regenerated_video, regenerated_audio, _ = self._engine.generate(
                    text_prompt=composed_prompt,
                    image_path=str(reference_frame_path),
                    video_frame_height_width=video_hw,
                    seed=base_seed + segment_index,
                    solver_name=solver_name,
                    sample_steps=sample_steps,
                    shift=shift,
                    video_guidance_scale=video_guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                    slg_layer=slg_layer,
                    video_negative_prompt=video_negative_prompt,
                    audio_negative_prompt=audio_negative_prompt,
                )
            
            if regenerated_video is None:
                raise RuntimeError("Segment regeneration failed.")
            
            # Trim regenerated segment to match original segment length
            regenerated_frames = regenerated_video.shape[1]
            target_segment_frames = segment_end_frame - segment_start_frame
            
            if regenerated_frames > target_segment_frames:
                regenerated_video = regenerated_video[:, :target_segment_frames, :, :]
                if regenerated_audio is not None:
                    target_samples = int(segment_duration * self._sample_rate)
                    current_samples = regenerated_audio.shape[0] if regenerated_audio.ndim == 1 else regenerated_audio.shape[1]
                    if current_samples > target_samples:
                        if regenerated_audio.ndim == 1:
                            regenerated_audio = regenerated_audio[:target_samples]
                        else:
                            regenerated_audio = regenerated_audio[:, :target_samples]
            
            # Replace the segment in the original video
            print(f"Replacing segment {segment_index + 1} in original video...")
            print(f"Original video shape: {original_video.shape}")
            print(f"Regenerated segment shape: {regenerated_video.shape}")
            print(f"Replacing frames {segment_start_frame} to {segment_start_frame + regenerated_video.shape[1]}")
            
            # Ensure we don't exceed original video bounds
            actual_end_frame = min(segment_start_frame + regenerated_video.shape[1], original_video.shape[1])
            actual_segment_frames = actual_end_frame - segment_start_frame
            
            # Replace the segment
            original_video[:, segment_start_frame:actual_end_frame, :, :] = regenerated_video[:, :actual_segment_frames, :, :]
            
            # Replace audio segment if available
            if regenerated_audio is not None and original_audio is not None:
                segment_start_sample = int(segment_start_time * self._sample_rate)
                segment_end_sample = int(segment_end_time * self._sample_rate)
                segment_end_sample = min(segment_end_sample, original_audio.shape[0] if original_audio.ndim == 1 else original_audio.shape[1])
                
                normalized_regenerated = self._normalize_audio(regenerated_audio)
                if normalized_regenerated.ndim == 1:
                    actual_audio_samples = min(normalized_regenerated.shape[0], segment_end_sample - segment_start_sample)
                    original_audio[segment_start_sample:segment_start_sample + actual_audio_samples] = normalized_regenerated[:actual_audio_samples]
                else:
                    actual_audio_samples = min(normalized_regenerated.shape[1], segment_end_sample - segment_start_sample)
                    original_audio[:, segment_start_sample:segment_start_sample + actual_audio_samples] = normalized_regenerated[:, :actual_audio_samples]
            
            # Save the modified video
            stem_source = video_prompt or "inpainted_video"
            stem = "".join(ch for ch in stem_source[:24] if ch.isalnum() or ch in ("-", "_"))
            if not stem:
                stem = "inpainted_video"
            unique_id = torch.randint(0, 10_000, (1,)).item()
            output_path = self._output_dir / f"{stem}_{os.getpid()}_{unique_id}.mp4"
            
            print(f"\nSaving inpainted video to: {output_path}")
            save_video(
                output_path=str(output_path),
                video_numpy=original_video,
                audio_numpy=original_audio,
                fps=self._fps,
                sample_rate=self._sample_rate,
            )
            print(f"Video saved successfully!")
            
        else:
            # Single segment video (5 seconds or less): use simple inpainting
            print(f"\nSingle segment video. Using simple inpainting...")
            
            # Priority: 1) provided reference_path, 2) provided frame_path, 3) extract from video
            if reference_path:
                reference, ref_cleanup = self._prepare_reference(reference_path)
                temp_paths.extend(ref_cleanup)
                print(f"Using provided reference asset: {reference_path}")
            elif frame_path:
        reference = frame_path
                print(f"Using provided frame: {reference}")
            else:
            print("No frame provided, extracting first frame from video...")
            reference = self._extract_first_frame(source_video)
            temp_paths.append(reference)
            print(f"Extracted frame: {reference}")

        extra_instruction = self._build_inpaint_instruction(reference)
        print(f"Inpainting instruction: {extra_instruction}")

            # Combine original prompt with inpainting prompt and bounding box instruction
            combined_video_prompt_parts = []
            if original_prompt:
                combined_video_prompt_parts.append(original_prompt.strip())
            combined_video_prompt_parts.append(video_prompt.strip())
            if extra_instruction:
                combined_video_prompt_parts.append(extra_instruction.strip())
            combined_video_prompt = ". ".join(part for part in combined_video_prompt_parts if part)
            
            print(f"Combined video prompt: {combined_video_prompt}")

            # Use inpainting audio prompt if provided, otherwise use original audio prompt
            final_audio_prompt = audio_prompt if audio_prompt else original_audio_prompt
            if final_audio_prompt:
                print(f"Using audio prompt: {final_audio_prompt}")

        output_path, cleanup = self.generate_video(
                video_prompt=combined_video_prompt,
                audio_prompt=final_audio_prompt,
            video_length=duration,
            reference_path=reference,
                extra_instruction=None,  # Already included in combined_video_prompt
        )
        temp_paths.extend(cleanup)
        
        print(f"{'='*60}\n")
        return output_path, temp_paths

    def _determine_video_duration(self, video_path: Path) -> float:
        with VideoFileClip(str(video_path)) as clip:
            return max(clip.duration, self._segment_duration)

    def _load_video_as_numpy(self, video_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load video from file and convert to numpy array format (C, F, H, W).
        Values are normalized to [-1, 1] range to match model output format.
        
        Returns:
            Tuple of (video_array, audio_array)
        """
        with VideoFileClip(str(video_path)) as clip:
            # Get video frames
            frames = []
            for frame in clip.iter_frames(fps=self._fps, dtype='uint8'):
                # Convert from (H, W, C) to (C, H, W)
                frame = frame.transpose(2, 0, 1)
                # Normalize from [0, 255] to [-1, 1]
                frame = (frame.astype(np.float32) / 255.0) * 2.0 - 1.0
                frames.append(frame)
            
            if not frames:
                raise RuntimeError("Failed to load video frames")
            
            # Stack frames: list of (C, H, W) -> (C, F, H, W)
            video_array = np.stack(frames, axis=1)
            
            # Get audio if available
            audio_array = None
            if clip.audio is not None:
                audio_array = clip.audio.to_soundarray(fps=self._sample_rate)
                # Normalize to [-1, 1] if needed
                if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                    # Already in correct range or needs normalization
                    if audio_array.max() > 1.0:
                        audio_array = audio_array / 255.0 * 2.0 - 1.0
                # Convert to mono if stereo
                if audio_array.ndim == 2:
                    if audio_array.shape[0] == 2:
                        audio_array = audio_array.mean(axis=0)
                    elif audio_array.shape[1] == 2:
                        audio_array = audio_array.mean(axis=1)
            
            return video_array, audio_array

    def _extract_frame_at_time(self, video_path: Path, time_seconds: float) -> Path:
        """
        Extract a frame from video at a specific time.
        
        Args:
            video_path: Path to video file
            time_seconds: Time in seconds to extract frame
        
        Returns:
            Path to saved frame image
        """
        with VideoFileClip(str(video_path)) as clip:
            if time_seconds > clip.duration:
                time_seconds = clip.duration
            frame = clip.get_frame(time_seconds)
            temp_file = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(temp_file), frame_bgr)
            return temp_file

    def _extract_segment_from_video(self, video_array: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        """
        Extract a segment from a video array.
        
        Args:
            video_array: Video array with shape (C, F, H, W)
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (exclusive)
        
        Returns:
            Segment array with shape (C, segment_frames, H, W)
        """
        return video_array[:, start_frame:end_frame, :, :]

    @staticmethod
    def safe_delete(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:
            print(f"WARNING: Failed to delete {path}: {exc}")


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
        video_prompt: str = Form(...),
        audio_prompt: Optional[str] = Form(None),
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
                video_prompt,
                audio_prompt,
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
        video_prompt: str = Form(...),
        audio_prompt: Optional[str] = Form(None),
        generated_video: UploadFile = File(...),
        frame: Optional[UploadFile] = File(None),
        original_prompt: Optional[str] = Form(None),
        original_audio_prompt: Optional[str] = Form(None),
        original_video_length: Optional[str] = Form(None),
        frame_time: Optional[str] = Form(None),
        reference: Optional[UploadFile] = File(None),
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

            reference_path: Optional[Path] = None
            if reference is not None:
                reference_path = Path(tempfile.NamedTemporaryFile(suffix=Path(reference.filename or '').suffix or ".dat", delete=False).name)
                with reference_path.open("wb") as buffer:
                    buffer.write(await reference.read())
                temp_paths.append(reference_path)

            # Convert string form parameters to float
            original_video_length_float = None
            if original_video_length is not None and original_video_length.strip():
                try:
                    original_video_length_float = float(original_video_length)
                except ValueError:
                    print(f"WARNING: Could not convert original_video_length '{original_video_length}' to float")
            
            frame_time_float = None
            if frame_time is not None and frame_time.strip():
                try:
                    frame_time_float = float(frame_time)
                except ValueError:
                    print(f"WARNING: Could not convert frame_time '{frame_time}' to float")

            output_path, additional_cleanup = await run_in_threadpool(
                service.inpaint_video,
                video_prompt,
                audio_prompt,
                source_video_path,
                frame_path,
                original_prompt,
                original_audio_prompt,
                original_video_length_float,
                frame_time_float,
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

    return app


def main() -> None:
    args = _parse_args()
    print(f"Starting OVI API server on {args.host}:{args.port}")
    print(f"Using config: {args.config}")
    print(f"Device index: {args.device_index}")
    app = create_app(config_path=args.config, device_index=args.device_index)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

