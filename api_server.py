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
    ""
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
        """
        Normalize audio to consistent format: (channels, samples) where channels is typically 1 (mono).
        
        Args:
            audio: Audio array in various formats
            
        Returns:
            Audio array with shape (channels, samples), typically (1, samples) for mono
        """
        if audio is None:
            return None
        
        if audio.ndim == 1:
            # 1D array: assume it's samples, convert to (1, samples)
            return audio.reshape(1, -1)
        
        if audio.ndim == 2:
            # 2D array: determine if it's (samples, channels) or (channels, samples)
            # Typically, channels will be 1-2 (mono/stereo), samples will be much larger
            # If first dimension is small (<= 2), assume it's (channels, samples)
            # If first dimension is large, assume it's (samples, channels) and transpose
            if audio.shape[0] <= 2:
                # Likely (channels, samples) - return as is
                return audio
            elif audio.shape[1] <= 2:
                # Likely (samples, channels) - transpose to (channels, samples)
                return audio.T
            else:
                # Ambiguous case - assume (samples, channels) if samples > channels
                # Otherwise assume (channels, samples)
                if audio.shape[0] > audio.shape[1]:
                    return audio.T
                return audio
        
        # Higher dimensions - flatten and reshape
        return audio.flatten().reshape(1, -1)

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
        # Ensure frame_index is valid
        num_frames = video_numpy.shape[1]
        if frame_index < 0:
            frame_index = num_frames + frame_index
        frame_index = max(0, min(frame_index, num_frames - 1))
        
        # Extract the frame: (C, F, H, W) -> (C, H, W)
        frame = video_numpy[:, frame_index, :, :]
        
        # Ensure frame is a numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Reorder to (H, W, C)
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        else:
            # Grayscale: (1, H, W) or (H, W)
            if frame.ndim == 3:
                frame = frame.squeeze(0)  # (1, H, W) -> (H, W)
            
            # Ensure frame is 2D before stacking
            if frame.ndim != 2:
                raise ValueError(f"Expected 2D frame after squeeze, got shape {frame.shape}")
            
            # Convert grayscale to RGB by stacking three copies
            # Ensure we pass a proper list/tuple to np.stack
            frame_list = [np.array(frame, copy=True), np.array(frame, copy=True), np.array(frame, copy=True)]
            # Convert to tuple to ensure it's a proper sequence
            frame_tuple = tuple(frame_list)
            frame = np.stack(frame_tuple, axis=-1)  # (H, W) -> (H, W, 3)
        
        # Normalize to [0, 255] if needed
        if frame.max() <= 1.0:
            frame = np.clip(frame, -1, 1)
            frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for cv2 (cv2 uses BGR)
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

    def _build_inpaint_instruction(self, frame_path: Path) -> str:
        return INPAINT_FALLBACK_INSTRUCTION

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

            # Handle audio concatenation with proper shape checking
            if generated_audio is not None:
                print(f"  Segment {segment_index + 1} audio shape: {generated_audio.shape}")
                normalized_audio = self._normalize_audio(generated_audio)
                print(f"  Normalized audio shape: {normalized_audio.shape}")
                
                # Ensure normalized_audio is 2D with shape (channels, samples)
                if normalized_audio.ndim == 1:
                    normalized_audio = normalized_audio.reshape(1, -1)
                
                if combined_audio is None:
                    combined_audio = normalized_audio
                    print(f"  Initialized combined_audio with shape: {combined_audio.shape}")
                else:
                    # Ensure combined_audio is also 2D
                    if combined_audio.ndim == 1:
                        combined_audio = combined_audio.reshape(1, -1)
                    
                    # Ensure shapes match for concatenation (same number of channels)
                    print(f"  Current combined_audio shape: {combined_audio.shape}")
                    if combined_audio.shape[0] != normalized_audio.shape[0]:
                        print(f"  WARNING: Channel mismatch! combined_audio.shape[0]={combined_audio.shape[0]}, normalized_audio.shape[0]={normalized_audio.shape[0]}")
                        # Convert to mono if needed (take mean across channels)
                        if combined_audio.shape[0] > 1:
                            combined_audio = combined_audio.mean(axis=0, keepdims=True)
                        if normalized_audio.shape[0] > 1:
                            normalized_audio = normalized_audio.mean(axis=0, keepdims=True)
                    
                    try:
                        combined_audio = np.concatenate([combined_audio, normalized_audio], axis=1)
                        print(f"  After concatenation, combined_audio shape: {combined_audio.shape}")
                    except ValueError as e:
                        print(f"  ERROR: Failed to concatenate audio: {e}")
                        print(f"    combined_audio shape: {combined_audio.shape}, normalized_audio shape: {normalized_audio.shape}")
                        raise
                total_samples = combined_audio.shape[1] if combined_audio.ndim == 2 else combined_audio.shape[0]
            else:
                print(f"  WARNING: Segment {segment_index + 1} has no audio!")
                # If we have combined_audio from previous segments, pad with silence
                if combined_audio is not None:
                    # Calculate expected audio length for this segment
                    segment_duration = self._segment_duration
                    expected_samples = int(segment_duration * self._sample_rate)
                    # Create silence (zeros) with same shape as combined_audio
                    if combined_audio.ndim == 2:
                        silence = np.zeros((combined_audio.shape[0], expected_samples), dtype=combined_audio.dtype)
                    else:
                        silence = np.zeros(expected_samples, dtype=combined_audio.dtype)
                        silence = silence.reshape(1, -1)
                        combined_audio = combined_audio.reshape(1, -1) if combined_audio.ndim == 1 else combined_audio
                    combined_audio = np.concatenate([combined_audio, silence], axis=1)
                    print(f"  Padded with silence. New combined_audio shape: {combined_audio.shape}")
            segment_index += 1

            if segment_index > 16:
                raise RuntimeError("Requested duration is too long for automated tiling.")
        
        # Clean up temporary reference frames
        for temp_frame in temp_reference_frames:
            self.safe_delete(temp_frame)

        if combined_video.shape[1] > target_frames:
            combined_video = combined_video[:, :target_frames, :, :]

        if combined_audio is not None:
            # Ensure combined_audio is in (channels, samples) format
            if combined_audio.ndim == 1:
                combined_audio = combined_audio.reshape(1, -1)
            
            print(f"Final combined_audio shape before trimming: {combined_audio.shape}")
            target_samples = math.ceil((target_frames / self._fps) * self._sample_rate)
            
            if combined_audio.shape[1] > target_samples:
                combined_audio = combined_audio[:, :target_samples]
                print(f"Trimmed audio to {target_samples} samples. New shape: {combined_audio.shape}")
            elif combined_audio.shape[1] < target_samples:
                pad_width = target_samples - combined_audio.shape[1]
                combined_audio = np.pad(combined_audio, ((0, 0), (0, pad_width)), mode="edge")
                print(f"Padded audio to {target_samples} samples. New shape: {combined_audio.shape}")

            # Convert to format expected by save_video: (samples,) for mono or (samples, channels) for stereo
            # scipy.io.wavfile.write expects (samples, channels) format
            if combined_audio.shape[0] == 1:
                # Mono: convert from (1, samples) to (samples,)
                combined_audio = combined_audio.squeeze(0)
            else:
                # Stereo: convert from (channels, samples) to (samples, channels)
                combined_audio = combined_audio.T
            
            print(f"Final combined_audio shape for saving: {combined_audio.shape}")

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
            
            # Ensure original_video is a numpy array
            if not isinstance(original_video, np.ndarray):
                if hasattr(original_video, 'cpu'):
                    # It's a torch tensor
                    original_video = original_video.cpu().numpy()
                else:
                    original_video = np.array(original_video)
            
            original_frames = original_video.shape[1]
            original_total_frames = int(original_video_length * self._fps)
            
            # Calculate frame ranges
            segment_start_frame = int(segment_start_time * self._fps)
            segment_end_frame = int(segment_end_time * self._fps)
            segment_end_frame = min(segment_end_frame, original_total_frames)
            
            # Get reference frame for regeneration
            # Priority: 1) provided reference_path, 2) last frame from previous segment, 3) extract from video at frame_time
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
                    # Fallback: extract frame from video at frame_time
                    if frame_time is not None:
                        reference_frame_path = self._extract_frame_at_time(source_video, frame_time)
                        temp_paths.append(reference_frame_path)
                        print(f"Extracting frame from video at {frame_time:.2f}s (first segment)")
                    else:
                        reference_frame_path = self._extract_first_frame(source_video)
                        temp_paths.append(reference_frame_path)
                        print(f"Extracting first frame from video (first segment, no frame_time)")
            else:
                # First segment, extract frame from video at frame_time
                if frame_time is not None:
                    reference_frame_path = self._extract_frame_at_time(source_video, frame_time)
                    temp_paths.append(reference_frame_path)
                    print(f"Extracting frame from video at {frame_time:.2f}s (first segment)")
                else:
                    reference_frame_path = self._extract_first_frame(source_video)
                    temp_paths.append(reference_frame_path)
                    print(f"Extracting first frame from video (first segment, no frame_time)")
            
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
            
            # Ensure regenerated_video is a numpy array
            if not isinstance(regenerated_video, np.ndarray):
                if hasattr(regenerated_video, 'cpu'):
                    # It's a torch tensor
                    regenerated_video = regenerated_video.cpu().numpy()
                else:
                    regenerated_video = np.array(regenerated_video)
            
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
            
            # Resize regenerated segment to match original video resolution if needed
            original_height = original_video.shape[2]
            original_width = original_video.shape[3]
            regenerated_height = regenerated_video.shape[2]
            regenerated_width = regenerated_video.shape[3]
            
            if regenerated_height != original_height or regenerated_width != original_width:
                print(f"Resizing regenerated segment from {regenerated_width}x{regenerated_height} to {original_width}x{original_height}")
                regenerated_video = self._resize_video_segment(regenerated_video, original_height, original_width)
                print(f"Resized regenerated segment shape: {regenerated_video.shape}")
            
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
            
            # Priority: 1) provided reference_path, 2) extract from video at frame_time, 3) extract first frame
            if reference_path:
                reference, ref_cleanup = self._prepare_reference(reference_path)
                temp_paths.extend(ref_cleanup)
                print(f"Using provided reference asset: {reference_path}")
            elif frame_time is not None:
                # Extract frame from video at the specified timestamp
                reference = self._extract_frame_at_time(source_video, frame_time)
                temp_paths.append(reference)
                print(f"Extracting frame from video at {frame_time:.2f}s")
                print(f"Extracted frame: {reference}")
            else:
                # Fallback to first frame if no frame_time provided
                reference = self._extract_first_frame(source_video)
                temp_paths.append(reference)
                print("No frame_time provided, extracting first frame from video...")
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
            
            # Ensure frames is a proper list and all elements are numpy arrays
            if not isinstance(frames, list):
                raise TypeError(f"frames must be a list, got {type(frames)}")
            
            # Verify all frames are numpy arrays
            for idx, frame in enumerate(frames):
                if not isinstance(frame, np.ndarray):
                    frames[idx] = np.asarray(frame)
            
            # Stack frames: list of (C, H, W) -> (C, F, H, W)
            # Convert to tuple to ensure it's a proper sequence
            try:
                frames_tuple = tuple(frames)
                video_array = np.stack(frames_tuple, axis=1)
            except (ValueError, TypeError) as e:
                shapes = [f.shape for f in frames]
                dtypes = [f.dtype for f in frames]
                raise RuntimeError(
                    f"Failed to stack {len(frames)} frames when loading video. "
                    f"Frame shapes: {shapes}, dtypes: {dtypes}. Error: {e}"
                ) from e
            
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

    def _resize_video_segment(self, video_array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Resize a video array to match target dimensions.
        
        Args:
            video_array: Video array with shape (C, F, H, W), values in [-1, 1] range
            target_height: Target height
            target_width: Target width
        
        Returns:
            Resized video array with shape (C, F, target_height, target_width)
        """
        # Ensure video_array is a numpy array
        if not isinstance(video_array, np.ndarray):
            video_array = np.array(video_array)
        
        C, F, H, W = video_array.shape
        
        # If already the correct size, return as is
        if H == target_height and W == target_width:
            return video_array
        
        # Handle edge case: no frames
        if F == 0:
            return video_array
        
        # Convert from [-1, 1] to [0, 255] for cv2
        video_normalized = ((video_array + 1) / 2 * 255).astype(np.uint8)
        
        # Resize each frame
        # Initialize as a proper Python list
        resized_frames: List[np.ndarray] = []
        for frame_idx in range(F):
            # Get frame: (C, H, W) -> (H, W, C)
            frame = video_normalized[:, frame_idx, :, :].transpose(1, 2, 0)
            
            # Ensure frame is contiguous and properly shaped
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Resize using cv2
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert back to (C, H, W)
            resized_frame = resized_frame.transpose(2, 0, 1)
            
            # Ensure it's a proper numpy array with correct dtype
            resized_frame = np.asarray(resized_frame, dtype=np.uint8)
            
            # Verify shape is correct
            if resized_frame.shape != (C, target_height, target_width):
                raise ValueError(
                    f"Frame {frame_idx} has incorrect shape after resize: "
                    f"expected ({C}, {target_height}, {target_width}), got {resized_frame.shape}"
                )
            
            resized_frames.append(resized_frame)
        
        # Ensure we have frames to stack and resized_frames is a proper list
        if not isinstance(resized_frames, list):
            raise TypeError(f"resized_frames must be a list, got {type(resized_frames)}")
        
        if len(resized_frames) == 0:
            raise ValueError(f"No frames to resize. Original video had {F} frames.")
        
        if len(resized_frames) != F:
            raise ValueError(
                f"Mismatch in frame count: expected {F} frames, got {len(resized_frames)}"
            )
        
        # Stack frames: list of (C, H, W) -> (C, F, H, W)
        # Ensure all frames are numpy arrays with consistent dtype and shape
        frames_to_stack = []
        for idx, frame in enumerate(resized_frames):
            # Ensure it's a numpy array
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame, dtype=np.uint8)
            # Ensure consistent dtype
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            # Ensure correct shape
            if frame.shape != (C, target_height, target_width):
                raise ValueError(
                    f"Frame {idx} has incorrect shape: "
                    f"expected ({C}, {target_height}, {target_width}), got {frame.shape}"
                )
            # Make a copy to ensure it's a proper independent array
            frame = np.array(frame, dtype=np.uint8, copy=True)
            frames_to_stack.append(frame)
        
        # Ensure frames_to_stack is a proper list (not empty, all numpy arrays)
        if not isinstance(frames_to_stack, (list, tuple)):
            raise TypeError(f"frames_to_stack must be a list or tuple, got {type(frames_to_stack)}")
        
        if len(frames_to_stack) == 0:
            raise ValueError(f"No frames to stack. Expected {F} frames.")
        
        # Verify all elements are numpy arrays
        for idx, frame in enumerate(frames_to_stack):
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"Frame {idx} is not a numpy array, got {type(frame)}")
        
        # Now stack using the properly formatted list
        # Convert to tuple explicitly to ensure it's a proper sequence
        try:
            frames_tuple = tuple(frames_to_stack)
            resized_video = np.stack(frames_tuple, axis=1)
        except (ValueError, TypeError) as e:
            # If stacking fails, provide more detailed error information
            shapes = [f.shape for f in frames_to_stack]
            dtypes = [f.dtype for f in frames_to_stack]
            types = [type(f).__name__ for f in frames_to_stack]
            raise ValueError(
                f"Failed to stack {len(frames_to_stack)} frames. "
                f"Frame shapes: {shapes}, dtypes: {dtypes}, types: {types}. "
                f"Target shape: ({C}, {target_height}, {target_width}). "
                f"Original error: {e}"
            ) from e
        
        # Convert back to [-1, 1] range
        resized_video = (resized_video.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        return resized_video

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
    parser.add_argument("--port", type=int, default=8001, help="Port for FastAPI server.")
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

    @app.post("/test")
    async def test_endpoint(
        background_tasks: BackgroundTasks,
        video_prompt: str = Form(...),
        audio_prompt: Optional[str] = Form(None),
        video_length: float = Form(5.0),
        reference: Optional[UploadFile] = File(None),
    ):
        """
        Test endpoint that returns a sample video without running the model.
        Accepts the same parameters as /generate_video for testing purposes.
        """
        # Look for sample video in parent folder first, then current folder
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        sample_video = None
        
        # Check parent folder for sample video files
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            parent_sample = parent_dir / f"sample{ext}"
            if parent_sample.exists():
                sample_video = parent_sample
                break
        
        # If not found in parent, check current folder
        if sample_video is None:
            for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                current_sample = current_dir / f"sample{ext}"
                if current_sample.exists():
                    sample_video = current_sample
                    break
        
        if sample_video is None:
            raise HTTPException(
                status_code=404,
                detail="Sample video not found. Please ensure 'sample.mp4' (or similar) exists in the parent or current folder."
            )
        
        return FileResponse(
            path=str(sample_video),
            media_type="video/mp4",
            filename=sample_video.name,
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
                None,  # frame_path is no longer used - we extract from video using frame_time
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

