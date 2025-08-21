"""Constants used throughout the HunyuanVideo-Foley project."""

from typing import Dict, List

# Model configuration
DEFAULT_AUDIO_SAMPLE_RATE = 48000
DEFAULT_VIDEO_FPS = 25
DEFAULT_AUDIO_CHANNELS = 2

# Video processing
MAX_VIDEO_DURATION_SECONDS = 15.0
MIN_VIDEO_DURATION_SECONDS = 1.0

# Audio processing
AUDIO_VAE_LATENT_DIM = 128
AUDIO_FRAME_RATE = 75  # frames per second in latent space

# Visual features
FPS_VISUAL: Dict[str, int] = {
    "siglip2": 8, 
    "synchformer": 25
}

# Model paths (can be overridden by environment variables)
DEFAULT_MODEL_PATH = "./pretrained_models/"
DEFAULT_CONFIG_PATH = "configs/hunyuanvideo-foley-xxl.yaml"

# Inference parameters
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_INFERENCE_STEPS = 50
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 10.0
MIN_INFERENCE_STEPS = 10
MAX_INFERENCE_STEPS = 100

# Text processing
MAX_TEXT_LENGTH = 100
DEFAULT_NEGATIVE_PROMPT = "noisy, harsh"

# File extensions
SUPPORTED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
SUPPORTED_AUDIO_EXTENSIONS: List[str] = [".wav", ".mp3", ".flac", ".aac"]

# Quality settings
AUDIO_QUALITY_SETTINGS: Dict[str, List[str]] = {
    "high": ["-b:a", "192k"],
    "medium": ["-b:a", "128k"], 
    "low": ["-b:a", "96k"]
}

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    "model_not_loaded": "Model is not loaded. Please load the model first.",
    "invalid_video_format": "Unsupported video format. Supported formats: {formats}",
    "video_too_long": f"Video duration exceeds maximum of {MAX_VIDEO_DURATION_SECONDS} seconds",
    "ffmpeg_not_found": "ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html"
}