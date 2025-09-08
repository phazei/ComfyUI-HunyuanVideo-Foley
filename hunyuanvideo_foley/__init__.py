"""
HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment 
for High-Fidelity Foley Audio Generation

This package provides tools for generating high-quality Foley audio effects 
from video content using multimodal diffusion models.
"""

__version__ = "1.0.0"
__author__ = "Tencent Hunyuan Team"
__email__ = "hunyuan@tencent.com"

# Import main components for easy access
try:
    from .utils.model_utils import load_model, denoise_process
    from .utils.feature_utils import feature_process
    from .utils.media_utils import merge_audio_video
    from .utils.config_utils import AttributeDict
    
    __all__ = [
        "__version__",
        "load_model", 
        "denoise_process",
        "feature_process", 
        "merge_audio_video",
        "AttributeDict"
    ]
except ImportError:
    # Handle missing dependencies gracefully during installation
    __all__ = ["__version__"]