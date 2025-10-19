__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

try:
    import audiotools
    audiotools.ml.BaseModel.INTERN += ["dac.**"]
    audiotools.ml.BaseModel.EXTERN += ["einops"]
except ImportError:
    # audiotools not available - skip configuration
    # This is fine for inference-only usage in ComfyUI
    pass


from . import nn
from . import model
from . import utils
from .model import DAC
from .model import DACFile
