"""
Stub classes to replace audiotools dependency.

These stubs allow the DAC VAE model to be imported and used for inference
without requiring the audiotools package. The compress/decompress methods
that depend on AudioSignal are not used in the ComfyUI workflow, which only
uses the internal encode/decode methods that work directly with tensors.
"""
import torch.nn as nn


class AudioSignal:
    """
    Stub class replacing audiotools.AudioSignal.
    
    The compress() and decompress() methods in the DAC model use this class,
    but these methods are not called in the ComfyUI workflow. The workflow
    only uses the internal decode() method which works with raw tensors.
    """
    pass


class BaseModel(nn.Module):
    """
    Stub class replacing audiotools.ml.BaseModel.
    
    This is used as a parent class for the DAC model. We inherit from nn.Module
    to provide the standard PyTorch module functionality like apply(), to(), etc.
    
    The audiotools.ml.BaseModel has some extra features for model management
    that we don't need for inference, so we just use nn.Module as the base.
    """
    pass
