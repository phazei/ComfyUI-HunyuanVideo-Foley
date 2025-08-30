# ComfyUI-HunyuanVideoFoley/__init__.py

# This line makes the 'hunyuanvideo_foley' sub-directory importable by your nodes.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# This line imports the node mappings from your nodes.py file
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# This is a Python convention that makes the mappings easily accessible
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
