#!/usr/bin/env python3
"""
HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment 
for High-Fidelity Foley Audio Generation

Setup script for building and installing the HunyuanVideo-Foley package.
"""

import os
import re
from typing import List
from setuptools import setup, find_packages

def read_file(filename: str) -> str:
    """Read content from a file."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), 'r', encoding='utf-8') as f:
        return f.read()

def get_version() -> str:
    """Extract version from constants.py or use default."""
    try:
        constants_path = os.path.join('hunyuanvideo_foley', 'constants.py')
        content = read_file(constants_path)
        version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", content)
        if version_match:
            return version_match.group(1)
    except FileNotFoundError:
        pass
    return "1.0.0"

def parse_requirements(filename: str) -> List[str]:
    """Parse requirements from requirements.txt file."""
    try:
        content = read_file(filename)
        lines = content.splitlines()
        requirements = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Handle git+https dependencies - convert to standard package names
            if line.startswith('git+'):
                if 'transformers' in line:
                    requirements.append('transformers>=4.49.0')
                elif 'audiotools' in line:
                    # Use a placeholder for audiotools since it's not on PyPI
                    # Users will need to install it separately
                    continue  # Skip for now
                else:
                    continue  # Skip other git dependencies
            else:
                requirements.append(line)
                
        return requirements
    except FileNotFoundError:
        return []

def get_long_description() -> str:
    """Get long description from README.md."""
    try:
        readme = read_file("README.md")
        # Remove HTML tags and excessive styling for PyPI compatibility
        readme = re.sub(r'<[^>]+>', '', readme)
        return readme
    except FileNotFoundError:
        return "Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation"

# Read requirements
install_requires = parse_requirements("requirements.txt")

# Separate development requirements
dev_requirements = [
    "black>=23.0.0",
    "isort>=5.12.0", 
    "flake8>=6.0.0",
    "mypy>=1.3.0",
    "pre-commit>=3.0.0",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

# Optional dependencies for different features
extras_require = {
    "dev": dev_requirements,
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    "gradio": [
        "gradio==3.50.2",
    ],
    "comfyui": [
        # ComfyUI specific dependencies can be added here
    ],
    "all": dev_requirements + ["gradio==3.50.2"],
}

setup(
    name="hunyuanvideo-foley",
    version=get_version(),
    
    # Package metadata
    author="Tencent Hunyuan Team",
    author_email="hunyuan@tencent.com", 
    description="Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley",
    project_urls={
        "Homepage": "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley",
        "Repository": "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley",
        "Documentation": "https://szczesnys.github.io/hunyuanvideo-foley",
        "Paper": "https://arxiv.org/abs/2508.16930",
        "Demo": "https://huggingface.co/spaces/tencent/HunyuanVideo-Foley",
        "Models": "https://huggingface.co/tencent/HunyuanVideo-Foley",
    },
    
    # Package discovery
    packages=find_packages(
        include=["hunyuanvideo_foley", "hunyuanvideo_foley.*"]
    ),
    include_package_data=True,
    
    # Package requirements
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points for command line scripts
    entry_points={
        "console_scripts": [
            "hunyuanvideo-foley=hunyuanvideo_foley.cli:main",
        ],
    },
    
    # Package data
    package_data={
        "hunyuanvideo_foley": [
            "configs/*.yaml",
            "configs/*.yml", 
            "*.yaml",
            "*.yml",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Video",
    ],
    
    # Keywords for discoverability
    keywords=[
        "artificial intelligence", 
        "machine learning", 
        "deep learning",
        "multimodal",
        "diffusion models",
        "audio generation", 
        "foley audio",
        "video-to-audio",
        "text-to-audio",
        "pytorch",
        "huggingface",
        "tencent",
        "hunyuan"
    ],
    
    # Licensing
    license="Apache-2.0",
    
    # Build configuration
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
)