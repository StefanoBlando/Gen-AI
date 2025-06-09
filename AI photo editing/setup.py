#!/usr/bin/env python

"""Setup script for AI Photo Editor package."""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file, excluding comments and dev dependencies."""
    with open(filename, "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            # Skip empty lines, comments, and conditional dependencies
            if line and not line.startswith("#") and ";" not in line:
                requirements.append(line)
        return requirements

requirements = read_requirements("requirements.txt")

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

setup(
    name="ai-photo-editor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered photo editing with SAM segmentation and Stable Diffusion inpainting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-photo-editor",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ai-photo-editor/issues",
        "Documentation": "https://github.com/yourusername/ai-photo-editor/docs",
        "Source Code": "https://github.com/yourusername/ai-photo-editor",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-photo-editor=src.ui.gradio_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords="ai, computer-vision, image-processing, segmentation, inpainting, stable-diffusion, sam",
)
