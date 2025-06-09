"""AI Photo Editor - Core modules."""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "AI-powered photo editing with SAM segmentation and Stable Diffusion inpainting"

# Import principali per facilitare l'uso
from .sam_processor import SAMProcessor
from .diffusion_pipeline import DiffusionInpainter
from .gradio_app import launch_app

__all__ = [
    "SAMProcessor", 
    "DiffusionInpainter", 
    "launch_app"
]
