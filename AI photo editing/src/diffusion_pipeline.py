# Module 1: Imports and Configuration (Diffusion part)

from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Configuration
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module 5: Inpainting Pipeline Setup

print("Loading inpainting pipeline...")

# Load the AutoPipelineForInpainting pipeline 
# (remember the diffusers demo in lesson 5)
# The checkpoint we want to use is 
# "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
# Remember to add torch_dtype=torch.float16 as an option

try:
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    print("SDXL inpainting pipeline loaded")
    
except Exception as e:
    print(f"SDXL loading failed: {e}")
    print("Loading standard stable-diffusion-inpainting...")
    
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("Standard inpainting pipeline loaded")

# This will make it more efficient on our hardware
pipeline.enable_model_cpu_offload()

# Additional optimizations
try:
    pipeline.enable_xformers_memory_efficient_attention()
    print("Memory efficient attention enabled")
except:
    print("Using default attention")

try:
    pipeline.enable_vae_slicing()
    print("VAE slicing enabled")
except:
    pass

# Pipeline info
print(f"Pipeline type: {type(pipeline).__name__}")
print(f"Components: {list(pipeline.components.keys())}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU Memory allocated: {allocated:.1f}GB")

print("Inpainting pipeline ready")

# Module 6: Inpainting Functions with Configuration Management

import json
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """Configuration for generation parameters"""
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    strength: float = 0.99
    negative_prompt: str = "blurry, low quality, distortion"

# Initialize default configuration
config = GenerationConfig()

def inpaint(raw_image, input_mask, prompt, negative_prompt=None, seed=74294536, cfgs=None):
    """
    Generate inpainting using diffusion pipeline with configuration support
    """
    mask_image = Image.fromarray(input_mask)
    
    # Use config defaults if not specified
    guidance_scale = cfgs if cfgs is not None else config.guidance_scale
    neg_prompt = negative_prompt if negative_prompt is not None else config.negative_prompt
    
    # Generator setup with proper device handling
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Parameter validation
    guidance_scale = np.clip(guidance_scale, 1.0, 20.0)
    
    print(f"Generating: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"CFG: {guidance_scale}, Steps: {config.num_inference_steps}, Seed: {seed}")
    
    # Use the pipeline we have created in the previous cell
    # Use "prompt" as prompt, 
    # "negative_prompt" as the negative prompt,
    # raw_image as the image,
    # mask_image as the mask_image,
    # generator as the generator and
    # cfgs as the guidance_scale
    
    try:
        with torch.autocast(device.type):
            image = pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=raw_image,
                mask_image=mask_image,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=config.num_inference_steps,
                strength=config.strength
            ).images[0]
        
        print("Generation completed")
        return image
        
    except Exception as e:
        print(f"Generation error: {e}")
        return raw_image


def update_config(guidance_scale=None, num_steps=None, strength=None, negative_prompt=None):
    """
    Update generation configuration parameters
    """
    global config
    
    if guidance_scale is not None:
        config.guidance_scale = guidance_scale
    if num_steps is not None:
        config.num_inference_steps = num_steps
    if strength is not None:
        config.strength = strength
    if negative_prompt is not None:
        config.negative_prompt = negative_prompt
    
    print(f"Configuration updated: CFG={config.guidance_scale}, Steps={config.num_inference_steps}")


def parameter_study_with_metrics(raw_image, input_mask, base_prompt, scales):
    """
    Parameter study with performance metrics tracking
    """
    results = []
    metrics = {
        'guidance_scales': scales,
        'generation_times': [],
        'memory_usage': []
    }
    
    for scale in scales:
        print(f"Testing CFG {scale}...")
        
        # Measure generation time
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = inpaint(raw_image, input_mask, base_prompt, 
                           config.negative_prompt, seed=42, cfgs=scale)
            end_event.record()
            
            torch.cuda.synchronize()
            generation_time = start_event.elapsed_time(end_event) / 1000
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            import time
            start_time = time.time()
            result = inpaint(raw_image, input_mask, base_prompt, 
                           config.negative_prompt, seed=42, cfgs=scale)
            generation_time = time.time() - start_time
            memory_used = 0
        
        results.append(result)
        metrics['generation_times'].append(generation_time)
        metrics['memory_usage'].append(memory_used)
    
    # Save metrics for analysis
    with open('parameter_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Parameter study completed. Metrics saved to parameter_metrics.json")
    return results, metrics

print("Inpainting functions loaded with configuration management")

# Module 7: Inpainting Testing with Statistical Analysis

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# print("Testing inpainting with statistical parameter analysis...")

# Define prompts
# prompt = "a car driving on Mars. Studio lights, 1970s, cinematic lighting, high quality"
# negative_prompt = "artifacts, low quality, distortion, blurry"

# Generate main result
# print("Generating Mars landscape...")
# image = inpaint(raw_image, mask, prompt, negative_prompt)

# Project requirement: 3-panel visualization
# print("Creating 3-panel result...")
# fig = make_image_grid([
#     raw_image, 
#     Image.fromarray(mask_to_rgb(mask)), 
#     image.resize((512, 512))
# ], rows=1, cols=3)

# print("Project requirement - 3-panel result:")
# display(fig)

# Statistical parameter analysis
# print("\nConducting statistical parameter analysis...")

# Test range of guidance scales
# guidance_scales = np.linspace(1.0, 20.0, 8)
# study_results, metrics = parameter_study_with_metrics(
#     raw_image, mask, prompt, guidance_scales
# )

class DiffusionInpainter:
    """Diffusion pipeline for inpainting."""
    
    def __init__(self):
        """Initialize diffusion inpainter."""
        self.pipeline = pipeline
        self.config = config
        self.device = device
    
    def inpaint(self, image, mask, prompt, **kwargs):
        """Generate inpainting result."""
        return inpaint(image, mask, prompt, **kwargs)
    
    def parameter_study(self, image, mask, prompt, scales):
        """Run parameter study."""
        return parameter_study_with_metrics(image, mask, prompt, scales)
