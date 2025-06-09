#!/usr/bin/env python3
"""Basic usage examples for AI Photo Editor."""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import requests
from io import BytesIO

from src.sam_processor import SAMProcessor
from src.diffusion_pipeline import DiffusionInpainter
from src.image_utils import create_image_grid, save_comparison_image


def download_sample_image(url: str, save_path: str) -> Image.Image:
    """Download a sample image for testing."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image.save(save_path)
        print(f"Downloaded sample image to {save_path}")
        return image
    except Exception as e:
        print(f"Failed to download image: {e}")
        return None


def example_1_basic_background_replacement():
    """Example 1: Basic background replacement."""
    print("Example 1: Basic Background Replacement")
    print("-" * 40)
    
    # Initialize processors
    sam = SAMProcessor()
    inpainter = DiffusionInpainter()
    
    # Load or download sample image
    sample_path = "examples/sample_car.jpg"
    if not os.path.exists(sample_path):
        # Use a car image URL for testing
        image_url = "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=512"
        image = download_sample_image(image_url, sample_path)
        if image is None:
            print("Skipping example - no sample image available")
            return
    else:
        image = Image.open(sample_path)
    
    # Resize for processing
    image = image.resize((512, 512))
    
    # Define points on the car (you may need to adjust these)
    points = [[150, 170], [300, 250]]
    print(f"Using segmentation points: {points}")
    
    # Segment the car
    print("Segmenting object...")
    mask = sam.segment(image, points)
    
    # Generate new background
    prompt = "beautiful sunset landscape with mountains, golden hour lighting"
    print(f"Generating new background: '{prompt}'")
    
    result = inpainter.inpaint(
        image=image,
        mask=mask,
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=20
    )
    
    # Save results
    os.makedirs("examples/outputs", exist_ok=True)
    result.save("examples/outputs/example1_result.jpg")
    save_comparison_image(
        image, mask, result, 
        "examples/outputs/example1_comparison.jpg",
        "Background Replacement Example"
    )
    
    print("✓ Example 1 completed. Check examples/outputs/")


def example_2_subject_replacement():
    """Example 2: Subject replacement (keeping background)."""
    print("\nExample 2: Subject Replacement")
    print("-" * 40)
    
    # Initialize processors
    sam = SAMProcessor()
    inpainter = DiffusionInpainter()
    
    # Use the same sample image
    sample_path = "examples/sample_car.jpg"
    if not os.path.exists(sample_path):
        print("Sample image not found. Run example 1 first.")
        return
    
    image = Image.open(sample_path).resize((512, 512))
    
    # Segment the car
    points = [[150, 170], [300, 250]]
    mask = sam.segment(image, points)
    
    # Invert mask to replace subject instead of background
    inverted_mask = 1 - mask
    
    # Generate new subject
    prompt = "red Ferrari sports car, sleek design, studio lighting"
    print(f"Generating new subject: '{prompt}'")
    
    result = inpainter.inpaint(
        image=image,
        mask=inverted_mask,
        prompt=prompt,
        guidance_scale=8.0,
        num_inference_steps=25
    )
    
    # Save results
    result.save("examples/outputs/example2_result.jpg")
    save_comparison_image(
        image, inverted_mask, result,
        "examples/outputs/example2_comparison.jpg", 
        "Subject Replacement Example"
    )
    
    print("✓ Example 2 completed. Check examples/outputs/")


def example_3_multiple_variations():
    """Example 3: Generate multiple variations with different prompts."""
    print("\nExample 3: Multiple Variations")
    print("-" * 40)
    
    # Initialize processors
    sam = SAMProcessor()
    inpainter = DiffusionInpainter()
    
    # Use sample image
    sample_path = "examples/sample_car.jpg"
    if not os.path.exists(sample_path):
        print("Sample image not found. Run example 1 first.")
        return
    
    image = Image.open(sample_path).resize((512, 512))
    
    # Segment once
    points = [[150, 170], [300, 250]]
    mask = sam.segment(image, points)
    
    # Different background prompts
    prompts = [
        "tropical beach with palm trees and blue ocean",
        "snowy mountain landscape with pine trees",
        "futuristic cyberpunk city with neon lights",
        "peaceful countryside with green fields"
    ]
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Generating variation {i+1}: {prompt[:30]}...")
        
        result = inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            guidance_scale=7.5,
            seed=42 + i  # Different seed for variation
        )
        
        results.append(result)
        result.save(f"examples/outputs/example3_variation_{i+1}.jpg")
    
    # Create grid of all variations
    grid = create_image_grid(results, rows=2, cols=2)
    grid.save("examples/outputs/example3_grid.jpg")
    
    print("✓ Example 3 completed. Check examples/outputs/")


def example_4_parameter_study():
    """Example 4: Study the effect of different guidance scales."""
    print("\nExample 4: Parameter Study")
    print("-" * 40)
    
    # Initialize processors
    sam = SAMProcessor()
    inpainter = DiffusionInpainter()
    
    # Use sample image
    sample_path = "examples/sample_car.jpg"
    if not os.path.exists(sample_path):
        print("Sample image not found. Run example 1 first.")
        return
    
    image = Image.open(sample_path).resize((512, 512))
    
    # Segment once
    points = [[150, 170], [300, 250]]
    mask = sam.segment(image, points)
    
    # Test different guidance scales
    guidance_scales = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    prompt = "Mars landscape with red rocks and dramatic sky"
    
    results = []
    for scale in guidance_scales:
        print(f"Testing guidance scale: {scale}")
        
        result = inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            guidance_scale=scale,
            seed=42  # Same seed for comparison
        )
        
        results.append(result)
        result.save(f"examples/outputs/example4_cfg_{scale}.jpg")
    
    # Create comparison grid
    grid = create_image_grid(results, rows=2, cols=3)
    grid.save("examples/outputs/example4_parameter_study.jpg")
    
    print("✓ Example 4 completed. Check examples/outputs/")


def main():
    """Run all examples."""
    print("AI Photo Editor - Basic Usage Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("examples/outputs", exist_ok=True)
    
    try:
        example_1_basic_background_replacement()
        example_2_subject_replacement()
        example_3_multiple_variations()
        example_4_parameter_study()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the examples/outputs/ directory for results.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. Installed all requirements: pip install -r requirements.txt")
        print("2. CUDA available (recommended)")
        print("3. Internet connection for model downloads")


if __name__ == "__main__":
    main()
