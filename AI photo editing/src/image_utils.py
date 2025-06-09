"""Image processing utilities for AI Photo Editor."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Union
import torch


def mask_to_rgb(mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Transform a binary mask into an RGBA image for visualization.
    
    Args:
        mask: Binary mask array
        color: RGB color for the mask area
        
    Returns:
        RGBA image array
    """
    bg_transparent = np.zeros(mask.shape + (4,), dtype=np.uint8)
    
    # Color the area we will replace
    bg_transparent[mask == 1] = [*color, 127]
    
    return bg_transparent


def create_overlay_visualization(
    image: Union[Image.Image, np.ndarray], 
    mask: np.ndarray, 
    alpha: float = 0.4,
    subject_color: Tuple[int, int, int] = (255, 100, 100),
    background_color: Tuple[int, int, int] = (0, 255, 100)
) -> Image.Image:
    """Create visualization overlay for mask inspection.
    
    Args:
        image: Input image
        mask: Binary mask
        alpha: Overlay transparency
        subject_color: Color for subject areas
        background_color: Color for background areas
        
    Returns:
        Image with overlay visualization
    """
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image.copy()
    
    overlay = np.zeros_like(image_array)
    overlay[mask == 1] = background_color
    overlay[mask == 0] = subject_color
    
    result = (image_array * (1-alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(result)


def resize_image(
    image: Image.Image, 
    target_size: Tuple[int, int] = (512, 512),
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """Resize image with optional aspect ratio preservation.
    
    Args:
        image: Input image
        target_size: Target dimensions (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if not maintain_aspect_ratio:
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Calculate dimensions maintaining aspect ratio
    width, height = image.size
    target_width, target_height = target_size
    
    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize and center on target canvas
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    if (new_width, new_height) == target_size:
        return resized
    
    # Create centered image on target canvas
    result = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    result.paste(resized, (paste_x, paste_y))
    
    return result


def validate_image(image: Union[Image.Image, str]) -> Image.Image:
    """Validate and load image from various input types.
    
    Args:
        image: PIL Image, file path, or URL
        
    Returns:
        Validated PIL Image
        
    Raises:
        ValueError: If image cannot be loaded or validated
    """
    if isinstance(image, str):
        try:
            if image.startswith(('http://', 'https://')):
                import requests
                from io import BytesIO
                response = requests.get(image)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image)
        except Exception as e:
            raise ValueError(f"Cannot load image from {image}: {e}")
    
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL Image, got {type(image)}")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Validate dimensions
    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError(f"Image too small: {width}x{height}, minimum 64x64")
    
    if width > 2048 or height > 2048:
        print(f"Warning: Large image {width}x{height}, consider resizing for better performance")
    
    return image


def create_image_grid(
    images: List[Image.Image], 
    rows: int, 
    cols: int,
    padding: int = 10
) -> Image.Image:
    """Create a grid of images.
    
    Args:
        images: List of PIL Images
        rows: Number of rows in grid
        cols: Number of columns in grid
        padding: Padding between images
        
    Returns:
        Grid image
    """
    if len(images) != rows * cols:
        raise ValueError(f"Number of images ({len(images)}) doesn't match grid size ({rows}x{cols})")
    
    if not images:
        raise ValueError("No images provided")
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create grid canvas
    grid_width = cols * img_width + (cols - 1) * padding
    grid_height = rows * img_height + (rows - 1) * padding
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        
        # Resize image if dimensions don't match
        if img.size != (img_width, img_height):
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        
        grid.paste(img, (x, y))
    
    return grid


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize mask to binary values (0 and 1).
    
    Args:
        mask: Input mask array
        
    Returns:
        Normalized binary mask
    """
    if mask.dtype == bool:
        return mask.astype(np.uint8)
    
    # Normalize to 0-1 range
    mask_normalized = mask.astype(np.float32)
    if mask_normalized.max() > 1.0:
        mask_normalized = mask_normalized / 255.0
    
    # Threshold to binary
    return (mask_normalized > 0.5).astype(np.uint8)


def get_mask_statistics(mask: np.ndarray) -> dict:
    """Calculate statistics for a segmentation mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Dictionary with mask statistics
    """
    mask_binary = normalize_mask(mask)
    total_pixels = mask_binary.size
    
    foreground_pixels = np.sum(mask_binary)
    background_pixels = total_pixels - foreground_pixels
    
    coverage = foreground_pixels / total_pixels
    
    return {
        'total_pixels': total_pixels,
        'foreground_pixels': int(foreground_pixels),
        'background_pixels': int(background_pixels),
        'coverage_ratio': float(coverage),
        'coverage_percent': float(coverage * 100)
    }


def apply_morphological_operations(
    mask: np.ndarray, 
    operation: str = 'close',
    kernel_size: int = 5
) -> np.ndarray:
    """Apply morphological operations to clean up mask.
    
    Args:
        mask: Binary mask
        operation: Type of operation ('open', 'close', 'erode', 'dilate')
        kernel_size: Size of morphological kernel
        
    Returns:
        Processed mask
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping morphological operations")
        return mask
    
    mask_binary = normalize_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'open':
        result = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    elif operation == 'erode':
        result = cv2.erode(mask_binary, kernel, iterations=1)
    elif operation == 'dilate':
        result = cv2.dilate(mask_binary, kernel, iterations=1)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result


def save_comparison_image(
    original: Image.Image,
    mask: np.ndarray,
    result: Image.Image,
    save_path: str,
    title: str = "Comparison"
):
    """Save a comparison visualization of original, mask, and result.
    
    Args:
        original: Original image
        mask: Segmentation mask
        result: Generated result
        save_path: Path to save comparison image
        title: Title for the comparison
    """
    # Create visualization
    mask_vis = Image.fromarray(mask_to_rgb(mask))
    
    # Create grid
    comparison = create_image_grid([original, mask_vis, result], rows=1, cols=3)
    
    # Add title if matplotlib is available
    try:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(comparison)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except ImportError:
        # Fallback: save without title
        comparison.save(save_path)
    
    print(f"Comparison saved to {save_path}")
