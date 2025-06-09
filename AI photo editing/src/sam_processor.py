# Module 1: Imports and Configuration (SAM part)

from PIL import Image, ImageDraw
import requests
from transformers import SamModel, SamProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Configuration
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Performance optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Set reproducible seeds
torch.manual_seed(42)
np.random.seed(42)

print("Environment configured")

# Module 2: SAM Model Loading

print("Loading SAM model...")

# Load the SAM model as we have seen in the class
# Remember to load it on the GPU by adding .to("cuda") at the end
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

# Load the SamProcessor using the facebook/sam-vit-base checkpoint
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Model validation
print(f"Model loaded: {type(model).__name__}")
print(f"Processor loaded: {type(processor).__name__}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print("SAM initialization complete")

# Module 3: SAM Functions with Mixed Precision

def mask_to_rgb(mask):
    """
    Transforms a binary mask into an RGBA image for visualization
    """
    bg_transparent = np.zeros(mask.shape + (4, ), dtype=np.uint8)
    
    # Color the area we will replace in green
    # (this vector is [Red, Green, Blue, Alpha])
    bg_transparent[mask == 1] = [0, 255, 0, 127]
    
    return bg_transparent


def get_processed_inputs(image, input_points):
    """
    Process image with SAM using mixed precision optimization
    """
    # Use the processor to generate the right inputs for SAM
    # Use "image" as your image
    # Use 'input_points' as your input_points,
    # and remember to use the option return_tensors='pt'
    # Also, remember to add .to("cuda") at the end
    inputs = processor(
        image, 
        input_points=input_points, 
        return_tensors='pt'
    ).to(device)
    
    # Call SAM with automatic mixed precision
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
    
    # Now let's post process the outputs of SAM to obtain the masks
    masks = processor.image_processor.post_process_masks(
       outputs.pred_masks.cpu(), 
       inputs["original_sizes"].cpu(), 
       inputs["reshaped_input_sizes"].cpu()
    )
    
    # Here we select the mask with the highest score
    # as the mask we will use. You can experiment with also
    # other selection criteria, for example the largest mask
    # instead of the most confident mask
    best_mask = masks[0][0][outputs.iou_scores.argmax()] 

    # Quality metrics with confidence scoring
    iou_scores = outputs.iou_scores.cpu().numpy()
    confidence_score = iou_scores.max()
    print(f"Mask confidence: {confidence_score:.3f}")
    print(f"Generated {len(masks[0][0])} candidates")
    
    # Memory usage tracking
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_memory:.2f}GB")
        torch.cuda.reset_peak_memory_stats()

    # NOTE: we invert the mask by using the ~ operator because
    # so that the subject pixels will have a value of 0 and the
    # background pixels a value of 1. This will make it more convenient
    # to infill the background
    inverted_mask = ~best_mask.cpu().numpy()
    
    mask_coverage = np.sum(inverted_mask) / inverted_mask.size
    print(f"Background coverage: {mask_coverage:.1%}")
    
    return inverted_mask


def create_overlay_visualization(image, mask, alpha=0.4):
    """
    Create visualization overlay for mask inspection
    """
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image.copy()
    
    overlay = np.zeros_like(image_array)
    overlay[mask == 1] = [0, 255, 100]
    overlay[mask == 0] = [255, 100, 100]
    
    result = (image_array * (1-alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(result)

print("SAM functions loaded with mixed precision optimization")

# Module 4: SAM Testing

print("Testing SAM segmentation...")

# Load and prepare test image
# raw_image = Image.open("car.png").convert("RGB").resize((512, 512))
# print(f"Image loaded: {raw_image.size}, mode: {raw_image.mode}")

# These are the coordinates of two points on the car
# input_points = [[[150, 170], [300, 250]]]
# print(f"Input points: {input_points[0]}")

# Generate segmentation mask
# mask = get_processed_inputs(raw_image, input_points)

# Create visualizations
# basic_viz = Image.fromarray(mask_to_rgb(mask)).resize((128, 128))
# overlay_viz = create_overlay_visualization(raw_image, mask)

# Display results
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# axes[0, 0].imshow(raw_image)
# axes[0, 0].set_title("Original Image")
# axes[0, 0].axis('off')

# axes[0, 1].imshow(mask, cmap='gray')
# axes[0, 1].set_title("Generated Mask")
# axes[0, 1].axis('off')

# axes[1, 0].imshow(overlay_viz)
# axes[1, 0].set_title("Overlay Visualization")
# axes[1, 0].axis('off')

# axes[1, 1].imshow(basic_viz)
# axes[1, 1].set_title("RGB Mask")
# axes[1, 1].axis('off')

# plt.suptitle("SAM Segmentation Results")
# plt.tight_layout()
# plt.show()

# Project requirement output
# print("Project requirement output:")
# basic_viz

class SAMProcessor:
    """SAM processor for object segmentation."""
    
    def __init__(self):
        """Initialize SAM processor."""
        self.model = model
        self.processor = processor
        self.device = device
    
    def segment(self, image, points):
        """Segment object using SAM."""
        return get_processed_inputs(image, points)
