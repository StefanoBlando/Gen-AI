# AI Photo Editor

**Advanced photo editing with SAM segmentation and Stable Diffusion inpainting**

Transform your photos by selecting objects and changing backgrounds or subjects using state-of-the-art AI models.

## Features

- **Precise Object Segmentation** - Using Meta's Segment Anything Model (SAM)
- **AI-Powered Inpainting** - Stable Diffusion XL for realistic background generation
- **Interactive Web Interface** - Easy-to-use Gradio application
- **Performance Optimized** - Mixed precision training and memory optimization
- **Iterative Workflows** - Refine results with multiple editing passes
- **Parameter Analysis** - Statistical studies and performance metrics
- **Batch Processing** - Handle multiple images efficiently

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- Git

### Setup

```bash
git clone https://github.com/yourusername/ai-photo-editor.git
cd ai-photo-editor
pip install -r requirements.txt
```

For development installation:
```bash
pip install -e .
```

## Quick Start

### Command Line Interface

Launch the web interface:
```bash
python -m src.ui.gradio_app
```

### Python API

```python
from src.segmentation import SAMProcessor
from src.inpainting import DiffusionInpainter
from PIL import Image

# Initialize processors
sam = SAMProcessor()
inpainter = DiffusionInpainter()

# Load and process image
image = Image.open("your_image.jpg")
mask = sam.segment(image, points=[[150, 170], [300, 250]])

# Generate new background
result = inpainter.inpaint(
    image=image,
    mask=mask,
    prompt="beautiful sunset landscape, high quality",
    guidance_scale=7.5
)

result.save("output.jpg")
```

### Advanced Usage

```python
from src.workflows import IterativeEditor

# Iterative refinement
editor = IterativeEditor(image, mask)
final_result = editor.refine(
    prompt="Mars landscape with red rocks",
    max_iterations=3,
    guidance_scale=7.5
)
```

## Configuration

The application uses YAML configuration files. Default settings are in `config/default_config.yaml`. You can create custom configurations:

```yaml
models:
  sam:
    checkpoint: "facebook/sam-vit-base"
  inpainting:
    checkpoint: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

generation:
  guidance_scale: 7.5
  num_inference_steps: 20
```

## Examples

### Background Replacement

Replace image backgrounds while preserving subjects:

```python
# Select car, replace with Mars landscape
result = inpainter.inpaint(
    image=car_image,
    mask=car_mask,
    prompt="Mars landscape with red rocks and dramatic sky"
)
```

### Subject Replacement

Replace subjects while keeping backgrounds:

```python
# Replace car with different vehicle
inverted_mask = ~car_mask
result = inpainter.inpaint(
    image=car_image,
    mask=inverted_mask,
    prompt="red Ferrari sports car, sleek design"
)
```

## Performance

### GPU Memory Requirements

- **Minimum**: 6GB VRAM
- **Recommended**: 8GB+ VRAM
- **Optimal**: 12GB+ VRAM for batch processing

### Optimization Features

- Mixed precision inference (FP16)
- Model CPU offloading
- VAE slicing for memory efficiency
- Automatic memory tracking and cleanup

## API Reference

### SAMProcessor

```python
class SAMProcessor:
    def segment(self, image, points):
        """Generate segmentation mask from input points"""
        
    def batch_segment(self, images, points_list):
        """Process multiple images in batch"""
```

### DiffusionInpainter

```python
class DiffusionInpainter:
    def inpaint(self, image, mask, prompt, **kwargs):
        """Generate inpainting result"""
        
    def batch_inpaint(self, images, masks, prompts):
        """Process multiple inpainting tasks"""
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific tests:

```bash
python -m pytest tests/test_segmentation.py
python -m pytest tests/test_inpainting.py
```

## Development

### Project Structure

```
ai-photo-editor/
├── src/
│   ├── segmentation/     # SAM integration
│   ├── inpainting/       # Diffusion models
│   ├── ui/              # Gradio interface
│   ├── utils/           # Utilities and config
│   └── workflows/       # Advanced workflows
├── examples/            # Usage examples
├── tests/              # Test suite
└── docs/               # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for public functions
- Include unit tests for new features

## Troubleshooting

### Common Issues

**CUDA out of memory**
- Reduce batch size
- Enable CPU offloading: `pipeline.enable_model_cpu_offload()`
- Use lower resolution images

**Model download fails**
- Check internet connection
- Verify Hugging Face access
- Try manual model download

**Poor segmentation quality**
- Adjust input points
- Try multiple point combinations
- Increase image resolution

### Performance Tips

- Use FP16 precision for faster inference
- Enable XFormers for attention optimization
- Batch process multiple images when possible
- Use SSD storage for model caching

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Meta AI for the Segment Anything Model
- Stability AI for Stable Diffusion
- Hugging Face for the diffusers library
- The open-source AI community

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ai_photo_editor,
  title={AI Photo Editor: Segmentation and Inpainting Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ai-photo-editor}
}
```
