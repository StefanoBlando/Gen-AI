"""Configuration management for AI Photo Editor."""

import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    checkpoint: str
    device: str = "auto"
    torch_dtype: str = "float16"
    cache_dir: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    strength: float = 0.99
    negative_prompt: str = "blurry, low quality, distortion, artifacts"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    mixed_precision: bool = True
    enable_cpu_offload: bool = True
    enable_xformers: bool = True
    enable_vae_slicing: bool = True
    memory_tracking: bool = True
    batch_size: int = 1


class ConfigManager:
    """Manages configuration loading and device detection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._setup_device()
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "config" / "default_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "models": {
                "sam": {
                    "checkpoint": "facebook/sam-vit-base",
                    "device": "auto",
                    "torch_dtype": "float16"
                },
                "inpainting": {
                    "checkpoint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    "device": "auto",
                    "torch_dtype": "float16"
                }
            },
            "generation": {
                "guidance_scale": 7.5,
                "num_inference_steps": 20,
                "strength": 0.99,
                "negative_prompt": "blurry, low quality, distortion, artifacts"
            },
            "performance": {
                "mixed_precision": True,
                "enable_cpu_offload": True,
                "enable_xformers": True,
                "enable_vae_slicing": True,
                "memory_tracking": True
            }
        }
    
    def _setup_device(self):
        """Setup device configuration based on available hardware."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float16
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32
            print("CUDA not available, using CPU")
    
    def get_sam_config(self) -> ModelConfig:
        """Get SAM model configuration."""
        sam_config = self.config["models"]["sam"]
        device = self.device if sam_config.get("device") == "auto" else sam_config["device"]
        
        return ModelConfig(
            checkpoint=sam_config["checkpoint"],
            device=str(device),
            torch_dtype=sam_config.get("torch_dtype", "float16"),
            cache_dir=sam_config.get("cache_dir")
        )
    
    def get_inpainting_config(self) -> ModelConfig:
        """Get inpainting model configuration."""
        inpainting_config = self.config["models"]["inpainting"]
        device = self.device if inpainting_config.get("device") == "auto" else inpainting_config["device"]
        
        return ModelConfig(
            checkpoint=inpainting_config["checkpoint"],
            device=str(device),
            torch_dtype=inpainting_config.get("torch_dtype", "float16"),
            cache_dir=inpainting_config.get("cache_dir")
        )
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration."""
        gen_config = self.config["generation"]
        return GenerationConfig(
            guidance_scale=gen_config.get("guidance_scale", 7.5),
            num_inference_steps=gen_config.get("num_inference_steps", 20),
            strength=gen_config.get("strength", 0.99),
            negative_prompt=gen_config.get("negative_prompt", "blurry, low quality, distortion")
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        perf_config = self.config["performance"]
        return PerformanceConfig(
            mixed_precision=perf_config.get("mixed_precision", True),
            enable_cpu_offload=perf_config.get("enable_cpu_offload", True),
            enable_xformers=perf_config.get("enable_xformers", True),
            enable_vae_slicing=perf_config.get("enable_vae_slicing", True),
            memory_tracking=perf_config.get("memory_tracking", True),
            batch_size=perf_config.get("batch_size", 1)
        )
    
    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value."""
        if section in self.config:
            self.config[section][key] = value
        else:
            self.config[section] = {key: value}
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving config: {e}")


# Global configuration instance
config_manager = ConfigManager()
