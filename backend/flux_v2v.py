"""Flux.1 Depth Model Integration for V2V Enhancement"""

import torch
import numpy as np
from PIL import Image
import io
import os
import logging
from typing import Optional, Dict, Any, Tuple

# Try to import diffusers components with fallbacks
try:
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    FluxPipeline = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxDepthProcessor:
    """Handles depth map processing and normalization for Flux.1 Depth model"""
    
    @staticmethod
    def normalize_depth_map(depth_image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """
        Normalize and prepare depth map for Flux.1 Depth
        
        Args:
            depth_image: PIL Image containing depth information
            target_size: Target resolution (default 1024x1024 as recommended by Flux)
            
        Returns:
            Normalized depth map as PIL Image
        """
        # Convert to numpy array
        depth_array = np.array(depth_image)
        
        # Handle different depth map formats
        if len(depth_array.shape) == 3:
            # If RGB, convert to grayscale (assuming all channels are the same)
            depth_array = depth_array[:, :, 0]
        
        # Normalize to 0-1 range
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        if depth_max > depth_min:
            depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        # Convert to uint8
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Convert to RGB (Flux expects 3-channel input)
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        
        # Create PIL Image and resize
        depth_pil = Image.fromarray(depth_rgb, mode='RGB')
        depth_pil = depth_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        return depth_pil
    
    @staticmethod
    def load_depth_from_exr(exr_path: str) -> Optional[np.ndarray]:
        """
        Load depth data from OpenEXR file
        
        Args:
            exr_path: Path to EXR file
            
        Returns:
            Depth array or None if failed
        """
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            
            # Get the data window
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            # Try to read Z channel (depth)
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            
            if 'Z' in header['channels']:
                depth_str = exr_file.channel('Z', FLOAT)
                depth = np.frombuffer(depth_str, dtype=np.float32)
                depth = depth.reshape(size[1], size[0])
                return depth
            else:
                logger.warning(f"No Z channel found in {exr_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading EXR file: {e}")
            return None


class FluxV2VModel:
    """Flux.1 Depth model for video-to-video enhancement"""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-Depth-dev-onnx", 
                 device: str = None,
                 use_fp16: bool = True):
        """
        Initialize Flux.1 Depth model
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use (cuda/cpu)
            use_fp16: Whether to use half precision
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if use_fp16 and self.device == 'cuda' else torch.float32
        self.model_id = model_id
        self.pipeline = None
        self.depth_processor = FluxDepthProcessor()
        
        logger.info(f"Initializing Flux.1 Depth model on {self.device} with dtype {self.dtype}")
    
    def load_model(self):
        """Load the Flux.1 Depth pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
            
        if self.pipeline is None:
            try:
                logger.info(f"Loading Flux.1 Depth model: {self.model_id}")
                logger.info("This may take several minutes for first download...")
                
                # Get Hugging Face token from environment
                hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
                if not hf_token:
                    logger.warning("No HUGGINGFACE_HUB_TOKEN found in environment. Model may fail if gated.")
                
                # Load using FluxPipeline with authentication
                self.pipeline = FluxPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    token=hf_token
                ).to(self.device)
                
                # Enable memory efficient attention if available
                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing()
                    logger.info("Attention slicing enabled")
                
                # Enable CPU offloading if available to reduce memory usage
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("CPU offloading enabled")
                
                logger.info("Flux.1 Depth model loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading Flux model: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                # Fallback to CPU if GPU fails
                if self.device == 'cuda':
                    logger.info("Attempting CPU fallback...")
                    self.device = 'cpu'
                    self.dtype = torch.float32
                    self.load_model()
                else:
                    raise RuntimeError(f"Failed to load Flux.1 Depth model: {str(e)}")
    
    def generate(self, 
                 depth_map: Image.Image,
                 prompt: str,
                 negative_prompt: str = "",
                 num_inference_steps: int = 28,
                 guidance_scale: float = 3.5,
                 seed: Optional[int] = None,
                 height: int = 1024,
                 width: int = 1024,
                 **kwargs) -> Image.Image:
        """
        Generate enhanced image using Flux.1 Depth
        
        Args:
            depth_map: PIL Image containing depth information
            prompt: Text prompt describing desired enhancement
            negative_prompt: What to avoid in generation
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            height: Output height
            width: Output width
            
        Returns:
            Enhanced image as PIL Image
        """
        # Load model if not already loaded
        if self.pipeline is None:
            self.load_model()
        
        # Normalize depth map
        depth_normalized = self.depth_processor.normalize_depth_map(
            depth_map, 
            target_size=(width, height)
        )
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate image
        logger.info(f"Generating with prompt: {prompt}")
        
        try:
            # Try different generation methods based on pipeline capabilities
            if hasattr(self.pipeline, '__call__'):
                # Check if pipeline supports control_image parameter
                try:
                    logger.info("Attempting generation with control_image (depth map)")
                    output = self.pipeline(
                        prompt=prompt,
                        control_image=depth_normalized,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        **kwargs
                    )
                except TypeError as te:
                    # Fallback if control_image not supported
                    logger.warning(f"control_image not supported ({te}), using basic generation")
                    output = self.pipeline(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        **kwargs
                    )
                except Exception as ge:
                    logger.error(f"Generation error: {ge}")
                    logger.error(f"Pipeline type: {type(self.pipeline)}")
                    raise RuntimeError(f"Flux generation failed: {str(ge)}")
            else:
                raise AttributeError("Pipeline not callable")
            
            if hasattr(output, 'images') and output.images:
                return output.images[0]
            else:
                logger.error(f"No images in output. Output type: {type(output)}")
                raise RuntimeError("No images generated by Flux pipeline")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    def enhance_frame(self, 
                      frame: Image.Image,
                      depth_map: Optional[Image.Image] = None,
                      prompt: str = "Enhanced cinematic quality, professional color grading",
                      **kwargs) -> Image.Image:
        """
        Enhance a single frame with optional depth map
        
        Args:
            frame: Input frame as PIL Image
            depth_map: Optional depth map (will extract if not provided)
            prompt: Enhancement prompt
            
        Returns:
            Enhanced frame
        """
        # If no depth map provided, extract from frame (simplified approach)
        if depth_map is None:
            # Convert to grayscale as simple depth approximation
            depth_array = np.array(frame.convert('L'))
            depth_map = Image.fromarray(depth_array)
            logger.info("No depth map provided, using grayscale approximation")
        
        # Generate enhanced frame
        enhanced = self.generate(
            depth_map=depth_map,
            prompt=prompt,
            height=frame.height,
            width=frame.width,
            **kwargs
        )
        
        return enhanced


# Global model instance
_global_flux_model = None

def get_flux_model() -> FluxV2VModel:
    """Get or create global Flux model instance"""
    global _global_flux_model
    if _global_flux_model is None:
        _global_flux_model = FluxV2VModel()
    return _global_flux_model


def enhance_with_flux(input_image: Image.Image, 
                      depth_map: Optional[Image.Image] = None,
                      params: Optional[Dict[str, Any]] = None) -> Image.Image:
    """
    Main enhancement function using Flux.1 Depth
    
    Args:
        input_image: Input image to enhance
        depth_map: Optional depth map
        params: Dictionary of parameters for Flux generation
        
    Returns:
        Enhanced image
    """
    model = get_flux_model()
    
    # Default parameters
    default_params = {
        'prompt': "Enhanced cinematic quality, professional VFX, high detail, photorealistic",
        'num_inference_steps': 28,
        'guidance_scale': 3.5,
        'seed': None
    }
    
    # Update with provided params
    if params:
        default_params.update(params)
    
    # Generate enhanced image
    enhanced = model.enhance_frame(
        frame=input_image,
        depth_map=depth_map,
        **default_params
    )
    
    return enhanced


def check_flux_status() -> Dict[str, Any]:
    """Check Flux model status and availability"""
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except:
        diffusers_version = "Not installed"
    
    return {
        'available': DIFFUSERS_AVAILABLE,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_type': 'Flux.1 Depth' if DIFFUSERS_AVAILABLE else 'Not Available',
        'diffusers_version': diffusers_version,
        'model_id': 'black-forest-labs/FLUX.1-Depth-dev' if DIFFUSERS_AVAILABLE else 'None',
        'diffusers_available': DIFFUSERS_AVAILABLE,
        'hf_token_configured': bool(os.getenv('HUGGINGFACE_HUB_TOKEN')),
        'note': 'Using FLUX.1-Depth-dev (requires HF authentication)'
    }