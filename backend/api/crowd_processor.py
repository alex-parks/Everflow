from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
from pathlib import Path
# import OpenImageIO as oiio  # Temporarily disabled
import asyncio
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

router = APIRouter()

UPLOAD_DIR = Path("/app/uploads/crowd_sequences")
TRAINING_DIR = Path("/app/training_data")
MODELS_DIR = Path("/app/models")

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {"status": "crowd API is working", "timestamp": str(np.datetime64('now'))}

@router.get("/test-pytorch")
async def test_pytorch():
    """Test PyTorch functionality"""
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = x.sum()
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        return {
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "test_tensor_sum": float(y.item()),
            "status": "PyTorch working correctly"
        }
    except Exception as e:
        return {"error": str(e), "status": "PyTorch test failed"}

@router.get("/test-v2v-model")
async def test_v2v_model():
    """Test creating and running a simple V2V model"""
    try:
        import torch
        import torch.nn as nn
        
        # Simple V2V model for testing
        class SimpleV2V(nn.Module):
            def __init__(self, input_channels=3, output_channels=3):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, output_channels, 3, padding=1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = torch.sigmoid(self.conv3(x))
                return x
        
        # Create model and test tensor
        model = SimpleV2V()
        test_input = torch.randn(1, 3, 64, 64)  # Batch=1, Channels=3, Height=64, Width=64
        
        # Run inference
        with torch.no_grad():
            output = model(test_input)
        
        return {
            "status": "V2V model test successful",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "input_shape": list(test_input.shape),
            "output_shape": list(output.shape),
            "output_mean": float(output.mean().item()),
            "model_name": "SimpleV2V"
        }
    except Exception as e:
        return {"error": str(e), "status": "V2V model test failed"}

@router.get("/test-simple-v2v")
async def test_simple_v2v():
    """Test the simple V2V model"""
    try:
        from simple_v2v import SimpleV2VModel, get_model, enhance_image
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color=(100, 150, 200))
        
        # Enhance it using our model
        enhanced_image = enhance_image(test_image)
        
        # Get model info
        model = get_model()
        param_count = sum(p.numel() for p in model.parameters())
        
        return {
            "status": "Simple V2V model test successful",
            "model_parameters": param_count,
            "input_size": test_image.size,
            "output_size": enhanced_image.size,
            "model_name": "SimpleV2VModel"
        }
    except Exception as e:
        return {"error": str(e), "status": "Simple V2V model test failed"}

@router.post("/upload-exr-sequence")
async def upload_exr_sequence(
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload EXR sequence for V2V processing"""
    try:
        # Create unique sequence ID
        sequence_id = str(uuid.uuid4())
        sequence_dir = UPLOAD_DIR / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        frame_count = 0
        for i, file in enumerate(files):
            if file.filename.lower().endswith(('.exr', '.jpg', '.png', '.tiff')):
                file_extension = Path(file.filename).suffix
                frame_path = sequence_dir / f"frame_{i:04d}{file_extension}"
                
                content = await file.read()
                with open(frame_path, "wb") as f:
                    f.write(content)
                frame_count += 1
        
        # Create metadata
        metadata = {
            "sequence_id": sequence_id,
            "name": name,
            "frame_count": frame_count,
            "created_at": str(np.datetime64('now')),
            "status": "uploaded"
        }
        
        # Save metadata
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "sequence_id": sequence_id,
            "message": f"Uploaded {frame_count} frames",
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/enhance-sequence/{sequence_id}")
async def enhance_sequence(sequence_id: str, background_tasks: BackgroundTasks):
    """Enhance an uploaded EXR sequence using V2V model"""
    try:
        from simple_v2v import enhance_image
        from PIL import Image
        import io
        
        sequence_dir = UPLOAD_DIR / sequence_id
        if not sequence_dir.exists():
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        # Load metadata
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create output directory
        output_dir = sequence_dir / "enhanced"
        output_dir.mkdir(exist_ok=True)
        
        # Process each frame
        frame_files = sorted(sequence_dir.glob("frame_*"))
        processed_count = 0
        
        for frame_file in frame_files:
            if frame_file.suffix.lower() in ['.exr', '.jpg', '.png', '.tiff']:
                try:
                    # For EXR files, actually read and process the EXR data
                    if frame_file.suffix.lower() == '.exr':
                        try:
                            # Try to read EXR file properly for enhancement
                            try:
                                # Method 1: Try using OpenEXR if available
                                import OpenEXR
                                import Imath
                                import struct
                                
                                # Open EXR file
                                exr_file = OpenEXR.InputFile(str(frame_file))
                                header = exr_file.header()
                                
                                # Get data window
                                dw = header['dataWindow']
                                width = dw.max.x - dw.min.x + 1
                                height = dw.max.y - dw.min.y + 1
                                
                                # Read RGB channels
                                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                                channels = ['R', 'G', 'B']
                                
                                # Read channel data
                                channel_data = {}
                                for channel in channels:
                                    if channel in exr_file.header()['channels']:
                                        channel_data[channel] = exr_file.channel(channel, FLOAT)
                                
                                exr_file.close()
                                
                                if len(channel_data) >= 3:
                                    # Convert to numpy arrays
                                    rgb_arrays = []
                                    for channel in ['R', 'G', 'B']:
                                        if channel in channel_data:
                                            data = struct.unpack('f' * width * height, channel_data[channel])
                                            array = np.array(data).reshape(height, width)
                                            rgb_arrays.append(array)
                                        else:
                                            rgb_arrays.append(np.zeros((height, width)))
                                    
                                    # Stack RGB channels
                                    rgb_image = np.stack(rgb_arrays, axis=2)
                                    
                                    # Simple tone mapping for display
                                    rgb_image = np.clip(rgb_image, 0, None)
                                    rgb_image = rgb_image ** (1.0/2.2)  # Gamma correction
                                    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
                                    
                                    # Convert to PIL Image
                                    input_image = Image.fromarray(rgb_image)
                                else:
                                    raise Exception("Not enough RGB channels found")
                                    
                            except (ImportError, Exception) as openexr_error:
                                # Method 2: Try using imageio
                                try:
                                    import imageio
                                    exr_data = imageio.imread(str(frame_file))
                                    
                                    # Handle different data types and ranges
                                    if exr_data.dtype == np.float32 or exr_data.dtype == np.float64:
                                        # Tone map HDR data
                                        exr_data = np.clip(exr_data, 0, None)
                                        exr_data = exr_data ** (1.0/2.2)  # Gamma correction
                                        exr_data = np.clip(exr_data * 255, 0, 255).astype(np.uint8)
                                    elif exr_data.dtype == np.uint16:
                                        # Convert 16-bit to 8-bit
                                        exr_data = (exr_data / 256).astype(np.uint8)
                                    
                                    input_image = Image.fromarray(exr_data)
                                    
                                except Exception as imageio_error:
                                    print(f"Could not read EXR {frame_file}: OpenEXR({openexr_error}), imageio({imageio_error})")
                                    # Create a fallback image with frame info
                                    input_image = Image.new('RGB', (512, 512), color=(60, 80, 120))
                                    from PIL import ImageDraw
                                    draw = ImageDraw.Draw(input_image)
                                    frame_num = frame_files.index(frame_file)
                                    draw.text((10, 10), f"EXR Frame {frame_num} (Fallback)", fill=(255, 255, 255))
                        
                        except Exception as e:
                            print(f"Error processing EXR {frame_file}: {e}")
                            # Final fallback
                            input_image = Image.new('RGB', (512, 512), color=(100, 100, 100))
                    else:
                        # Load regular image files
                        input_image = Image.open(frame_file)
                        input_image = input_image.convert('RGB')
                    
                    # Enhance the image
                    enhanced_image = enhance_image(input_image)
                    
                    # Save enhanced frame
                    output_path = output_dir / f"enhanced_{frame_file.stem}.jpg"
                    enhanced_image.save(output_path, quality=90)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing frame {frame_file}: {e}")
                    continue
        
        # Update metadata
        metadata["enhanced_frames"] = processed_count
        metadata["enhancement_status"] = "completed"
        metadata["enhanced_at"] = str(np.datetime64('now'))
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "sequence_id": sequence_id,
            "processed_frames": processed_count,
            "status": "Enhancement completed",
            "output_directory": str(output_dir)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@router.get("/sequences/{sequence_id}")
async def get_sequence_info(sequence_id: str):
    """Get information about a sequence"""
    try:
        sequence_dir = UPLOAD_DIR / sequence_id
        if not sequence_dir.exists():
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        # Load metadata
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check for enhanced frames
        enhanced_dir = sequence_dir / "enhanced"
        enhanced_count = len(list(enhanced_dir.glob("*.jpg"))) if enhanced_dir.exists() else 0
        
        metadata["enhanced_frames_available"] = enhanced_count
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sequence info: {str(e)}")

@router.get("/sequences/{sequence_id}/frame/{frame_index}")
async def get_enhanced_frame(sequence_id: str, frame_index: int, enhanced: bool = False):
    """Get a specific frame (original or enhanced)"""
    try:
        from fastapi.responses import FileResponse, Response
        from PIL import Image
        import io
        
        sequence_dir = UPLOAD_DIR / sequence_id
        if not sequence_dir.exists():
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        if enhanced:
            # Get enhanced frame (already JPG)
            enhanced_dir = sequence_dir / "enhanced"
            if not enhanced_dir.exists():
                raise HTTPException(status_code=404, detail="No enhanced frames available")
            
            enhanced_files = sorted(enhanced_dir.glob("enhanced_*.jpg"))
            if frame_index >= len(enhanced_files):
                raise HTTPException(status_code=404, detail="Frame not found")
            
            frame_path = enhanced_files[frame_index]
            return FileResponse(frame_path)
        else:
            # Get original frame and convert EXR to displayable format
            original_files = sorted(sequence_dir.glob("frame_*"))
            if frame_index >= len(original_files):
                raise HTTPException(status_code=404, detail="Frame not found")
            
            frame_path = original_files[frame_index]
            
            # If it's an EXR file, convert to PNG for display
            if frame_path.suffix.lower() == '.exr':
                try:
                    # Try to read EXR file properly
                    try:
                        # Method 1: Try using OpenEXR if available
                        import OpenEXR
                        import Imath
                        
                        # Open EXR file
                        exr_file = OpenEXR.InputFile(str(frame_path))
                        header = exr_file.header()
                        
                        # Get data window
                        dw = header['dataWindow']
                        width = dw.max.x - dw.min.x + 1
                        height = dw.max.y - dw.min.y + 1
                        
                        # Read RGB channels
                        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                        channels = ['R', 'G', 'B']
                        
                        # Read channel data
                        channel_data = {}
                        for channel in channels:
                            if channel in exr_file.header()['channels']:
                                channel_data[channel] = exr_file.channel(channel, FLOAT)
                        
                        exr_file.close()
                        
                        if len(channel_data) >= 3:
                            # Convert to numpy arrays
                            import struct
                            import numpy as np
                            
                            rgb_arrays = []
                            for channel in ['R', 'G', 'B']:
                                if channel in channel_data:
                                    data = struct.unpack('f' * width * height, channel_data[channel])
                                    array = np.array(data).reshape(height, width)
                                    rgb_arrays.append(array)
                                else:
                                    # If channel missing, use zeros
                                    rgb_arrays.append(np.zeros((height, width)))
                            
                            # Stack RGB channels
                            rgb_image = np.stack(rgb_arrays, axis=2)
                            
                            # Simple tone mapping
                            # Apply gamma correction and exposure
                            rgb_image = np.clip(rgb_image, 0, None)  # Remove negative values
                            rgb_image = rgb_image ** (1.0/2.2)  # Gamma correction
                            rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
                            
                            # Convert to PIL Image
                            img = Image.fromarray(rgb_image)
                        else:
                            raise Exception("Not enough RGB channels found")
                            
                    except (ImportError, Exception) as openexr_error:
                        # Method 2: Try using imageio (supports some EXR files)
                        try:
                            import imageio
                            exr_data = imageio.imread(str(frame_path))
                            
                            # Handle different data types and ranges
                            if exr_data.dtype == np.float32 or exr_data.dtype == np.float64:
                                # Tone map HDR data
                                exr_data = np.clip(exr_data, 0, None)
                                exr_data = exr_data ** (1.0/2.2)  # Gamma correction
                                exr_data = np.clip(exr_data * 255, 0, 255).astype(np.uint8)
                            elif exr_data.dtype == np.uint16:
                                # Convert 16-bit to 8-bit
                                exr_data = (exr_data / 256).astype(np.uint8)
                            
                            img = Image.fromarray(exr_data)
                            
                        except (ImportError, Exception) as imageio_error:
                            # Method 3: Fallback - create preview with filename info
                            print(f"Could not read EXR with OpenEXR ({openexr_error}) or imageio ({imageio_error})")
                            
                            # Try to get some basic info about the file
                            file_size = frame_path.stat().st_size
                            file_size_mb = file_size / (1024 * 1024)
                            
                            img = Image.new('RGB', (512, 512), color=(60, 80, 120))
                            
                            # Add informative text overlay
                            from PIL import ImageDraw, ImageFont
                            draw = ImageDraw.Draw(img)
                            font = ImageFont.load_default()
                            
                            text_lines = [
                                f"EXR Frame {frame_index}",
                                f"Size: {file_size_mb:.1f} MB",
                                f"File: {frame_path.name}",
                                "",
                                "EXR Preview",
                                "(Install OpenEXR for full support)"
                            ]
                            
                            y_offset = 150
                            for line in text_lines:
                                if line:  # Skip empty lines for spacing
                                    bbox = draw.textbbox((0, 0), line, font=font)
                                    text_width = bbox[2] - bbox[0]
                                    x = (512 - text_width) // 2
                                    draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
                                y_offset += 25
                    
                    # Convert to bytes and return
                    img_io = io.BytesIO()
                    img.save(img_io, format='PNG')
                    img_io.seek(0)
                    
                    return Response(content=img_io.getvalue(), media_type="image/png")
                    
                except Exception as e:
                    print(f"Error converting EXR: {e}")
                    # Fallback to file response (browser may not display but will download)
                    return FileResponse(frame_path)
            else:
                # For non-EXR files, return directly
                return FileResponse(frame_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get frame: {str(e)}")

# Create directories
for dir_path in [UPLOAD_DIR, TRAINING_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class PassType(str, Enum):
    BEAUTY = "beauty"
    DEPTH = "depth"
    NORMAL = "normal"
    OBJECT_ID = "object_id"
    MATERIAL_ID = "material_id"
    DIFFUSE = "diffuse"
    SPECULAR = "specular"
    EMISSION = "emission"
    UV = "uv"
    MOTION_VECTOR = "motion_vector"

@dataclass
class CrowdSequenceMetadata:
    sequence_id: str
    name: str
    frame_count: int
    resolution: tuple
    frame_rate: float
    camera_data: Dict[str, Any]
    passes: Dict[PassType, Dict[str, Any]]
    crowd_config: Dict[str, Any]
    created_at: str

@dataclass
class CrowdPersonConfig:
    person_id: str
    shirt_color: tuple  # RGB emissive color
    position_keyframes: List[Dict[str, Any]]  # frame, x, y, z, rotation
    scale: float
    proxy_mesh: str

def read_exr_channels(file_path: Path) -> Dict[str, np.ndarray]:
    """Read all channels from an EXR file"""
    try:
        img_input = oiio.ImageInput.open(str(file_path))
        if not img_input:
            raise Exception(f"Could not open {file_path}")
        
        spec = img_input.spec()
        width, height = spec.width, spec.height
        
        # Get all channel names
        channel_names = [ch.name for ch in spec.channelnames]
        
        # Read all channels
        all_data = img_input.read_image()
        img_input.close()
        
        if all_data is None:
            raise Exception("Failed to read image data")
        
        # Convert to numpy and separate channels
        all_data = np.array(all_data)
        channels = {}
        
        for i, channel_name in enumerate(channel_names):
            if len(all_data.shape) == 3:
                channels[channel_name] = all_data[:, :, i]
            else:
                channels[channel_name] = all_data
        
        return {
            'channels': channels,
            'width': width,
            'height': height,
            'channel_names': channel_names
        }
        
    except Exception as e:
        raise Exception(f"Error reading EXR channels: {str(e)}")

def analyze_crowd_proxy_data(beauty_data: np.ndarray, emission_data: np.ndarray) -> List[CrowdPersonConfig]:
    """Analyze proxy render to extract crowd person configurations"""
    crowd_people = []
    
    # Find unique emissive colors (shirt colors)
    emission_rgb = emission_data[:, :, :3] if len(emission_data.shape) == 3 else emission_data
    
    # Threshold to find emissive regions
    emission_mask = np.any(emission_rgb > 0.1, axis=2)
    
    if np.any(emission_mask):
        # Simple person detection without scipy for now
        # TODO: Implement proper connected components analysis
        
        # For now, just create a single person config if emission is detected
        person_config = CrowdPersonConfig(
            person_id="person_001",
            shirt_color=(1.0, 0.5, 0.0),  # Orange placeholder
            position_keyframes=[{
                'frame': 0,
                'screen_x': emission_data.shape[1] / 2,
                'screen_y': emission_data.shape[0] / 2,
                'bbox': {
                    'min_x': 0,
                    'max_x': emission_data.shape[1],
                    'min_y': 0,
                    'max_y': emission_data.shape[0]
                }
            }],
            scale=1.0,
            proxy_mesh="crowd_proxy_v1"
        )
        crowd_people.append(person_config)
    
    return crowd_people

@router.post("/upload-crowd-sequence")
async def upload_crowd_sequence(
    name: str = Form(...),
    frame_rate: float = Form(24.0),
    camera_data: str = Form("{}"),
    files: List[UploadFile] = File(...)
):
    """Upload a multi-pass EXR crowd sequence"""
    
    sequence_id = str(uuid.uuid4())
    sequence_dir = UPLOAD_DIR / sequence_id
    sequence_dir.mkdir(parents=True, exist_ok=True)
    
    # Group files by pass type and frame
    frame_passes = {}
    
    for file in files:
        filename = file.filename
        if not filename.endswith('.exr'):
            continue
            
        # Parse filename: beauty_0001.exr, depth_0001.exr, etc.
        parts = filename.split('_')
        if len(parts) < 2:
            continue
            
        pass_name = parts[0].lower()
        frame_num_str = parts[-1].replace('.exr', '')
        
        try:
            frame_num = int(frame_num_str)
        except ValueError:
            continue
        
        if frame_num not in frame_passes:
            frame_passes[frame_num] = {}
        
        # Save file
        file_path = sequence_dir / filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        frame_passes[frame_num][pass_name] = str(file_path)
    
    if not frame_passes:
        raise HTTPException(status_code=400, detail="No valid EXR files found")
    
    # Analyze first frame to get sequence info
    first_frame = min(frame_passes.keys())
    first_frame_passes = frame_passes[first_frame]
    
    # Read beauty pass to get resolution
    if 'beauty' in first_frame_passes:
        beauty_data = read_exr_channels(Path(first_frame_passes['beauty']))
        resolution = (beauty_data['width'], beauty_data['height'])
        
        # Analyze crowd if we have emission data
        crowd_people = []
        if 'emission' in first_frame_passes:
            emission_data = read_exr_channels(Path(first_frame_passes['emission']))
            # Combine beauty and emission for analysis
            beauty_rgb = np.dstack([
                beauty_data['channels'].get('R', np.zeros((resolution[1], resolution[0]))),
                beauty_data['channels'].get('G', np.zeros((resolution[1], resolution[0]))),
                beauty_data['channels'].get('B', np.zeros((resolution[1], resolution[0])))
            ])
            emission_rgb = np.dstack([
                emission_data['channels'].get('R', np.zeros((resolution[1], resolution[0]))),
                emission_data['channels'].get('G', np.zeros((resolution[1], resolution[0]))),
                emission_data['channels'].get('B', np.zeros((resolution[1], resolution[0])))
            ])
            crowd_people = analyze_crowd_proxy_data(beauty_rgb, emission_rgb)
    else:
        raise HTTPException(status_code=400, detail="Beauty pass required")
    
    # Create metadata
    passes_info = {}
    for pass_name in set().union(*[frame_passes[f].keys() for f in frame_passes.keys()]):
        passes_info[pass_name] = {
            'available_frames': [f for f in frame_passes.keys() if pass_name in frame_passes[f]],
            'total_frames': len([f for f in frame_passes.keys() if pass_name in frame_passes[f]])
        }
    
    metadata = CrowdSequenceMetadata(
        sequence_id=sequence_id,
        name=name,
        frame_count=len(frame_passes),
        resolution=resolution,
        frame_rate=frame_rate,
        camera_data=json.loads(camera_data),
        passes=passes_info,
        crowd_config={
            'people': [asdict(person) for person in crowd_people],
            'total_people': len(crowd_people)
        },
        created_at=str(np.datetime64('now'))
    )
    
    # Save metadata
    metadata_path = sequence_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2, default=str)
    
    return {
        "sequence_id": sequence_id,
        "message": f"Uploaded crowd sequence with {len(frame_passes)} frames",
        "passes": list(passes_info.keys()),
        "crowd_analysis": {
            "people_detected": len(crowd_people),
            "resolution": resolution
        }
    }

@router.get("/crowd-sequences")
async def list_crowd_sequences():
    """List all uploaded crowd sequences"""
    sequences = []
    
    if UPLOAD_DIR.exists():
        for seq_dir in UPLOAD_DIR.iterdir():
            if seq_dir.is_dir():
                metadata_path = seq_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        sequences.append(metadata)
    
    return sequences

@router.get("/crowd-sequences/{sequence_id}")
async def get_crowd_sequence(sequence_id: str):
    """Get detailed information about a crowd sequence"""
    sequence_dir = UPLOAD_DIR / sequence_id
    metadata_path = sequence_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Sequence not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

@router.get("/crowd-sequences/{sequence_id}/frame/{frame_num}/{pass_type}")
async def get_crowd_frame_pass(sequence_id: str, frame_num: int, pass_type: str, channel: Optional[str] = None):
    """Get a specific pass from a crowd sequence frame"""
    sequence_dir = UPLOAD_DIR / sequence_id
    
    # Find the correct file
    pattern = f"{pass_type}_{frame_num:04d}.exr"
    file_path = sequence_dir / pattern
    
    if not file_path.exists():
        # Try alternative naming
        for file in sequence_dir.glob(f"*{pass_type}*{frame_num:04d}*.exr"):
            file_path = file
            break
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame {frame_num} {pass_type} pass not found")
    
    try:
        exr_data = read_exr_channels(file_path)
        
        if channel:
            if channel not in exr_data['channels']:
                available_channels = list(exr_data['channels'].keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Channel '{channel}' not found. Available: {available_channels}"
                )
            
            # Return specific channel as image
            channel_data = exr_data['channels'][channel]
            # Convert to 8-bit for display (you might want to keep HDR for processing)
            display_data = np.clip(channel_data * 255, 0, 255).astype(np.uint8)
            
            from PIL import Image
            import io
            img = Image.fromarray(display_data, mode='L' if len(display_data.shape) == 2 else 'RGB')
            output = io.BytesIO()
            img.save(output, format='PNG')
            
            from fastapi import Response
            return Response(content=output.getvalue(), media_type="image/png")
        else:
            # Return channel information
            return {
                'width': exr_data['width'],
                'height': exr_data['height'],
                'channels': exr_data['channel_names'],
                'pass_type': pass_type,
                'frame': frame_num
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading EXR: {str(e)}")

@router.post("/crowd-sequences/{sequence_id}/prepare-training-data")
async def prepare_training_data(sequence_id: str, background_tasks: BackgroundTasks):
    """Prepare training data from crowd sequence"""
    background_tasks.add_task(generate_training_data, sequence_id)
    return {"message": "Training data preparation started"}

def generate_training_data(sequence_id: str):
    """Background task to generate training data pairs"""
    sequence_dir = UPLOAD_DIR / sequence_id
    training_output_dir = TRAINING_DIR / sequence_id
    training_output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = sequence_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Process each frame
    for frame_num in range(metadata['frame_count']):
        try:
            # Read all passes for this frame
            frame_data = {}
            for pass_type in metadata['passes'].keys():
                if frame_num in metadata['passes'][pass_type]['available_frames']:
                    file_pattern = f"{pass_type}_{frame_num:04d}.exr"
                    file_path = sequence_dir / file_pattern
                    
                    if file_path.exists():
                        frame_data[pass_type] = read_exr_channels(file_path)
            
            # Create training pair
            if 'beauty' in frame_data and 'depth' in frame_data:
                training_sample = {
                    'frame': frame_num,
                    'input': {
                        'beauty': frame_data['beauty'],
                        'depth': frame_data.get('depth'),
                        'emission': frame_data.get('emission'),
                        'normal': frame_data.get('normal')
                    },
                    'metadata': {
                        'sequence_id': sequence_id,
                        'camera_data': metadata['camera_data'],
                        'crowd_config': metadata['crowd_config']
                    }
                }
                
                # Save training sample
                sample_path = training_output_dir / f"frame_{frame_num:04d}.npz"
                np.savez_compressed(sample_path, **training_sample)
                
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            continue

@router.post("/crowd-sequences/{sequence_id}/enhance")
async def enhance_crowd_sequence(sequence_id: str, model_path: str = "best_model.pth"):
    """Enhance crowd sequence using trained V2V model"""
    try:
        from ..ai.inference_pipeline import SequenceProcessor
        
        sequence_dir = UPLOAD_DIR / sequence_id
        if not sequence_dir.exists():
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        # Initialize processor
        model_full_path = MODELS_DIR / model_path
        if not model_full_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        output_dir = UPLOAD_DIR / f"enhanced_{sequence_id}"
        processor = SequenceProcessor(
            model_path=model_full_path,
            output_dir=output_dir
        )
        
        # Process sequence
        result = processor.process_exr_sequence(sequence_dir)
        
        return {
            "message": "Sequence enhancement completed",
            "output_directory": str(result['output_directory']),
            "frames_processed": len(result['frames_saved']),
            "performance_stats": result['processing_metadata']['performance_stats']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@router.get("/crowd-sequences/{sequence_id}/enhanced/{frame_num}")
async def get_enhanced_frame(sequence_id: str, frame_num: int, format: str = "png"):
    """Get enhanced frame from processed sequence"""
    enhanced_dir = UPLOAD_DIR / f"enhanced_{sequence_id}"
    
    if format.lower() == "exr":
        frame_path = enhanced_dir / f"enhanced_{frame_num:04d}.exr"
    else:
        frame_path = enhanced_dir / f"enhanced_{frame_num:04d}.{format}"
    
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Enhanced frame not found")
    
    from fastapi import FileResponse
    return FileResponse(frame_path)

@router.get("/crowd-sequences/{sequence_id}/enhanced/download")
async def download_enhanced_sequence(sequence_id: str):
    """Download entire enhanced sequence as ZIP"""
    import zipfile
    import io
    from fastapi.responses import StreamingResponse
    
    enhanced_dir = UPLOAD_DIR / f"enhanced_{sequence_id}"
    if not enhanced_dir.exists():
        raise HTTPException(status_code=404, detail="Enhanced sequence not found")
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in enhanced_dir.glob("enhanced_*.exr"):
            zip_file.write(file_path, file_path.name)
        
        # Add processing metadata if available
        metadata_path = enhanced_dir / "processing_metadata.json"
        if metadata_path.exists():
            zip_file.write(metadata_path, "processing_metadata.json")
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=enhanced_{sequence_id}.zip"}
    )

@router.get("/models")
async def list_available_models():
    """List available trained models"""
    models = []
    
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.glob("*.pth"):
            try:
                # Get model info
                model_info = {
                    "filename": model_file.name,
                    "path": str(model_file),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                    "modified": model_file.stat().st_mtime
                }
                models.append(model_info)
            except Exception:
                continue
    
    return {"models": models}

@router.post("/train")
async def start_training(
    background_tasks: BackgroundTasks,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.0002
):
    """Start training a new V2V model"""
    try:
        from ..ai.training_pipeline import TrainingConfig, main
        
        # Create training config
        config = TrainingConfig()
        
        # Override with provided parameters
        if data_dir:
            config.data_dir = Path(data_dir)
        else:
            config.data_dir = TRAINING_DIR
            
        config.output_dir = MODELS_DIR
        config.num_epochs = epochs
        config.batch_size = batch_size
        config.learning_rate = learning_rate
        
        # Start training in background
        background_tasks.add_task(run_training, config)
        
        return {
            "message": "Training started",
            "config": {
                "data_dir": str(config.data_dir),
                "output_dir": str(config.output_dir),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

def run_training(config):
    """Run training in background"""
    try:
        import sys
        import subprocess
        
        # Save config to file
        config_path = config.output_dir / "training_config.yaml"
        config.save_to_file(config_path)
        
        # Run training script
        cmd = [
            sys.executable, 
            "-m", "backend.ai.training_pipeline",
            "--config", str(config_path)
        ]
        
        subprocess.run(cmd, cwd="/app", check=True)
        
    except Exception as e:
        print(f"Training failed: {e}")

@router.get("/training/status")
async def get_training_status():
    """Get current training status"""
    # Check for running training processes
    import psutil
    
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'training_pipeline' in ' '.join(proc.info['cmdline'] or []):
                training_processes.append({
                    'pid': proc.info['pid'],
                    'command': ' '.join(proc.info['cmdline'])
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Check for recent checkpoints
    latest_checkpoint = None
    if MODELS_DIR.exists():
        checkpoints = list(MODELS_DIR.glob("checkpoints/*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return {
        "active_training": len(training_processes) > 0,
        "training_processes": training_processes,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "models_dir": str(MODELS_DIR)
    }