# Everflow AI VFX Enhancement Platform

A multi-container AI platform for advanced VFX generation using Hunyuan Image 3.0 and Hunyuan3D 2.1 models.

## Features

- **Hunyuan Image 3.0**: State-of-the-art text-to-image generation
- **Hunyuan3D 2.1**: Image-to-3D model conversion
- **Multi-container Architecture**: Isolated services for different PyTorch/CUDA requirements
- **Web Interface**: React-based UI for easy interaction

## Architecture

The platform uses a multi-container architecture to avoid dependency conflicts:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Frontend   │────▶│  API Gateway │────▶│ Hunyuan Image   │
│  (Port 5173)│     │  (Port 4005) │     │ (Port 4006)     │
└─────────────┘     └──────────────┘     │ PyTorch 2.5.1   │
                            │             └─────────────────┘
                            │             ┌─────────────────┐
                            └────────────▶│ Hunyuan3D       │
                                         │ (Port 4007)     │
                                         │ PyTorch 2.1.2   │
                                         └─────────────────┘
```

## Prerequisites

- Docker or Podman
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- 200GB+ free disk space for models
- Python 3.10+
- Node.js 18+

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/everflow.git
   cd everflow
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Hugging Face token and other settings
   ```

3. **Download models** (if not already present)
   ```bash
   # Models should be placed in:
   # backend/HunyuanImage-3-Model/
   # backend/HunyuanImage-3-Official/
   # backend/Hunyuan3D-2.1/
   ```

4. **Build and start services**
   ```bash
   # Using Docker Compose
   docker-compose up -d

   # Or using Podman Compose
   podman-compose up -d
   ```

5. **Access the application**
   - Frontend: http://localhost:5173
   - API Gateway: http://localhost:4005
   - Hunyuan Image Service: http://localhost:4006
   - Hunyuan3D Service: http://localhost:4007

## Services

### API Gateway (Port 4005)
Routes requests to appropriate backend services.

### Hunyuan Image Service (Port 4006)
- Handles text-to-image generation
- Uses PyTorch 2.5.1
- Requires ~14GB GPU memory

### Hunyuan3D Service (Port 4007)
- Handles image-to-3D conversion
- Uses PyTorch 2.1.2 with xformers
- Optimized for 3D model generation

### Frontend (Port 5173)
- React-based user interface
- Real-time job status tracking
- Image upload and preview

## API Endpoints

### Image Generation
```bash
POST /api/hunyuan3d/generate-image
{
  "prompt": "A futuristic city",
  "width": 512,
  "height": 512,
  "num_inference_steps": 50
}
```

### 3D Generation
```bash
POST /api/hunyuan3d/generate-3d/{image_id}
```

### Job Status
```bash
GET /api/hunyuan3d/job-status/{job_id}
```

## GPU Memory Requirements

- **Hunyuan Image 3.0**: ~14-16GB VRAM
- **Hunyuan3D 2.1**: ~8-10GB VRAM

For GPUs with less memory, the models will automatically offload to CPU (slower performance).

## Development

### Running services individually

```bash
# Hunyuan Image service
python backend/hunyuan_image_service_final.py

# Hunyuan3D service
python backend/hunyuan3d_service.py

# API Gateway
python backend/gateway.py

# Frontend
cd frontend && npm run dev
```

### Building Docker images

```bash
# Build all images
./build_docker.sh

# Build individual images
docker build -f Dockerfile.gateway -t everflow-gateway .
docker build -f Dockerfile.hunyuan-image -t everflow-hunyuan-image .
docker build -f Dockerfile.hunyuan3d -t everflow-hunyuan3d .
```

## Troubleshooting

### CUDA Out of Memory
- Reduce image resolution
- Reduce inference steps
- Enable CPU offloading in service configuration

### Model Loading Issues
- Ensure model files are in correct directories
- Check file permissions
- Verify Hugging Face token is set

### Container Issues
```bash
# Check container logs
podman logs vfx-hunyuan-image
podman logs vfx-hunyuan3d
podman logs vfx-backend-gateway

# Restart services
podman-compose restart
```

## Important Notes

- **Model Files**: The Hunyuan model files (160GB+) are NOT included in the repository
- **GPU Memory**: A GPU with 24GB+ VRAM is recommended for optimal performance
- **First Run**: Initial model loading can take 2-3 minutes

## License

This project uses models under their respective licenses:
- Hunyuan Image 3.0: [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE)
- Hunyuan3D 2.1: [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/Hunyuan3D/blob/main/LICENSE)

## Contributing

Contributions are welcome! Please ensure you:
1. Don't commit model files or sensitive data
2. Follow the existing code style
3. Test your changes thoroughly

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check existing issues for solutions