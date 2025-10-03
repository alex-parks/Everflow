import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Download, FileImage, Loader2, CheckCircle, AlertCircle, Eye, Zap, Settings, Type, Box, Sparkles, FileDown, Package } from 'lucide-react';
import './Hunyuan3DWorkflow.css';
import Mesh3DViewer from './Mesh3DViewer';

const Hunyuan3DWorkflow = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [inputMethod, setInputMethod] = useState('text'); // 'text' or 'upload'
  const [textPrompt, setTextPrompt] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [generated3DModel, setGenerated3DModel] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [imageGenerationProgress, setImageGenerationProgress] = useState(0);
  const [modelGenerationProgress, setModelGenerationProgress] = useState(0);
  const [modelProgressMessage, setModelProgressMessage] = useState('');
  const [lastLoggedMessage, setLastLoggedMessage] = useState('');
  const [debugLogs, setDebugLogs] = useState([]);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [isGenerating3D, setIsGenerating3D] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const debugLogRef = useRef(null);

  const addDebugLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setDebugLogs(prev => {
      const newLogs = [...prev, { timestamp, message, type }];
      setTimeout(() => {
        if (debugLogRef.current) {
          debugLogRef.current.scrollTop = debugLogRef.current.scrollHeight;
        }
      }, 100);
      return newLogs;
    });
  };

  const resetWorkflow = () => {
    addDebugLog('🔄 Resetting 3D generation workflow...', 'info');
    setCurrentStep(1);
    setInputMethod('text');
    setTextPrompt('');
    setUploadedImage(null);
    setGeneratedImage(null);
    setGenerated3DModel(null);
    setUploadProgress(0);
    setImageGenerationProgress(0);
    setModelGenerationProgress(0);
    setModelProgressMessage('');
    setLastLoggedMessage('');
    setDebugLogs([]);
    setIsGeneratingImage(false);
    setIsGenerating3D(false);
    setIsUploading(false);
    setTimeout(() => {
      addDebugLog('✨ Workflow reset - ready for new 3D generation', 'info');
    }, 100);
  };

  const handleTextToImage = async () => {
    if (!textPrompt.trim()) return;

    setIsGeneratingImage(true);
    setImageGenerationProgress(0);
    addDebugLog(`🎨 Starting text-to-image generation with prompt: "${textPrompt}"`, 'info');

    try {
      // TODO: Replace with actual HunyuanImage-3.0 API call
      addDebugLog('⚠️ HunyuanImage-3.0 endpoint not yet configured', 'info');
      addDebugLog('📝 This is a placeholder for Hunyuan text-to-image generation', 'info');

      // Simulate image generation
      setImageGenerationProgress(20);
      addDebugLog('🔧 Initializing HunyuanImage-3.0 model...', 'info');
      await new Promise(resolve => setTimeout(resolve, 1500));

      setImageGenerationProgress(60);
      addDebugLog('🎨 Generating high-quality image from text...', 'info');
      await new Promise(resolve => setTimeout(resolve, 2000));

      setImageGenerationProgress(100);
      addDebugLog('✅ Image generation complete!', 'success');

      // Mock generated image
      const mockImage = {
        url: `data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTEyIiBoZWlnaHQ9IjUxMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJsaW5lYXItZ3JhZGllbnQoNDVkZWcsICM4YjVjZjYsICM3YzNhZWQpIi8+PHRleHQgeD0iNTAlIiB5PSI0NSUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIyNCIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkdlbmVyYXRlZCBJbWFnZTwvdGV4dD48dGV4dCB4PSI1MCUiIHk9IjU1JSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE0IiBmaWxsPSJyZ2JhKDI1NSwyNTUsMjU1LDAuNykiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5mcm9tIEh1bnl1YW5JbWFnZS0zLjA8L3RleHQ+PC9zdmc+`,
        metadata: {
          prompt: textPrompt,
          resolution: '512x512',
          model: 'HunyuanImage-3.0'
        }
      };

      setGeneratedImage(mockImage);
      addDebugLog('⏭️ Transitioning to Step 2: Image-to-3D Generation', 'info');
      setCurrentStep(2);

    } catch (error) {
      addDebugLog(`❌ Image generation failed: ${error.message}`, 'error');
      console.error('Image generation error:', error);
    } finally {
      setIsGeneratingImage(false);
    }
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    addDebugLog(`📁 Uploading image: ${file.name}`, 'info');

    try {
      // Create a new Image object to get dimensions
      const img = new Image();
      const imageUrl = URL.createObjectURL(file);

      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imageUrl;
      });

      setUploadProgress(30);
      addDebugLog('📤 Uploading to Hunyuan3D server...', 'info');

      // Upload to backend API
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch('http://localhost:4005/api/hunyuan3d/upload-image', {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      const uploadResult = await uploadResponse.json();
      setUploadProgress(100);

      const uploadedImageData = {
        url: imageUrl,
        imageId: uploadResult.image_id,
        metadata: {
          filename: file.name,
          size: file.size,
          type: file.type,
          width: img.naturalWidth,
          height: img.naturalHeight,
          aspectRatio: (img.naturalWidth / img.naturalHeight).toFixed(2),
          serverPath: uploadResult.upload_path
        }
      };

      setUploadedImage(uploadedImageData);
      addDebugLog('✅ Image uploaded to server successfully!', 'success');
      addDebugLog(`🆔 Server Image ID: ${uploadResult.image_id}`, 'info');
      addDebugLog(`📐 Image dimensions: ${img.naturalWidth}x${img.naturalHeight} (${uploadedImageData.metadata.aspectRatio}:1)`, 'info');
      addDebugLog('⏭️ Transitioning to Step 2: Image-to-3D Generation', 'info');
      setCurrentStep(2);

    } catch (error) {
      addDebugLog(`❌ Upload failed: ${error.message}`, 'error');
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleImageTo3D = async () => {
    const sourceImage = generatedImage || uploadedImage;
    if (!sourceImage) return;

    setIsGenerating3D(true);
    setModelGenerationProgress(0);
    setModelProgressMessage('Initializing 3D generation...');
    setCurrentStep(3);

    addDebugLog('⏭️ Transitioning to Step 3: 3D Model Generation', 'info');
    addDebugLog('🚀 Starting image-to-3D conversion with Hunyuan3D-2.1...', 'info');

    try {
      // Get image ID for uploaded image or use generated image
      let imageId = null;
      if (uploadedImage?.imageId) {
        imageId = uploadedImage.imageId;
        addDebugLog(`🆔 Using uploaded image ID: ${imageId}`, 'info');
      } else {
        // For generated images, we would need to upload them first
        addDebugLog('📝 Generated image workflow not yet implemented', 'info');
        throw new Error('Generated image to 3D not yet supported');
      }

      setModelGenerationProgress(10);
      addDebugLog('🔧 Starting Hunyuan3D-2.1 processing...', 'info');

      // Start 3D generation job
      const generationResponse = await fetch(`http://localhost:4005/api/hunyuan3d/generate-3d/${imageId}`, {
        method: 'POST'
      });

      if (!generationResponse.ok) {
        throw new Error(`3D generation failed: ${generationResponse.statusText}`);
      }

      const generationResult = await generationResponse.json();
      const jobId = generationResult.job_id;

      addDebugLog(`✅ 3D generation job started: ${jobId}`, 'success');
      addDebugLog('⏳ Monitoring processing progress...', 'info');
      console.log(`Frontend: Starting to poll job ${jobId}`);

      // Poll job status
      let completed = false;
      let attempts = 0;
      const maxAttempts = 120; // 10 minutes max

      while (!completed && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second for more frequent updates
        attempts++;

        const statusResponse = await fetch(`http://localhost:4005/api/hunyuan3d/job-status/${jobId}`);
        if (!statusResponse.ok) {
          if (statusResponse.status === 404) {
            throw new Error(`Job ${jobId} not found - possibly due to server restart`);
          }
          throw new Error(`Failed to get job status: ${statusResponse.statusText}`);
        }

        const statusResult = await statusResponse.json();
        const { status, progress, progress_message, error, result } = statusResult;

        // Debug logging for troubleshooting
        console.log(`Job ${jobId}: Status=${status}, Progress=${progress}%, Message="${progress_message}"`);

        setModelGenerationProgress(progress);

        // Update the progress message for display
        if (progress_message) {
          setModelProgressMessage(progress_message);
          // Only log if message changed
          if (lastLoggedMessage !== progress_message) {
            addDebugLog(`⚙️ ${progress_message}`, 'info');
            setLastLoggedMessage(progress_message);
          }
        }

        if (status === 'completed') {
          completed = true;
          setModelGenerationProgress(100);
          setModelProgressMessage('3D generation complete!');
          addDebugLog('✅ 3D model generation complete!', 'success');

          // Set the generated 3D model data
          const generated3DModel = {
            id: jobId,
            jobId: jobId,
            meshUrl: result?.mesh_path || null,
            exports: result?.exports || {},
            metadata: {
              vertices: result?.metadata?.model_vertices || 'Unknown',
              faces: result?.metadata?.model_faces || 'Unknown',
              materials: ['PBR_Material'],
              format: 'Multiple formats available'
            }
          };

          setGenerated3DModel(generated3DModel);
          addDebugLog('🎉 3D asset creation complete!', 'success');
          addDebugLog(`📁 Available formats: ${Object.keys(result?.exports || {}).join(', ')}`, 'info');
          addDebugLog('⏭️ Transitioning to Step 4: 3D Preview & Download', 'info');
          setCurrentStep(4);

        } else if (status === 'failed') {
          throw new Error(error || 'Unknown processing error');
        }
      }

      if (!completed) {
        throw new Error('3D generation timeout - process took longer than expected');
      }

    } catch (error) {
      addDebugLog(`❌ 3D generation failed: ${error.message}`, 'error');
      console.error('3D generation error:', error);
    } finally {
      setIsGenerating3D(false);
    }
  };

  const handleDownload = async (format) => {
    try {
      if (!generated3DModel?.jobId) {
        throw new Error('No job ID available for download');
      }

      addDebugLog(`📥 Starting download of ${format.toUpperCase()} file...`, 'info');

      const downloadUrl = `http://localhost:4005/api/hunyuan3d/download/${generated3DModel.jobId}/${format}`;

      // Create a temporary link and trigger download
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `model_${generated3DModel.jobId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      addDebugLog(`✅ ${format.toUpperCase()} download started`, 'success');

    } catch (error) {
      addDebugLog(`❌ Download failed: ${error.message}`, 'error');
      console.error('Download error:', error);
    }
  };

  const handleDownloadTextures = async () => {
    try {
      if (!generated3DModel || !generated3DModel.id) {
        throw new Error('No 3D model data available');
      }

      addDebugLog('📥 Starting texture download...', 'info');

      // For now, download the processed image as the texture
      // In a full implementation, this would download a texture pack
      const textureUrl = `http://localhost:4005/api/hunyuan3d/download/${generated3DModel.id}/texture`;

      const link = document.createElement('a');
      link.href = textureUrl;
      link.download = `textures_${generated3DModel.id}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      addDebugLog('✅ Texture download started', 'success');

    } catch (error) {
      addDebugLog(`❌ Texture download failed: ${error.message}`, 'error');
      console.error('Texture download error:', error);
    }
  };

  const getStepStatus = (step) => {
    if (step < currentStep) return 'completed';
    if (step === currentStep) return 'active';
    return 'pending';
  };

  const getCurrentImage = () => generatedImage || uploadedImage;

  // Initialize debug log
  useEffect(() => {
    addDebugLog('🚀 Hunyuan 3D Generation Workflow initialized', 'info');
    addDebugLog('🎯 Text→Image→3D→Download pipeline ready', 'info');
    addDebugLog('⚡ Supports text prompts or direct image upload', 'info');
  }, []);

  return (
    <div className="simple-v2v-workflow">
      <div className="workflow-header">
        <h1>3D AI Generation Workflow</h1>
        <p>Create 3D assets from text prompts or images using Hunyuan AI models</p>
      </div>

      {/* Progress Steps */}
      <div className="workflow-progress">
        <div className="progress-steps">
          {[
            { number: 1, title: 'Text/Image Input', icon: inputMethod === 'text' ? Type : Upload },
            { number: 2, title: 'Image Ready', icon: Eye },
            { number: 3, title: '3D Generation', icon: Box },
            { number: 4, title: '3D Preview & Download', icon: Download }
          ].map(({ number, title, icon: Icon }) => {
            const status = getStepStatus(number);
            return (
              <div key={number} className={`progress-step ${status}`}>
                <div className="step-icon">
                  {status === 'completed' ? (
                    <CheckCircle className="w-6 h-6" />
                  ) : (
                    <Icon className="w-6 h-6" />
                  )}
                </div>
                <span className="step-title">{title}</span>
              </div>
            );
          })}
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${((currentStep - 1) / 3) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Step 1: Input Method */}
      {currentStep === 1 && (
        <div className="step-content">
          <h3>
            <Sparkles className="w-6 h-6" />
            Choose Input Method
          </h3>

          {/* Input Method Toggle */}
          <div className="input-method-toggle">
            <button
              className={`method-button ${inputMethod === 'text' ? 'active' : ''}`}
              onClick={() => {
                setInputMethod('text');
                addDebugLog('📝 Switched to text-to-image mode', 'info');
              }}
            >
              <Type className="w-5 h-5" />
              Text-to-Image
            </button>
            <button
              className={`method-button ${inputMethod === 'upload' ? 'active' : ''}`}
              onClick={() => {
                setInputMethod('upload');
                addDebugLog('📁 Switched to image upload mode', 'info');
              }}
            >
              <Upload className="w-5 h-5" />
              Upload Image
            </button>
          </div>

          {/* Text Input */}
          {inputMethod === 'text' && (
            <div className="text-input-section">
              <label>Describe the image you want to generate:</label>
              <textarea
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                placeholder="e.g., A futuristic robot standing in a cyberpunk city, highly detailed, 8k resolution..."
                rows={4}
                className="text-prompt-input"
              />

              <button
                className="primary-button"
                onClick={handleTextToImage}
                disabled={!textPrompt.trim() || isGeneratingImage}
              >
                {isGeneratingImage ? (
                  <Loader2 className="w-5 h-5 spinner" />
                ) : (
                  <Sparkles className="w-5 h-5" />
                )}
                {isGeneratingImage ? 'Generating Image...' : 'Generate Image'}
              </button>

              {isGeneratingImage && (
                <div className="upload-progress">
                  <div className="upload-progress-bar">
                    <div
                      className="upload-progress-fill"
                      style={{ width: `${imageGenerationProgress}%` }}
                    ></div>
                  </div>
                  <p>{imageGenerationProgress}% generated</p>
                </div>
              )}
            </div>
          )}

          {/* Image Upload */}
          {inputMethod === 'upload' && (
            <div className={`upload-zone ${isUploading ? 'uploading' : ''}`}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.tiff,.tif,.bmp"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />

              <FileImage className="w-16 h-16" />
              <h4>Upload Source Image</h4>
              <p>Select an image to convert to 3D (JPG, PNG, WebP, TIFF, BMP)<br/>
            <span style={{fontSize: '0.9rem', color: 'rgba(255,255,255,0.6)'}}>Any aspect ratio and resolution supported</span></p>

              <button
                className="upload-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                {isUploading ? (
                  <Loader2 className="w-5 h-5 spinner" />
                ) : (
                  <Upload className="w-5 h-5" />
                )}
                {isUploading ? 'Uploading...' : 'Select Image'}
              </button>

              {isUploading && (
                <div className="upload-progress">
                  <div className="upload-progress-bar">
                    <div
                      className="upload-progress-fill"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <p>{uploadProgress}% uploaded</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Step 2: Image Ready */}
      {currentStep === 2 && getCurrentImage() && (
        <div className="step-content">
          <h3>
            <Eye className="w-6 h-6" />
            Image Ready for 3D Generation
          </h3>

          <div className="image-preview">
            <h4>Source Image</h4>
            <div className="image-container">
              <img
                src={getCurrentImage().url}
                alt="Source for 3D generation"
                className="preview-image"
              />
            </div>

            <div className="image-info">
              {generatedImage && (
                <>
                  <p><strong>Generated from:</strong> "{generatedImage.metadata.prompt}"</p>
                  <p><strong>Resolution:</strong> {generatedImage.metadata.resolution}</p>
                </>
              )}
              {uploadedImage && (
                <>
                  <p><strong>Uploaded:</strong> {uploadedImage.metadata.filename}</p>
                  <p><strong>Dimensions:</strong> {uploadedImage.metadata.width}×{uploadedImage.metadata.height}px</p>
                  <p><strong>Aspect Ratio:</strong> {uploadedImage.metadata.aspectRatio}:1</p>
                  <p><strong>File Size:</strong> {(uploadedImage.metadata.size / 1024 / 1024).toFixed(2)} MB</p>
                </>
              )}
            </div>
          </div>

          <div className="action-buttons">
            <button
              className="primary-button"
              onClick={handleImageTo3D}
              disabled={isGenerating3D}
            >
              <Box className="w-5 h-5" />
              Generate 3D Model
            </button>

            <button
              className="secondary-button"
              onClick={resetWorkflow}
            >
              <Upload className="w-5 h-5" />
              Start Over
            </button>
          </div>
        </div>
      )}

      {/* Step 3: 3D Generation */}
      {currentStep === 3 && (
        <div className="step-content">
          <h3>
            <Box className="w-6 h-6" />
            Generating 3D Model
          </h3>

          <div className="processing-info">
            <div className="processing-status">
              <Loader2 className="w-6 h-6 spinner" />
              Converting image to 3D mesh with PBR textures...
            </div>

            <div className="processing-progress">
              <div
                className="processing-progress-fill"
                style={{ width: `${modelGenerationProgress}%` }}
              ></div>
            </div>
            <p className="processing-percentage">{modelGenerationProgress}% complete</p>
            {modelProgressMessage && (
              <p className="processing-details">{modelProgressMessage}</p>
            )}
            <p style={{fontSize: '12px', color: '#666', marginTop: '10px'}}>
              Debug: Progress={modelGenerationProgress}%, Message="{modelProgressMessage}"
            </p>
          </div>
        </div>
      )}

      {/* Step 4: 3D Preview & Download */}
      {currentStep === 4 && generated3DModel && (
        <div className="step-content">
          <h3>
            <CheckCircle className="w-6 h-6" />
            3D Model Ready
          </h3>

          <div className="results-summary">
            <div className="results-summary-content">
              <CheckCircle className="w-5 h-5" style={{ color: '#059669' }} />
              <div>
                <h4>3D Generation Complete!</h4>
                <p>Successfully created 3D model with PBR textures</p>
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="model-info">
            <h4>Model Information</h4>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Vertices:</span>
                <span className="info-value">{generated3DModel.metadata.vertices.toLocaleString()}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Faces:</span>
                <span className="info-value">{generated3DModel.metadata.faces.toLocaleString()}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Format:</span>
                <span className="info-value">{generated3DModel.metadata.format}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Materials:</span>
                <span className="info-value">PBR Textures</span>
              </div>
            </div>
          </div>

          {/* 3D Mesh Viewer */}
          <div className="model-preview">
            <h4>3D Preview</h4>
            <div className="mesh-viewer-wrapper">
              {generated3DModel.exports?.ply ? (
                <Mesh3DViewer
                  meshUrl={`http://localhost:4005/api/hunyuan3d/download/${generated3DModel.jobId}/ply`}
                  className="interactive-viewer"
                />
              ) : (
                <div className="model-viewer-placeholder">
                  <Box className="w-12 h-12" style={{ color: 'rgba(255, 255, 255, 0.5)' }} />
                  <p>3D Preview Loading...</p>
                  <p className="text-sm">PLY mesh file is being processed</p>
                </div>
              )}
            </div>
          </div>

          {/* Download Options */}
          <div className="download-section">
            <h4>Download Assets</h4>
            <div className="download-buttons">
              {/* FBX Download */}
              {generated3DModel.exports?.fbx && (
                <button
                  className="download-button fbx-download"
                  onClick={() => handleDownload('fbx')}
                >
                  <Package className="w-4 h-4" />
                  Download FBX Model
                </button>
              )}

              {/* OBJ Download (as fallback if no FBX) */}
              {!generated3DModel.exports?.fbx && generated3DModel.exports?.obj && (
                <button
                  className="download-button obj-download"
                  onClick={() => handleDownload('obj')}
                >
                  <Package className="w-4 h-4" />
                  Download OBJ Model
                </button>
              )}

              {/* Textures Download */}
              <button
                className="download-button textures-download"
                onClick={() => handleDownloadTextures()}
              >
                <FileDown className="w-4 h-4" />
                Download Textures
              </button>

              {/* PLY for 3D Viewer (hidden but functional) */}
              {generated3DModel.exports?.ply && (
                <button
                  className="download-button ply-download secondary"
                  onClick={() => handleDownload('ply')}
                >
                  <Download className="w-4 h-4" />
                  Download PLY (Raw Mesh)
                </button>
              )}

              {(!generated3DModel.exports || Object.keys(generated3DModel.exports).length === 0) && (
                <div className="download-placeholder">
                  <p>3D model assets will appear here after generation</p>
                </div>
              )}
            </div>
          </div>

          <div className="action-buttons">
            <button
              onClick={resetWorkflow}
              className="secondary-button"
            >
              <Sparkles className="w-5 h-5" />
              Generate New 3D Model
            </button>
          </div>
        </div>
      )}

      {/* Debug Log Panel */}
      <div className="debug-log">
        <h4>
          <Settings className="w-4 h-4" />
          Live Debug Console
          <span className="debug-counter">({debugLogs.length} events)</span>
        </h4>
        <div className="debug-log-content" ref={debugLogRef}>
          {debugLogs.length === 0 ? (
            <div className="debug-log-entry">
              <span className="debug-timestamp">{new Date().toLocaleTimeString()}</span>
              <span className="debug-message info">
                🔧 Debug console ready - choose input method to begin...
              </span>
            </div>
          ) : (
            debugLogs.map((log, index) => (
              <div key={index} className="debug-log-entry">
                <span className="debug-timestamp">{log.timestamp}</span>
                <span className={`debug-message ${log.type}`}>
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default Hunyuan3DWorkflow;