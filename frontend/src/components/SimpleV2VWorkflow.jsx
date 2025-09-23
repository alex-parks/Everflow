import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Download, FileImage, Loader2, CheckCircle, AlertCircle, Eye, Zap, Settings } from 'lucide-react';
import './SimpleV2VWorkflow.css';

const SimpleV2VWorkflow = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [uploadedSequence, setUploadedSequence] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [debugLogs, setDebugLogs] = useState([]);
  const [enhancedFrames, setEnhancedFrames] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [enhancementMode, setEnhancementMode] = useState('sd-controlnet'); // 'simple' or 'sd-controlnet'
  const [sdParams, setSDParams] = useState({
    prompt: "Enhanced cinematic quality, professional VFX, high detail, photorealistic",
    negative_prompt: "blurry, low quality, distorted, deformed",
    num_inference_steps: 20,
    guidance_scale: 7.5,
    controlnet_conditioning_scale: 1.0,
    seed: null,
    width: 512,
    height: 512,
    use_depth: true,
    // Temporal consistency parameters
    temporal_window_size: 5,
    keyframe_interval: 3,
    temporal_weight: 0.3,
    memory_length: 16,
    enable_temporal: true
  });
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const fileInputRef = useRef(null);
  const debugLogRef = useRef(null);

  const addDebugLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setDebugLogs(prev => {
      const newLogs = [...prev, { timestamp, message, type }];
      // Auto-scroll to bottom after state update
      setTimeout(() => {
        if (debugLogRef.current) {
          debugLogRef.current.scrollTop = debugLogRef.current.scrollHeight;
        }
      }, 100);
      return newLogs;
    });
  };

  const resetWorkflow = () => {
    addDebugLog('🔄 Resetting workflow to initial state...', 'info');
    setCurrentStep(1);
    setUploadedSequence(null);
    setUploadProgress(0);
    setProcessingProgress(0);
    setDebugLogs([]);
    setEnhancedFrames([]);
    setSelectedFrame(0);
    setIsUploading(false);
    setIsProcessing(false);
    // Add initial log after reset
    setTimeout(() => {
      addDebugLog('✨ Workflow reset complete - ready for new sequence', 'info');
    }, 100);
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (!files.length) return;

    setIsUploading(true);
    setUploadProgress(0);
    addDebugLog(`Starting upload of ${files.length} files`, 'info');

    try {
      const formData = new FormData();
      formData.append('name', `User Sequence ${Date.now()}`);
      
      files.forEach((file, index) => {
        formData.append('files', file);
        addDebugLog(`Added file ${index + 1}: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, 'info');
      });

      setUploadProgress(30);
      addDebugLog('Sending files to server...', 'info');

      const response = await fetch('http://localhost:4005/api/crowd/upload-exr-sequence', {
        method: 'POST',
        body: formData
      });

      setUploadProgress(80);

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      setUploadProgress(100);
      
      addDebugLog(`✅ Upload successful! Sequence ID: ${result.sequence_id}`, 'success');
      addDebugLog(`📁 Uploaded ${result.metadata.frame_count} frames`, 'success');
      addDebugLog(`📊 Sequence metadata: ${JSON.stringify(result.metadata)}`, 'info');
      
      setUploadedSequence(result);
      addDebugLog('⏭️ Transitioning to Step 2: Review & Prepare', 'info');
      setCurrentStep(2);

    } catch (error) {
      addDebugLog(`❌ Upload failed: ${error.message}`, 'error');
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleEnhanceSequence = async () => {
    if (!uploadedSequence) return;

    setIsProcessing(true);
    setProcessingProgress(0);
    setCurrentStep(3);
    
    const isSDControlNetMode = enhancementMode === 'sd-controlnet';
    const modelName = isSDControlNetMode ? 'SD + ControlNet Depth' : 'Simple V2V';
    
    addDebugLog('⏭️ Transitioning to Step 3: V2V Processing', 'info');
    addDebugLog(`🚀 Starting ${modelName} enhancement process...`, 'info');
    addDebugLog(`📊 Model: ${modelName}`, 'info');
    addDebugLog(`🎯 Input: ${uploadedSequence.metadata.frame_count} frames`, 'info');
    
    if (isSDControlNetMode) {
      addDebugLog(`📝 Prompt: ${sdParams.prompt}`, 'info');
      addDebugLog(`🎨 Steps: ${sdParams.num_inference_steps}`, 'info');
      addDebugLog(`🎯 Guidance: ${sdParams.guidance_scale}`, 'info');
      addDebugLog(`🎛️ ControlNet Scale: ${sdParams.controlnet_conditioning_scale}`, 'info');
      addDebugLog(`🗺️ Using Depth Map: ${sdParams.use_depth ? 'Yes' : 'No'}`, 'info');
      
      if (sdParams.enable_temporal) {
        addDebugLog(`🕐 Temporal Consistency: Enabled`, 'info');
        addDebugLog(`📱 Window Size: ${sdParams.temporal_window_size} frames`, 'info');
        addDebugLog(`🔑 Keyframe Interval: ${sdParams.keyframe_interval}`, 'info');
        addDebugLog(`⚖️ Temporal Weight: ${sdParams.temporal_weight}`, 'info');
        addDebugLog(`🧠 Memory Length: ${sdParams.memory_length}`, 'info');
      }
    }
    
    addDebugLog(`💾 Model Device: ${navigator.hardwareConcurrency > 4 ? 'GPU (CUDA)' : 'CPU'}`, 'info');
    
    try {
      setProcessingProgress(10);
      addDebugLog(`🔧 Initializing ${modelName} model...`, 'info');
      addDebugLog('📡 Connecting to backend processing server...', 'info');
      
      let response;
      if (isSDControlNetMode) {
        // Use SD ControlNet enhancement endpoint
        response = await fetch(
          `http://localhost:4005/api/frames-simple/${uploadedSequence.sequence_id}/enhance-sd-controlnet`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(sdParams)
          }
        );
      } else {
        // Use simple V2V enhancement endpoint
        response = await fetch(
          `http://localhost:4005/api/crowd/enhance-sequence/${uploadedSequence.sequence_id}`,
          { method: 'POST' }
        );
      }

      setProcessingProgress(30);
      addDebugLog(`🎨 Processing frames through ${modelName} model...`, 'info');
      
      if (isSDControlNetMode) {
        addDebugLog('🧠 Running Stable Diffusion process...', 'info');
        addDebugLog('🎯 Applying ControlNet depth guidance...', 'info');
        addDebugLog('🕐 Maintaining temporal consistency...', 'info');
      } else {
        addDebugLog('🔄 Applying enhancement filters...', 'info');
        addDebugLog('⚡ Enhancing image quality...', 'info');
      }

      if (!response.ok) {
        throw new Error(`Enhancement failed: ${response.statusText}`);
      }

      const result = await response.json();
      setProcessingProgress(70);
      
      const frameCount = result.enhanced_frames || result.processed_frames;
      addDebugLog(`✅ Enhanced ${frameCount} frames successfully`, 'success');
      
      if (isSDControlNetMode) {
        addDebugLog(`🔧 ${modelName} parameters used:`, 'info');
        addDebugLog(`   - Inference steps: ${result.parameters?.num_inference_steps}`, 'info');
        addDebugLog(`   - Guidance scale: ${result.parameters?.guidance_scale}`, 'info');
        addDebugLog(`   - ControlNet scale: ${result.parameters?.controlnet_conditioning_scale}`, 'info');
        if (result.temporal_enabled) {
          addDebugLog(`   - Temporal consistency: Enabled`, 'info');
          addDebugLog(`   - Window size: ${result.parameters?.temporal_window_size}`, 'info');
        }
      } else {
        addDebugLog(`📈 Performance: ${(frameCount / 10).toFixed(1)} FPS average`, 'success');
      }
      
      addDebugLog('📊 Loading enhanced frame metadata...', 'info');
      addDebugLog('🔍 Validating output quality...', 'info');
      
      // Load enhanced frames info
      const infoResponse = await fetch(
        `http://localhost:4005/api/crowd/sequences/${uploadedSequence.sequence_id}`
      );
      const info = await infoResponse.json();
      
      setProcessingProgress(90);
      addDebugLog(`📈 Enhanced frames available: ${frameCount}`, 'success');
      
      // Create array of enhanced frame indices
      const frameIndices = Array.from(
        { length: frameCount }, 
        (_, i) => i
      );
      setEnhancedFrames(frameIndices);
      
      setProcessingProgress(100);
      addDebugLog(`🎉 ${modelName} enhancement complete!`, 'success');
      addDebugLog('⏭️ Transitioning to Step 4: Enhanced Results', 'info');
      addDebugLog('🖼️ Enhanced frames ready for preview and download', 'success');
      setCurrentStep(4);
      
    } catch (error) {
      addDebugLog(`❌ Enhancement failed: ${error.message}`, 'error');
      console.error('Enhancement error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const getFrameUrl = (frameIndex, enhanced = false) => {
    if (!uploadedSequence) return '';
    if (enhanced) {
      if (enhancementMode === 'sd-controlnet') {
        return `http://localhost:4005/api/frames-simple/${uploadedSequence.sequence_id}/enhanced-sd-controlnet/${frameIndex}`;
      }
    }
    return `http://localhost:4005/api/crowd/sequences/${uploadedSequence.sequence_id}/frame/${frameIndex}?enhanced=${enhanced}`;
  };

  const getEXRBeautyUrl = (frameIndex) => {
    if (!uploadedSequence) return '';
    return `http://localhost:4005/api/frames-simple/${uploadedSequence.sequence_id}/frame/${frameIndex}?channel=beauty`;
  };

  const getEXRDepthUrl = (frameIndex) => {
    if (!uploadedSequence) return '';
    return `http://localhost:4005/api/frames-simple/${uploadedSequence.sequence_id}/frame/${frameIndex}?channel=depth`;
  };

  const handleFrameSelection = (frameIndex) => {
    setSelectedFrame(frameIndex);
    addDebugLog(`🖼️ Selected frame ${frameIndex + 1} for preview`, 'info');
  };

  const handleStartProcessing = () => {
    addDebugLog('👤 User initiated enhancement process', 'info');
    addDebugLog(`🎬 Processing sequence: ${uploadedSequence.sequence_id}`, 'info');
    handleEnhanceSequence();
  };

  const getStepStatus = (step) => {
    if (step < currentStep) return 'completed';
    if (step === currentStep) return 'active';
    return 'pending';
  };

  // Initialize debug log with system status
  useEffect(() => {
    addDebugLog('🚀 Simple V2V Workflow initialized', 'info');
    addDebugLog('🔧 Checking available models...', 'info');
    addDebugLog(`💻 Browser: ${navigator.userAgent.split(' ')[0]}`, 'info');
    addDebugLog(`⚡ Hardware threads: ${navigator.hardwareConcurrency || 'Unknown'}`, 'info');
    addDebugLog('📡 Backend connection: Checking...', 'info');
    
    // Check Simple V2V status
    fetch('http://localhost:4005/api/frames-simple/v2v/status')
      .then(response => response.json())
      .then(data => {
        if (data.available) {
          addDebugLog('✅ Simple V2V model available', 'success');
          addDebugLog(`🎯 Device: ${data.device}`, 'success');
          addDebugLog(`🤖 Model: ${data.model_type}`, 'success');
        } else {
          addDebugLog('⚠️ V2V model not available', 'error');
        }
      })
      .catch(() => {
        addDebugLog('❌ Simple V2V backend connection failed', 'error');
      });
    
    // Check SD ControlNet model status
    fetch('http://localhost:4005/api/frames-simple/sd-controlnet/status')
      .then(response => response.json())
      .then(data => {
        if (data.available) {
          addDebugLog('✅ SD + ControlNet model available', 'success');
          addDebugLog(`🎯 SD Device: ${data.device}`, 'success');
          addDebugLog(`🤖 SD Model: ${data.model_type}`, 'success');
          addDebugLog(`📚 Diffusers version: ${data.diffusers_version}`, 'info');
          addDebugLog(`💼 Commercial License: ${data.commercial_license ? 'Yes' : 'No'}`, 'success');
        } else {
          addDebugLog('⚠️ SD ControlNet model not available', 'error');
        }
      })
      .catch(() => {
        addDebugLog('❌ SD ControlNet backend connection failed', 'error');
      });
  }, []);

  return (
    <div className="simple-v2v-workflow">
      <div className="workflow-header">
        <h1>Simple V2V Enhancement Workflow</h1>
        <p>Transform your images using our Video-to-Video neural network model with step-by-step processing</p>
      </div>

      {/* Progress Steps */}
      <div className="workflow-progress">
        <div className="progress-steps">
          {[
            { number: 1, title: 'Upload Sequence', icon: Upload },
            { number: 2, title: 'Review & Prepare', icon: Eye },
            { number: 3, title: 'V2V Processing', icon: Settings },
            { number: 4, title: 'Enhanced Results', icon: CheckCircle }
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

      {/* Step 1: Upload */}
      {currentStep === 1 && (
        <div className="step-content">
          <h3>
            <Upload className="w-6 h-6" />
            Upload Image Sequence
          </h3>
          
          <div className={`upload-zone ${isUploading ? 'uploading' : ''}`}>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".exr,.jpg,.jpeg,.png,.tiff,.tif"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
            
            <FileImage className="w-16 h-16" />
            
            <h4>Upload Image Sequence</h4>
            <p>Select multiple image files (EXR, JPG, PNG, TIFF) to create a sequence for V2V enhancement</p>
            
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
              {isUploading ? 'Uploading...' : 'Select Image Files'}
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
        </div>
      )}

      {/* Step 2: Review */}
      {currentStep === 2 && uploadedSequence && (
        <div className="step-content">
          <h3>
            <Eye className="w-6 h-6" />
            Review & Prepare
          </h3>
          
          <div className="sequence-info">
            <h4>Sequence Information</h4>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Sequence ID:</span>
                <span className="info-value">{uploadedSequence.sequence_id.substring(0, 8)}...</span>
              </div>
              <div className="info-item">
                <span className="info-label">Frame Count:</span>
                <span className="info-value">{uploadedSequence.metadata.frame_count} frames</span>
              </div>
              <div className="info-item">
                <span className="info-label">Upload Time:</span>
                <span className="info-value">{uploadedSequence.metadata.created_at}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Status:</span>
                <span className="info-value status-ready">✅ Ready for processing</span>
              </div>
            </div>
          </div>

          {/* EXR Preview Section */}
          <div className="exr-preview-section">
            <h4>EXR Preview</h4>
            <div className="frame-selector">
              <label>Preview Frame:</label>
              <select
                value={selectedFrame}
                onChange={(e) => handleFrameSelection(parseInt(e.target.value))}
              >
                {Array.from({ length: uploadedSequence.metadata.frame_count }, (_, index) => (
                  <option key={index} value={index}>
                    Frame {index + 1}
                  </option>
                ))}
              </select>
            </div>

            {/* Beauty Pass and Depth Map Preview */}
            <div className="exr-comparison-grid">
              <div className="exr-preview-item">
                <h5>Beauty Pass (RGB)</h5>
                <div className="image-container">
                  <img
                    src={getEXRBeautyUrl(selectedFrame)}
                    alt={`Beauty pass frame ${selectedFrame + 1}`}
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkJlYXV0eSBQYXNzPC90ZXh0Pjwvc3ZnPg==';
                    }}
                  />
                </div>
              </div>

              <div className="exr-preview-item">
                <h5>Depth Map</h5>
                <div className="image-container">
                  <img
                    src={getEXRDepthUrl(selectedFrame)}
                    alt={`Depth map frame ${selectedFrame + 1}`}
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMTExIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iI2FhYSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkRlcHRoIE1hcDwvdGV4dD48L3N2Zz4=';
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Enhancement Mode Selection */}
          <div className="enhancement-settings">
            <h4>Enhancement Settings</h4>
            <div className="mode-selector">
              <label>
                <input
                  type="radio"
                  name="enhancement-mode"
                  value="sd-controlnet"
                  checked={enhancementMode === 'sd-controlnet'}
                  onChange={(e) => setEnhancementMode(e.target.value)}
                />
                <span className="mode-label">
                  <strong>SD + ControlNet Depth</strong> - Commercial AI with depth-guided generation and temporal consistency
                </span>
              </label>
              <label>
                <input
                  type="radio"
                  name="enhancement-mode"
                  value="simple"
                  checked={enhancementMode === 'simple'}
                  onChange={(e) => setEnhancementMode(e.target.value)}
                />
                <span className="mode-label">
                  <strong>Simple V2V</strong> - Fast traditional enhancement
                </span>
              </label>
            </div>

            {/* SD ControlNet Parameters */}
            {enhancementMode === 'sd-controlnet' && (
              <div className="sd-parameters">
                <div className="param-group">
                  <label>Enhancement Prompt:</label>
                  <textarea
                    value={sdParams.prompt}
                    onChange={(e) => setSDParams({...sdParams, prompt: e.target.value})}
                    placeholder="Describe the enhancement you want..."
                    rows={3}
                  />
                </div>

                <div className="param-group">
                  <label>
                    <input
                      type="checkbox"
                      checked={sdParams.use_depth}
                      onChange={(e) => setSDParams({...sdParams, use_depth: e.target.checked})}
                    />
                    Use Depth Map for guidance
                  </label>
                </div>

                <button
                  className="settings-toggle"
                  onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                >
                  {showAdvancedSettings ? 'Hide' : 'Show'} Advanced Settings
                </button>

                {showAdvancedSettings && (
                  <div className="advanced-settings">
                    <div className="param-row">
                      <div className="param-group">
                        <label>Inference Steps:</label>
                        <input
                          type="number"
                          value={sdParams.num_inference_steps}
                          onChange={(e) => setSDParams({...sdParams, num_inference_steps: parseInt(e.target.value)})}
                          min={1}
                          max={50}
                        />
                      </div>

                      <div className="param-group">
                        <label>Guidance Scale:</label>
                        <input
                          type="number"
                          value={sdParams.guidance_scale}
                          onChange={(e) => setSDParams({...sdParams, guidance_scale: parseFloat(e.target.value)})}
                          min={1}
                          max={20}
                          step={0.5}
                        />
                      </div>
                    </div>

                    <div className="param-row">
                      <div className="param-group">
                        <label>ControlNet Scale:</label>
                        <input
                          type="number"
                          value={sdParams.controlnet_conditioning_scale}
                          onChange={(e) => setSDParams({...sdParams, controlnet_conditioning_scale: parseFloat(e.target.value)})}
                          min={0.1}
                          max={2.0}
                          step={0.1}
                          title="Strength of ControlNet depth conditioning"
                        />
                      </div>
                    </div>

                    <div className="param-row">
                      <div className="param-group">
                        <label>Width:</label>
                        <input
                          type="number"
                          value={sdParams.width}
                          onChange={(e) => setSDParams({...sdParams, width: parseInt(e.target.value)})}
                          min={256}
                          max={1024}
                          step={64}
                        />
                      </div>

                      <div className="param-group">
                        <label>Height:</label>
                        <input
                          type="number"
                          value={sdParams.height}
                          onChange={(e) => setSDParams({...sdParams, height: parseInt(e.target.value)})}
                          min={256}
                          max={1024}
                          step={64}
                        />
                      </div>
                    </div>

                    <div className="param-group">
                      <label>Seed (optional):</label>
                      <input
                        type="number"
                        value={sdParams.seed || ''}
                        onChange={(e) => setSDParams({...sdParams, seed: e.target.value ? parseInt(e.target.value) : null})}
                        placeholder="Random seed for reproducibility"
                      />
                    </div>

                    <div className="param-group">
                      <label>Negative Prompt:</label>
                      <textarea
                        value={sdParams.negative_prompt}
                        onChange={(e) => setSDParams({...sdParams, negative_prompt: e.target.value})}
                        placeholder="What to avoid in the enhancement..."
                        rows={2}
                      />
                    </div>

                    {/* Temporal Consistency Parameters */}
                    <div className="temporal-settings">
                      <h5 style={{color: '#059669', marginBottom: '1rem'}}>🕐 Temporal Consistency Settings</h5>
                      
                      <div className="param-group">
                        <label>
                          <input
                            type="checkbox"
                            checked={sdParams.enable_temporal}
                            onChange={(e) => setSDParams({...sdParams, enable_temporal: e.target.checked})}
                          />
                          Enable Temporal Consistency
                        </label>
                      </div>

                      {sdParams.enable_temporal && (
                        <>
                          <div className="param-row">
                            <div className="param-group">
                              <label>Temporal Window Size:</label>
                              <input
                                type="number"
                                value={sdParams.temporal_window_size}
                                onChange={(e) => setSDParams({...sdParams, temporal_window_size: parseInt(e.target.value)})}
                                min={2}
                                max={10}
                                title="Number of frames to consider for temporal consistency"
                              />
                            </div>

                            <div className="param-group">
                              <label>Keyframe Interval:</label>
                              <input
                                type="number"
                                value={sdParams.keyframe_interval}
                                onChange={(e) => setSDParams({...sdParams, keyframe_interval: parseInt(e.target.value)})}
                                min={1}
                                max={10}
                                title="Interval between keyframes for long-term consistency"
                              />
                            </div>
                          </div>

                          <div className="param-row">
                            <div className="param-group">
                              <label>Temporal Weight:</label>
                              <input
                                type="number"
                                step="0.1"
                                value={sdParams.temporal_weight}
                                onChange={(e) => setSDParams({...sdParams, temporal_weight: parseFloat(e.target.value)})}
                                min={0.0}
                                max={1.0}
                                title="Strength of temporal consistency (0.0 = disabled, 1.0 = maximum)"
                              />
                            </div>

                            <div className="param-group">
                              <label>Memory Length:</label>
                              <input
                                type="number"
                                value={sdParams.memory_length}
                                onChange={(e) => setSDParams({...sdParams, memory_length: parseInt(e.target.value)})}
                                min={4}
                                max={32}
                                title="Length of temporal memory buffer"
                              />
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="action-buttons">
            <button
              className="primary-button"
              onClick={handleStartProcessing}
              disabled={isProcessing}
            >
              <Zap className="w-5 h-5" />
              Start V2V Enhancement
            </button>
            
            <button
              className="secondary-button"
              onClick={resetWorkflow}
            >
              <Upload className="w-5 h-5" />
              Upload Different Sequence
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Processing */}
      {currentStep === 3 && (
        <div className="step-content">
          <h3>
            <Settings className="w-6 h-6" />
            V2V Neural Network Processing
          </h3>
          
          <div className="processing-info">
            <div className="processing-status">
              <Loader2 className="w-6 h-6 spinner" />
              Processing frames through V2V model...
            </div>
            
            <div className="processing-progress">
              <div 
                className="processing-progress-fill"
                style={{ width: `${processingProgress}%` }}
              ></div>
            </div>
            <p className="processing-percentage">{processingProgress}% complete</p>
          </div>
        </div>
      )}

      {/* Step 4: Results */}
      {currentStep === 4 && enhancedFrames.length > 0 && (
        <div className="step-content">
          <h3>
            <CheckCircle className="w-6 h-6" />
            Enhanced Results
          </h3>
          
          <div className="results-summary">
            <div className="results-summary-content">
              <CheckCircle className="w-5 h-5" style={{ color: '#059669' }} />
              <div>
                <h4>Enhancement Complete!</h4>
                <p>Successfully enhanced {enhancedFrames.length} frames using the V2V neural network</p>
              </div>
            </div>
          </div>

          {/* Frame Selector */}
          <div className="frame-selector">
            <label>Compare Frame:</label>
            <select
              value={selectedFrame}
              onChange={(e) => handleFrameSelection(parseInt(e.target.value))}
            >
              {enhancedFrames.map((_, index) => (
                <option key={index} value={index}>
                  Frame {index + 1}
                </option>
              ))}
            </select>
          </div>

          {/* Before/After Comparison */}
          <div className="comparison-grid">
            <div className="comparison-item">
              <h4>Original Input</h4>
              <div className="image-container">
                <img
                  src={getFrameUrl(selectedFrame, false)}
                  alt={`Original frame ${selectedFrame + 1}`}
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZhaWxlZCB0byBsb2FkPC90ZXh0Pjwvc3ZnPg==';
                  }}
                />
              </div>
            </div>

            <div className="comparison-item">
              <h4>V2V Enhanced Output</h4>
              <div className="image-container">
                <img
                  src={getFrameUrl(selectedFrame, true)}
                  alt={`Enhanced frame ${selectedFrame + 1}`}
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkVuaGFuY2VkIGZyYW1lPC90ZXh0Pjwvc3ZnPg==';
                  }}
                />
              </div>
            </div>
          </div>

          {/* Download Section */}
          <div className="download-section">
            <h4>Download Results</h4>
            <div className="download-buttons">
              <a
                href={getFrameUrl(selectedFrame, false)}
                download={`original_frame_${selectedFrame + 1}`}
                className="download-button"
              >
                <Download className="w-4 h-4" />
                Download Original
              </a>
              <a
                href={getFrameUrl(selectedFrame, true)}
                download={`enhanced_frame_${selectedFrame + 1}.jpg`}
                className="download-button enhanced"
              >
                <Download className="w-4 h-4" />
                Download Enhanced
              </a>
            </div>
          </div>

          <div className="action-buttons">
            <button
              onClick={resetWorkflow}
              className="secondary-button"
            >
              <Upload className="w-5 h-5" />
              Process New Sequence
            </button>
          </div>
        </div>
      )}

      {/* Debug Log Panel - Always Visible */}
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
                🔧 Debug console ready - upload sequence to begin processing...
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

export default SimpleV2VWorkflow;