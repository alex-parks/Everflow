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
      // TODO: Replace with your new AI model's upload endpoint
      addDebugLog('⚠️ Upload endpoint not yet configured for new AI model', 'info');
      addDebugLog('📝 This is a placeholder for your new workflow', 'info');

      // Simulate upload for UI demonstration
      setUploadProgress(30);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setUploadProgress(70);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setUploadProgress(100);

      // Mock successful upload
      const mockResult = {
        sequence_id: 'demo-sequence-' + Date.now(),
        metadata: {
          frame_count: files.length,
          created_at: new Date().toISOString()
        }
      };

      addDebugLog(`✅ Mock upload successful! Sequence ID: ${mockResult.sequence_id}`, 'success');
      addDebugLog(`📁 Uploaded ${mockResult.metadata.frame_count} frames`, 'success');

      setUploadedSequence(mockResult);
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

    addDebugLog('⏭️ Transitioning to Step 3: AI Processing', 'info');
    addDebugLog('🚀 Starting AI enhancement process...', 'info');
    addDebugLog('⚠️ Processing endpoint not yet configured for new AI model', 'info');

    try {
      // TODO: Replace with your new AI model's processing endpoint
      addDebugLog('📝 This is a placeholder for your new AI workflow', 'info');

      // Simulate processing for UI demonstration
      setProcessingProgress(10);
      addDebugLog('🔧 Initializing AI model...', 'info');
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingProgress(30);
      addDebugLog('🎨 Processing frames through AI model...', 'info');
      await new Promise(resolve => setTimeout(resolve, 2000));

      setProcessingProgress(70);
      addDebugLog('⚡ Applying enhancements...', 'info');
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingProgress(100);
      addDebugLog('✅ Enhanced frames successfully', 'success');

      // Mock enhanced frames
      const frameCount = uploadedSequence.metadata.frame_count;
      const frameIndices = Array.from({ length: frameCount }, (_, i) => i);
      setEnhancedFrames(frameIndices);

      addDebugLog('🎉 AI enhancement complete!', 'success');
      addDebugLog('⏭️ Transitioning to Step 4: Enhanced Results', 'info');
      setCurrentStep(4);

    } catch (error) {
      addDebugLog(`❌ Enhancement failed: ${error.message}`, 'error');
      console.error('Enhancement error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const getFrameUrl = (frameIndex, enhanced = false) => {
    // TODO: Replace with your new AI model's frame endpoints
    return `data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPiR7ZW5oYW5jZWQgPyAnRW5oYW5jZWQnIDogJ09yaWdpbmFsJ30gRnJhbWUgJHtmcmFtZUluZGV4ICsgMX08L3RleHQ+PC9zdmc+`;
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

  // Initialize debug log
  useEffect(() => {
    addDebugLog('🚀 AI Enhancement Workflow initialized', 'info');
    addDebugLog('⚠️ Ready for new AI model integration', 'info');
    addDebugLog(`💻 Browser: ${navigator.userAgent.split(' ')[0]}`, 'info');
    addDebugLog(`⚡ Hardware threads: ${navigator.hardwareConcurrency || 'Unknown'}`, 'info');
  }, []);

  return (
    <div className="simple-v2v-workflow">
      <div className="workflow-header">
        <h1>AI Enhancement Workflow</h1>
        <p>Transform your images using our AI model with step-by-step processing</p>
      </div>

      {/* Progress Steps */}
      <div className="workflow-progress">
        <div className="progress-steps">
          {[
            { number: 1, title: 'Upload Sequence', icon: Upload },
            { number: 2, title: 'Review & Prepare', icon: Eye },
            { number: 3, title: 'AI Processing', icon: Settings },
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
            <p>Select multiple image files (EXR, JPG, PNG, TIFF) to create a sequence for AI enhancement</p>

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
                <span className="info-value">{uploadedSequence.sequence_id.substring(0, 16)}...</span>
              </div>
              <div className="info-item">
                <span className="info-label">Frame Count:</span>
                <span className="info-value">{uploadedSequence.metadata.frame_count} frames</span>
              </div>
              <div className="info-item">
                <span className="info-label">Upload Time:</span>
                <span className="info-value">{new Date(uploadedSequence.metadata.created_at).toLocaleString()}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Status:</span>
                <span className="info-value status-ready">✅ Ready for processing</span>
              </div>
            </div>
          </div>

          <div className="action-buttons">
            <button
              className="primary-button"
              onClick={handleStartProcessing}
              disabled={isProcessing}
            >
              <Zap className="w-5 h-5" />
              Start AI Enhancement
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
            AI Processing
          </h3>

          <div className="processing-info">
            <div className="processing-status">
              <Loader2 className="w-6 h-6 spinner" />
              Processing frames through AI model...
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
                <p>Successfully enhanced {enhancedFrames.length} frames using AI</p>
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
                />
              </div>
            </div>

            <div className="comparison-item">
              <h4>AI Enhanced Output</h4>
              <div className="image-container">
                <img
                  src={getFrameUrl(selectedFrame, true)}
                  alt={`Enhanced frame ${selectedFrame + 1}`}
                />
              </div>
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