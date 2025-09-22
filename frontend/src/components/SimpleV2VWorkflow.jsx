import React, { useState, useRef } from 'react';
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

  const addDebugLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setDebugLogs(prev => [...prev, { timestamp, message, type }]);
  };

  const resetWorkflow = () => {
    setCurrentStep(1);
    setUploadedSequence(null);
    setUploadProgress(0);
    setProcessingProgress(0);
    setDebugLogs([]);
    setEnhancedFrames([]);
    setSelectedFrame(0);
    setIsUploading(false);
    setIsProcessing(false);
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
      
      setUploadedSequence(result);
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
    
    addDebugLog('🚀 Starting V2V enhancement process...', 'info');
    addDebugLog(`📊 Model: SimpleV2VModel (6M+ parameters)`, 'info');
    addDebugLog(`🎯 Input: ${uploadedSequence.metadata.frame_count} frames`, 'info');
    
    try {
      setProcessingProgress(10);
      addDebugLog('🔧 Initializing PyTorch V2V model...', 'info');
      
      const response = await fetch(
        `http://localhost:4005/api/crowd/enhance-sequence/${uploadedSequence.sequence_id}`,
        { method: 'POST' }
      );

      setProcessingProgress(30);
      addDebugLog('🎨 Processing frames through neural network...', 'info');

      if (!response.ok) {
        throw new Error(`Enhancement failed: ${response.statusText}`);
      }

      const result = await response.json();
      setProcessingProgress(70);
      
      addDebugLog(`✅ Enhanced ${result.processed_frames} frames successfully`, 'success');
      addDebugLog('📊 Loading enhanced frame metadata...', 'info');
      
      // Load enhanced frames info
      const infoResponse = await fetch(
        `http://localhost:4005/api/crowd/sequences/${uploadedSequence.sequence_id}`
      );
      const info = await infoResponse.json();
      
      setProcessingProgress(90);
      addDebugLog(`📈 Enhanced frames available: ${info.enhanced_frames_available}`, 'success');
      
      // Create array of enhanced frame indices
      const frameIndices = Array.from(
        { length: info.enhanced_frames_available }, 
        (_, i) => i
      );
      setEnhancedFrames(frameIndices);
      
      setProcessingProgress(100);
      addDebugLog('🎉 V2V enhancement complete!', 'success');
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
    return `http://localhost:4005/api/crowd/sequences/${uploadedSequence.sequence_id}/frame/${frameIndex}?enhanced=${enhanced}`;
  };

  const getStepStatus = (step) => {
    if (step < currentStep) return 'completed';
    if (step === currentStep) return 'active';
    return 'pending';
  };

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

          <div className="action-buttons">
            <button
              className="primary-button"
              onClick={handleEnhanceSequence}
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
              onChange={(e) => setSelectedFrame(parseInt(e.target.value))}
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

      {/* Debug Log Panel */}
      {debugLogs.length > 0 && (
        <div className="debug-log">
          <h4>
            <Settings className="w-4 h-4" />
            Debug Log
          </h4>
          <div className="debug-log-content">
            {debugLogs.map((log, index) => (
              <div key={index} className="debug-log-entry">
                <span className="debug-timestamp">{log.timestamp}</span>
                <span className={`debug-message ${log.type}`}>
                  {log.message}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleV2VWorkflow;