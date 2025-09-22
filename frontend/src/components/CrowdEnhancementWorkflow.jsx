import React, { useState, useEffect, useCallback } from 'react'
import { Upload, Play, Download, Settings, Eye, Brain, Zap } from 'lucide-react'
import VideoSequenceUploader from './VideoSequenceUploader'
import './CrowdEnhancementWorkflow.css'

const CrowdEnhancementWorkflow = () => {
  const [currentStep, setCurrentStep] = useState(1)
  const [uploadedSequence, setUploadedSequence] = useState(null)
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [enhancementProgress, setEnhancementProgress] = useState(null)
  const [enhancedSequence, setEnhancedSequence] = useState(null)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isEnhancing, setIsEnhancing] = useState(false)
  
  // Load available models on component mount
  useEffect(() => {
    checkBackendHealth()
    loadAvailableModels()
    checkTrainingStatus()
  }, [])

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:4005/api/health')
      if (!response.ok) {
        console.error('Backend health check failed:', response.status)
      } else {
        const data = await response.json()
        console.log('Backend is healthy:', data)
      }
    } catch (error) {
      console.error('Cannot connect to backend:', error)
      alert('Cannot connect to backend API on port 4005. Please ensure Docker containers are running.')
    }
  }

  const loadAvailableModels = async () => {
    try {
      const response = await fetch('http://localhost:4005/api/crowd/models')
      const data = await response.json()
      setAvailableModels(data.models || [])
      if (data.models && data.models.length > 0) {
        setSelectedModel(data.models[0].filename)
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch('http://localhost:4005/api/crowd/training/status')
      const data = await response.json()
      setTrainingStatus(data)
    } catch (error) {
      console.error('Failed to check training status:', error)
    }
  }

  const handleSequenceReady = async (sequenceData) => {
    setUploadedSequence(sequenceData)
    setCurrentStep(2)
    
    // Automatically prepare training data
    if (sequenceData.sequence_id) {
      await prepareTrainingData(sequenceData.sequence_id)
    }
  }

  const prepareTrainingData = async (sequenceId) => {
    try {
      await fetch(`http://localhost:4005/api/crowd/crowd-sequences/${sequenceId}/prepare-training-data`, {
        method: 'POST'
      })
    } catch (error) {
      console.error('Failed to prepare training data:', error)
    }
  }

  const handleEnhancement = async () => {
    if (!uploadedSequence || !selectedModel) return

    setIsEnhancing(true)
    setEnhancementProgress({ current: 0, total: uploadedSequence.crowd_analysis?.frame_count || 1 })

    try {
      const response = await fetch(
        `http://localhost:4005/api/crowd/crowd-sequences/${uploadedSequence.sequence_id}/enhance?model_path=${selectedModel}`,
        { method: 'POST' }
      )

      if (response.ok) {
        const result = await response.json()
        setEnhancedSequence(result)
        setCurrentStep(4)
      } else {
        throw new Error('Enhancement failed')
      }
    } catch (error) {
      console.error('Enhancement error:', error)
      alert('Enhancement failed. Please try again.')
    } finally {
      setIsEnhancing(false)
      setEnhancementProgress(null)
    }
  }

  const startTraining = async () => {
    try {
      const response = await fetch('http://localhost:4005/api/crowd/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          epochs: 50,
          batch_size: 2,
          learning_rate: 0.0001
        })
      })

      if (response.ok) {
        alert('Training started! This will take several hours.')
        checkTrainingStatus()
      }
    } catch (error) {
      console.error('Training start error:', error)
      alert('Failed to start training.')
    }
  }

  return (
    <div className="crowd-enhancement-workflow">
      <div className="workflow-header">
        <h1>AI Crowd Enhancement Pipeline</h1>
        <p>Transform low-poly proxy crowds into photorealistic people with V2V AI</p>
      </div>

      <div className="workflow-progress">
        <div className="progress-steps">
          {[
            { id: 1, title: 'Upload Sequence', icon: Upload },
            { id: 2, title: 'Review & Prepare', icon: Eye },
            { id: 3, title: 'AI Enhancement', icon: Brain },
            { id: 4, title: 'Download Results', icon: Download }
          ].map(step => (
            <div 
              key={step.id}
              className={`progress-step ${currentStep >= step.id ? 'active' : ''} ${currentStep === step.id ? 'current' : ''}`}
            >
              <div className="step-icon">
                <step.icon size={20} />
              </div>
              <span className="step-title">{step.title}</span>
            </div>
          ))}
        </div>
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${(currentStep - 1) * 33.33}%` }}
          />
        </div>
      </div>

      <div className="workflow-content">
        {currentStep === 1 && (
          <UploadStep 
            onUpload={handleSequenceReady} 
            isUploading={isUploading}
          />
        )}

        {currentStep === 2 && uploadedSequence && (
          <ReviewStep 
            sequence={uploadedSequence}
            models={availableModels}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            onProceed={() => setCurrentStep(3)}
            onStartTraining={startTraining}
            trainingStatus={trainingStatus}
          />
        )}

        {currentStep === 3 && (
          <EnhancementStep
            onEnhance={handleEnhancement}
            isEnhancing={isEnhancing}
            progress={enhancementProgress}
            selectedModel={selectedModel}
          />
        )}

        {currentStep === 4 && enhancedSequence && (
          <ResultsStep 
            sequence={uploadedSequence}
            enhancedData={enhancedSequence}
            onStartNew={() => {
              setCurrentStep(1)
              setUploadedSequence(null)
              setEnhancedSequence(null)
            }}
          />
        )}
      </div>
    </div>
  )
}

const UploadStep = ({ onUpload, isUploading }) => {
  return (
    <div className="upload-step">
      <VideoSequenceUploader
        onSequenceReady={onUpload}
        isUploading={isUploading}
      />
    </div>
  )
}

const ReviewStep = ({ sequence, models, selectedModel, onModelChange, onProceed, onStartTraining, trainingStatus }) => {
  return (
    <div className="review-step">
      <h2>Review Uploaded Sequence</h2>
      
      <div className="sequence-info">
        <div className="info-card">
          <h3>Sequence Details</h3>
          <div className="info-grid">
            <div>
              <label>Name:</label>
              <span>{sequence.name || 'Unnamed Sequence'}</span>
            </div>
            <div>
              <label>Frames:</label>
              <span>{sequence.crowd_analysis?.frame_count || 0}</span>
            </div>
            <div>
              <label>Resolution:</label>
              <span>{sequence.crowd_analysis?.resolution?.join('x') || 'Unknown'}</span>
            </div>
            <div>
              <label>People Detected:</label>
              <span>{sequence.crowd_analysis?.people_detected || 0}</span>
            </div>
          </div>
        </div>

        <div className="info-card">
          <h3>Available Passes</h3>
          <div className="passes-list">
            {sequence.passes?.map(pass => (
              <div key={pass} className="pass-tag">{pass}</div>
            ))}
          </div>
        </div>
      </div>

      <div className="model-selection">
        <h3>Select AI Model</h3>
        {models.length > 0 ? (
          <div className="model-options">
            <select 
              value={selectedModel} 
              onChange={(e) => onModelChange(e.target.value)}
              className="model-select"
            >
              {models.map(model => (
                <option key={model.filename} value={model.filename}>
                  {model.filename} ({model.size_mb}MB)
                </option>
              ))}
            </select>
            <button 
              className="proceed-button primary"
              onClick={onProceed}
            >
              Proceed to Enhancement
            </button>
          </div>
        ) : (
          <div className="no-models">
            <p>No trained models available. You need to train a model first.</p>
            <div className="training-section">
              <h4>Training Status</h4>
              {trainingStatus?.active_training ? (
                <div className="training-active">
                  <Zap className="training-icon" />
                  <span>Training in progress...</span>
                </div>
              ) : (
                <button 
                  className="train-button"
                  onClick={onStartTraining}
                >
                  Start Training New Model
                </button>
              )}
              {trainingStatus?.latest_checkpoint && (
                <p className="checkpoint-info">
                  Latest checkpoint: {trainingStatus.latest_checkpoint}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const EnhancementStep = ({ onEnhance, isEnhancing, progress, selectedModel }) => {
  return (
    <div className="enhancement-step">
      <h2>AI Enhancement</h2>
      
      <div className="enhancement-info">
        <div className="model-info">
          <Brain size={32} />
          <div>
            <h3>Selected Model: {selectedModel}</h3>
            <p>This process will transform your proxy crowd into photorealistic people while maintaining exact camera movement and timing.</p>
          </div>
        </div>

        <div className="enhancement-features">
          <h4>Enhancement Features:</h4>
          <ul>
            <li>Person-specific clothing based on emissive colors</li>
            <li>Depth-aware realistic rendering</li>
            <li>Temporal consistency across frames</li>
            <li>Camera motion preservation</li>
          </ul>
        </div>
      </div>

      {isEnhancing ? (
        <div className="enhancement-progress">
          <div className="progress-circle">
            <div className="spinner large"></div>
          </div>
          <h3>Enhancing Sequence...</h3>
          {progress && (
            <div className="progress-details">
              <div className="progress-bar-container">
                <div 
                  className="progress-bar-fill"
                  style={{ width: `${(progress.current / progress.total) * 100}%` }}
                />
              </div>
              <span>{progress.current} / {progress.total} frames</span>
            </div>
          )}
          <p>This may take several minutes depending on sequence length...</p>
        </div>
      ) : (
        <div className="enhancement-controls">
          <button 
            className="enhance-button primary large"
            onClick={onEnhance}
          >
            <Brain size={20} />
            Start AI Enhancement
          </button>
        </div>
      )}
    </div>
  )
}

const ResultsStep = ({ sequence, enhancedData, onStartNew }) => {
  const [previewFrame, setPreviewFrame] = useState(0)
  const [showComparison, setShowComparison] = useState(true)

  const downloadResults = async () => {
    try {
      // Create download link for the enhanced sequence
      const link = document.createElement('a')
      link.href = `http://localhost:4005/api/crowd/crowd-sequences/${sequence.sequence_id}/enhanced/download`
      link.download = `enhanced_${sequence.name || 'sequence'}.zip`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      console.error('Download error:', error)
      alert('Download failed. Please try again.')
    }
  }

  return (
    <div className="results-step">
      <h2>Enhancement Complete!</h2>
      
      <div className="results-summary">
        <div className="summary-card">
          <h3>Processing Results</h3>
          <div className="summary-stats">
            <div className="stat">
              <label>Frames Processed:</label>
              <span>{enhancedData.frames_processed}</span>
            </div>
            <div className="stat">
              <label>Average FPS:</label>
              <span>{enhancedData.performance_stats?.average_fps?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="stat">
              <label>Total Time:</label>
              <span>{enhancedData.performance_stats?.total_time?.toFixed(1) || 'N/A'}s</span>
            </div>
          </div>
        </div>
      </div>

      <div className="results-preview">
        <div className="preview-controls">
          <h3>Preview Results</h3>
          <div className="preview-options">
            <label>
              <input 
                type="checkbox" 
                checked={showComparison}
                onChange={(e) => setShowComparison(e.target.checked)}
              />
              Show Before/After Comparison
            </label>
            <div className="frame-selector">
              <label>Frame:</label>
              <input 
                type="range"
                min="0"
                max={enhancedData.frames_processed - 1}
                value={previewFrame}
                onChange={(e) => setPreviewFrame(parseInt(e.target.value))}
              />
              <span>{previewFrame + 1} / {enhancedData.frames_processed}</span>
            </div>
          </div>
        </div>

        <div className="preview-images">
          {showComparison ? (
            <div className="comparison-view">
              <div className="image-container">
                <h4>Original Proxy</h4>
                <img 
                  src={`http://localhost:4005/api/crowd/crowd-sequences/${sequence.sequence_id}/frame/${previewFrame}/beauty`}
                  alt="Original"
                  onError={(e) => {
                    e.target.src = '/placeholder-image.png'
                  }}
                />
              </div>
              <div className="image-container">
                <h4>AI Enhanced</h4>
                <img 
                  src={`http://localhost:4005/api/crowd/crowd-sequences/${sequence.sequence_id}/enhanced/${previewFrame}`}
                  alt="Enhanced"
                  onError={(e) => {
                    e.target.src = '/placeholder-image.png'
                  }}
                />
              </div>
            </div>
          ) : (
            <div className="single-view">
              <img 
                src={`http://localhost:4005/api/crowd/crowd-sequences/${sequence.sequence_id}/enhanced/${previewFrame}`}
                alt="Enhanced Result"
                onError={(e) => {
                  e.target.src = '/placeholder-image.png'
                }}
              />
            </div>
          )}
        </div>
      </div>

      <div className="results-actions">
        <button 
          className="download-button primary"
          onClick={downloadResults}
        >
          <Download size={20} />
          Download Enhanced Sequence
        </button>
        <button 
          className="new-sequence-button secondary"
          onClick={onStartNew}
        >
          Process New Sequence
        </button>
      </div>
    </div>
  )
}

export default CrowdEnhancementWorkflow