import React, { useState, useCallback } from 'react'
import { Upload, Film, AlertCircle, CheckCircle } from 'lucide-react'
import './VideoSequenceUploader.css'

const VideoSequenceUploader = ({ onSequenceReady }) => {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('idle') // idle, uploading, processing, complete, error
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [processingStep, setProcessingStep] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = Array.from(e.dataTransfer.files)
    handleFileUpload(files)
  }, [])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files)
    handleFileUpload(files)
  }, [])

  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return

    // Validate files
    const validFiles = files.filter(file => {
      const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|avi|mkv|exr)$/i)
      const isEXR = file.name.toLowerCase().endsWith('.exr')
      return isVideo || isEXR
    })

    if (validFiles.length === 0) {
      setErrorMessage('Please upload video files (MP4, MOV, AVI, MKV) or EXR sequences')
      setUploadStatus('error')
      return
    }

    setUploadedFiles(validFiles)
    setUploadStatus('uploading')
    setUploadProgress(0)
    setErrorMessage('')

    // Check if files are EXR sequence or video
    const isEXRSequence = validFiles.every(file => file.name.toLowerCase().endsWith('.exr'))
    
    if (isEXRSequence) {
      await handleEXRSequenceUpload(validFiles)
    } else {
      await handleVideoUpload(validFiles[0]) // Take first video file
    }
  }

  const handleEXRSequenceUpload = async (exrFiles) => {
    try {
      setProcessingStep('Uploading EXR sequence...')
      
      const formData = new FormData()
      formData.append('name', `EXR Sequence ${Date.now()}`)
      formData.append('frame_rate', '24.0')
      formData.append('camera_data', JSON.stringify({
        location: [0, 0, 10],
        rotation_euler: [0, 0, 0],
        angle: 45,
        clip_start: 0.1,
        clip_end: 1000,
        resolution: [1920, 1080],
        matrix_world: Array(16).fill().map((_, i) => i === 0 || i === 5 || i === 10 || i === 15 ? 1 : 0),
        projection_matrix: Array(16).fill(0),
        world_to_screen: Array(16).fill(0)
      }))

      // Add all EXR files
      exrFiles.forEach(file => {
        formData.append('files', file)
      })

      const response = await fetch('http://localhost:4005/api/crowd/upload-crowd-sequence', {
        method: 'POST',
        body: formData
      }).catch(error => {
        console.error('Fetch error:', error)
        throw new Error(`Network error: ${error.message}. Is the backend running on port 4005?`)
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Response error:', errorText)
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`)
      }

      const result = await response.json()
      
      setUploadProgress(100)
      setProcessingStep('EXR sequence uploaded successfully!')
      setUploadStatus('complete')
      
      if (onSequenceReady) {
        onSequenceReady(result)
      }

    } catch (error) {
      console.error('EXR upload error:', error)
      setErrorMessage(error.message || 'Failed to upload EXR sequence')
      setUploadStatus('error')
    }
  }

  const handleVideoUpload = async (videoFile) => {
    try {
      setProcessingStep('Extracting frames from video...')
      
      // First, upload the video for frame extraction
      const formData = new FormData()
      formData.append('video', videoFile)
      formData.append('target_fps', '24')
      formData.append('extract_proxy_passes', 'true') // Extract beauty, depth, emission passes
      
      const uploadResponse = await fetch('http://localhost:4005/api/crowd/upload-video-sequence', {
        method: 'POST',
        body: formData,
        // Add progress tracking
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(progress * 0.5) // First 50% for upload
        }
      })

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload video')
      }

      const uploadResult = await uploadResponse.json()
      setUploadProgress(50)
      setProcessingStep('Processing video into multi-pass renders...')

      // Poll for processing completion
      let processingComplete = false
      let attempts = 0
      const maxAttempts = 60 // 5 minutes max

      while (!processingComplete && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000)) // Wait 5 seconds
        
        try {
          const statusResponse = await fetch(
            `http://localhost:4005/api/crowd/video-processing-status/${uploadResult.processing_id}`
          )
          
          if (statusResponse.ok) {
            const status = await statusResponse.json()
            
            if (status.status === 'completed') {
              processingComplete = true
              setUploadProgress(100)
              setProcessingStep('Video processing complete!')
              setUploadStatus('complete')
              
              if (onSequenceReady) {
                onSequenceReady(status.result)
              }
            } else if (status.status === 'failed') {
              throw new Error(status.error || 'Video processing failed')
            } else {
              // Update progress
              const progress = 50 + (status.progress || 0) * 0.5
              setUploadProgress(progress)
              setProcessingStep(status.current_step || 'Processing...')
            }
          }
        } catch (error) {
          console.error('Status check error:', error)
        }
        
        attempts++
      }

      if (!processingComplete) {
        throw new Error('Video processing timed out')
      }

    } catch (error) {
      console.error('Video upload error:', error)
      setErrorMessage(error.message || 'Failed to process video')
      setUploadStatus('error')
    }
  }

  const resetUpload = () => {
    setUploadStatus('idle')
    setUploadProgress(0)
    setUploadedFiles([])
    setProcessingStep('')
    setErrorMessage('')
  }

  return (
    <div className="video-sequence-uploader">
      <div className="uploader-header">
        <Film size={32} />
        <div>
          <h2>Upload Video or EXR Sequence</h2>
          <p>Upload your crowd footage for AI enhancement processing</p>
        </div>
      </div>

      {uploadStatus === 'idle' && (
        <div 
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <Upload size={48} />
          <h3>Drag & Drop Your Files Here</h3>
          <p>Supported formats:</p>
          <div className="supported-formats">
            <div className="format-group">
              <strong>Video Files:</strong>
              <span>MP4, MOV, AVI, MKV</span>
            </div>
            <div className="format-group">
              <strong>EXR Sequences:</strong>
              <span>Multi-pass renders (beauty, depth, emission, normal)</span>
            </div>
          </div>
          
          <div className="upload-actions">
            <label className="upload-button">
              <input 
                type="file" 
                multiple 
                accept=".mp4,.mov,.avi,.mkv,.exr"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
              Choose Files
            </label>
          </div>
          
          <div className="upload-hints">
            <div className="hint">
              <strong>Video Upload:</strong> We'll automatically extract proxy crowd renders with depth and emission data
            </div>
            <div className="hint">
              <strong>EXR Sequence:</strong> Upload your pre-rendered multi-pass sequences directly
            </div>
          </div>
        </div>
      )}

      {(uploadStatus === 'uploading' || uploadStatus === 'processing') && (
        <div className="upload-progress">
          <div className="progress-header">
            <div className="progress-icon">
              <div className="spinner"></div>
            </div>
            <div className="progress-info">
              <h3>Processing Your Upload</h3>
              <p className="progress-step">{processingStep}</p>
            </div>
          </div>
          
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <div className="progress-text">{uploadProgress}%</div>
          
          <div className="file-list">
            <h4>Files:</h4>
            {uploadedFiles.map((file, index) => (
              <div key={index} className="file-item">
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {uploadStatus === 'complete' && (
        <div className="upload-complete">
          <div className="success-icon">
            <CheckCircle size={48} />
          </div>
          <h3>Upload Successful!</h3>
          <p>{processingStep}</p>
          <div className="file-summary">
            <strong>Files processed:</strong> {uploadedFiles.length}
          </div>
          <button className="reset-button" onClick={resetUpload}>
            Upload Another Sequence
          </button>
        </div>
      )}

      {uploadStatus === 'error' && (
        <div className="upload-error">
          <div className="error-icon">
            <AlertCircle size={48} />
          </div>
          <h3>Upload Failed</h3>
          <p className="error-message">{errorMessage}</p>
          <button className="reset-button" onClick={resetUpload}>
            Try Again
          </button>
        </div>
      )}
    </div>
  )
}

export default VideoSequenceUploader