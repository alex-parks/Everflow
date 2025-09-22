import React, { useState } from 'react'
import axios from 'axios'
import './SequenceUploader.css'

function SequenceUploader({ onUpload }) {
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const handleFiles = async (files) => {
    if (files.length === 0) return

    setUploading(true)
    
    try {
      const formData = new FormData()
      Array.from(files).forEach(file => {
        formData.append('files', file)
      })

      const response = await axios.post('/api/sequences/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      onUpload(response.data)
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const files = e.dataTransfer.files
    handleFiles(files)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleFileInput = (e) => {
    const files = e.target.files
    handleFiles(files)
  }

  return (
    <div className="uploader-container">
      <div 
        className={`dropzone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {uploading ? (
          <div className="upload-status">
            <div className="spinner"></div>
            <p>Uploading image sequence...</p>
          </div>
        ) : (
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <h3>Upload Image Sequence</h3>
            <p>Drag and drop your image files here, or click to browse</p>
            <p className="file-info">Supports: EXR, JPG, PNG, TIFF</p>
            <input
              type="file"
              multiple
              accept=".exr,.jpg,.jpeg,.png,.tiff,.tif"
              onChange={handleFileInput}
              className="file-input"
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default SequenceUploader