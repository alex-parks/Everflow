import React, { useState, useEffect } from 'react'
import './FrameViewer.css'

function FrameViewer({ sequenceId, frameNum }) {
  const [imageUrl, setImageUrl] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!sequenceId) return

    setLoading(true)
    setError(null)

    const url = `/api/frames/${sequenceId}/frame/${frameNum}`
    setImageUrl(url)

    const img = new Image()
    img.onload = () => setLoading(false)
    img.onerror = () => {
      setError('Failed to load frame')
      setLoading(false)
    }
    img.src = url
  }, [sequenceId, frameNum])

  return (
    <div className="frame-viewer">
      <div className="frame-container">
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Loading frame {frameNum}...</p>
          </div>
        )}
        
        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}
        
        {!loading && !error && (
          <img 
            src={imageUrl} 
            alt={`Frame ${frameNum}`}
            className="frame-image"
          />
        )}
      </div>
      
      <div className="frame-info">
        <span>Frame: {frameNum}</span>
        <span>Sequence: {sequenceId}</span>
      </div>
    </div>
  )
}

export default FrameViewer