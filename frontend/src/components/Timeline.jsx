import React, { useState, useRef, useEffect } from 'react'
import './Timeline.css'

const Timeline = ({ currentFrame, totalFrames, onFrameChange, sequenceId, exposure }) => {
  const [isDragging, setIsDragging] = useState(false)
  const [thumbnails, setThumbnails] = useState({})
  const [loadingThumbnails, setLoadingThumbnails] = useState(false)
  const timelineRef = useRef(null)
  const thumbnailSize = 32

  // Calculate visible thumbnail range based on timeline width
  const maxVisibleThumbnails = 20
  const step = Math.max(1, Math.floor(totalFrames / maxVisibleThumbnails))

  useEffect(() => {
    // Load thumbnails for visible frames
    const loadThumbnails = async () => {
      setLoadingThumbnails(true)
      const newThumbnails = {}
      
      for (let i = 0; i < totalFrames; i += step) {
        try {
          const url = `http://localhost:4005/api/frames/${sequenceId}/frame/${i}/thumbnail?size=${thumbnailSize}&exposure=${exposure}`
          newThumbnails[i] = url
        } catch (error) {
          console.error(`Failed to load thumbnail for frame ${i}:`, error)
        }
      }
      
      setThumbnails(newThumbnails)
      setLoadingThumbnails(false)
    }

    if (sequenceId && totalFrames > 0) {
      loadThumbnails()
    }
  }, [sequenceId, totalFrames, exposure, step])

  const handleMouseDown = (e) => {
    setIsDragging(true)
    updateFrameFromPosition(e)
  }

  const handleMouseMove = (e) => {
    if (isDragging) {
      updateFrameFromPosition(e)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const updateFrameFromPosition = (e) => {
    if (!timelineRef.current) return
    
    const rect = timelineRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = Math.max(0, Math.min(1, x / rect.width))
    const frame = Math.round(percentage * (totalFrames - 1))
    onFrameChange(frame)
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging])

  const progressPercentage = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0

  return (
    <div className="timeline">
      <div className="timeline-header">
        <span className="timeline-title">Timeline</span>
        <span className="frame-info">
          Frame {currentFrame + 1} / {totalFrames}
        </span>
      </div>
      
      <div 
        className="timeline-container"
        ref={timelineRef}
        onMouseDown={handleMouseDown}
      >
        <div className="timeline-track">
          <div 
            className="timeline-progress"
            style={{ width: `${progressPercentage}%` }}
          />
          <div 
            className="timeline-handle"
            style={{ left: `${progressPercentage}%` }}
          />
        </div>
        
        <div className="timeline-thumbnails">
          {Object.entries(thumbnails).map(([frameIndex, url]) => {
            const frame = parseInt(frameIndex)
            const position = totalFrames > 1 ? (frame / (totalFrames - 1)) * 100 : 0
            
            return (
              <div
                key={frame}
                className={`timeline-thumbnail ${frame === currentFrame ? 'active' : ''}`}
                style={{ left: `${position}%` }}
                onClick={(e) => {
                  e.stopPropagation()
                  onFrameChange(frame)
                }}
              >
                <img
                  src={url}
                  alt={`Frame ${frame}`}
                  className="thumbnail-image"
                  onError={(e) => {
                    e.target.style.display = 'none'
                  }}
                />
                <div className="thumbnail-frame-number">{frame + 1}</div>
              </div>
            )
          })}
        </div>
      </div>
      
      <div className="timeline-markers">
        {Array.from({ length: Math.min(11, totalFrames) }, (_, i) => {
          const frame = Math.round((i / 10) * (totalFrames - 1))
          return (
            <div
              key={i}
              className="timeline-marker"
              style={{ left: `${(i / 10) * 100}%` }}
            >
              {frame + 1}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default Timeline