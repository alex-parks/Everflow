import React, { useState, useEffect, useCallback, useRef } from 'react'
import Timeline from './Timeline'
import ViewerControls from './ViewerControls'
import { imageCache } from '../utils/imageCache'
import './SequenceViewer.css'

const SequenceViewer = ({ sequence }) => {
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [exposure, setExposure] = useState(0.0)
  const [playbackSpeed, setPlaybackSpeed] = useState(24) // fps
  const [loading, setLoading] = useState(false)
  const [imageUrl, setImageUrl] = useState('')
  const [cacheStats, setCacheStats] = useState({ size: 0, maxSize: 150 })
  const intervalRef = useRef(null)
  const imageRef = useRef(null)
  const preloadTimeoutRef = useRef(null)

  const maxFrame = sequence.frame_count - 1

  // Load image with caching
  const loadCurrentFrame = useCallback(async () => {
    try {
      setLoading(true)
      const url = await imageCache.loadImage(sequence.id, currentFrame, exposure)
      setImageUrl(url)
      
      // Update cache stats
      setCacheStats(imageCache.getCacheStats())
      
      // Trigger intelligent preloading
      schedulePreloading()
    } catch (error) {
      console.error('Error loading frame:', error)
    } finally {
      setLoading(false)
    }
  }, [sequence.id, currentFrame, exposure])

  // Intelligent preloading based on playback state
  const schedulePreloading = useCallback(() => {
    if (preloadTimeoutRef.current) {
      clearTimeout(preloadTimeoutRef.current)
    }

    preloadTimeoutRef.current = setTimeout(async () => {
      if (isPlaying) {
        // When playing, preload upcoming frames in playback direction
        const preloadCount = Math.min(10, Math.ceil(playbackSpeed / 4))
        const framesToPreload = []
        
        for (let i = 1; i <= preloadCount; i++) {
          const nextFrame = currentFrame + i
          if (nextFrame <= maxFrame) {
            framesToPreload.push(nextFrame)
          }
        }
        
        if (framesToPreload.length > 0) {
          imageCache.preloadFrames(sequence.id, framesToPreload, exposure)
          
          // Also request backend preloading for smoother experience
          const endFrame = Math.min(maxFrame, currentFrame + preloadCount * 2)
          imageCache.requestBackendPreload(sequence.id, currentFrame + 1, endFrame, exposure)
        }
      } else {
        // When paused, preload frames around current position
        await imageCache.preloadAdjacent(sequence.id, currentFrame, sequence.frame_count, exposure, 8)
      }
    }, 100) // Small delay to avoid excessive preloading during rapid changes
  }, [sequence.id, currentFrame, exposure, isPlaying, playbackSpeed, maxFrame])

  // Load frame when dependencies change
  useEffect(() => {
    loadCurrentFrame()
  }, [loadCurrentFrame])

  // Clear cache when sequence changes
  useEffect(() => {
    imageCache.clearSequence(sequence.id)
    setCurrentFrame(0)
    setImageUrl('')
  }, [sequence.id])

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      switch (e.key) {
        case ' ':
          e.preventDefault()
          setIsPlaying(prev => !prev)
          break
        case 'ArrowLeft':
          e.preventDefault()
          setCurrentFrame(prev => Math.max(0, prev - 1))
          break
        case 'ArrowRight':
          e.preventDefault()
          setCurrentFrame(prev => Math.min(maxFrame, prev + 1))
          break
        case 'Home':
          e.preventDefault()
          setCurrentFrame(0)
          break
        case 'End':
          e.preventDefault()
          setCurrentFrame(maxFrame)
          break
        default:
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [maxFrame])

  // Handle playback
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentFrame(prev => {
          if (prev >= maxFrame) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 1000 / playbackSpeed)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isPlaying, playbackSpeed, maxFrame])

  const handleFrameChange = useCallback((newFrame) => {
    setCurrentFrame(Math.max(0, Math.min(maxFrame, newFrame)))
  }, [maxFrame])

  const handlePlayPause = () => {
    setIsPlaying(prev => !prev)
  }

  const handlePreviousFrame = () => {
    setCurrentFrame(prev => Math.max(0, prev - 1))
  }

  const handleNextFrame = () => {
    setCurrentFrame(prev => Math.min(maxFrame, prev + 1))
  }

  const handleFirstFrame = () => {
    setCurrentFrame(0)
  }

  const handleLastFrame = () => {
    setCurrentFrame(maxFrame)
  }

  const handlePreloadSequence = async () => {
    try {
      setLoading(true)
      // Preload the entire sequence in chunks to avoid overwhelming the system
      const chunkSize = 20
      for (let start = 0; start < sequence.frame_count; start += chunkSize) {
        const end = Math.min(start + chunkSize - 1, maxFrame)
        await imageCache.preloadRange(sequence.id, start, end, exposure)
        setCacheStats(imageCache.getCacheStats())
        
        // Small delay between chunks to keep UI responsive
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      
      // Also trigger backend preloading
      await imageCache.requestBackendPreload(sequence.id, 0, maxFrame, exposure)
    } catch (error) {
      console.error('Error preloading sequence:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="sequence-viewer">
      <div className="viewer-main">
        <div className="image-container">
          {imageUrl && (
            <img
              ref={imageRef}
              src={imageUrl}
              alt={`Frame ${currentFrame}`}
              className="sequence-image"
            />
          )}
          {loading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
            </div>
          )}
          <div className="frame-info">
            Frame {currentFrame + 1} of {sequence.frame_count}
          </div>
          <div className="cache-info">
            Cache: {cacheStats.size}/{cacheStats.maxSize}
          </div>
        </div>
      </div>

      <div className="viewer-controls-panel">
        <ViewerControls
          isPlaying={isPlaying}
          onPlayPause={handlePlayPause}
          onPreviousFrame={handlePreviousFrame}
          onNextFrame={handleNextFrame}
          onFirstFrame={handleFirstFrame}
          onLastFrame={handleLastFrame}
          currentFrame={currentFrame}
          totalFrames={sequence.frame_count}
          exposure={exposure}
          onExposureChange={setExposure}
          playbackSpeed={playbackSpeed}
          onPlaybackSpeedChange={setPlaybackSpeed}
          onPreloadSequence={handlePreloadSequence}
          cacheStats={cacheStats}
        />
        
        <Timeline
          currentFrame={currentFrame}
          totalFrames={sequence.frame_count}
          onFrameChange={handleFrameChange}
          sequenceId={sequence.id}
          exposure={exposure}
        />
      </div>

      <div className="keyboard-shortcuts">
        <div className="shortcuts-info">
          <span>⌨️ Shortcuts:</span>
          <span>Space = Play/Pause</span>
          <span>← → = Frame by frame</span>
          <span>Home/End = First/Last frame</span>
        </div>
      </div>
    </div>
  )
}

export default SequenceViewer