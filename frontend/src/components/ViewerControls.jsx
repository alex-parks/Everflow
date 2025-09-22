import React from 'react'
import './ViewerControls.css'

const ViewerControls = ({
  isPlaying,
  onPlayPause,
  onPreviousFrame,
  onNextFrame,
  onFirstFrame,
  onLastFrame,
  currentFrame,
  totalFrames,
  exposure,
  onExposureChange,
  playbackSpeed,
  onPlaybackSpeedChange,
  onPreloadSequence,
  cacheStats
}) => {
  const playbackSpeeds = [1, 6, 12, 24, 30, 48, 60]

  return (
    <div className="viewer-controls">
      <div className="playback-controls">
        <button onClick={onFirstFrame} className="control-btn" title="First Frame (Home)">
          ⏮
        </button>
        <button onClick={onPreviousFrame} className="control-btn" title="Previous Frame (←)">
          ⏪
        </button>
        <button onClick={onPlayPause} className="play-pause-btn" title="Play/Pause (Space)">
          {isPlaying ? '⏸' : '▶️'}
        </button>
        <button onClick={onNextFrame} className="control-btn" title="Next Frame (→)">
          ⏩
        </button>
        <button onClick={onLastFrame} className="control-btn" title="Last Frame (End)">
          ⏭
        </button>
      </div>

      <div className="frame-counter">
        <span className="current-frame">{currentFrame + 1}</span>
        <span className="frame-separator">/</span>
        <span className="total-frames">{totalFrames}</span>
      </div>

      <div className="settings-controls">
        <div className="control-group">
          <label htmlFor="exposure-slider">Exposure:</label>
          <input
            id="exposure-slider"
            type="range"
            min="-3"
            max="3"
            step="0.1"
            value={exposure}
            onChange={(e) => onExposureChange(parseFloat(e.target.value))}
            className="exposure-slider"
          />
          <span className="exposure-value">{exposure.toFixed(1)}</span>
        </div>

        <div className="control-group">
          <label htmlFor="speed-select">Speed:</label>
          <select
            id="speed-select"
            value={playbackSpeed}
            onChange={(e) => onPlaybackSpeedChange(parseInt(e.target.value))}
            className="speed-select"
          >
            {playbackSpeeds.map(speed => (
              <option key={speed} value={speed}>
                {speed} fps
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <button 
            onClick={onPreloadSequence}
            className="preload-btn"
            title="Preload entire sequence for smooth playback"
          >
            🚀 Preload All
          </button>
        </div>
      </div>
    </div>
  )
}

export default ViewerControls