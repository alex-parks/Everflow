import React from 'react'
import './SequenceSelector.css'

const SequenceSelector = ({ sequences, onSelect, loading }) => {
  if (loading) {
    return (
      <div className="sequence-selector">
        <h2>Available Sequences</h2>
        <div className="loading">Loading sequences...</div>
      </div>
    )
  }

  if (sequences.length === 0) {
    return (
      <div className="sequence-selector">
        <h2>Available Sequences</h2>
        <div className="empty-state">
          <p>No sequences found. Upload some EXR sequences to get started.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="sequence-selector">
      <h2>Available Sequences ({sequences.length})</h2>
      <div className="sequences-grid">
        {sequences.map((sequence) => (
          <div
            key={sequence.id}
            className="sequence-card"
            onClick={() => onSelect(sequence)}
          >
            <div className="sequence-info">
              <div className="sequence-id">
                ID: {sequence.id.substring(0, 8)}...
              </div>
              <div className="sequence-details">
                <span className="frame-count">{sequence.frame_count} frames</span>
                <span className="format">{sequence.format.toUpperCase()}</span>
              </div>
            </div>
            <div className="sequence-preview">
              {sequence.frame_count > 0 && (
                <img
                  src={`http://localhost:4005/api/frames/${sequence.id}/frame/0/thumbnail?size=128`}
                  alt="First frame"
                  className="thumbnail"
                  onError={(e) => {
                    e.target.style.display = 'none'
                  }}
                />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SequenceSelector