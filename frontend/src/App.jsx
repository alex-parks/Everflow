import React, { useState, useEffect } from 'react'
import SequenceUploader from './components/SequenceUploader'
import SequenceViewer from './components/SequenceViewer'
import SequenceSelector from './components/SequenceSelector'
import CrowdEnhancementWorkflow from './components/CrowdEnhancementWorkflow'
import SimpleV2VWorkflow from './components/SimpleV2VWorkflow'
import Sidebar from './components/Sidebar'
import './App.css'

function App() {
  const [currentView, setCurrentView] = useState('simple-v2v')
  const [sequences, setSequences] = useState([])
  const [currentSequence, setCurrentSequence] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchSequences = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:4005/api/sequences/')
      const data = await response.json()
      setSequences(data)
    } catch (error) {
      console.error('Error fetching sequences:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (currentView === 'sequence-viewer') {
      fetchSequences()
    }
  }, [currentView])

  const handleSequenceUpload = (newSequence) => {
    fetchSequences() // Refresh the list
  }

  const handleSequenceSelect = (sequence) => {
    setCurrentSequence(sequence)
  }

  const handleBackToList = () => {
    setCurrentSequence(null)
  }

  return (
    <div className="app">
      <Sidebar 
        currentView={currentView} 
        onViewChange={setCurrentView} 
      />
      
      <div className="app-content">
        {currentView === 'sequence-viewer' && currentSequence && (
          <div className="app-header">
            <button onClick={handleBackToList} className="back-button">
              ← Back to Sequences
            </button>
          </div>
        )}
        
        <main className="app-main">
          {currentView === 'simple-v2v' && (
            <SimpleV2VWorkflow />
          )}

          {currentView === 'crowd-enhancement' && (
            <CrowdEnhancementWorkflow />
          )}

          {currentView === 'sequence-viewer' && (
            <div className="sequence-viewer-container">
              {!currentSequence ? (
                <div className="sequence-management">
                  <SequenceUploader onUpload={handleSequenceUpload} />
                  <SequenceSelector 
                    sequences={sequences}
                    onSelect={handleSequenceSelect}
                    loading={loading}
                  />
                </div>
              ) : (
                <SequenceViewer sequence={currentSequence} />
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App