import React, { useState } from 'react';
import { Brain, Users, Eye, ChevronRight } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ currentView, onViewChange }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const workflows = [
    {
      id: 'simple-v2v',
      title: 'Simple V2V Enhancement',
      description: 'Video-to-Video neural network model for basic image enhancement',
      icon: Brain,
      badge: 'V2V'
    },
    {
      id: '3d-ai-generation',
      title: '3D AI Generation',
      description: 'Text-to-Image-to-3D using Hunyuan models for 3D asset creation',
      icon: Brain,
      badge: '3D'
    },
    {
      id: 'sequence-viewer',
      title: 'EXR Sequence Viewer',
      description: 'View and manage EXR image sequences',
      icon: Eye,
      badge: 'TOOL'
    }
  ];

  return (
    <div 
      className={`sidebar ${isExpanded ? 'expanded' : 'collapsed'}`}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon">
            <Brain className="w-6 h-6" />
          </div>
          <div className={`logo-text ${isExpanded ? 'visible' : 'hidden'}`}>
            <h2>VFX AI</h2>
            <p>Enhancement Platform</p>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section">
          <div className={`section-title ${isExpanded ? 'visible' : 'hidden'}`}>
            AI Models
          </div>
          
          {workflows.map((workflow) => {
            const IconComponent = workflow.icon;
            const isActive = currentView === workflow.id;
            
            return (
              <button
                key={workflow.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => onViewChange(workflow.id)}
                title={!isExpanded ? workflow.title : ''}
              >
                <div className="nav-item-content">
                  <div className="nav-icon">
                    <IconComponent className="w-5 h-5" />
                  </div>
                  
                  <div className={`nav-details ${isExpanded ? 'visible' : 'hidden'}`}>
                    <div className="nav-title">
                      {workflow.title}
                      <span className={`nav-badge badge-${workflow.id}`}>
                        {workflow.badge}
                      </span>
                    </div>
                    <div className="nav-description">
                      {workflow.description}
                    </div>
                  </div>
                  
                  {isActive && (
                    <div className="nav-indicator">
                      <ChevronRight className="w-4 h-4" />
                    </div>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </nav>

      <div className="sidebar-footer">
        <div className={`footer-content ${isExpanded ? 'visible' : 'hidden'}`}>
          <div className="status-indicator">
            <div className="status-dot"></div>
            <span>AI Models Ready</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;