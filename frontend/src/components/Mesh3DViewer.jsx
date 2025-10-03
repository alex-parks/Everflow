import React, { Suspense, useRef, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Center } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import * as THREE from 'three';
import { RotateCcw, ZoomIn, ZoomOut, RotateCw } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Mesh3DViewer error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="mesh-viewer-error">
          <div className="error-icon">⚠️</div>
          <p>3D Viewer Error</p>
          <p className="error-details">{this.state.error?.message || 'Unknown error'}</p>
        </div>
      );
    }

    return this.props.children;
  }
}

function MeshModel({ url, onLoad, onError }) {
  const meshRef = useRef();
  const [rotation, setRotation] = useState([0, 0, 0]);

  // Load the PLY mesh with error handling
  const geometry = useLoader(PLYLoader, url);

  // Call onLoad when geometry is loaded
  React.useEffect(() => {
    if (geometry && onLoad) {
      onLoad();
    }
  }, [geometry, onLoad]);

  // Auto-rotate the mesh
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
  });

  // Create material with proper lighting
  const material = new THREE.MeshStandardMaterial({
    color: '#8b5cf6',
    roughness: 0.3,
    metalness: 0.1,
    flatShading: false,
  });

  return (
    <Center>
      <mesh ref={meshRef} geometry={geometry} material={material} scale={[1, 1, 1]} />
    </Center>
  );
}

function LoadingFallback() {
  return (
    <div className="mesh-viewer-loading">
      <div className="loading-spinner"></div>
      <p>Loading 3D mesh...</p>
    </div>
  );
}

export default function Mesh3DViewer({ meshUrl, className = "" }) {
  const [autoRotate, setAutoRotate] = useState(true);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const controlsRef = useRef();

  const handleResetView = () => {
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  };

  const handleMeshLoad = () => {
    setLoading(false);
    setError(null);
  };

  const handleMeshError = (error) => {
    setLoading(false);
    setError(error);
  };

  if (error) {
    return (
      <div className={`mesh-viewer ${className}`}>
        <div className="mesh-viewer-container">
          <div className="mesh-viewer-error">
            <div className="error-icon">⚠️</div>
            <p>Failed to load 3D mesh</p>
            <p className="error-details">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className={`mesh-viewer ${className}`}>
        <div className="mesh-viewer-container">
          {loading && (
            <div className="mesh-viewer-loading">
              <div className="loading-spinner"></div>
              <p>Loading 3D mesh...</p>
            </div>
          )}
          <Canvas>
          <PerspectiveCamera makeDefault fov={50} position={[3, 2, 3]} />

          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <pointLight position={[-10, -10, -10]} />

          {/* Environment for reflections */}
          <Environment preset="studio" />

          {/* Controls */}
          <OrbitControls
            ref={controlsRef}
            autoRotate={autoRotate}
            autoRotateSpeed={0.5}
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={1}
            maxDistance={10}
          />

          {/* 3D Model */}
          <Suspense fallback={null}>
            {meshUrl && (
              <MeshModel
                url={meshUrl}
                onLoad={handleMeshLoad}
                onError={handleMeshError}
              />
            )}
          </Suspense>
        </Canvas>
      </div>

      {/* 3D Viewer Controls */}
      <div className="mesh-viewer-controls">
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className="viewer-control-button"
          title={autoRotate ? "Stop rotation" : "Auto rotate"}
        >
          {autoRotate ? <RotateCcw className="w-4 h-4" /> : <RotateCw className="w-4 h-4" />}
        </button>

        <button
          onClick={handleResetView}
          className="viewer-control-button"
          title="Reset view"
        >
          <ZoomOut className="w-4 h-4" />
        </button>

        <span className="viewer-hint">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </span>
      </div>
    </div>
    </ErrorBoundary>
  );
}