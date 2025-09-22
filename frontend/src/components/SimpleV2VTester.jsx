import React, { useState, useRef } from 'react';
import { Upload, Play, Download, FileImage, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

const SimpleV2VTester = () => {
  // For testing, start with the existing sequence
  const [uploadedSequence, setUploadedSequence] = useState({
    sequence_id: "82d94644-9cb0-46e0-b608-b4f9f7bcea5a",
    metadata: { frame_count: 10 }
  });
  const [processingStatus, setProcessingStatus] = useState('completed');
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [enhancedFrames, setEnhancedFrames] = useState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (!files.length) return;

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('name', `Test Sequence ${Date.now()}`);
      
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch('http://localhost:4005/api/crowd/upload-exr-sequence', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      setUploadedSequence(result);
      setProcessingStatus('uploaded');
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleEnhanceSequence = async () => {
    if (!uploadedSequence) return;

    setIsProcessing(true);
    setProcessingStatus('processing');
    
    try {
      const response = await fetch(
        `http://localhost:4005/api/crowd/enhance-sequence/${uploadedSequence.sequence_id}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error(`Enhancement failed: ${response.statusText}`);
      }

      const result = await response.json();
      setProcessingStatus('completed');
      
      // Load enhanced frames info
      const infoResponse = await fetch(
        `http://localhost:4005/api/crowd/sequences/${uploadedSequence.sequence_id}`
      );
      const info = await infoResponse.json();
      
      // Create array of enhanced frame indices
      const frameIndices = Array.from(
        { length: info.enhanced_frames_available }, 
        (_, i) => i
      );
      setEnhancedFrames(frameIndices);
      
    } catch (error) {
      console.error('Enhancement error:', error);
      alert(`Enhancement failed: ${error.message}`);
      setProcessingStatus('error');
    } finally {
      setIsProcessing(false);
    }
  };

  const getFrameUrl = (frameIndex, enhanced = false) => {
    return `http://localhost:4005/api/crowd/sequences/${uploadedSequence.sequence_id}/frame/${frameIndex}?enhanced=${enhanced}`;
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <FileImage className="w-6 h-6" />
          Simple V2V Model Tester
        </h2>
        
        <p className="text-gray-600 mb-6">
          Upload image files (EXR, JPG, PNG, TIFF) to test the V2V enhancement model
        </p>

        {/* Upload Section */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-6">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".exr,.jpg,.jpeg,.png,.tiff,.tif"
            onChange={handleFileUpload}
            className="hidden"
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {isUploading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            {isUploading ? 'Uploading...' : 'Select Files'}
          </button>
          
          <p className="text-sm text-gray-500 mt-2">
            Select multiple image files to create a sequence
          </p>
        </div>

        {/* Status Display */}
        {processingStatus && (
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              {processingStatus === 'uploaded' && (
                <>
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-green-700">Sequence uploaded successfully</span>
                </>
              )}
              {processingStatus === 'processing' && (
                <>
                  <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                  <span className="text-blue-700">Processing with V2V model...</span>
                </>
              )}
              {processingStatus === 'completed' && (
                <>
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-green-700">Enhancement completed</span>
                </>
              )}
              {processingStatus === 'error' && (
                <>
                  <AlertCircle className="w-5 h-5 text-red-500" />
                  <span className="text-red-700">Enhancement failed</span>
                </>
              )}
            </div>

            {uploadedSequence && (
              <div className="text-sm text-gray-600">
                Sequence ID: {uploadedSequence.sequence_id}
                <br />
                Frames: {uploadedSequence.metadata.frame_count}
              </div>
            )}
          </div>
        )}

        {/* Enhancement Button */}
        {uploadedSequence && processingStatus === 'uploaded' && (
          <div className="mb-6">
            <button
              onClick={handleEnhanceSequence}
              disabled={isProcessing}
              className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isProcessing ? 'Enhancing...' : 'Enhance with V2V Model'}
            </button>
          </div>
        )}

        {/* Results Display */}
        {enhancedFrames.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Enhanced Results</h3>
            
            {/* Frame Selector */}
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Frame:</label>
              <select
                value={selectedFrame}
                onChange={(e) => setSelectedFrame(parseInt(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1"
              >
                {enhancedFrames.map((_, index) => (
                  <option key={index} value={index}>
                    Frame {index + 1}
                  </option>
                ))}
              </select>
            </div>

            {/* Before/After Comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <h4 className="text-md font-medium mb-2">Original</h4>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={getFrameUrl(selectedFrame, false)}
                    alt={`Original frame ${selectedFrame + 1}`}
                    className="w-full h-auto"
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjMuNGY0LWY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZhaWxlZCB0byBsb2FkPC90ZXh0Pjwvc3ZnPg==';
                    }}
                  />
                </div>
              </div>

              <div>
                <h4 className="text-md font-medium mb-2">Enhanced (V2V Model)</h4>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={getFrameUrl(selectedFrame, true)}
                    alt={`Enhanced frame ${selectedFrame + 1}`}
                    className="w-full h-auto"
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjMuNGY0LWY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZhaWxlZCB0byBsb2FkPC90ZXh0Pjwvc3ZnPg==';
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Download Links */}
            <div className="flex gap-2">
              <a
                href={getFrameUrl(selectedFrame, false)}
                download={`original_frame_${selectedFrame + 1}`}
                className="inline-flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded hover:bg-gray-50"
              >
                <Download className="w-4 h-4" />
                Download Original
              </a>
              <a
                href={getFrameUrl(selectedFrame, true)}
                download={`enhanced_frame_${selectedFrame + 1}.jpg`}
                className="inline-flex items-center gap-2 px-3 py-2 text-sm bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                <Download className="w-4 h-4" />
                Download Enhanced
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SimpleV2VTester;