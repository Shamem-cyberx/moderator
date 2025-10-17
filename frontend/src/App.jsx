import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:5000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [textContent, setTextContent] = useState('');

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
      if (!validTypes.includes(file.type)) {
        setUploadStatus('error');
        setUploadResult({
          error: 'Invalid file type. Please upload JPEG, PNG, GIF, or WebP images.'
        });
        return;
      }

      if (file.size > 16 * 1024 * 1024) {
        setUploadStatus('error');
        setUploadResult({
          error: 'File too large. Maximum size is 16MB.'
        });
        return;
      }

      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setUploadStatus('');
      setUploadResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('error');
      setUploadResult({ error: 'Please select a file first' });
      return;
    }

    setIsUploading(true);
    setUploadStatus('uploading');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('text_content', textContent);

    try {
      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000,
      });

      setUploadStatus('success');
      setUploadResult(response.data);
      loadStats();
    } catch (error) {
      setUploadStatus('error');
      if (error.response) {
        setUploadResult(error.response.data);
      } else if (error.request) {
        setUploadResult({ error: 'Network error. Please check if the server is running.' });
      } else {
        setUploadResult({ error: 'An unexpected error occurred.' });
      }
    } finally {
      setIsUploading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setTextContent('');
    setUploadStatus('');
    setUploadResult(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  const getCategoryIcon = (category) => {
    const icons = {
      'explicit_nudity': '🔞',
      'suggestive_content': '🧍',
      'offensive_text': '🤬',
      'violence': '💉',
      'hate_content': '💀',
      'property_relevance': '🏘️🚗'
    };
    return icons[category] || '📊';
  };

  const getCategoryName = (category) => {
    const names = {
      'explicit_nudity': 'Explicit Nudity',
      'suggestive_content': 'Suggestive Content',
      'offensive_text': 'Offensive Text',
      'violence': 'Violence',
      'hate_content': 'Hate Content',
      'property_relevance': 'Property/Vehicle Relevance'
    };
    return names[category] || category;
  };

  const getScoreColor = (score) => {
    if (score >= 70) return '#ff4444';
    if (score >= 30) return '#ffaa00';
    return '#00c851';
  };

  React.useEffect(() => {
    loadStats();
  }, []);

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🏠🚗 AI Property & Vehicle Moderation</h1>
          <p className="subtitle">
            Advanced content analysis for property and vehicle listings with strict safety assessment
          </p>
        </header>

        <div className="upload-area">
          {!selectedFile ? (
            <div className="file-selector">
              <input
                type="file"
                id="file-input"
                accept="image/jpeg,image/png,image/gif,image/webp"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input" className="file-label">
                <div className="upload-icon">🤖</div>
                <h3>Select Property or Vehicle Image</h3>
                <p>AI will analyze content for safety and relevance to property/vehicle listings</p>
                <div className="file-requirements">
                  <strong>Supported formats:</strong> JPEG, PNG, GIF, WebP<br />
                  <strong>Max size:</strong> 16MB<br />
                  <strong>AI Analysis:</strong> 6 safety categories + scoring
                </div>
              </label>
            </div>
          ) : (
            <div className="preview-section">
              <div className="image-preview">
                <img src={previewUrl} alt="Preview" />
                <button 
                  className="remove-btn" 
                  onClick={resetUpload}
                  title="Remove image"
                >
                  ×
                </button>
              </div>
              
              <div className="text-input-section">
                <label htmlFor="text-content" className="text-label">
                  Description (AI will analyze this text):
                </label>
                <textarea
                  id="text-content"
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  placeholder="Describe your property or vehicle... This text will be analyzed for offensive content, hate speech, etc."
                  className="text-input"
                  rows="3"
                />
              </div>
              
              <div className="upload-controls">
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className={`upload-btn ${isUploading ? 'uploading' : ''}`}
                >
                  {isUploading ? (
                    <>
                      <span className="spinner"></span>
                      AI Analyzing...
                    </>
                  ) : (
                    'Upload & Analyze'
                  )}
                </button>
                <button
                  onClick={resetUpload}
                  disabled={isUploading}
                  className="cancel-btn"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>

        {uploadStatus && (
          <div className={`result-message ${uploadStatus}`}>
            {uploadStatus === 'uploading' && (
              <div className="loading-spinner">
                <div className="spinner"></div>
                <p>AI is analyzing content across 6 safety categories...</p>
              </div>
            )}
            
            {uploadResult && (
              <>
                {!uploadResult.success && (
                  <div className="message-content">
                    <div className="message-icon">
                      {uploadResult.decision === 'blocked' ? '🚫' : 
                       uploadResult.decision === 'manual_review' ? '⚠️' : '❌'}
                    </div>
                    <div className="message-details">
                      <strong>{uploadResult.error}</strong>
                      <div className="detail-message">{uploadResult.message}</div>
                      
                      {uploadResult.risk_score !== undefined && (
                        <div className="risk-score-display">
                          <div className="risk-score-header">
                            <span>Overall Risk Score:</span>
                            <span className="risk-score-value" style={{color: getScoreColor(uploadResult.risk_score)}}>
                              {uploadResult.risk_score}/100
                            </span>
                          </div>
                          <div className="risk-level-badge">
                            Risk Level: <strong>{uploadResult.risk_level?.toUpperCase()}</strong>
                          </div>
                        </div>
                      )}
                      
                      {uploadResult.categories && (
                        <div className="categories-analysis">
                          <h4>Category Analysis:</h4>
                          <div className="categories-grid">
                            {Object.entries(uploadResult.categories).map(([category, data]) => (
                              <div key={category} className="category-card">
                                <div className="category-header">
                                  <span className="category-icon">{getCategoryIcon(category)}</span>
                                  <span className="category-name">{getCategoryName(category)}</span>
                                </div>
                                <div className="score-bar">
                                  <div 
                                    className="score-fill"
                                    style={{
                                      width: `${data.score}%`,
                                      backgroundColor: getScoreColor(data.score)
                                    }}
                                  ></div>
                                  <span className="score-text">{data.score}/100</span>
                                </div>
                                {data.reason && (
                                  <div className="category-reason">{data.reason}</div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {uploadResult.detailed_explanation && (
                        <div className="detailed-explanation">
                          <strong>AI Analysis:</strong> {uploadResult.detailed_explanation}
                        </div>
                      )}
                      
                      {uploadResult.confidence && (
                        <div className="confidence-level">
                          Confidence: <strong>{uploadResult.confidence}</strong>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {uploadResult.success && (
                  <div className="message-content">
                    <div className="message-icon">✅</div>
                    <div className="message-details">
                      <strong>Upload Successful!</strong>
                      <div className="detail-message">{uploadResult.message}</div>
                      
                      {uploadResult.risk_score !== undefined && (
                        <div className="risk-score-display success">
                          <div className="risk-score-header">
                            <span>Safety Score:</span>
                            <span className="risk-score-value" style={{color: '#00c851'}}>
                              {100 - uploadResult.risk_score}/100
                            </span>
                          </div>
                          <div className="risk-level-badge safe">
                            Status: <strong>SAFE</strong>
                          </div>
                        </div>
                      )}
                      
                      {uploadResult.categories && (
                        <div className="categories-analysis">
                          <h4>Safety Analysis:</h4>
                          <div className="categories-grid">
                            {Object.entries(uploadResult.categories).map(([category, data]) => (
                              <div key={category} className="category-card">
                                <div className="category-header">
                                  <span className="category-icon">{getCategoryIcon(category)}</span>
                                  <span className="category-name">{getCategoryName(category)}</span>
                                </div>
                                <div className="score-bar">
                                  <div 
                                    className="score-fill safe"
                                    style={{
                                      width: `${data.score}%`,
                                      backgroundColor: getScoreColor(data.score)
                                    }}
                                  ></div>
                                  <span className="score-text">{data.score}/100</span>
                                </div>
                                {data.reason && data.reason !== 'Not analyzed' && (
                                  <div className="category-reason">{data.reason}</div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {uploadResult.moderation_notes && (
                        <div className="moderation-notes">
                          {uploadResult.moderation_notes}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {stats && (
          <div className="stats-section">
            <h3>Moderation Statistics</h3>
            <div className="stats-grid">
              <div className="stat-card approved">
                <div className="stat-number">{stats.approved_property}</div>
                <div className="stat-label">Approved</div>
                <div className="stat-rate">{stats.property_approval_rate}</div>
              </div>
              <div className="stat-card blocked">
                <div className="stat-number">{stats.blocked_violations}</div>
                <div className="stat-label">Blocked</div>
              </div>
              <div className="stat-card review">
                <div className="stat-number">{stats.manual_review}</div>
                <div className="stat-label">Under Review</div>
              </div>
              <div className="stat-card total">
                <div className="stat-number">{stats.total_processed}</div>
                <div className="stat-label">Total Processed</div>
              </div>
            </div>
          </div>
        )}

        <div className="info-section">
          <h3>AI Safety Categories</h3>
          <div className="categories-info">
            <div className="category-info">
              <span className="category-icon">🔞</span>
              <div>
                <strong>Explicit Nudity</strong>
                <p>Nude persons, pornographic content, explicit sexual acts</p>
              </div>
            </div>
            <div className="category-info">
              <span className="category-icon">🧍</span>
              <div>
                <strong>Suggestive Content</strong>
                <p>Underwear, bikinis, revealing clothes, suggestive poses</p>
              </div>
            </div>
            <div className="category-info">
              <span className="category-icon">🤬</span>
              <div>
                <strong>Offensive Text</strong>
                <p>Profanity, explicit language, sexual references in text</p>
              </div>
            </div>
            <div className="category-info">
              <span className="category-icon">💉</span>
              <div>
                <strong>Violence</strong>
                <p>Weapons, fights, blood, injuries, threatening imagery</p>
              </div>
            </div>
            <div className="category-info">
              <span className="category-icon">💀</span>
              <div>
                <strong>Hate Content</strong>
                <p>Racist symbols, discriminatory content, hate speech</p>
              </div>
            </div>
            <div className="category-info">
              <span className="category-icon">🏘️🚗</span>
              <div>
                <strong>Property/Vehicle Relevance</strong>
                <p>Relevance to real estate or vehicle listings (houses, apartments, cars, trucks, etc.)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;