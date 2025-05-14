// frontend/src/App.js
import React, { useState } from 'react';
import './App.css';

function App() {
  const [logInput, setLogInput] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError('');
    setPrediction(null);

    if (!logInput.trim()) {
      setError('Log input cannot be empty.');
      setIsLoading(false);
      return;
    }

    let requestBody;
    try {
      // Try to parse as JSON (for structured input)
      const jsonData = JSON.parse(logInput);
      requestBody = { log_data: jsonData };
    } catch (e) {
      // If not JSON, treat as raw string
      requestBody = { log_entry: logInput };
    }

    console.log("Sending to backend:", JSON.stringify(requestBody));

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message || 'Failed to get prediction. Check console for details.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🛡️ AI Cybersecurity Threat Detector</h1>
        <form onSubmit={handleSubmit} className="log-form">
          <textarea
            value={logInput}
            onChange={(e) => setLogInput(e.target.value)}
            placeholder="Enter log data here as a single string (e.g., 'srcip:1.2.3.4 ...') or as JSON ({ \"srcip\": \"1.2.3.4\", ... })"
            rows="10"
            cols="80"
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Detecting...' : 'Detect Threat'}
          </button>
        </form>

        {error && <p className="error-message">Error: {error}</p>}

        {prediction && (
          <div className={`prediction-result ${prediction.toLowerCase()}`}>
            <h2>Prediction: <span className={prediction.toLowerCase()}>{prediction}</span></h2>
          </div>
        )}

        <div className="example-logs">
          <p><strong>Example Malicious-looking Log (string format):</strong></p>
          <pre><code>srcip:175.45.176.0 sport:11873 dstip:149.171.126.8 dport:53 proto:udp state:CON service:dns sbytes:158 dbytes:234 sttl:31 dttl:29 sloss:0 dloss:0 Sload:6396.774414 Dload:9252.198242 Spkts:2 Dpkts:2</code></pre>
          <p><strong>Example Normal-looking Log (JSON format for structured input):</strong></p>
          <pre><code>{'{'}"srcip": "192.168.1.10", "sport": 54321, "dstip": "8.8.8.8", "dport": 53, "proto": "udp", "state": "INT", "service": "dns", "sbytes": 74, "dbytes": 130, "sttl": 64, "dttl": 58, "sloss": 0, "dloss": 0, "Sload": 12000, "Dload": 18000, "Spkts": 1, "Dpkts": 1{'}'}</code></pre>
          <p><em>Note: These are simplified examples. The model's accuracy depends on the training data and feature engineering. Use features that were part of the training.</em></p>
        </div>
      </header>
    </div>
  );
}

export default App;