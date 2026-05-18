import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [status, setStatus] = useState({
    pods: 0,
    cpu_usage: 0.0,
    ram_usage: 0.0,
    cpu_bucket: 0,
    ram_bucket: 0,
    action: "Waiting...",
    reward: 0.0,
    q_values: [0, 0, 0, 0]
  });

  const [logs, setLogs] = useState([]);
  const [isCooldown, setIsCooldown] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentView, setCurrentView] = useState('dashboard');

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        setIsConnected(true);
        setIsCooldown(data.action && data.action.includes("Resting"));
      } else {
        setIsConnected(false);
      }
    } catch (error) {
      setIsConnected(false);
      setIsCooldown(false);
    }
  };

  const fetchLogs = async () => {
    try {
      const response = await fetch('http://localhost:8000/logs-data');
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || []);
      }
    } catch (error) {}
  };

  useEffect(() => {
    fetchStatus();
    fetchLogs();
    const interval = setInterval(() => {
      fetchStatus();
      fetchLogs();
    }, 2000); 
    return () => clearInterval(interval);
  }, []);

  const triggerLoad = async (type) => {
    if (!isConnected) return;
    try {
      await fetch(`http://localhost:8000/${type}`, { method: 'POST' });
      fetchStatus();
    } catch (error) {
      console.error(`Error triggering ${type}:`, error);
    }
  };

  // --- Logs View ---
  if (currentView === 'logs') {
    return (
      <div className="app-container">
        <header className="app-header">
          <div className="header-titles">
            <h1>System Terminal</h1>
          </div>
          <button className="btn btn-small btn-secondary" onClick={() => setCurrentView('dashboard')}>
            Back to Dashboard
          </button>
        </header>

        <div className={`card full-page-terminal ${!isConnected ? 'dimmed' : ''}`}>
          <div className="terminal-window full-height">
            {!isConnected ? (
              <span className="log-placeholder" style={{color: '#ef4444'}}>Disconnected from Python API...</span>
            ) : logs.length === 0 ? (
              <span className="log-placeholder">Waiting for logs...</span>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="log-line">{log}</div>
              ))
            )}
          </div>
        </div>
      </div>
    );
  }

  // --- Dashboard View ---
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-titles">
          <h1>K8s RL Autoscaler Dashboard</h1>
        </div>
        
        <div className="header-controls">
          <button className="btn btn-small btn-secondary" onClick={() => setCurrentView('logs')}>
            Open Logs
          </button>
          <div className={`status-badge ${isConnected ? 'online' : 'offline'}`}>
            <span className="dot"></span>
            {isConnected ? 'Online' : 'Disconnected'}
          </div>
        </div>
      </header>

      {isCooldown && isConnected && (
        <div className="cooldown-banner">
          System is in Cooldown period to stabilize cluster state...
        </div>
      )}

      {/* Action Buttons Row */}
      <section className={`controls-row ${!isConnected ? 'dimmed' : ''}`}>
        <button className="btn btn-action btn-red" disabled={!isConnected} onClick={() => triggerLoad('start-load')}>
          Inject High Load
        </button>
        <button className="btn btn-action btn-blue" disabled={!isConnected} onClick={() => triggerLoad('stop-load')}>
          Stop Load (Idle)
        </button>
        <button className="btn btn-action btn-purple" disabled={!isConnected} onClick={() => triggerLoad('scale-min')}>
          Force Scale Min
        </button>
        <button className="btn btn-action btn-orange" disabled={!isConnected} onClick={() => triggerLoad('scale-max')}>
          Force Scale Max
        </button>
      </section>

      {/* Metrics Grid */}
      <section className={`metrics-grid ${!isConnected ? 'dimmed' : ''}`}>
        <div className="card metric-card">
          <span className="metric-label">ACTIVE PODS</span>
          <div className="metric-value">{isConnected ? status.pods : '-'}</div>
        </div>

        <div className="card metric-card">
          <span className="metric-label">CPU LOAD</span>
          <div className="metric-value">{isConnected ? status.cpu_usage.toFixed(1) : '-'} <span className="small-percent">%</span></div>
          <div className="metric-sub">Bucket: {isConnected ? status.cpu_bucket : '-'}</div>
          <div className="progress-track"><div className="progress-fill cpu-fill" style={{ width: `${isConnected ? Math.min(status.cpu_usage, 100) : 0}%` }}></div></div>
        </div>

        <div className="card metric-card">
          <span className="metric-label">RAM LOAD</span>
          <div className="metric-value">{isConnected ? status.ram_usage.toFixed(1) : '-'} <span className="small-percent">%</span></div>
          <div className="metric-sub">Bucket: {isConnected ? status.ram_bucket : '-'}</div>
          <div className="progress-track"><div className="progress-fill ram-fill" style={{ width: `${isConnected ? Math.min(status.ram_usage, 100) : 0}%` }}></div></div>
        </div>

        <div className="card metric-card">
          <span className="metric-label">BRAIN DECISION</span>
          <div className="metric-value text-xl">{isConnected ? status.action : '-'}</div>
        </div>

        <div className="card metric-card">
          <span className="metric-label">CURRENT REWARD</span>
          <div className="metric-value">{isConnected ? status.reward.toFixed(1) : '-'}</div>
        </div>
      </section>

      {/* Q-Table Knowledge */}
      <section className={`q-table-section ${!isConnected ? 'dimmed' : ''}`}>
        <div className="card">
          <h2 className="card-title">Q-Table Knowledge (Current State)</h2>
          <div className="q-grid">
            <div className="q-box">
              <span className="q-label">Scale Up (0)</span>
              <span className="q-val">{isConnected ? status.q_values[0]?.toFixed(2) : '0.00'}</span>
            </div>
            <div className="q-box">
              <span className="q-label">Scale Down (1)</span>
              <span className="q-val">{isConnected ? status.q_values[1]?.toFixed(2) : '0.00'}</span>
            </div>
            <div className="q-box">
              <span className="q-label">No Action (2)</span>
              <span className="q-val">{isConnected ? status.q_values[2]?.toFixed(2) : '0.00'}</span>
            </div>
            <div className="q-box">
              <span className="q-label">Restart (3)</span>
              <span className="q-val">{isConnected ? status.q_values[3]?.toFixed(2) : '0.00'}</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;