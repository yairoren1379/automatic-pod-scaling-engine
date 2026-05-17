import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [status, setStatus] = useState({
    pods: 0,
    cpu_usage: 0.0,
    ram_usage: 0.0,
    cpu_bucket: 0,
    ram_bucket: 0,
    action: "Loading...",
    reward: 0.0,
    q_values: [0, 0, 0, 0]
  });

  const [logs, setLogs] = useState([]);
  const [isCooldown, setIsCooldown] = useState(false);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/status');
      const data = await response.json();
      setStatus(data);
      
      if (data.action && data.action.includes("Resting")) {
        setIsCooldown(true);
      } else {
        setIsCooldown(false);
      }
    } catch (error) {
      console.error("Error fetching cluster status:", error);
    }
  };

  const fetchLogs = async () => {
    try {
      const response = await fetch('http://localhost:8000/logs-data');
      const data = await response.json();
      setLogs(data.logs || []);
    } catch (error) {
      console.error("Error fetching brain logs:", error);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchLogs();
    const interval = setInterval(() => {
      fetchStatus();
      fetchLogs();
    }, 2000); // Polls every 2 seconds for real-time update
    return () => clearInterval(interval);
  }, []);

  const triggerLoad = async (type) => {
    try {
      await fetch(`http://localhost:8000/${type}`, { method: 'POST' });
      fetchStatus();
    } catch (error) {
      console.error(`Error triggering ${type}:`, error);
    }
  };

  return (
    <div className="dashboard-container" dir="rtl">
      <header className="dashboard-header">
        <h1>מנוע סקיילינג אוטונומי מבוסס AI & Reinforcement Learning</h1>
        <p>בקרת אשכול קוברנטיס בזמן אמת</p>
      </header>

      {isCooldown && (
        <div className="cooldown-banner">
          ⚠️ המערכת נמצאת בתקופת צינון (Cooldown) של 30 שניות לצורך התייצבות האשכול...
        </div>
      )}

      <div className="metrics-grid">
        {/* Pods Card */}
        <div className="metric-card">
          <h3>כמות פודים פעילה</h3>
          <div className="metric-value " style={{ color: '#38bdf8' }}>{status.pods}</div>
          <p className="metric-sub">רפליקות ב-Deployment</p>
        </div>

        {/* CPU Card */}
        <div className="metric-card">
          <h3>עומס מעבד (CPU)</h3>
          <div className="metric-value" style={{ color: '#ef4444' }}>{status.cpu_usage.toFixed(1)}%</div>
          <div className="progress-bg">
            <div className="progress-bar cpu-bar" style={{ width: `${status.cpu_usage}%` }}></div>
          </div>
          <p className="metric-sub">אינדקס דלי בטבלה: {status.cpu_bucket}</p>
        </div>

        {/* RAM Card */}
        <div className="metric-card">
          <h3>עומס זיכרון (RAM)</h3>
          <div className="metric-value" style={{ color: '#a855f7' }}>{status.ram_usage.toFixed(1)}%</div>
          <div className="progress-bg">
            <div className="progress-bar ram-bar" style={{ width: `${status.ram_usage}%` }}></div>
          </div>
          <p className="metric-sub">אינדקס דלי בטבלה: {status.ram_bucket}</p>
        </div>

        {/* Action Card */}
        <div className="metric-card">
          <h3>החלטת ה-AI הנוכחית</h3>
          <div className="metric-value action-highlight">{status.action}</div>
          <p className="metric-sub">תגמול (Reward): <span style={{ direction: 'ltr', display: 'inline-block' }}>{status.reward.toFixed(1)}</span></p>
        </div>
      </div>

      <div className="lower-sections">
        {/* Simulation Controls */}
        <div className="panel-card simulation-panel">
          <h2>מערכת הזרקת עומסים (Simulation)</h2>
          <div className="button-group">
            <button className="btn btn-danger" onClick={() => triggerLoad('start-load')}>הזרק עומס מעבד וזיכרון</button>
            <button className="btn btn-success" onClick={() => triggerLoad('stop-load')}>עצור עומס והחזר לשפל</button>
            <button className="btn btn-warning" onClick={() => triggerLoad('scale-max')}>Scale Up מאולץ (15 פודים)</button>
            <button className="btn btn-secondary" onClick={() => triggerLoad('scale-min')}>Scale Down מאולץ (0 פודים)</button>
          </div>

          <div className="q-values-section">
            <h3>ערכי ה-Q הנוכחיים של המצב (Brain Knowledge)</h3>
            <div className="q-grid">
              <div className="q-item"><span>Scale Up (0):</span> <strong>{status.q_values[0]?.toFixed(2)}</strong></div>
              <div className="q-item"><span>Scale Down (1):</span> <strong>{status.q_values[1]?.toFixed(2)}</strong></div>
              <div className="q-item"><span>No Action (2):</span> <strong>{status.q_values[2]?.toFixed(2)}</strong></div>
              <div className="q-item"><span>Restart (3):</span> <strong>{status.q_values[3]?.toFixed(2)}</strong></div>
            </div>
          </div>
        </div>

        {/* Brain Logs Console */}
        <div className="panel-card logs-panel">
          <h2>יומן החלטות ותהליך מחשבה (Live AI Logs)</h2>
          <div className="console-box">
            {logs.length === 0 ? (
              <p className="empty-logs">ממתין להזרמת לוגים מהבקר האוטונומי...</p>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="log-line">{log}</div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;