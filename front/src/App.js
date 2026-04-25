import React, { useState, useEffect, useRef } from 'react';

const API_URL = "http://127.0.0.1:8000";
const MAX_PODS = 15;

function LogsPage() {
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const response = await fetch(`${API_URL}/logs-data`);
        const data = await response.json();
        setLogs(data.logs);
      } catch (error) { console.error("Error fetching logs"); }
    };
    const intervalId = setInterval(fetchLogs, 1000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="bg-black text-emerald-400 font-mono min-h-screen p-8">
      <div className="flex justify-between items-center mb-6 border-b border-emerald-900 pb-4">
        <h1 className="text-2xl font-bold"><i className="fa-solid fa-terminal mr-3 animate-pulse"></i>LIVE BRAIN LOGS</h1>
        <button
          onClick={() => window.location.href = '/'}
          className="bg-gray-800 hover:bg-gray-700 text-white py-2 px-4 rounded transition-colors">
          <i className="fa-solid fa-arrow-left mr-2"></i> Back to Dashboard
        </button>
      </div>
      <div className="whitespace-pre-wrap text-sm leading-relaxed">
        {logs.length === 0 ? "Waiting for the brain to learn..." : logs.join('\n\n')}
        <div ref={logEndRef}></div>
      </div>
    </div>
  );
}

function Dashboard() {
  const [status, setStatus] = useState({
    pods: 0, cpu_usage: 0.0, cpu_level: 0, action: "Waiting...", reward: 0.0, q_values: [0, 0, 0, 0]
  });
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_URL}/status`);
        if (!response.ok) throw new Error("Network response was not ok");
        const data = await response.json();
        setStatus(data);
        setIsConnected(true);
      } catch (error) { setIsConnected(false); }
    };
    const intervalId = setInterval(fetchData, 1000);
    return () => clearInterval(intervalId);
  }, []);

  const getActionColor = (action) => {
    if (action === "ScaleUp") return "bg-green-500 text-white";
    if (action === "ScaleDown") return "bg-red-500 text-white";
    if (action === "Restart") return "bg-yellow-500 text-white";
    return "bg-gray-700 text-gray-300";
  };

  const getTextColor = (val) => {
    if (val > 0) return "text-green-400";
    if (val < 0) return "text-red-400";
    return "text-white";
  };

  const handleControlAction = async (endpoint) => {
    try { await fetch(`${API_URL}/${endpoint}`, { method: 'POST' }); }
    catch (error) { console.error(`Error triggering ${endpoint}:`, error); }
  };

  return (
    <div className="bg-gray-900 text-gray-100 font-sans min-h-screen p-8">
      <div className="max-w-6xl mx-auto">

        { }
        <div className="flex items-center justify-between mb-8 border-b border-gray-700 pb-4 gap-4">
          <div className="flex items-center space-x-4">
            <i className="fa-solid fa-brain text-4xl text-blue-500"></i>
            <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              K8s RL Autoscaler Dashboard
            </h1>
          </div>
          <div className="flex items-center gap-4">
            { }
            <button
              onClick={() => window.open('/logs', '_blank')}
              className="bg-gray-700 hover:bg-gray-600 text-white py-2 px-5 rounded-lg text-sm flex items-center gap-2 cursor-pointer transition-colors border border-gray-600">
              <i className="fa-solid fa-up-right-from-square"></i> Open Logs Page
            </button>
            <div className={`w-3 h-3 rounded-full animate-pulse ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-400">{isConnected ? 'Live Connected' : 'Disconnected'}</span>
          </div>
        </div>

        { }
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 bg-gray-800 p-4 rounded-xl border border-gray-700">
          <button onClick={() => handleControlAction('start-load')} className="bg-red-600 hover:bg-red-500 text-white font-bold py-3 px-4 rounded-lg shadow-[0_0_15px_rgba(220,38,38,0.4)] transition-all flex justify-center items-center gap-2 cursor-pointer">
            <i className="fa-solid fa-fire"></i> High Load
          </button>
          <button onClick={() => handleControlAction('stop-load')} className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded-lg shadow-[0_0_15px_rgba(37,99,235,0.4)] transition-all flex justify-center items-center gap-2 cursor-pointer">
            <i className="fa-solid fa-snowflake"></i> Stop Load
          </button>
          <button onClick={() => handleControlAction('scale-min')} className="bg-purple-600 hover:bg-purple-500 text-white font-bold py-3 px-4 rounded-lg shadow-[0_0_15px_rgba(147,51,234,0.4)] transition-all flex justify-center items-center gap-2 cursor-pointer">
            <i className="fa-solid fa-skull"></i> Kill All (0)
          </button>
          <button onClick={() => handleControlAction('scale-max')} className="bg-orange-600 hover:bg-orange-500 text-white font-bold py-3 px-4 rounded-lg shadow-[0_0_15px_rgba(234,88,12,0.4)] transition-all flex justify-center items-center gap-2 cursor-pointer">
            <i className="fa-solid fa-rocket"></i> Max Pods (15)
          </button>
        </div>

        { }
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
            <div className="text-gray-400 text-sm font-semibold uppercase mb-2"><i className="fa-solid fa-server mr-2"></i> Active Pods</div>
            <div className="text-5xl font-black text-white">{status.pods}</div>
            <div className="w-full bg-gray-700 h-2 mt-4 rounded overflow-hidden">
              <div className="bg-blue-500 h-2 rounded transition-all duration-300" style={{ width: `${(status.pods / MAX_PODS) * 100}%` }}></div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
            <div className="text-gray-400 text-sm font-semibold uppercase mb-2"><i className="fa-solid fa-microchip mr-2"></i> Real CPU Load</div>
            <div className="flex items-end space-x-2">
              <div className="text-5xl font-black text-white">{status.cpu_usage.toFixed(1)}</div>
              <div className="text-xl text-gray-400 mb-1">%</div>
            </div>
            <div className="text-xs text-gray-500 mt-2">Level: {status.cpu_level} ({status.cpu_level === 2 ? 'High' : status.cpu_level === 1 ? 'Medium' : 'Low'})</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-[0_0_15px_rgba(59,130,246,0.3)] flex flex-col justify-center items-center text-center">
            <div className="text-gray-400 text-sm font-semibold uppercase mb-2">Brain Decision</div>
            <div className={`text-3xl font-bold px-4 py-2 rounded-lg transition-colors duration-300 ${getActionColor(status.action)}`}>{status.action}</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-[0_0_15px_rgba(59,130,246,0.3)] flex flex-col justify-center items-center text-center">
            <div className="text-gray-400 text-sm font-semibold uppercase mb-2">Last Reward</div>
            <div className={`text-5xl font-black ${getTextColor(status.reward)}`}>{status.reward > 0 ? '+' : ''}{status.reward.toFixed(1)}</div>
          </div>
        </div>

        { }
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
          <h2 className="text-xl font-bold mb-6 text-gray-200"><i className="fa-solid fa-table-list mr-2 text-emerald-400"></i> Q-Table Knowledge (Current State)</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {['Scale Up (0)', 'Scale Down (1)', 'No Action (2)', 'Restart (3)'].map((actionName, index) => (
              <div key={index} className="bg-gray-900 rounded p-4 border border-gray-700">
                <div className="text-sm text-gray-400 mb-1">{actionName}</div>
                <div className={`text-2xl font-bold ${getTextColor(status.q_values[index])}`}>{status.q_values[index].toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  if (window.location.pathname === '/logs') {
    return <LogsPage />;
  }
  return <Dashboard />;
}