import React, { useState, useEffect } from 'react';

const API_URL = "http://127.0.0.1:8000/status";
const MAX_PODS = 15;

function App() {
  const [status, setStatus] = useState({
    pods: 0,
    cpu_usage: 0.0,
    cpu_level: 0,
    action: "Waiting...",
    reward: 0.0,
    q_values: [0, 0, 0, 0]
  });
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(API_URL);
        const data = await response.json();
        setStatus(data);
        setIsConnected(true);
      } catch (error) {
        setIsConnected(false);
      }
    };
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const getActionColor = (action) => {
    if (action.includes("Up")) return "bg-green-600 text-white";
    if (action.includes("Down")) return "bg-red-600 text-white";
    if (action.includes("Restart")) return "bg-yellow-600 text-black";
    return "bg-gray-700 text-gray-300";
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen p-10 font-sans">
      <div className="max-w-5xl mx-auto">
        <header className="flex justify-between items-center border-b border-gray-700 pb-5 mb-10">
          <h1 className="text-3xl font-bold text-blue-400">K8s RL Controller</h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">{isConnected ? "Connected" : "Offline"}</span>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
            <p className="text-gray-400 text-sm uppercase">Pods Count</p>
            <p className="text-5xl font-bold">{status.pods}</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
            <p className="text-gray-400 text-sm uppercase">CPU Usage</p>
            <p className="text-5xl font-bold text-emerald-400">{status.cpu_usage.toFixed(1)}%</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
            <p className="text-gray-400 text-sm uppercase">Last Decision</p>
            <p className={`text-2xl font-bold mt-2 py-1 px-3 rounded inline-block ${getActionColor(status.action)}`}>
              {status.action}
            </p>
          </div>
        </div>

        <div className="bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-lg">
          <h2 className="text-xl font-semibold mb-6">Brain Knowledge (Q-Values)</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {status.q_values.map((val, i) => (
              <div key={i} className="bg-gray-900 p-4 rounded border border-gray-700 text-center">
                <p className="text-xs text-gray-500 mb-1">Action {i}</p>
                <p className={`text-xl font-mono ${val >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {val.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
          <div className="mt-8 pt-6 border-t border-gray-700 flex justify-between items-center">
            <span className="text-gray-400">Current Reward:</span>
            <span className={`text-2xl font-bold ${status.reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {status.reward.toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;