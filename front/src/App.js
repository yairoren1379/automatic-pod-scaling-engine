import React, { useEffect, useState } from 'react';
import './App.css';
import { API_URL } from './api';
import useLiveFeed from './useLiveFeed';
import LiveView from './views/LiveView';
import TrainingView from './views/TrainingView';
import ControlView from './views/ControlView';

const TABS = [
  { id: 'live', label: 'Live dashboard' },
  { id: 'training', label: 'Training analytics' },
  { id: 'control', label: 'Simulation & control' },
];

const THEMES = ['system', 'light', 'dark'];

function readStored(key, fallback) {
  try {
    return window.localStorage.getItem(key) || fallback;
  } catch (e) {
    return fallback;
  }
}

function store(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (e) {
    // storage unavailable (private window): the choice just isn't remembered
  }
}

export default function App() {
  const feed = useLiveFeed();
  // A link like /#training opens that tab; otherwise the last tab used.
  const [tab, setTab] = useState(() => {
    const fromUrl = window.location.hash.slice(1);
    return TABS.some((t) => t.id === fromUrl) ? fromUrl : readStored('tab', 'live');
  });
  const [theme, setTheme] = useState(() => readStored('theme', 'system'));

  // Follow the URL hash (links and the browser's back/forward buttons).
  useEffect(() => {
    const onHashChange = () => {
      const fromUrl = window.location.hash.slice(1);
      if (TABS.some((t) => t.id === fromUrl)) setTab(fromUrl);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (theme === 'system') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    store('theme', theme);
  }, [theme]);

  const selectTab = (id) => {
    setTab(id);
    store('tab', id);
    window.history.replaceState(null, '', '#' + id);
  };

  const status = feed.status;
  const controllerOnline = status && status.connected;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Autonomous Pod Scaling</h1>
          <p className="app-subtitle">Deep RL autoscaler · DQN + LSTM · Kubernetes</p>
        </div>
        <div className="header-status">
          {status && <span className="mode-pill">Mode: {status.mode}</span>}
          <span className="connection">
            <span className={feed.connected ? 'dot dot-good' : 'dot dot-critical'} aria-hidden="true" />
            {feed.connected ? 'Engine connected' : 'Engine offline'}
          </span>
          <span className="connection">
            <span className={controllerOnline ? 'dot dot-good' : 'dot dot-warning'} aria-hidden="true" />
            {controllerOnline ? 'Controller reporting' : 'No controller'}
          </span>
          <button className="chip" onClick={() => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length])}>
            Theme: {theme}
          </button>
        </div>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'tab tab-active' : 'tab'}
            onClick={() => selectTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {!feed.connected && (
        <div className="notice notice-critical" role="alert">
          ✕ Can't reach the learning engine at {API_URL}. Start it with <code>python -m api.server</code> in back/python.
        </div>
      )}
      {feed.connected && !controllerOnline && (
        <div className="notice" role="status">
          ! The engine is up but no controller is reporting. Check the Go controller in the cluster
          (<code>kubectl logs deployment/autoscaler-controller</code>) or start the simulated cluster
          (<code>python simulate_controller.py</code>).
        </div>
      )}

      {feed.status && tab === 'live' && <LiveView feed={feed} />}
      {tab === 'training' && <TrainingView feed={feed} />}
      {feed.config && tab === 'control' && <ControlView feed={feed} />}
    </div>
  );
}
