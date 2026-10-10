import React, { useState } from 'react';
import { api } from '../api';
import { clock } from '../format';

const MODES = [
  {
    id: 'baseline',
    title: 'Baseline',
    text: 'The threshold controller scales the cluster. The DQN only watches; its proposals are recorded for comparison.',
  },
  {
    id: 'shadow',
    title: 'Shadow',
    text: 'The threshold controller still scales the cluster, and the DQN learns from what happens (off-policy). Nothing it decides is applied.',
  },
  {
    id: 'active',
    title: 'Active',
    text: 'The DQN scales the cluster through the guardrails, and keeps fine-tuning with a low learning rate and epsilon.',
  },
];

const PROFILES = [
  { id: 'constant', title: 'Constant', text: 'Steady traffic at the base rate.' },
  { id: 'spike', title: 'Sudden spike', text: 'Base rate, then a jump for a while, then back.' },
  { id: 'periodic', title: 'Periodic', text: 'A smooth wave around the base rate, like daily traffic.' },
  { id: 'off', title: 'Stop', text: 'Back to idle background traffic.' },
];

// The settings that can be changed live from the dashboard (all stored in ZooKeeper).
const FIELDS = [
  ['guardrails', 'min_replicas', 'Minimum replicas', 1],
  ['guardrails', 'max_replicas', 'Maximum replicas', 1],
  ['controller', 'loop_delay_seconds', 'Control loop delay (s)', 1],
  ['baseline', 'scale_up_cpu_percent', 'Baseline: scale up above CPU %', 1],
  ['baseline', 'scale_down_cpu_percent', 'Baseline: scale down below CPU %', 1],
  ['reward', 'latency_slo_ms', 'Latency SLO (ms)', 1],
  ['dqn', 'gamma', 'Discount gamma', 0.01],
  ['online', 'shadow_learning_rate', 'Learning rate (shadow)', 0.0001],
  ['online', 'active_learning_rate', 'Learning rate (active)', 0.0001],
  ['online', 'active_epsilon', 'Epsilon (active)', 0.01],
];

export default function ControlView({ feed }) {
  const { config, load } = feed;
  const [baseQps, setBaseQps] = useState(null); // null until the slider is moved
  const [draft, setDraft] = useState({});
  const [message, setMessage] = useState('');

  if (!config) return null;
  const mode = config.controller.mode;
  // Until the slider is moved, it shows the running profile's rate (or the configured default).
  const runningQps = load && load.profile !== 'off' ? load.base_qps : null;
  const rate = baseQps ?? runningQps ?? config.load_profiles.default_base_qps;

  const run = async (action, success) => {
    try {
      await action();
      setMessage(success);
    } catch (e) {
      setMessage('Error: ' + e.message);
    }
  };

  const saveConfig = () => {
    const changes = {};
    Object.entries(draft).forEach(([path, value]) => {
      const [section, key] = path.split('.');
      changes[section] = { ...(changes[section] || {}), [key]: Number(value) };
    });
    run(() => api.updateConfig(changes), 'Configuration saved to ZooKeeper and applied.').then(() => setDraft({}));
  };

  return (
    <div className="view">
      {message && <div className="notice" role="status">{message}</div>}

      <h2 className="section-title">Controller mode</h2>
      <div className="option-grid">
        {MODES.map((m) => (
          <button key={m.id} className={m.id === mode ? 'option option-active' : 'option'} aria-pressed={m.id === mode}
            onClick={() => run(() => api.setMode(m.id), `Mode set to ${m.title}.`)}>
            <span className="option-title">{m.id === mode ? '✓ ' : ''}{m.title}</span>
            <span className="option-text">{m.text}</span>
          </button>
        ))}
      </div>

      <h2 className="section-title">Simulation panel: load profile</h2>
      <p className="section-intro">
        Written to ZooKeeper; the load generator (or the simulated cluster) follows it immediately.
        {load && ` Current: ${load.profile}` + (load.base_qps ? ` at ${load.base_qps} requests/s` : '') +
          (load.started_at ? `, since ${clock(load.started_at)}.` : '.')}
      </p>
      <div className="filter-row">
        <label className="field-inline">
          Base rate: <strong>{rate}</strong> requests/s
          <input type="range" min="50" max="1200" step="50" value={rate} onChange={(e) => setBaseQps(Number(e.target.value))} />
        </label>
      </div>
      <div className="option-grid four">
        {PROFILES.map((p) => (
          <button key={p.id} className={load && load.profile === p.id ? 'option option-active' : 'option'}
            onClick={() => run(() => api.setLoad(p.id, rate), `Load profile: ${p.title}.`)}>
            <span className="option-title">{p.title}</span>
            <span className="option-text">{p.text}</span>
          </button>
        ))}
      </div>

      <h2 className="section-title">Live configuration</h2>
      <p className="section-intro">Stored in ZooKeeper. The controller and the learning engine watch it and apply changes without a restart.</p>
      <section className="card">
        <div className="form-grid">
          {FIELDS.map(([section, key, label, step]) => {
            const path = `${section}.${key}`;
            return (
              <label key={path} className="field">
                <span>{label}</span>
                <input type="number" step={step} value={draft[path] ?? config[section][key]}
                  onChange={(e) => setDraft({ ...draft, [path]: e.target.value })} />
              </label>
            );
          })}
        </div>
        <div className="form-actions">
          <button className="button-primary" disabled={!Object.keys(draft).length} onClick={saveConfig}>Save changes</button>
          <button className="button-secondary" disabled={!Object.keys(draft).length} onClick={() => setDraft({})}>Discard</button>
        </div>
      </section>
    </div>
  );
}
