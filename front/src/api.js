// REST client for the learning engine (back/python/api/server.py).

// In development (npm start on :3000) the API runs on :8000. When the built dashboard is served
// by the API itself, everything is on the same origin.
export const API_URL =
  process.env.REACT_APP_API_URL ||
  (window.location.port === '3000' ? 'http://localhost:8000' : window.location.origin);

export const WS_URL = API_URL.replace(/^http/, 'ws') + '/ws';

async function request(path, options = {}) {
  const response = await fetch(API_URL + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (e) {
      // not JSON
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  setMode: (mode) => request('/api/mode', { method: 'POST', body: JSON.stringify({ mode }) }),
  setLoad: (profile, baseQps) =>
    request('/api/load', { method: 'POST', body: JSON.stringify({ profile, base_qps: baseQps }) }),
  updateConfig: (changes) => request('/api/config', { method: 'PUT', body: JSON.stringify(changes) }),
  runs: () => request('/api/runs'),
  run: (name) => request('/api/runs/' + name),
  experiments: () => request('/api/experiments'),
  models: () => request('/api/models'),
};
