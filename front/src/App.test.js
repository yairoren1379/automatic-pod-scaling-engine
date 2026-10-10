import { act, render, screen } from '@testing-library/react';
import App from './App';

// A fake WebSocket that lets the test push server messages.
class FakeSocket {
  static instances = [];
  constructor() {
    FakeSocket.instances.push(this);
    setTimeout(() => this.onopen && this.onopen(), 0);
  }
  close() {}
  push(message) {
    this.onmessage({ data: JSON.stringify(message) });
  }
}

const metrics = { cpu_percent: 72, ram_percent: 61, latency_ms: 95, qps: 640, error_rate_percent: 0.4, replicas: 7, pod_crashed: false };

const snapshot = {
  type: 'snapshot',
  status: {
    mode: 'shadow', connected: true, metrics, decision: 'ScaleUp', q_values: [-3.2, -5.1, -4.0], epsilon: 0,
    regime: 1, forecast: { cpu_percent: 80, ram_percent: 63, latency_ms: 110, qps: 700 },
    reward: { total: 0.12, severe_failure: false, latency_improvement: 0.12, slo_violation: 0, error_increase: 0, waste: 0, blocked_action: 0 },
  },
  history: [1, 2, 3].map((i) => ({ timestamp: 1000 + 30 * i, ...metrics, forecast_qps: 650, executed_action: 'Hold', mode: 'shadow' })),
  audit: [{ timestamp: 1090, decision_id: 3, mode: 'shadow', dqn_action: 'ScaleUp', executed_action: 'ScaleUp', source: 'baseline', blocked: true, block_reason: 'max_replicas', replicas_before: 7, replicas_after: 7 }],
  learning: [],
  config: {
    guardrails: { min_replicas: 1, max_replicas: 15 }, reward: { latency_slo_ms: 200 }, predictor: { horizon_steps: 3 },
    controller: { mode: 'shadow', loop_delay_seconds: 30 }, load_profiles: { default_base_qps: 400 },
  },
  load: { profile: 'off' },
};

beforeEach(() => {
  FakeSocket.instances = [];
  global.WebSocket = FakeSocket;
  global.fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve([]) }));
});

test('shows live metrics, the agent decision and guardrail blocks from the WebSocket', async () => {
  render(<App />);
  expect(screen.getByText('Autonomous Pod Scaling')).toBeInTheDocument();

  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
    FakeSocket.instances[0].push(snapshot);
  });

  expect(screen.getByText('Mode: shadow')).toBeInTheDocument();
  expect(screen.getByText('Engine connected')).toBeInTheDocument();
  expect(screen.getByText('▲ High load')).toBeInTheDocument();
  expect(screen.getAllByText('ScaleUp').length).toBeGreaterThan(0);
  expect(screen.getAllByText(/max_replicas/).length).toBeGreaterThan(0);
});
