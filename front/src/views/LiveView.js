import React, { useMemo, useState } from 'react';
import LineChart from '../components/LineChart';
import { clock, compact, shortClock, signed } from '../format';

const WINDOWS = [
  { label: 'Last 60 steps', size: 60 },
  { label: 'Last 240 steps', size: 240 },
  { label: 'All (720)', size: 720 },
];

const REGIMES = { '-1': '▼ Low load', 0: '● Normal', 1: '▲ High load' };

const REWARD_TERMS = [
  ['latency_improvement', 'Latency improvement'],
  ['slo_violation', 'Latency above SLO'],
  ['error_increase', 'New errors'],
  ['waste', 'Idle replicas (waste)'],
  ['blocked_action', 'Guardrail block'],
];

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function StatTile({ label, value, unit, detail }) {
  return (
    <div className="stat-tile">
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </div>
      {detail && <div className="stat-detail">{detail}</div>}
    </div>
  );
}

function ChartCard({ title, subtitle, children }) {
  return (
    <section className="card">
      <h3 className="card-title">{title}</h3>
      {subtitle && <p className="card-subtitle">{subtitle}</p>}
      {children}
    </section>
  );
}

function DecisionPanel({ status, lastAudit }) {
  const q = status.q_values || [];
  const names = ['ScaleUp', 'ScaleDown', 'Hold'];
  const worst = Math.min(...q);
  const best = Math.max(...q);
  const reward = status.reward;
  return (
    <section className="card decision-card">
      <h3 className="card-title">Agent decision</h3>
      <div className="decision-row">
        <div>
          <div className="stat-label">DQN proposes</div>
          <div className="decision-action">{status.decision || '–'}</div>
          <div className="stat-detail">ε = {status.epsilon ?? 0} · mode {status.mode}</div>
        </div>
        {lastAudit && (
          <div>
            <div className="stat-label">Controller executed</div>
            <div className="decision-action">{lastAudit.executed_action}</div>
            <div className={lastAudit.blocked ? 'badge badge-critical' : 'badge badge-good'}>
              {lastAudit.blocked ? `✕ Blocked: ${lastAudit.block_reason}` : `✓ ${lastAudit.source === 'dqn' ? 'DQN decision' : 'Baseline decision'}`}
            </div>
          </div>
        )}
      </div>

      <h4 className="mini-title">Q-values (expected future reward)</h4>
      <div className="q-list">
        {names.map((name, i) => {
          const value = q[i];
          const share = best === worst ? 1 : (value - worst) / (best - worst);
          return (
            <div key={name} className="q-row">
              <span className="q-name">{name}</span>
              <span className="q-track">
                <span className={value === best ? 'q-bar q-best' : 'q-bar'} style={{ width: `${4 + share * 96}%` }} />
              </span>
              <span className="q-value">{value == null ? '–' : value.toFixed(2)}</span>
            </div>
          );
        })}
      </div>

      <h4 className="mini-title">Reward of the previous action</h4>
      {reward ? (
        <table className="data-table compact-table">
          <tbody>
            {reward.severe_failure ? (
              <tr>
                <td>Severe failure (crash, extreme latency or error spike)</td>
                <td className="num">{signed(reward.total, 1)}</td>
              </tr>
            ) : (
              REWARD_TERMS.map(([key, label]) => (
                <tr key={key}>
                  <td>{label}</td>
                  <td className="num">{signed(reward[key])}</td>
                </tr>
              ))
            )}
            <tr className="total-row">
              <td>Total</td>
              <td className="num">{signed(reward.total)}</td>
            </tr>
          </tbody>
        </table>
      ) : (
        <p className="muted">Available after the first completed step.</p>
      )}
    </section>
  );
}

export default function LiveView({ feed }) {
  const [windowSize, setWindowSize] = useState(240);
  const [showTable, setShowTable] = useState(false);
  const { status, history, audit, config } = feed;
  const points = useMemo(() => history.slice(-windowSize), [history, windowSize]);

  const charts = useMemo(() => {
    const at = (key) => points.map((p) => ({ x: p.timestamp, y: p[key] }));
    // The LSTM forecast made at time t is for t + horizon steps: draw it where it applies.
    const horizon = config ? config.predictor.horizon_steps : 3;
    const stepSeconds = median(points.slice(1).map((p, i) => p.timestamp - points[i].timestamp));
    const forecastQps = points.map((p) => ({ x: p.timestamp + horizon * stepSeconds, y: p.forecast_qps }));
    return {
      resources: [
        { label: 'CPU', color: 'var(--series-1)', points: at('cpu_percent') },
        { label: 'Memory', color: 'var(--series-2)', points: at('ram_percent') },
      ],
      latency: [{ label: 'Latency', color: 'var(--series-1)', points: at('latency_ms') }],
      qps: [
        { label: 'Actual', color: 'var(--series-1)', points: at('qps') },
        { label: `LSTM forecast (${horizon} steps ahead)`, color: 'var(--series-2)', dashed: true, points: forecastQps },
      ],
      replicas: [{ label: 'Replicas', color: 'var(--series-1)', step: true, points: at('replicas') }],
      errors: [{ label: 'Error rate', color: 'var(--series-1)', points: at('error_rate_percent') }],
    };
  }, [points, config]);

  if (!status) return null;
  const m = status.metrics;
  const slo = config ? config.reward.latency_slo_ms : 200;
  const maxReplicas = config ? config.guardrails.max_replicas : 15;

  return (
    <div className="view">
      <div className="stat-row">
        <StatTile label="Replicas" value={m ? m.replicas : '–'} detail={config && `limits ${config.guardrails.min_replicas}–${maxReplicas}`} />
        <StatTile label="CPU" value={m ? m.cpu_percent.toFixed(0) : '–'} unit="%" />
        <StatTile label="Memory" value={m ? m.ram_percent.toFixed(0) : '–'} unit="%" />
        <StatTile label="Latency" value={m ? compact(m.latency_ms, 0) : '–'} unit="ms" detail={`SLO ${slo} ms`} />
        <StatTile label="Requests" value={m ? compact(m.qps, 0) : '–'} unit="/s"
          detail={status.forecast && `forecast ${compact(status.forecast.qps, 0)}/s`} />
        <StatTile label="Errors" value={m ? m.error_rate_percent.toFixed(1) : '–'} unit="%" />
        <StatTile label="Load regime" value={REGIMES[status.regime] || '–'} detail="hysteresis on smoothed CPU" />
      </div>

      <div className="filter-row">
        {WINDOWS.map((w) => (
          <button key={w.size} className={w.size === windowSize ? 'chip chip-active' : 'chip'} onClick={() => setWindowSize(w.size)}>
            {w.label}
          </button>
        ))}
        <button className="chip" onClick={() => setShowTable(!showTable)}>
          {showTable ? 'Hide data table' : 'Show data table'}
        </button>
      </div>

      <div className="live-layout">
        <div className="chart-grid">
          <ChartCard title="CPU and memory" subtitle="Average of ready pods, % of the pod limit">
            <LineChart series={charts.resources} yMin={0} yMax={100} unit="%" formatX={shortClock} />
          </ChartCard>
          <ChartCard title="Latency" subtitle="Average per request, ms">
            <LineChart series={charts.latency} yMin={0} unit=" ms" formatX={shortClock}
              reference={{ value: slo, label: `SLO ${slo} ms` }} />
          </ChartCard>
          <ChartCard title="Requests per second" subtitle="Actual traffic and the LSTM forecast">
            <LineChart series={charts.qps} yMin={0} formatX={shortClock} />
          </ChartCard>
          <ChartCard title="Replicas" subtitle="Desired replica count of the Deployment">
            <LineChart series={charts.replicas} yMin={0} yMax={maxReplicas} formatX={shortClock}
              formatY={(v) => v.toFixed(0)} reference={{ value: maxReplicas, label: 'max_replicas' }} />
          </ChartCard>
          <ChartCard title="Error rate" subtitle="Failed requests, %">
            <LineChart series={charts.errors} yMin={0} unit="%" formatX={shortClock} />
          </ChartCard>
        </div>
        <DecisionPanel status={status} lastAudit={audit[audit.length - 1]} />
      </div>

      {showTable && (
        <section className="card">
          <h3 className="card-title">Recent data</h3>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th><th>Mode</th><th className="num">CPU %</th><th className="num">Memory %</th>
                  <th className="num">Latency ms</th><th className="num">QPS</th><th className="num">QPS forecast</th>
                  <th className="num">Errors %</th><th className="num">Replicas</th><th>Executed</th><th className="num">Reward</th>
                </tr>
              </thead>
              <tbody>
                {points.slice(-30).reverse().map((p) => (
                  <tr key={p.timestamp}>
                    <td>{clock(p.timestamp)}</td><td>{p.mode}</td>
                    <td className="num">{p.cpu_percent?.toFixed(1)}</td><td className="num">{p.ram_percent?.toFixed(1)}</td>
                    <td className="num">{p.latency_ms?.toFixed(0)}</td><td className="num">{p.qps?.toFixed(0)}</td>
                    <td className="num">{p.forecast_qps?.toFixed(0)}</td><td className="num">{p.error_rate_percent?.toFixed(2)}</td>
                    <td className="num">{p.replicas}</td><td>{p.executed_action}</td>
                    <td className="num">{p.reward == null ? '–' : signed(p.reward)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card">
        <h3 className="card-title">Decisions vs guardrails</h3>
        <p className="card-subtitle">Every DQN proposal, what the controller executed, and every guardrail intervention (safety audit log).</p>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th><th>Mode</th><th>DQN proposed</th><th>Executed</th><th>Decided by</th>
                <th>Guardrails</th><th className="num">Replicas</th><th className="num">CPU %</th><th className="num">QPS</th>
              </tr>
            </thead>
            <tbody>
              {audit.slice(-15).reverse().map((a) => (
                <tr key={a.decision_id + '-' + a.timestamp}>
                  <td>{clock(a.timestamp)}</td><td>{a.mode}</td><td>{a.dqn_action}</td><td>{a.executed_action}</td>
                  <td>{a.source === 'dqn' ? 'DQN' : 'Baseline'}</td>
                  <td>
                    {a.blocked ? (
                      <span className="badge badge-critical">✕ {a.block_reason}</span>
                    ) : (
                      <span className="badge badge-good">✓ allowed</span>
                    )}
                  </td>
                  <td className="num">{a.replicas_before} → {a.replicas_after}</td>
                  <td className="num">{a.cpu_percent?.toFixed(0)}</td><td className="num">{a.qps?.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
