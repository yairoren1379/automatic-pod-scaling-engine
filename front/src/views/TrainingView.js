import React, { useEffect, useMemo, useState } from 'react';
import LineChart from '../components/LineChart';
import { api } from '../api';
import { compact, movingAverage, shortClock } from '../format';

const PROFILES = ['constant', 'spike', 'periodic'];
const EVAL_COLUMNS = [
  ['total_reward', 'Reward', 1],
  ['severe_failures', 'Severe failures', 2],
  ['slo_violation_percent', 'SLO misses %', 1],
  ['avg_latency_ms', 'Latency ms', 0],
  ['avg_replicas', 'Replicas', 2],
  ['scale_actions', 'Scalings', 1],
  ['flaps', 'Flaps', 1],
  ['blocked_actions', 'Blocked', 1],
];

function Card({ title, subtitle, children }) {
  return (
    <section className="card">
      <h3 className="card-title">{title}</h3>
      {subtitle && <p className="card-subtitle">{subtitle}</p>}
      {children}
    </section>
  );
}

function OnlineLearning({ learning }) {
  const charts = useMemo(() => {
    const rewards = learning.map((p) => p.reward);
    const average = movingAverage(rewards, 20);
    return {
      reward: [
        { label: 'per step', color: 'var(--series-1)', faint: true, points: learning.map((p) => ({ x: p.timestamp, y: p.reward })) },
        { label: '20-step average', color: 'var(--series-1)', points: learning.map((p, i) => ({ x: p.timestamp, y: average[i] })) },
      ],
      loss: [{ label: 'Loss', color: 'var(--series-1)', points: learning.filter((p) => p.loss != null).map((p) => ({ x: p.timestamp, y: p.loss })) }],
      epsilon: [{ label: 'Epsilon', color: 'var(--series-1)', step: true, points: learning.map((p) => ({ x: p.timestamp, y: p.epsilon })) }],
    };
  }, [learning]);
  const last = learning[learning.length - 1];

  return (
    <>
      <h2 className="section-title">Online learning on the cluster</h2>
      <p className="section-intro">
        Shadow and active modes keep learning from real transitions.
        {last && ` ${last.online_steps} transitions learned, replay buffer ${last.replay_size}.`}
      </p>
      <div className="chart-grid three">
        <Card title="Reward per control step">
          <LineChart series={charts.reward} formatX={shortClock} emptyText="No completed steps yet" />
        </Card>
        <Card title="Loss (Huber)" subtitle="Gap between predicted Q and the TD target">
          <LineChart series={charts.loss} yMin={0} formatX={shortClock} emptyText="Learning starts once the replay buffer is warm" />
        </Card>
        <Card title="Epsilon" subtitle="Exploration rate (only active mode explores)">
          <LineChart series={charts.epsilon} yMin={0} formatX={shortClock} emptyText="No completed steps yet" />
        </Card>
      </div>
    </>
  );
}

function EvaluationTable({ evaluation }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Load profile</th>
            <th>Controller</th>
            {EVAL_COLUMNS.map(([key, label]) => <th key={key} className="num">{label}</th>)}
          </tr>
        </thead>
        <tbody>
          {PROFILES.filter((p) => evaluation[p]).map((profile) =>
            ['dqn', 'threshold'].map((policy) => (
              <tr key={profile + policy} className={policy === 'threshold' ? 'row-muted' : ''}>
                <td>{policy === 'dqn' ? profile : ''}</td>
                <td>{policy === 'dqn' ? 'DQN' : 'Threshold baseline'}</td>
                {EVAL_COLUMNS.map(([key, , digits]) => (
                  <td key={key} className="num">
                    {evaluation[profile][policy][key] == null ? '–' : evaluation[profile][policy][key].toFixed(digits)}
                  </td>
                ))}
              </tr>
            )),
          )}
        </tbody>
      </table>
      <p className="muted">Per-episode averages on the same test loads. Higher reward is better; everything else lower is better.</p>
    </div>
  );
}

function OfflineRuns() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState('');
  const [run, setRun] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.runs()
      .then((list) => {
        setRuns(list);
        const preferred = list.find((r) => r.name === 'experiments/lr0.0005_g0.99_s0') || list[0];
        if (preferred) setSelected(preferred.name);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api.run(selected).then(setRun).catch((e) => setError(e.message));
  }, [selected]);

  const charts = useMemo(() => {
    if (!run) return null;
    const rows = run.episodes;
    const at = (key) => rows.map((r) => ({ x: r.episode, y: typeof r[key] === 'number' ? r[key] : null }));
    const rewardAverage = movingAverage(rows.map((r) => r.total_reward), 50);
    const failureAverage = movingAverage(rows.map((r) => r.severe_failures), 50);
    const validation = rows.filter((r) => typeof r.validation_reward === 'number')
      .map((r) => ({ x: r.episode, y: r.validation_reward }));
    return {
      reward: [
        { label: 'per episode', color: 'var(--series-1)', faint: true, points: at('total_reward') },
        { label: '50-episode average', color: 'var(--series-1)', points: rows.map((r, i) => ({ x: r.episode, y: rewardAverage[i] })) },
        ...(validation.length ? [{ label: 'validation (greedy)', color: 'var(--series-2)', points: validation }] : []),
      ],
      loss: [{ label: 'Loss', color: 'var(--series-1)', points: at('avg_loss') }],
      epsilon: [{ label: 'Epsilon', color: 'var(--series-1)', points: at('epsilon') }],
      failures: [{ label: 'Severe failures', color: 'var(--series-1)', points: rows.map((r, i) => ({ x: r.episode, y: failureAverage[i] })) }],
    };
  }, [run]);

  const episodeLabel = (x) => Math.round(x).toString();

  return (
    <>
      <h2 className="section-title">Pre-training in the simulation</h2>
      <div className="filter-row">
        <label className="field-inline">
          Training run
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {runs.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} ({r.epsilon_mode} epsilon{r.lstm ? ', LSTM' : ', no LSTM'})
              </option>
            ))}
          </select>
        </label>
        {run && <span className="muted">learning rate {run.dqn.learning_rate} · gamma {run.dqn.gamma} · {run.episodes.length} episodes</span>}
      </div>
      {error && <p className="error-text">{error}</p>}
      {!runs.length && !error && <p className="muted">No training runs yet: run train_dqn.py or run_experiments.py.</p>}
      {charts && (
        <div className="chart-grid">
          <Card title="Total reward per episode" subtitle="dqn.pt keeps the network with the best validation reward">
            <LineChart series={charts.reward} formatX={episodeLabel} />
          </Card>
          <Card title="Loss (Huber)">
            <LineChart series={charts.loss} yMin={0} formatX={episodeLabel} />
          </Card>
          <Card title="Epsilon decay" subtitle="eps = max(eps_min, eps_0 / (k·t + 1))">
            <LineChart series={charts.epsilon} yMin={0} yMax={1} formatX={episodeLabel} />
          </Card>
          <Card title="Severe failures per episode" subtitle="50-episode average">
            <LineChart series={charts.failures} yMin={0} formatX={episodeLabel} />
          </Card>
        </div>
      )}
      {run && run.evaluation && (
        <Card title="Evaluation: DQN vs threshold baseline" subtitle="Greedy policy (epsilon = 0) on unseen episodes">
          <EvaluationTable evaluation={run.evaluation} />
        </Card>
      )}
    </>
  );
}

function Experiments() {
  const [data, setData] = useState(null);
  useEffect(() => {
    api.experiments().then(setData).catch(() => setData(null));
  }, []);
  if (!data || !data.summary.length) return null;
  const t = data.threshold;
  return (
    <Card title="Experiments" subtitle="Mean ± standard deviation over seeds, averaged over the three load profiles">
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Configuration</th><th className="num">Reward</th><th className="num">Severe failures</th>
              <th className="num">Replicas</th><th className="num">Flaps</th><th className="num">Seeds</th>
            </tr>
          </thead>
          <tbody>
            {data.summary.map((row) => (
              <tr key={row.config}>
                <td>{row.config}</td>
                <td className="num">{compact(row.total_reward_mean)} ± {compact(row.total_reward_std)}</td>
                <td className="num">{row.severe_failures_mean.toFixed(2)} ± {row.severe_failures_std.toFixed(2)}</td>
                <td className="num">{row.avg_replicas_mean.toFixed(2)} ± {row.avg_replicas_std.toFixed(2)}</td>
                <td className="num">{row.flaps_mean.toFixed(1)} ± {row.flaps_std.toFixed(1)}</td>
                <td className="num">{row.seeds}</td>
              </tr>
            ))}
            {t && (
              <tr className="row-muted">
                <td>Threshold baseline</td><td className="num">{compact(t.total_reward)}</td>
                <td className="num">{t.severe_failures.toFixed(2)}</td><td className="num">{t.avg_replicas.toFixed(2)}</td>
                <td className="num">{t.flaps.toFixed(1)}</td><td className="num">–</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function ForecasterAccuracy() {
  const [models, setModels] = useState(null);
  useEffect(() => {
    api.models().then(setModels).catch(() => setModels(null));
  }, []);
  const lstm = models && models.lstm;
  if (!lstm) return null;
  const names = { cpu_percent: 'CPU %', ram_percent: 'Memory %', latency_ms: 'Latency ms', qps: 'Requests/s' };
  return (
    <Card title="LSTM forecaster accuracy"
      subtitle={`Error ${lstm.horizon_steps} steps ahead on unseen data (RMSE), vs the naive forecast "the future equals now". Trained on ${lstm.source}.`}>
      <table className="data-table">
        <thead>
          <tr><th>Metric</th><th className="num">LSTM</th><th className="num">Naive</th><th className="num">Better by</th></tr>
        </thead>
        <tbody>
          {Object.entries(lstm.validation_rmse).map(([key, row]) => (
            <tr key={key}>
              <td>{names[key] || key}</td><td className="num">{row.lstm_rmse.toFixed(2)}</td>
              <td className="num">{row.naive_rmse.toFixed(2)}</td><td className="num">{row.improvement_percent.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

export default function TrainingView({ feed }) {
  return (
    <div className="view">
      <OnlineLearning learning={feed.learning} />
      <OfflineRuns />
      <div className="two-columns">
        <Experiments />
        <ForecasterAccuracy />
      </div>
    </div>
  );
}
