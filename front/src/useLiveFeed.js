import { useEffect, useState } from 'react';
import { WS_URL } from './api';

const MAX_POINTS = 720;
const MAX_AUDIT = 100;

const initialState = {
  connected: false, // WebSocket to the API
  status: null, // latest engine snapshot (mode, metrics, decision, q-values, reward...)
  history: [], // one point per control loop
  audit: [], // decisions vs guardrail interventions
  learning: [], // online learning curve
  config: null,
  load: null,
};

function append(list, item, max) {
  const next = list.length >= max ? list.slice(list.length - max + 1) : list.slice();
  next.push(item);
  return next;
}

// Applies one message from the server to the dashboard state.
function reduce(state, message) {
  switch (message.type) {
    case 'snapshot':
      return {
        ...state,
        status: message.status,
        history: message.history,
        audit: message.audit,
        learning: message.learning,
        config: message.config,
        load: message.load,
      };
    case 'step': {
      const last = state.learning[state.learning.length - 1];
      const isNew = message.learning && (!last || last.timestamp !== message.learning.timestamp);
      return {
        ...state,
        status: message.status,
        history: append(state.history, message.point, MAX_POINTS),
        audit: append(state.audit, message.audit, MAX_AUDIT),
        learning: isNew ? append(state.learning, message.learning, MAX_POINTS) : state.learning,
      };
    }
    case 'config':
      return { ...state, config: message.config };
    case 'load':
      return { ...state, load: message.load };
    default:
      return state;
  }
}

// Live data over a WebSocket: the server pushes every control step, so nothing is polled.
// Reconnects automatically if the API restarts.
export default function useLiveFeed() {
  const [state, setState] = useState(initialState);

  useEffect(() => {
    let socket;
    let retryTimer;
    let stopped = false;

    const connect = () => {
      socket = new WebSocket(WS_URL);
      socket.onopen = () => setState((s) => ({ ...s, connected: true }));
      socket.onmessage = (event) => setState((s) => reduce(s, JSON.parse(event.data)));
      socket.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!stopped) retryTimer = setTimeout(connect, 2000);
      };
      socket.onerror = () => socket.close();
    };
    connect();

    return () => {
      stopped = true;
      clearTimeout(retryTimer);
      if (socket) socket.close();
    };
  }, []);

  return state;
}
