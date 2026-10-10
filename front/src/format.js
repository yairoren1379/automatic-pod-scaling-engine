// Number and time formatting shared by the views.

export function compact(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return '–';
  const abs = Math.abs(value);
  if (abs >= 1e6) return (value / 1e6).toFixed(digits) + 'M';
  if (abs >= 1e4) return (value / 1e3).toFixed(digits) + 'K';
  if (abs >= 100) return Math.round(value).toLocaleString();
  if (abs >= 10) return value.toFixed(digits);
  return value.toFixed(Math.max(digits, 2));
}

export function signed(value, digits = 2) {
  if (value == null) return '–';
  const text = Math.abs(value).toFixed(digits);
  return value > 0 ? '+' + text : value < 0 ? '−' + text : text;
}

export function clock(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function shortClock(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function movingAverage(values, window) {
  const result = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i] ?? 0;
    if (i >= window) sum -= values[i - window] ?? 0;
    result.push(i >= window - 1 ? sum / window : null);
  }
  return result;
}
