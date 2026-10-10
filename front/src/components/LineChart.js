import React, { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { compact } from '../format';

const MARGIN = { top: 10, right: 14, bottom: 26, left: 50 };

// Axis ticks are round numbers: show 0, 25, 1,000 rather than 0.00, 25.0, 1.0K.
function tickLabel(value) {
  return Number.isInteger(value) ? value.toLocaleString() : compact(value);
}

// A "nice" step (1, 2, 2.5 or 5 times a power of ten) so axis ticks are round numbers.
function niceStep(range, count) {
  if (!(range > 0)) return 1;
  const raw = range / count;
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const fraction = raw / power;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
  return nice * power;
}

function nearestIndex(sorted, value) {
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(sorted[lo - 1] - value) <= Math.abs(sorted[lo] - value)) return lo - 1;
  return lo;
}

function nearestPoint(points, x, tolerance) {
  if (!points.length) return null;
  const xs = points.map((p) => p.x);
  const point = points[nearestIndex(xs, x)];
  return Math.abs(point.x - x) <= tolerance ? point : null;
}

function LineKey({ color, dashed, faint }) {
  return (
    <svg width="16" height="8" aria-hidden="true" className="line-key">
      <line x1="1" y1="4" x2="15" y2="4" stroke={color} strokeWidth="2" strokeLinecap="round"
        strokeDasharray={dashed ? '3 3' : undefined} opacity={faint ? 0.4 : 1} />
    </svg>
  );
}

/**
 * series: [{ label, color, points: [{x, y}] (sorted by x), dashed?, step?, faint? }]
 * reference: { value, label } draws a labeled horizontal line (for example the latency SLO).
 */
export default function LineChart({
  series, height = 170, yMin, yMax, reference, unit = '', formatX = (x) => x, formatY = compact,
  ariaLabel, emptyText = 'Waiting for data…',
}) {
  const wrapRef = useRef(null);
  const [width, setWidth] = useState(600);
  const [hoverX, setHoverX] = useState(null);

  useLayoutEffect(() => {
    const element = wrapRef.current;
    if (!element || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const geometry = useMemo(() => {
    const clean = series.map((s) => ({ ...s, points: s.points.filter((p) => p.y != null && Number.isFinite(p.y)) }));
    const xs = [...new Set(clean.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
    if (!xs.length) return null;
    const ys = clean.flatMap((s) => s.points.map((p) => p.y));
    if (reference) ys.push(reference.value);

    let lo = yMin !== undefined ? yMin : Math.min(...ys);
    let hi = yMax !== undefined ? yMax : Math.max(...ys);
    if (hi === lo) hi = lo + 1;
    const yStep = niceStep(hi - lo, 4);
    if (yMin === undefined) lo = Math.floor(lo / yStep) * yStep;
    if (yMax === undefined) hi = Math.ceil(hi / yStep) * yStep;

    const xMin = xs[0];
    const xMax = xs[xs.length - 1];
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    const x = (v) => MARGIN.left + (xMax === xMin ? innerW / 2 : ((v - xMin) / (xMax - xMin)) * innerW);
    const y = (v) => MARGIN.top + innerH - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * innerH;

    const yTicks = [];
    for (let v = lo; v <= hi + yStep / 1000; v += yStep) yTicks.push(v);
    const xStep = niceStep(xMax - xMin, Math.max(2, Math.floor(innerW / 110)));
    const xTicks = [];
    for (let v = Math.ceil(xMin / xStep) * xStep; v <= xMax; v += xStep) xTicks.push(v);

    const spacing = xs.length > 1 ? (xMax - xMin) / (xs.length - 1) : 1;
    const paths = clean.map((s) =>
      s.points
        .map((p, i) => {
          if (i === 0) return `M${x(p.x)},${y(p.y)}`;
          return s.step ? `H${x(p.x)}V${y(p.y)}` : `L${x(p.x)},${y(p.y)}`;
        })
        .join(''),
    );
    return { clean, xs, x, y, yTicks, xTicks, innerW, xMin, xMax, paths, tolerance: spacing * 1.5 };
  }, [series, width, height, yMin, yMax, reference]);

  if (!geometry) {
    return (
      <div ref={wrapRef} className="chart-empty" style={{ height }}>
        {emptyText}
      </div>
    );
  }

  const { clean, xs, x, y, yTicks, xTicks, innerW, xMin, xMax, paths, tolerance } = geometry;
  const right = width - MARGIN.right;
  const bottom = height - MARGIN.bottom;

  const moveTo = (clientX, rect) => {
    const fraction = (clientX - rect.left - MARGIN.left) / innerW;
    const value = xMin + Math.min(1, Math.max(0, fraction)) * (xMax - xMin);
    setHoverX(xs[nearestIndex(xs, value)]);
  };
  const onKeyDown = (event) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    const index = hoverX == null ? xs.length - 1 : nearestIndex(xs, hoverX);
    const next = Math.min(xs.length - 1, Math.max(0, index + (event.key === 'ArrowRight' ? 1 : -1)));
    setHoverX(xs[next]);
  };

  const hovered = hoverX == null ? [] : clean.map((s) => ({ s, point: nearestPoint(s.points, hoverX, tolerance) }));
  const tooltipLeft = hoverX == null ? 0 : x(hoverX) + 12 > width - 190 ? x(hoverX) - 190 : x(hoverX) + 12;

  return (
    <div ref={wrapRef} className="chart">
      {clean.length > 1 && (
        <div className="legend">
          {clean.map((s) => (
            <span key={s.label} className="legend-item">
              <LineKey color={s.color} dashed={s.dashed} faint={s.faint} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <div className="chart-plot">
        <svg
          width={width} height={height} role="img" tabIndex={0}
          aria-label={ariaLabel || clean.map((s) => s.label).join(', ')}
          onPointerMove={(e) => moveTo(e.clientX, e.currentTarget.getBoundingClientRect())}
          onPointerLeave={() => setHoverX(null)}
          onKeyDown={onKeyDown}
          onBlur={() => setHoverX(null)}
        >
          {yTicks.map((v) => (
            <g key={'y' + v}>
              <line x1={MARGIN.left} x2={right} y1={y(v)} y2={y(v)} className="grid-line" />
              <text x={MARGIN.left - 6} y={y(v)} className="tick" textAnchor="end" dominantBaseline="middle">
                {tickLabel(Math.round(v * 1e6) / 1e6)}
              </text>
            </g>
          ))}
          <line x1={MARGIN.left} x2={right} y1={bottom} y2={bottom} className="axis-line" />
          {xTicks.map((v) => (
            <text key={'x' + v} x={x(v)} y={bottom + 16} className="tick" textAnchor="middle">
              {formatX(v)}
            </text>
          ))}
          {reference && (
            <g>
              <line x1={MARGIN.left} x2={right} y1={y(reference.value)} y2={y(reference.value)} className="reference-line" />
              {/* Label above the line, or below it when the line is at the top of the plot. */}
              <text x={right - 4} className="reference-label" textAnchor="end"
                y={y(reference.value) - MARGIN.top < 14 ? y(reference.value) + 14 : y(reference.value) - 5}>
                {reference.label}
              </text>
            </g>
          )}
          {clean.map((s, i) => (
            <path key={s.label} d={paths[i]} fill="none" stroke={s.color} strokeWidth="2"
              strokeLinejoin="round" strokeLinecap="round" opacity={s.faint ? 0.35 : 1}
              strokeDasharray={s.dashed ? '5 4' : undefined} />
          ))}
          {hoverX != null && (
            <g>
              <line x1={x(hoverX)} x2={x(hoverX)} y1={MARGIN.top} y2={bottom} className="crosshair" />
              {hovered.map(({ s, point }) =>
                point ? (
                  <circle key={s.label} cx={x(point.x)} cy={y(point.y)} r="4" fill={s.color} className="hover-dot" />
                ) : null,
              )}
            </g>
          )}
        </svg>
        {hoverX != null && (
          <div className="tooltip" style={{ left: tooltipLeft, top: MARGIN.top }}>
            <div className="tooltip-title">{formatX(hoverX)}</div>
            {hovered.map(({ s, point }) => (
              <div key={s.label} className="tooltip-row">
                <LineKey color={s.color} dashed={s.dashed} faint={s.faint} />
                <strong>{point ? formatY(point.y) + unit : '–'}</strong>
                <span>{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
