/**
 * Canvas 2D matplotlib-style line plotter.
 *
 * Designed for low-overhead rendering of transient, harmonic, and realtime
 * waveform tiles. Keeps a consistent flat-design look (axes, gridlines,
 * muted tick labels, colored series, and a legend).
 */

export type PlotSeries = {
  label: string;
  values: number[];
  /** Optional explicit color; if omitted a palette is used. */
  color?: string;
};

export type PlotData = {
  x: number[];
  series: PlotSeries[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
  /** Line for sweeps/waveforms, stem for discrete spectra, bar for categorical summaries. */
  kind?: "line" | "stem" | "bar";
  /** Optional labels for categorical x values such as .op solution entries. */
  xTickLabels?: string[];
  /** Display x in log scale (AC sweeps). */
  logX?: boolean;
  /** Display y in log scale. */
  logY?: boolean;
};

const PALETTE = [
  "#6ea8ff",
  "#ffa56b",
  "#7ad97a",
  "#ff86c7",
  "#ffd26d",
  "#9bb7ff",
  "#82e0d4",
  "#e08cff"
];

export function drawPlot(ctx: CanvasRenderingContext2D, w: number, h: number, data: PlotData) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1422";
  ctx.fillRect(0, 0, w, h);

  const padL = 52;
  const padR = 16;
  const padT = data.title ? 26 : 14;
  const padB = 38;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  if (plotW <= 10 || plotH <= 10) return;

  // Axis ranges
  const xs = data.x;
  const yRange = finiteSeriesRange(data.series, data.kind === "stem" || data.kind === "bar");
  if ((xs.length < 2 && data.kind !== "bar") || !yRange) {
    drawEmptyState(ctx, w, h, "no samples yet");
    return;
  }

  const useLogX = !!data.logX && xs.some((v) => v > 0);
  const xRange = finiteArrayRange(xs, useLogX);
  if (!xRange) {
    drawEmptyState(ctx, w, h, "no samples yet");
    return;
  }
  let [xMin, xMax] = xRange;
  if (data.kind === "bar") {
    xMin -= 0.5;
    xMax += 0.5;
  }
  if (xMin === xMax) xMax = xMin + 1;

  let [yMin, yMax] = yRange;
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  // Gridlines
  ctx.save();
  ctx.strokeStyle = "rgba(110,168,255,0.08)";
  ctx.fillStyle = "#7f8fa6";
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.lineWidth = 1;

  const xTicks =
    data.kind === "bar"
      ? xs.filter((v) => Number.isFinite(v))
      : useLogX
        ? logTicks(xMin, xMax)
        : niceTicks(xMin, xMax, plotW < 360 ? 4 : 6);
  const yTicks = niceTicks(yMin, yMax, 5);
  const yStep = yTicks.length > 1 ? Math.abs(yTicks[1] - yTicks[0]) : Math.abs(yMax - yMin);
  const xStep = xTicks.length > 1 ? Math.abs(xTicks[1] - xTicks[0]) : Math.abs(xMax - xMin);

  ctx.beginPath();
  for (const tv of xTicks) {
    const px = scaleX(tv, xMin, xMax, plotW, padL, useLogX);
    ctx.moveTo(px, padT);
    ctx.lineTo(px, padT + plotH);
  }
  for (const tv of yTicks) {
    const py = padT + plotH - ((tv - yMin) / (yMax - yMin)) * plotH;
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
  }
  ctx.stroke();
  ctx.restore();

  // Axes
  ctx.save();
  ctx.strokeStyle = "rgba(230,237,247,0.4)";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.restore();

  // Tick labels
  ctx.fillStyle = "#9aa7bf";
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tv of xTicks) {
    const px = scaleX(tv, xMin, xMax, plotW, padL, useLogX);
    const label = data.kind === "bar" ? data.xTickLabels?.[Math.round(tv)] ?? formatTick(tv, xStep) : formatTick(tv, xStep);
    const trimmed = label.length > 11 ? `${label.slice(0, 10)}…` : label;
    ctx.fillText(trimmed, px, padT + plotH + 6);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const tv of yTicks) {
    const py = padT + plotH - ((tv - yMin) / (yMax - yMin)) * plotH;
    ctx.fillText(formatTick(tv, yStep), padL - 6, py);
  }

  // Title
  if (data.title) {
    ctx.fillStyle = "#d6dcea";
    ctx.font = "600 12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(data.title, padL, 6);
  }

  // Axis labels (explicit baseline so they never collide with tick labels)
  if (data.xLabel) {
    ctx.fillStyle = "#7f8fa6";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(data.xLabel, padL + plotW / 2, h - 5);
  }
  if (data.yLabel) {
    ctx.save();
    ctx.fillStyle = "#7f8fa6";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(data.yLabel, 0, 0);
    ctx.restore();
  }

  // Series
  data.series.forEach((series, sIdx) => {
    if (series.values.length < 2 && data.kind !== "bar") return;
    const color = series.color ?? PALETTE[sIdx % PALETTE.length];
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = data.kind === "stem" ? 1.2 : 1.6;
    ctx.lineJoin = "round";
    const n = Math.min(series.values.length, xs.length);
    if (data.kind === "bar") {
      const baseline = padT + plotH - ((0 - yMin) / (yMax - yMin)) * plotH;
      const groupW = Math.max(8, plotW / Math.max(1, xs.length) * 0.7);
      const barW = Math.max(3, groupW / Math.max(1, data.series.length));
      const groupOffset = (sIdx - (data.series.length - 1) / 2) * barW;
      ctx.fillStyle = color;
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(xs[i]) || !Number.isFinite(series.values[i])) continue;
        const x = scaleX(xs[i], xMin, xMax, plotW, padL, false) + groupOffset - barW / 2;
        const y = padT + plotH - ((series.values[i] - yMin) / (yMax - yMin)) * plotH;
        const top = Math.min(y, baseline);
        const height = Math.max(1, Math.abs(baseline - y));
        ctx.globalAlpha = 0.88;
        ctx.fillRect(x, top, barW, height);
      }
      ctx.globalAlpha = 1;
    } else if (data.kind === "stem") {
      const baseline = padT + plotH - ((0 - yMin) / (yMax - yMin)) * plotH;
      const offset = (sIdx - (data.series.length - 1) / 2) * 5;
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(xs[i]) || !Number.isFinite(series.values[i])) continue;
        const x = scaleX(xs[i], xMin, xMax, plotW, padL, useLogX) + offset;
        const y = padT + plotH - ((series.values[i] - yMin) / (yMax - yMin)) * plotH;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(x, baseline);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    } else {
      ctx.beginPath();
      let started = false;
      const stride = lineStride(n, plotW);
      for (let i = 0; i < n; i += stride) {
        if (!Number.isFinite(xs[i]) || !Number.isFinite(series.values[i])) continue;
        if (useLogX && xs[i] <= 0) continue;
        const x = scaleX(xs[i], xMin, xMax, plotW, padL, useLogX);
        const y = padT + plotH - ((series.values[i] - yMin) / (yMax - yMin)) * plotH;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      if (stride > 1 && n > 0) {
        const i = n - 1;
        if (Number.isFinite(xs[i]) && Number.isFinite(series.values[i]) && (!useLogX || xs[i] > 0)) {
          const x = scaleX(xs[i], xMin, xMax, plotW, padL, useLogX);
          const y = padT + plotH - ((series.values[i] - yMin) / (yMax - yMin)) * plotH;
          if (!started) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
    ctx.restore();
  });

  // Legend (top-left with a translucent backing so traces stay readable)
  const legendX = padL + 8;
  let legendY = padT + 8;
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  if (data.series.length > 0) {
    const widest = Math.max(...data.series.map((series) => ctx.measureText(series.label).width));
    ctx.fillStyle = "rgba(15,20,34,0.78)";
    ctx.fillRect(legendX - 4, legendY - 9, widest + 22, data.series.length * 14 + 4);
  }
  data.series.forEach((series, sIdx) => {
    const color = series.color ?? PALETTE[sIdx % PALETTE.length];
    ctx.fillStyle = color;
    ctx.fillRect(legendX, legendY - 4, 10, 3);
    ctx.fillStyle = "#d6dcea";
    ctx.fillText(series.label, legendX + 14, legendY - 2);
    legendY += 14;
  });
}

function finiteArrayRange(values: number[], positiveOnly = false): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (positiveOnly && value <= 0) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return min === Infinity ? null : [min, max];
}

function finiteSeriesRange(series: PlotSeries[], includeZero = false): [number, number] | null {
  let min = includeZero ? 0 : Infinity;
  let max = includeZero ? 0 : -Infinity;
  let found = false;
  for (const entry of series) {
    for (const value of entry.values) {
      if (!Number.isFinite(value)) continue;
      found = true;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return found ? [min, max] : null;
}

function lineStride(pointCount: number, plotWidth: number): number {
  const target = Math.max(800, Math.floor(plotWidth * 2));
  return Math.max(1, Math.ceil(pointCount / target));
}

function scaleX(
  value: number,
  min: number,
  max: number,
  width: number,
  offset: number,
  log: boolean
): number {
  if (!log) return offset + ((value - min) / (max - min)) * width;
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  return offset + ((Math.log10(value) - lo) / (hi - lo)) * width;
}

function logTicks(min: number, max: number): number[] {
  const start = Math.floor(Math.log10(min));
  const end = Math.ceil(Math.log10(max));
  const ticks: number[] = [];
  for (let e = start; e <= end; e++) {
    const value = Math.pow(10, e);
    if (value >= min && value <= max) ticks.push(value);
  }
  if (ticks.length < 2) return niceTicks(min, max, 6);
  return ticks;
}

function drawEmptyState(ctx: CanvasRenderingContext2D, w: number, h: number, text: string) {
  ctx.fillStyle = "#7f8fa6";
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
}

function niceTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const span = max - min;
  const step = niceStep(span / count);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + 1e-12; v += step) ticks.push(v);
  return ticks;
}

function niceStep(raw: number): number {
  const exp = Math.floor(Math.log10(raw));
  const f = raw / Math.pow(10, exp);
  let nf = 1;
  if (f >= 5) nf = 5;
  else if (f >= 2) nf = 2;
  return nf * Math.pow(10, exp);
}

/**
 * Engineering-notation tick labels (2m, 20n, 1.5k) like a SPICE waveform
 * viewer. ``step`` is the tick spacing: values far below it are floating
 * point noise around zero and print as "0" instead of "-2.8e-17".
 */
function formatTick(v: number, step = 0): string {
  if (v === 0 || (step > 0 && Math.abs(v) < step * 1e-6)) return "0";
  const abs = Math.abs(v);
  const units: [number, string][] = [
    [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""],
    [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"], [1e-15, "f"]
  ];
  if (abs >= 1e15 || abs < 1e-16) return v.toExponential(1);
  const [scale, suffix] = units.find(([candidate]) => abs >= candidate * 0.9999) ?? [1e-15, "f"];
  const scaled = v / scale;
  const stepScaled = step > 0 ? step / scale : 0;
  // Keep enough decimals to separate neighbouring ticks, but no trailing zeros.
  const decimals = stepScaled > 0 ? Math.min(4, Math.max(0, Math.ceil(-Math.log10(stepScaled)) + 1)) : 2;
  const text = Number(scaled.toFixed(decimals)).toString();
  return `${text}${suffix}`;
}
