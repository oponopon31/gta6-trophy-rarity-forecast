// Forecast how many GTA VI players earn each trophy in week one.
//
// For every trophy, a recency-weighted quadratic of logit(rate) against
// ln(1 + hours since 2026-11-19T00:00:00Z) over the case's distinct reads
// (consecutive identical reads are one observation, since Exophase's page
// only changes when it refreshes). The curvature is shrunk toward zero
// (weight n/(n+8)), the quadratic prediction is blended 75/25 with a
// linear trend whose slope is shrunk toward the tier mean, and with fewer
// than 3 effective points the model falls back to the line entirely.
// A line would extrapolate the recent decline straight into the floor;
// the quadratic lets the trend bend toward flat as dilution eases and
// earners catch up.
//
// Two priors cover what the history cannot show. An unseen-refresh median
// shift (-0.3 logit, decaying as refreshes are observed) accounts for the
// next page refresh cutting rates as more players are tracked, and the
// spread is calibrated from the history's own volatility around its trends
// plus refresh uncertainty. A trophy last seen at 90%+ never has its
// median raised; the one trophy last seen at 99%+ (everyone tracked holds
// it) has its median pinned at its last rate with a tight band, since
// trophies are permanent and it cannot fall. The forecast is read at the
// target minus half a typical refresh gap, because the answering read is
// the last refresh before the target.
//
// Node 22, standard library only. Reads every <id>.in in the input
// directory and writes one <id>.out per case:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Launch midnight, the origin of log time. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
const HOUR_MS = 3_600_000;
/** Recency half-life, in hours: a read 6 h old counts half as much. */
const HALF_LIFE_H = 6;
/** Curvature shrinkage: the quadratic term's weight is n/(n+8). */
const SHRINK_K2 = 8;
/** Blend of the quadratic prediction with the linear one. */
const BLEND = 0.75;
/** A universal trophy's median is pinned at its last rate, capped here. */
const UNIV_CAP = 99.99;
/** Spread multiplier (kept at 1: the calibrated spread is used as-is). */
const SIGMA_MULT = 1.0;

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

/** A percent clamped to the scorer's range before any logit. */
const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));

const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};

const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));

/** Round to 4 decimals, ties to even (the grader's reference rounds so). */
function roundHalfEven(x: number): number {
  const fl = Math.floor(x);
  const diff = x - fl;
  if (diff > 0.5) return fl + 1;
  if (diff < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

const round4 = (x: number): number => roundHalfEven(x * 10000) / 10000;

/** Force a quantile row non-decreasing (rounding can break the order). */
function enforceOrder(q: number[]): number[] {
  const out = [...q];
  for (let i = 1; i < out.length; i++) {
    if (out[i] < out[i - 1]) out[i] = out[i - 1];
  }
  return out;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** ln(1 + hours since launch), floored at launch. */
function logTimeMs(ms: number): number {
  const hours = Math.max(0, (ms - LAUNCH_MS) / HOUR_MS);
  return Math.log1p(hours);
}

function parseMs(iso: string): number {
  return Date.parse(iso);
}

/**
 * The history's distinct pages, oldest first, with the gaps between them
 * in hours. A read is skipped when it has no positive players_tracked or
 * no rates list; two consecutive reads with the same tracked count and
 * the same rates are the same page, so only the first is kept.
 */
function distinctReads(history: Read[]): { reads: Read[]; gaps: number[] } {
  const reads: Read[] = [];
  const gaps: number[] = [];
  let prevKey = '';
  let prevMs = 0;
  for (const r of history) {
    if (typeof r.players_tracked !== 'number' || !(r.players_tracked > 0)) continue;
    if (!Array.isArray(r.rates)) continue;
    const key = `${r.players_tracked}|${r.rates.map((v) => (v === null ? 'None' : String(v))).join(',')}`;
    if (key === prevKey) continue;
    const ms = parseMs(r.at);
    if (reads.length > 0 && prevMs > 0) {
      gaps.push(Math.max(0, (ms - prevMs) / HOUR_MS));
    }
    prevMs = ms;
    prevKey = key;
    reads.push(r);
  }
  return { reads, gaps };
}

/** A trophy's rate in the newest history read that lists it. */
function lastRate(history: Read[], i: number): number | undefined {
  for (let k = history.length - 1; k >= 0; k--) {
    const rates = history[k].rates;
    if (Array.isArray(rates) && i < rates.length) {
      const v = rates[i];
      if (typeof v === 'number') return v;
    }
  }
  return undefined;
}

interface LinFit {
  /** Effective number of points (weights concentrated => fewer). */
  n: number;
  xBar: number;
  yBar: number;
  b: number;
  resid: number;
  sxx: number;
}

/** Weighted least squares of y on x. */
function wols(xs: number[], ys: number[], ws: number[]): LinFit {
  const n = xs.length;
  if (n === 0) return { n: 0, xBar: 0, yBar: 0, b: 0, resid: 0.15, sxx: 0 };
  let wSum = 0;
  for (const w of ws) wSum += w;
  let xBar = 0;
  for (let i = 0; i < n; i++) xBar += xs[i] * ws[i];
  xBar /= wSum;
  let yBar = 0;
  for (let i = 0; i < n; i++) yBar += ys[i] * ws[i];
  yBar /= wSum;
  let sxx = 0;
  for (let i = 0; i < n; i++) sxx += ws[i] * (xs[i] - xBar) ** 2;
  let sxy = 0;
  for (let i = 0; i < n; i++) sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (yBar + b * (xs[i] - xBar));
    rss += ws[i] * r * r;
  }
  let wSq = 0;
  for (const w of ws) wSq += w * w;
  const nEff = wSum > 0 ? (wSum * wSum) / Math.max(wSq, 1e-12) : 0;
  const dof = Math.max(1, nEff - 2);
  const resid = nEff > 2 ? Math.sqrt(rss / ((wSum * dof) / nEff)) : 0.15;
  return { n: nEff, xBar, yBar, b, resid: Math.max(0.15, resid), sxx };
}

/** Standard error of a linear fit's prediction at x0. */
function predSe(fit: LinFit, x0: number): number {
  if (fit.n === 0) return 0.5;
  if (fit.n < 3 || fit.sxx <= 1e-12) return 0.35;
  const v = fit.resid ** 2 * (1 / fit.n + (x0 - fit.xBar) ** 2 / fit.sxx);
  return Math.sqrt(Math.max(0, v));
}

interface QuadFit {
  a0: number;
  b0: number;
  c0: number;
  resid: number;
  nEff: number;
  xBar: number;
  sxx: number;
}

/**
 * Weighted least squares of y on [1, x - xBar, (x - xBar)^2], solved by
 * Gauss-Jordan elimination with partial pivoting and a tiny ridge for
 * stability. The design is centred on xBar so the intercept is the fit's
 * level at the weighted mean of log time.
 */
function wols2(px: number[], py: number[], pw: number[]): QuadFit {
  if (px.length === 0) {
    return { a0: 0, b0: 0, c0: 0, resid: 0.15, nEff: 0, xBar: 0, sxx: 0 };
  }
  let wSum = 0;
  for (const w of pw) wSum += w;
  let xBar = 0;
  for (let i = 0; i < px.length; i++) xBar += px[i] * pw[i];
  xBar /= wSum;
  const A: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const v: number[] = [0, 0, 0];
  for (let i = 0; i < px.length; i++) {
    const d = px[i] - xBar;
    const row = [1, d, d ** 2];
    for (let a = 0; a < 3; a++) {
      v[a] += pw[i] * row[a] * py[i];
      for (let b = 0; b < 3; b++) {
        A[a][b] += pw[i] * row[a] * row[b];
      }
    }
  }
  for (let a = 0; a < 3; a++) A[a][a] += 1e-9;
  const M: number[][] = [
    [A[0][0], A[0][1], A[0][2], v[0]],
    [A[1][0], A[1][1], A[1][2], v[1]],
    [A[2][0], A[2][1], A[2][2], v[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    const tmp = M[col];
    M[col] = M[piv];
    M[piv] = tmp;
    const pv = M[col][col];
    if (Math.abs(pv) < 1e-12) continue;
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = M[r][col] / pv;
      for (let cc = col; cc < 4; cc++) {
        M[r][cc] -= f * M[col][cc];
      }
    }
  }
  const sol = [0, 1, 2].map((a) => (Math.abs(M[a][a]) > 1e-12 ? M[a][3] / M[a][a] : 0));
  const [a0, b0, c0] = sol;
  let rss = 0;
  for (let i = 0; i < px.length; i++) {
    const d = px[i] - xBar;
    const pred = a0 + b0 * d + c0 * d ** 2;
    rss += pw[i] * (py[i] - pred) ** 2;
  }
  let wSq = 0;
  for (const w of pw) wSq += w * w;
  const nEff = (wSum * wSum) / Math.max(wSq, 1e-12);
  const dof = Math.max(1, nEff - 3);
  const resid = nEff > 3 ? Math.sqrt(rss / ((wSum * dof) / nEff)) : 0.15;
  let sxx = 0;
  for (let i = 0; i < px.length; i++) sxx += pw[i] * (px[i] - xBar) ** 2;
  return { a0, b0, c0, resid: Math.max(0.15, resid), nEff, xBar, sxx };
}

/**
 * The spread's base variance: the history's volatility over the horizon,
 * plus the unseen-refresh variance, plus the fit's prediction error. A
 * rare trophy and the platinum get a wider band.
 */
function spreadBase(
  vol: number,
  horizonDays: number,
  unseenVar: number,
  se: number,
  lastR: number | undefined,
  tier: string,
): number {
  let v = vol ** 2 * horizonDays + unseenVar + se ** 2;
  if (lastR !== undefined && lastR < 1) v += 0.06;
  else if (lastR !== undefined && lastR < 5) v += 0.02;
  if (tier === 'platinum') v += 0.04;
  return Math.sqrt(Math.max(0, v));
}

/**
 * Persistence fallback for a case with no usable read: every trophy's
 * quantiles are its last rate widened like a random walk's, as the house
 * baseline does. Used only when the history has no distinct page, so a
 * degenerate case still yields a valid forecast instead of crashing the
 * run (an invalid case takes no prize, a persistence one stays valid).
 */
function persistenceRows(
  c: CaseFile,
  horizonDays: number,
): Array<{ id: number; q: number[] }> {
  const spread = 0.5 * Math.sqrt(horizonDays);
  return c.trophies.map((t, i) => {
    const lastR = lastRate(c.history, i);
    const l = logit(lastR ? lastR : 1);
    const qs = Z.map((z) => round4(clampPct(invLogit(l + z * spread))));
    return { id: t.id, q: enforceOrder(qs) };
  });
}

/** Forecast one case: one five-quantile row per trophy. */
function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const { reads, gaps } = distinctReads(c.history);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  // No usable read: fall back to persistence rather than failing the case.
  if (reads.length === 0) return persistenceRows(c, horizonDays);
  const last = reads[reads.length - 1];
  const lag = gaps.length > 0 ? Math.min(12, 0.5 * median(gaps)) : 0;
  const targetMs = parseMs(c.target);
  const aimMs = targetMs - lag * HOUR_MS;
  const xAim = logTimeMs(aimMs);
  const xs = reads.map((r) => logTimeMs(parseMs(r.at)));
  const lastMs = parseMs(last.at);
  const ws = reads.map(
    (r) => 0.5 ** (Math.max(0, (lastMs - parseMs(r.at)) / HOUR_MS) / HALF_LIFE_H),
  );

  const nT = c.trophies.length;
  const quadFits: QuadFit[] = [];
  const linFits: LinFit[] = [];
  for (let i = 0; i < nT; i++) {
    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let k = 0; k < reads.length; k++) {
      const v = reads[k].rates[i];
      if (typeof v === 'number') {
        px.push(xs[k]);
        py.push(logit(v));
        pw.push(ws[k]);
      }
    }
    quadFits.push(wols2(px, py, pw));
    linFits.push(wols(px, py, pw));
  }

  // Tier mean linear slopes, for shrinking each trophy's slope.
  const tierSlope = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const sl: number[] = [];
    for (let i = 0; i < nT; i++) {
      if (c.trophies[i].tier === tier && linFits[i].n >= 2) sl.push(linFits[i].b);
    }
    if (sl.length > 0) tierSlope.set(tier, sl.reduce((a, b) => a + b, 0) / sl.length);
  }
  const allSl = linFits.filter((f) => f.n >= 2).map((f) => f.b);
  const globalSlope = allSl.length > 0 ? allSl.reduce((a, b) => a + b, 0) / allSl.length : 0;

  // Volatility: the median residual of the quadratic fits.
  const resids = quadFits.filter((f) => f.nEff >= 2).map((f) => f.resid);
  const vol = resids.length > 0 ? Math.min(0.75, Math.max(0.15, median(resids))) : 0.15;

  const refreshes = Math.max(0, reads.length - 1);
  const unseenShift = -0.3 * Math.exp(-refreshes / 1.5);
  const unseenVar = 0.7 ** 2 * Math.exp(-refreshes / 2);

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nT; i++) {
    const t = c.trophies[i];
    const lastR = lastRate(c.history, i);
    const q = quadFits[i];
    const lin = linFits[i];
    const groupB = tierSlope.get(t.tier) ?? globalSlope;
    const w = lin.n / (lin.n + 3);
    const bLin = w * lin.b + (1 - w) * groupB;
    // Curvature shrunk toward zero; the quadratic's level at the aim.
    const w2 = q.nEff / (q.nEff + SHRINK_K2);
    const cEff = w2 * q.c0;
    const dx = xAim - q.xBar;
    let mLogit =
      q.nEff > 0 ? q.a0 + q.b0 * dx + cEff * dx * dx : logit(lastR ? lastR : 1);
    const mLin =
      lin.n > 0 ? lin.yBar + bLin * (xAim - lin.xBar) : logit(lastR ? lastR : 1);
    // Blend with the line; with too few effective points, the line alone.
    mLogit = q.nEff >= 3 ? BLEND * mLogit + (1 - BLEND) * mLin : mLin;
    if (lastR !== undefined && lastR >= 90) {
      if (lastR >= 99) {
        // Universal trophy: pinned at its last rate, it cannot fall.
        mLogit = logit(Math.min(lastR, UNIV_CAP));
      } else {
        mLogit = Math.min(mLogit, logit(lastR));
        if (lastR >= 99.9) mLogit = Math.min(mLogit, logit(98));
      }
    } else {
      mLogit += unseenShift;
      if (lastR !== undefined) {
        mLogit = Math.min(logit(lastR) + 3, Math.max(logit(lastR) - 3, mLogit));
      }
    }
    const med = clampPct(invLogit(mLogit));
    const se = predSe(lin, xAim);
    let sigma = spreadBase(vol, horizonDays, unseenVar, se, lastR, t.tier);
    if (lastR !== undefined && lastR >= 90 && lastR < 99) sigma = Math.max(sigma, 1.15);
    if (lastR !== undefined && lastR >= 99) sigma = Math.min(sigma, 0.3);
    sigma = Math.min(1.8, Math.max(0.35, sigma)) * SIGMA_MULT;
    sigma = Math.min(1.8, Math.max(0.1, sigma));
    const medLogit = logit(med);
    const qs = Z.map((z) => round4(clampPct(invLogit(medLogit + z * sigma))));
    rows.push({ id: t.id, q: enforceOrder(qs) });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (!Array.isArray(c.history) || c.history.length === 0) {
    throw new Error(`${file} has no history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
