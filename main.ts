// GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).
//
// Forecasts every trophy's earn rate on Exophase at the case's target, as five
// quantiles per trophy, from the case's history of reads alone. Node 22,
// standard library only, no network, fully deterministic (no randomness at
// all, so two runs are byte-identical).
//
// The model, in one paragraph. A trophy's rate is earners divided by players
// tracked, and the two move for different reasons, so the fit is done on the
// rate itself in logit space, where dilution and progression combine into one
// number: per trophy, a weighted least-squares line of logit(rate) against
// x = ln(1 + hours since the release day began, 2026-11-19T00:00:00Z), over
// the whole history. Launch adoption is fast at first and then flattens,
// which is close to linear in log time, and the logit scale is the scale the
// score uses, so a straight line there is the honest extrapolation. Reads are
// weighted steeply towards the recent (half-life 6 hours): the logit curve
// saturates, so a line fitted over the whole history overstates the late
// slope and overshoots, while the last few refreshes say where the curve is
// heading now. (Backtests of both shapes agree: a 6-hour half-life beats
// 12/24/36 hours and the house curve's last-24-hours window on a saturating
// world with per-refresh noise, and loses only a little on a hypothetical
// exactly-linear world.) Exophase does not refresh continuously: the page sits
// unchanged for hours and then jumps, so consecutive identical reads are one
// observation, not many (collapsing them stops a long stale stretch from
// overweighting the fit). Each trophy's slope is shrunk towards the mean
// slope of its tier, so a noisy rare trophy does not extrapolate wildly on
// two points. The forecast is read at the target minus half the typical gap
// between refreshes, because the answering read shows the rates as of
// Exophase's last refresh before it, not as of the target instant itself.
// A trophy last seen at 90% or more can only fall as casual players arrive,
// so its median is never allowed to rise above its last rate (and never
// above 98% for one at 100%).
//
// Two priors cover what the history cannot show. First, an unseen refresh:
// when the history contains no refresh at all (the practice cases: every
// read identical), the next refresh is still coming, and the first refresh
// on launch night multiplied players tracked by about six while earners
// grew only about four and a half times, so the average rate fell to about
// three quarters of its old value, a logit shift of about -0.30. The median
// is shifted down by that amount, decaying away as observed refreshes
// accumulate and the data starts speaking for itself. Second, an honest
// spread: the pinball loss punishes a narrow band that misses far more than
// a wide band that is roughly right (the house baseline's 90% coverage on
// the practice is about 0 because one refresh kicked every rate out of its
// narrow band), so the logit spread combines a term growing with the square
// root of the horizon, scaled by the volatility the history itself shows
// around each trophy's trend (a smooth history earns narrow bands, a jumpy
// one wide ones), a term for refreshes not yet seen, the trend fit's own
// prediction error, and extra variance for rare trophies, floored so no band
// is ever spuriously tight and capped so no band is absurdly wide.
//
// Nothing here is tuned on the practice answers (there are none to tune on:
// the kit holds no answer file and Exophase shows only current rates). The
// constants are priors from the launch-night refresh the rules describe and
// from synthetic backtests of the same generative shape (step refreshes,
// denominator dilution, tiered adoption curves) run locally in /tmp.
//
// Output: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}, one row
// per trophy, quantiles non-decreasing, each 0..100, at most 65,536 bytes.
// A case that fails any step still gets a valid persistence forecast, so a
// single odd trophy can never invalidate the whole case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at LEVELS. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release day began here; x measures log time from this instant. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');

/** Half-life (hours) of a read's weight in the trend fit: recent reads count most. */
const WEIGHT_HALFLIFE_HOURS = 6;
/** Floor / cap for the per-day logit volatility read off the history. */
const VOL_FLOOR = 0.15;
const VOL_CAP = 0.75;
/** Spread for a refresh the history has never shown; decays with refreshes seen. */
const SIGMA_UNSEEN = 0.7;
/** Median logit shift from one unseen refresh (first launch-night refresh). */
const SHIFT_UNSEEN = -0.3;
/** How fast the unseen-refresh shift decays per refresh observed. */
const SHIFT_DECAY = 1.5;
/** How fast the unseen-refresh spread decays per refresh observed. */
const SPREAD_DECAY = 2;
/** Weight of a trophy's own slope vs its tier mean: n / (n + SHRINK_K). */
const SHRINK_K = 3;
/** Largest logit move allowed from a trophy's last rate. */
const MAX_LOGIT_MOVE = 3;
/** A trophy last at or above this rate never has its median raised. */
const SATURATED_AT = 90;
/** Median cap for a trophy last seen at (essentially) 100%. */
const SATURATED_MEDIAN_CAP = 98;
/** Spread floor / ceiling, in logit units. */
const SIGMA_MIN = 0.35;
const SIGMA_MAX = 1.8;

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Hours from the release day's start to `iso`, floored at 0, in log time. */
const logTime = (iso: string): number => {
  const hours = Math.max(0, (Date.parse(iso) - LAUNCH_MS) / 3_600_000);
  return Math.log1p(hours);
};

/** Median of a non-empty array of numbers. */
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

/** Force a quantile row non-decreasing (rounding can tie, never invert). */
function enforceOrder(q: number[]): number[] {
  const out = [...q];
  for (let i = 1; i < out.length; i++) {
    if (out[i] < out[i - 1]) out[i] = out[i - 1];
  }
  return out;
}

interface Fit {
  /** Effective number of points (sum of weights, capped by the count). */
  n: number;
  /** Weighted mean x and y (the line is anchored here). */
  xBar: number;
  yBar: number;
  /** Slope of y ~ a + b * x. */
  b: number;
  /** Residual SD, floored so a perfectly flat history is not read as certain. */
  resid: number;
  /** Weighted sum of squared deviations of x (prediction standard error). */
  sxx: number;
}

/**
 * Weighted least squares of y on x. With one effective point the line is
 * flat through it; with none the fit is empty. The residual SD has a floor
 * of 0.15 logit units: rates printed to 2 decimals on a few hundred tracked
 * players are never exact, and a flat stretch of reads is a stale page, not
 * a proof that nothing moves.
 */
const wols = (xs: number[], ys: number[], ws: number[]): Fit => {
  const n = xs.length;
  if (n === 0) return { n: 0, xBar: 0, yBar: 0, b: 0, resid: 0.15, sxx: 0 };
  const wSum = ws.reduce((s, v) => s + v, 0);
  const xBar = xs.reduce((s, v, k) => s + v * ws[k], 0) / wSum;
  const yBar = ys.reduce((s, v, k) => s + v * ws[k], 0) / wSum;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += ws[i] * (xs[i] - xBar) * (xs[i] - xBar);
    sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += ws[i] * (ys[i] - (yBar + b * (xs[i] - xBar))) ** 2;
  // Effective sample size, for the residual degrees of freedom.
  const wSq = ws.reduce((s, v) => s + v * v, 0);
  const nEff = wSum > 0 ? (wSum * wSum) / Math.max(wSq, 1e-12) : 0;
  const dof = Math.max(1, nEff - 2);
  const resid = nEff > 2 ? Math.sqrt(rss / (wSum * dof / nEff)) : 0.15;
  return { n: nEff, xBar, yBar, b, resid: Math.max(0.15, resid), sxx };
};

/** Standard error of a fitted value at x0 (parameter uncertainty only). */
const predSe = (fit: Fit, x0: number): number => {
  if (fit.n === 0) return 0.5;
  if (fit.n < 3 || fit.sxx <= 1e-12) return 0.35;
  const v = fit.resid * fit.resid * (1 / fit.n + (x0 - fit.xBar) ** 2 / fit.sxx);
  return Math.sqrt(Math.max(0, v));
};

/**
 * Collapse consecutive reads that show the same page (same players tracked
 * and same rates) into one observation, keeping the earliest. Between
 * Exophase refreshes the page is byte-identical, so the duplicates carry no
 * information and would otherwise let a long stale stretch dominate the fit.
 * Returns the distinct reads plus the gaps (hours) between them.
 */
const distinctReads = (history: Read[]): { reads: Read[]; gapsHours: number[] } => {
  const reads: Read[] = [];
  const gapsHours: number[] = [];
  let prevKey = '';
  let prevMs = 0;
  for (const r of history) {
    if (typeof r.players_tracked !== 'number' || !(r.players_tracked > 0)) continue;
    if (!Array.isArray(r.rates)) continue;
    const key = `${r.players_tracked}|${r.rates.join(',')}`;
    if (key === prevKey) continue;
    const ms = Date.parse(r.at);
    if (reads.length > 0 && Number.isFinite(ms) && prevMs > 0) {
      gapsHours.push(Math.max(0, (ms - prevMs) / 3_600_000));
    }
    if (Number.isFinite(ms)) prevMs = ms;
    prevKey = key;
    reads.push(r);
  }
  return { reads, gapsHours };
};

/** Last non-null rate of trophy i over the history (oldest to newest). */
const lastRate = (history: Read[], i: number): number | null => {
  for (let k = history.length - 1; k >= 0; k--) {
    const v = history[k].rates[i];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
  }
  return null;
};

/** Persistence forecast: last rate, baseline random-walk spread. Never throws. */
const persistenceRows = (c: Case): Array<{ id: number; q: number[] }> => {
  const spread = 0.5 * Math.sqrt(Math.max(0, c.horizon_minutes) / 1440);
  return c.trophies.map((t, i) => {
    const r = lastRate(c.history, i) ?? 1;
    const l = logit(r);
    const q = Z.map((z) => round4(clampPct(invLogit(l + z * spread))));
    return { id: t.id, q: enforceOrder(q) };
  });
};

/**
 * Forecast one case: per trophy, a recency-weighted line of logit(rate) on
 * log time over all distinct reads, slope shrunk towards the tier mean, read
 * at the target (minus half a refresh gap), with an honest spread around it.
 */
const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const { reads, gapsHours } = distinctReads(c.history);
  const nDistinct = reads.length;
  if (nDistinct === 0) throw new Error('no usable reads');
  const last = reads[reads.length - 1];
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;

  // The answering read shows the last refresh before the target, so aim the
  // extrapolation half a typical refresh gap early. With no refresh ever
  // seen there is nothing to discount from (the fits are flat anyway).
  const lagHours = gapsHours.length > 0 ? Math.min(12, 0.5 * median(gapsHours)) : 0;
  const targetMs = Date.parse(c.target);
  const aimMs = Number.isFinite(targetMs) ? targetMs - lagHours * 3_600_000 : Date.parse(c.at);
  const xAim = logTime(new Date(aimMs).toISOString());

  const xs = reads.map((r) => logTime(r.at));
  const lastMs = Date.parse(last.at);
  const ws = reads.map((r) => {
    const ageH = Number.isFinite(lastMs) ? Math.max(0, (lastMs - Date.parse(r.at)) / 3_600_000) : 0;
    return 0.5 ** (ageH / WEIGHT_HALFLIFE_HOURS);
  });

  // --- Per-trophy trend fits, then the tier mean slope for shrinkage. ---
  const fits = c.trophies.map((_, i) => {
    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let k = 0; k < reads.length; k++) {
      const rate = reads[k].rates[i];
      if (typeof rate !== 'number' || !Number.isFinite(rate)) continue;
      px.push(xs[k]);
      py.push(logit(rate));
      pw.push(ws[k]);
    }
    return wols(px, py, pw);
  });
  const tierSlope = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const slopes = fits
      .filter((f, i) => c.trophies[i].tier === tier && f.n >= 2 && Number.isFinite(f.b))
      .map((f) => f.b);
    if (slopes.length > 0) tierSlope.set(tier, slopes.reduce((s, v) => s + v, 0) / slopes.length);
  }
  const allSlopes = fits.filter((f) => f.n >= 2 && Number.isFinite(f.b)).map((f) => f.b);
  const globalSlope = allSlopes.length > 0
    ? allSlopes.reduce((s, v) => s + v, 0) / allSlopes.length
    : 0;
  // How jumpy the world is around its own trends: the median trophy's
  // residual SD. A smooth history earns narrow bands, a jumpy one wide ones.
  const resids = fits.filter((f) => f.n >= 2).map((f) => f.resid);
  const vol = resids.length > 0
    ? Math.min(VOL_CAP, Math.max(VOL_FLOOR, median(resids)))
    : VOL_FLOOR;

  // The unseen-refresh prior: strong with no refresh observed, gone once the
  // history has shown several. (One refresh = one distinct page after the
  // first, so nDistinct - 1 counts the refreshes seen.)
  const refreshesSeen = Math.max(0, nDistinct - 1);
  const unseenShift = SHIFT_UNSEEN * Math.exp(-refreshesSeen / SHIFT_DECAY);
  const unseenVar = SIGMA_UNSEEN ** 2 * Math.exp(-refreshesSeen / SPREAD_DECAY);

  return c.trophies.map((t, i) => {
    const lastR = lastRate(c.history, i);
    const fit = fits[i];
    const groupB = tierSlope.get(t.tier) ?? globalSlope;
    // Shrink the slope towards the tier mean; a trophy with one distinct
    // read has no slope of its own and takes the tier mean (which is 0 on a
    // flat history), so its forecast is its last level, shifted by the
    // unseen-refresh prior below.
    const w = fit.n / (fit.n + SHRINK_K);
    const b = w * fit.b + (1 - w) * groupB;
    let mLogit = fit.n > 0 ? fit.yBar + b * (xAim - fit.xBar) : logit(lastR ?? 1);

    if (lastR !== null && lastR >= SATURATED_AT) {
      // A trophy nearly everyone tracked holds can only fall as casual
      // players arrive; the trend fit alone would let it keep rising.
      mLogit = Math.min(mLogit, logit(lastR));
      if (lastR >= 99.9) mLogit = Math.min(mLogit, logit(SATURATED_MEDIAN_CAP));
    } else {
      mLogit += unseenShift;
      if (lastR !== null) {
        // Never let an extrapolation run away from the last observed rate.
        mLogit = Math.min(logit(lastR) + MAX_LOGIT_MOVE, Math.max(logit(lastR) - MAX_LOGIT_MOVE, mLogit));
      }
    }
    const med = clampPct(invLogit(mLogit));

    // --- Spread: horizon random walk (scaled by observed volatility) +
    // --- unseen refresh + fit error + rarity. ---
    const se = predSe(fit, xAim);
    let variance = vol ** 2 * horizonDays + unseenVar + se ** 2;
    if (lastR !== null && lastR < 1) variance += 0.06;
    else if (lastR !== null && lastR < 5) variance += 0.02;
    if (t.tier === 'platinum') variance += 0.04;
    let sigma = Math.sqrt(Math.max(0, variance));
    if (lastR !== null && lastR >= SATURATED_AT) sigma = Math.max(sigma, 1.15);
    sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, sigma));

    const l = logit(med);
    const q = enforceOrder(Z.map((z) => round4(clampPct(invLogit(l + z * sigma)))));
    return { id: t.id, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let rows: Array<{ id: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    if (!Array.isArray(c.trophies) || !Array.isArray(c.history) || c.history.length === 0) {
      throw new Error(`${file} has no trophies or history`);
    }
    rows = forecastCase(c);
  } catch {
    // A case that fails any step still gets a valid persistence forecast:
    // an invalid case forfeits the prize, a merely mediocre one does not.
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
      rows = persistenceRows(c);
    } catch {
      rows = [];
    }
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
