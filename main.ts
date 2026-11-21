// GTA VI trophy rarity forecaster — M5 structural Weibull E/N model.
// Node 22, standard library only, deterministic, no network, no randomness.
// Reads every *.in case in the input directory and writes one <id>.out per case.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const LAUNCH_MS = Date.parse("2026-11-19T00:00:00Z");
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const HALF_LIFE_H = 6;
const BETA_GRID = [1.0, 1.5, 2.0];
const TAU_GRID = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];
const TAU_N_GRID = [6, 12, 24, 48, 96, 192, 384, 768, 1e9];
const UNIV_CAP = 99.99;
const UNSEEN_SHIFT0 = -0.3;
const MAX_MOVE = 3.0;
const LAG_MULT = 0.5;

interface TrophyRow {
  id: number;
  tier: string;
}

interface HistoryRead {
  at: string;
  players_tracked: number;
  rates: (number | null)[];
}

interface CaseInput {
  id: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  trophies: TrophyRow[];
  history: HistoryRead[];
}

interface DistinctRead {
  t: number;
  n: number;
  rates: (number | null)[];
}

function clampPct(x: number): number {
  return Math.min(UNIV_CAP, Math.max(0.01, x));
}

function logitPct(p: number): number {
  const c = clampPct(p);
  const q = c / 100;
  return Math.log(q / (1 - q));
}

function invLogit(l: number): number {
  return 100 / (1 + Math.exp(-l));
}

// Round to 4 decimals, ties to even.
function round4(x: number): number {
  const s = x * 10000;
  const f = Math.floor(s);
  const d = s - f;
  if (d > 0.5) return (f + 1) / 10000;
  if (d < 0.5) return f / 10000;
  return (f % 2 === 0 ? f : f + 1) / 10000;
}

function enforceOrder(q: number[]): number[] {
  for (let i = 1; i < q.length; i++) {
    if (q[i] < q[i - 1]) q[i] = q[i - 1];
  }
  return q;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

interface TrophyFit {
  b: number;
  tau: number;
  a: number;
  sse: number;
  resid: number;
  nObs: number;
}

function forecastCase(c: CaseInput): { trophies: { id: number; q: number[] }[] } {
  const expected = [0.05, 0.25, 0.5, 0.75, 0.95];
  if (
    !Array.isArray(c.levels) ||
    c.levels.length !== 5 ||
    c.levels.some((v, i) => v !== expected[i])
  ) {
    throw new Error("levels must be [0.05, 0.25, 0.5, 0.75, 0.95]");
  }
  if (!Array.isArray(c.history) || c.history.length === 0) {
    throw new Error("empty history");
  }

  const trophies = c.trophies;
  const horizonDays = c.horizon_minutes / 1440;
  const targetMs = Date.parse(c.target);

  // Last rate per trophy: newest history read (any read) that lists it.
  const lastR: (number | undefined)[] = trophies.map((_, i) => {
    for (let r = c.history.length - 1; r >= 0; r--) {
      const rates = c.history[r].rates;
      if (Array.isArray(rates) && typeof rates[i] === "number") {
        return rates[i] as number;
      }
    }
    return undefined;
  });

  // Distinct reads: drop repeats of the previous kept read's key.
  const distinct: DistinctRead[] = [];
  let prevKey: string | null = null;
  for (const h of c.history) {
    if (typeof h.players_tracked !== "number" || !(h.players_tracked > 0)) continue;
    if (!Array.isArray(h.rates)) continue;
    const key =
      h.players_tracked +
      "|" +
      h.rates.map((v) => (v === null ? "None" : String(v))).join(",");
    if (prevKey !== null && key === prevKey) continue;
    prevKey = key;
    distinct.push({
      t: (Date.parse(h.at) - LAUNCH_MS) / 3600000,
      n: h.players_tracked,
      rates: h.rates,
    });
  }

  // Persistence fallback: no distinct reads at all.
  if (distinct.length === 0) {
    const spread = 0.5 * Math.sqrt(horizonDays);
    const rows = trophies.map((tr, i) => {
      const l = logitPct(lastR[i] === undefined ? 1 : lastR[i]);
      const q = enforceOrder(
        Z.map((z) => round4(clampPct(invLogit(l + z * spread))))
      );
      return { id: tr.id, q };
    });
    return { trophies: rows };
  }

  const gaps: number[] = [];
  for (let i = 1; i < distinct.length; i++) gaps.push(distinct[i].t - distinct[i - 1].t);
  const lagHours = gaps.length ? Math.min(12, LAG_MULT * median(gaps)) : 0;
  const aimMs = targetMs - lagHours * 3600000;
  const aimH = (aimMs - LAUNCH_MS) / 3600000;

  const tLast = distinct[distinct.length - 1].t;
  const w = distinct.map((d) => Math.pow(0.5, (tLast - d.t) / HALF_LIFE_H));

  // Tracked-players curve N(t): N_k - N0 = R0 * f_k + r1 * t_k, least squares.
  const N0 = distinct[0].n;
  let bestN: { score: number; r0: number; tau: number; r1: number } | null = null;
  for (const tauN of TAU_N_GRID) {
    const f = distinct.map((d) => (tauN >= 1e8 ? 0 : 1 - Math.exp(-d.t / tauN)));
    let sff = 0;
    let sft = 0;
    let stt = 0;
    let sfy = 0;
    let sty = 0;
    for (let k = 0; k < distinct.length; k++) {
      const y = distinct[k].n - N0;
      sff += f[k] * f[k];
      sft += f[k] * distinct[k].t;
      stt += distinct[k].t * distinct[k].t;
      sfy += f[k] * y;
      sty += distinct[k].t * y;
    }
    const det = sff * stt - sft * sft;
    if (Math.abs(det) < 1e-12) continue;
    let r0 = (sfy * stt - sft * sty) / det;
    let r1 = (sff * sty - sft * sfy) / det;
    r0 = Math.max(0, r0);
    r1 = Math.max(0, r1);
    let score = 0;
    for (let k = 0; k < distinct.length; k++) {
      const pred = Math.max(1, N0 + r0 * f[k] + r1 * distinct[k].t);
      const d = Math.log(pred) - Math.log(Math.max(1, distinct[k].n));
      score += d * d;
    }
    if (bestN === null || score < bestN.score) {
      bestN = { score, r0, tau: tauN, r1 };
    }
  }
  const nFit = bestN === null ? { r0: 0, tau: 24, r1: 0 } : bestN;
  const fAim = nFit.tau >= 1e8 ? 0 : 1 - Math.exp(-aimH / nFit.tau);
  const nAim = Math.max(1, N0 + nFit.r0 * fAim + nFit.r1 * aimH);

  // Per-trophy Weibull fit for one shared beta.
  function fitTrophy(beta: number, i: number): TrophyFit {
    const obs: { t: number; w: number; e: number }[] = [];
    for (let k = 0; k < distinct.length; k++) {
      const r = distinct[k].rates[i];
      if (typeof r === "number") {
        obs.push({ t: distinct[k].t, w: w[k], e: (r / 100) * distinct[k].n });
      }
    }
    if (obs.length < 2) {
      return { b: obs.length ? obs[0].e : 0, tau: TAU_GRID[0], a: 0, sse: 0, resid: 0.35, nObs: obs.length };
    }
    const b = obs[0].e;
    let sumW = 0;
    for (const o of obs) sumW += o.w;
    let best: { sse: number; tau: number; a: number } | null = null;
    for (const tau of TAU_GRID) {
      const f = obs.map((o) => 1 - Math.exp(-Math.pow(Math.max(o.t, 0) / tau, beta)));
      let num = 0;
      let den = 0;
      for (let k = 0; k < obs.length; k++) {
        num += obs[k].w * f[k] * (obs[k].e - b);
        den += obs[k].w * f[k] * f[k];
      }
      let a = den < 1e-12 ? 0 : num / den;
      a = Math.max(0, a);
      let sse = 0;
      for (let k = 0; k < obs.length; k++) {
        const d =
          Math.log(Math.max(0.05, b + a * f[k])) - Math.log(Math.max(0.05, obs[k].e));
        sse += obs[k].w * d * d;
      }
      if (best === null || sse < best.sse) best = { sse, tau, a };
    }
    const resid = Math.sqrt((best as { sse: number }).sse / sumW);
    return {
      b,
      tau: (best as { tau: number }).tau,
      a: (best as { a: number }).a,
      sse: (best as { sse: number }).sse,
      resid,
      nObs: obs.length,
    };
  }

  // Shared beta: smallest total best-tau SSE over all trophies.
  let bestBeta = BETA_GRID[0];
  let bestTotal = Infinity;
  const fitsByBeta = new Map<number, TrophyFit[]>();
  for (const beta of BETA_GRID) {
    const fits = trophies.map((_, i) => fitTrophy(beta, i));
    fitsByBeta.set(beta, fits);
    let total = 0;
    for (const ft of fits) total += ft.sse;
    if (total < bestTotal) {
      bestTotal = total;
      bestBeta = beta;
    }
  }
  const fits = fitsByBeta.get(bestBeta) as TrophyFit[];

  const refreshes = Math.max(0, distinct.length - 1);
  // vol is the median residual over fitted trophies only: a trophy with
  // fewer than 2 observations carries no fit residual, so it does not enter
  // this median (when no trophy has a fit, vol falls back to 0.15).
  const resids = fits.filter((f) => f.nObs >= 2).map((f) => f.resid);
  let vol = resids.length ? median(resids) : 0.15;
  vol = Math.min(0.75, Math.max(0.15, vol));

  const rows = trophies.map((tr, i) => {
    const ft = fits[i];
    const lr = lastR[i];
    let m: number;
    if (ft.nObs < 2) {
      m = logitPct(lr === undefined ? 1 : lr);
    } else {
      const fAimT = 1 - Math.exp(-Math.pow(Math.max(aimH, 0) / ft.tau, bestBeta));
      const eAim = ft.b + ft.a * fAimT;
      const p = eAim / nAim;
      if (p <= 0) m = Math.log(1e-9);
      else if (p >= 1) m = Math.log(1e9);
      else m = Math.log(p / (1 - p));
      if (lr !== undefined && lr >= 99) {
        m = logitPct(Math.min(lr, UNIV_CAP));
      } else if (lr !== undefined && lr >= 90) {
        m = Math.min(m, logitPct(lr));
      } else {
        m += UNSEEN_SHIFT0 * Math.exp(-refreshes / 1.5);
        if (lr !== undefined) {
          m = Math.min(logitPct(lr) + MAX_MOVE, Math.max(logitPct(lr) - MAX_MOVE, m));
        }
      }
    }
    const med = clampPct(invLogit(m));
    const unseenVar = 0.49 * Math.exp(-refreshes / 2);
    let variance = vol * vol * horizonDays + unseenVar + 0.35 * 0.35;
    if (lr !== undefined && lr < 1) variance += 0.06;
    else if (lr !== undefined && lr < 5) variance += 0.02;
    if (tr.tier === "platinum") variance += 0.04;
    let sigma = Math.sqrt(Math.max(0, variance));
    if (lr !== undefined && lr >= 90 && !(lr >= 99)) sigma = Math.max(sigma, 1.15);
    if (lr !== undefined && lr >= 99) sigma = Math.min(sigma, 0.3);
    sigma = Math.min(1.8, Math.max(0.35, sigma));
    sigma = Math.min(1.8, Math.max(0.1, sigma));
    const lm = logitPct(med);
    const q = enforceOrder(Z.map((z) => round4(clampPct(invLogit(lm + z * sigma)))));
    return { id: tr.id, q };
  });

  return { trophies: rows };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ? process.env.MAKEFUN_INPUT_DIR : "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ? process.env.MAKEFUN_OUTPUT_DIR : "/output";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const c = JSON.parse(readFileSync(join(inputDir, file), "utf8")) as CaseInput;
    const out = forecastCase(c);
    const id =
      typeof c.id === "string" && c.id ? c.id : file.replace(/\.in$/, "");
    const text = JSON.stringify(out) + "\n";
    if (Buffer.byteLength(text, "utf8") > 65536) {
      throw new Error("output too large");
    }
    writeFileSync(join(outputDir, id + ".out"), text, "utf8");
  }
}

main();
