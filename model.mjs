// marrow/src/model.mjs
// A per-user scorer. No training framework, no dependencies: robust statistics
// over behavioural features, one fixed threshold per profile.
//
// Why this shape: a stranger's session must not open an enrolled profile, so the
// model has to be narrow around ONE person's habits rather than a classifier that
// separates "humans" from "not humans". Median/MAD per feature is exactly that:
// it describes the enrolled user's own spread and rejects everything outside it.
//
// The threshold is computed during enrolment and written into profile.json. It is
// never re-fitted on the evaluation recordings: evaluate.mjs only reads it.

import { BASE_FEATURES, SOURCE_FEATURES } from "./features.mjs";

const EPS = 1e-9;

const quantile = (arr, q) => {
  const a = [...arr].sort((x, y) => x - y);
  if (!a.length) return NaN;
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? a[lo] : a[lo] + (a[hi] - a[lo]) * (pos - lo);
};

// Monotone transform so that heavy-tailed rates/timings behave. Proportions stay
// as-is; counts and durations are compressed with log1p.
const LOG_FEATURES = new Set([
  "moveRate", "medSpeed", "p90Speed", "clickRate", "hoverBeforeClick",
  "ikiMed", "burstLen",
]);
export const transform = (name, v) => {
  const x = Number.isFinite(v) ? Math.max(0, v) : 0;
  return LOG_FEATURES.has(name) ? Math.log1p(x) : x;
};

export function featuresFor(set) {
  return set === "full" ? [...BASE_FEATURES, ...SOURCE_FEATURES] : [...BASE_FEATURES];
}

/**
 * Train one profile from the windows of the enrolled user's own sessions.
 * Returns { featureSet, features: [{name, median, sigma}], threshold, ... }.
 */
export function trainProfile(windows, { featureSet = "base", quantile: q = 0.95, slack = 1.2 } = {}) {
  const names = featuresFor(featureSet);
  const features = names.map((name) => {
    const vals = windows.map((w) => transform(name, w[name]));
    const med = quantile(vals, 0.5);
    const mad = quantile(vals.map((v) => Math.abs(v - med)), 0.5);
    // 1.4826 * MAD is a normal-consistent scale estimate; the floor keeps a
    // feature that never varies in enrolment from becoming an infinite veto.
    // Scale floor: a session-to-session MAD of a few milliseconds would otherwise
    // turn an ordinary tired day into a z-score of six. 4% of the enrolled median
    // keeps every feature's tolerance proportional to its own size.
    const sigma = Math.max(1.4826 * mad, 0.015, 0.04 * Math.abs(med));
    return { name, median: med, sigma };
  });

  const profile = { featureSet, features, threshold: null, calibration: {} };
  const dists = windows.map((w) => distance(profile, w).distance);
  const cut = quantile(dists, q);
  profile.threshold = Math.max(1.0, cut * slack);
  profile.calibration = {
    method: `owner-window distance q${q} * ${slack}`,
    windows: windows.length,
    q,
    slack,
    ownerDistance: { median: quantile(dists, 0.5), p90: quantile(dists, 0.9), p99: cut, max: Math.max(...dists) },
  };
  return profile;
}

/** Per-window distance from the profile: mean of |z|, with z clipped at 6. */
export function distance(profile, window) {
  const parts = [];
  for (const f of profile.features) {
    const v = transform(f.name, window[f.name]);
    const z = Math.max(-6, Math.min(6, (v - f.median) / f.sigma));
    parts.push(Math.abs(z));
  }
  // The three largest deviations decide the score. A person is recognised by the
  // handful of habits that are hardest to copy; averaging all features would let
  // six ordinary ones drown out the two that actually separate a mimic.
  const top = [...parts].sort((a, b) => b - a).slice(0, 3);
  const rms = Math.sqrt(top.reduce((a, b) => a + b * b, 0) / top.length);
  return { distance: rms, parts };
}

/** A recording is scored by the median window distance: one odd window cannot
 *  decide the session, and every window is judged by the same fixed threshold. */
export function scoreRecording(profile, windows) {
  if (!windows.length) return { windows: 0, median: Infinity, accepted: false, distances: [] };
  const ds = windows.map((w) => distance(profile, w).distance);
  const med = quantile(ds, 0.5);
  return {
    windows: ds.length,
    median: med,
    distances: ds,
    accepted: med <= profile.threshold,
    margin: profile.threshold - med,
  };
}
