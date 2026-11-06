// Edit operations for a 30 Hz G1 reference (36 columns: root pos 3, quat xyzw 4, 29 joints in SDK order).
// tools/ref_edit.py is a line-by-line port of this file; keep the two in step.
(function (root) {
  const W = 36;
  const SDK = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"];
  const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
  const PARTS = {
    everything: range(0, 36), joints: range(7, 36), root: range(0, 7),
    legs: range(7, 19), left_leg: range(7, 13), right_leg: range(13, 19),
    waist: range(19, 22), arms: range(22, 36), left_arm: range(22, 29), right_arm: range(29, 36),
  };

  function clone(m) { return { fps: m.fps, n: m.n, data: Float64Array.from(m.data), src: Float64Array.from(m.src) }; }

  function fixQuats(m) {
    const d = m.data;
    for (let f = 0; f < m.n; f++) {
      const o = f * W + 3;
      let s = Math.hypot(d[o], d[o + 1], d[o + 2], d[o + 3]) || 1;
      if (f > 0) { const p = o - W; if (d[o] * d[p] + d[o + 1] * d[p + 1] + d[o + 2] * d[p + 2] + d[o + 3] * d[p + 3] < 0) s = -s; }
      for (let k = 0; k < 4; k++) d[o + k] /= s;
    }
  }

  const idx = (m, t) => Math.max(0, Math.min(m.n - 1, Math.round(t * m.fps)));
  // 1 inside [i0, i1], raised-cosine ramps of r frames outside it
  function weight(f, i0, i1, r) {
    if (f >= i0 && f <= i1) return 1;
    if (r <= 0) return 0;
    if (f < i0 && f > i0 - r) return 0.5 - 0.5 * Math.cos(Math.PI * (f - (i0 - r)) / r);
    if (f > i1 && f < i1 + r) return 0.5 - 0.5 * Math.cos(Math.PI * ((i1 + r) - f) / r);
    return 0;
  }
  // velocity per frame, central difference
  function vel(m, f, c) {
    const a = Math.max(0, f - 1), b = Math.min(m.n - 1, f + 1);
    return b > a ? (m.data[b * W + c] - m.data[a * W + c]) / (b - a) : 0;
  }
  function hermite(p0, v0, p1, v1, T, s) {
    const s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * T * v1;
  }
  function sample(m, tau) {
    const k = Math.max(0, Math.min(m.n - 1, Math.floor(tau))), k1 = Math.min(m.n - 1, k + 1), u = tau - k;
    const out = new Float64Array(W);
    for (let c = 0; c < W; c++) out[c] = (1 - u) * m.data[k * W + c] + u * m.data[k1 * W + c];
    return { row: out, src: (1 - u) * m.src[k] + u * m.src[k1] };
  }
  function build(fps, rows, src) {
    const n = rows.length, data = new Float64Array(n * W);
    rows.forEach((r, f) => data.set(r, f * W));
    return { fps, n, data, src: Float64Array.from(src) };
  }
  const rowOf = (m, f) => m.data.slice(f * W, f * W + W);

  function needRange(m, op, minFrames) {
    const i0 = idx(m, op.t0), i1 = idx(m, op.t1);
    if (i1 - i0 < minFrames) throw new Error(`range ${op.t0}-${op.t1} s is too short`);
    return [i0, i1];
  }

  const OPS = {
    // Gaussian low-pass of one body part inside the range
    smooth(m, op) {
      const [i0, i1] = needRange(m, op, 1), cols = PARTS[op.part], sf = op.sigma * m.fps, r = Math.round((op.ramp ?? 0.2) * m.fps);
      const rad = Math.ceil(3 * sf), ker = [];
      for (let k = -rad; k <= rad; k++) ker.push(Math.exp(-0.5 * (k / sf) ** 2));
      const ks = ker.reduce((a, b) => a + b, 0), old = m.data.slice();
      for (let f = Math.max(0, i0 - r); f <= Math.min(m.n - 1, i1 + r); f++) {
        const w = weight(f, i0, i1, r);
        if (w <= 0) continue;
        for (const c of cols) {
          let acc = 0;
          for (let k = -rad; k <= rad; k++) acc += ker[k + rad] * old[Math.max(0, Math.min(m.n - 1, f + k)) * W + c];
          m.data[f * W + c] = old[f * W + c] + w * (acc / ks - old[f * W + c]);
        }
      }
    },
    // pull one body part towards a smooth path between the poses at t0 and t1 (amount 1 = only that path)
    reduce(m, op) {
      const [i0, i1] = needRange(m, op, 2), cols = PARTS[op.part], T = i1 - i0, a = op.amount;
      for (const c of cols) {
        const p0 = m.data[i0 * W + c], p1 = m.data[i1 * W + c], v0 = vel(m, i0, c), v1 = vel(m, i1, c);
        for (let f = i0 + 1; f < i1; f++) {
          const h = hermite(p0, v0, p1, v1, T, (f - i0) / T), x = m.data[f * W + c];
          m.data[f * W + c] = h + (1 - a) * (x - h);
        }
      }
    },
    // add delta (rad) to one joint inside the range, ramped in and out
    offset(m, op) {
      const i0 = idx(m, op.t0), i1 = idx(m, op.t1), c = 7 + SDK.indexOf(op.joint), r = Math.round((op.ramp ?? 0.3) * m.fps);
      if (c < 7) throw new Error(`unknown joint ${op.joint}`);
      for (let f = Math.max(0, i0 - r); f <= Math.min(m.n - 1, i1 + r); f++) m.data[f * W + c] += weight(f, i0, i1, r) * op.delta;
    },
    // move a hand or foot: several joints get the deltas (rad) found by IK in the editor, ramped in and out
    pose(m, op) {
      const i0 = idx(m, op.t0), i1 = idx(m, op.t1), r = Math.round((op.ramp ?? 0.3) * m.fps);
      for (const [name, delta] of Object.entries(op.deltas)) {
        const c = 7 + SDK.indexOf(name);
        if (c < 7) throw new Error(`unknown joint ${name}`);
        for (let f = Math.max(0, i0 - r); f <= Math.min(m.n - 1, i1 + r); f++) m.data[f * W + c] += weight(f, i0, i1, r) * delta;
      }
    },
    // play the range `factor` times slower (factor < 1 = faster), easing in and out of the new speed
    speed(m, op) {
      const [i0, i1] = needRange(m, op, 2), L = i1 - i0, Nn = Math.max(1, Math.round(L * op.factor));
      const r = Math.min(Math.round((op.ramp ?? 0.3) * m.fps), Math.floor(Nn / 3));
      const ss = x => x * x * (3 - 2 * x), b = [];
      for (let u = 0; u < Nn; u++) { const c = u + 0.5; b.push(r > 0 && c < r ? ss(c / r) : r > 0 && c > Nn - r ? ss((Nn - c) / r) : 1); }
      const B = b.reduce((x, y) => x + y, 0), a = (Nn - L) / B;
      if (a > 0.95) throw new Error(`range ${op.t0}-${op.t1} s is too short to slow down x${op.factor}; make it longer or the ramp shorter`);
      const rows = [], src = [];
      for (let f = 0; f < i0; f++) { rows.push(rowOf(m, f)); src.push(m.src[f]); }
      let tau = i0;
      for (let u = 0; u < Nn; u++) { const s = sample(m, tau); rows.push(s.row); src.push(s.src); tau += 1 - a * b[u]; }
      for (let f = i1; f < m.n; f++) { rows.push(rowOf(m, f)); src.push(m.src[f]); }
      return build(m.fps, rows, src);
    },
    // smooth tempo change over the whole dance: knots [[t, slow], ...] (slow 1.2 = 1.2x slower there, linear
    // between knots, 1 outside them). Made by tools/tempo_smooth.py from the hand speed, so the tempo never jumps.
    tempo(m, op) {
      const K = op.knots, n = m.n, fps = m.fps;
      const slow = s => {
        if (!K.length || s <= K[0][0] || s >= K[K.length - 1][0]) return 1;
        let i = 0; while (K[i + 1][0] < s) i++;
        const [t0, a] = K[i], [t1, b] = K[i + 1]; return a + (b - a) * (s - t0) / (t1 - t0);
      };
      const rows = [], src = [];
      let tau = 0, last = 0;
      while (tau <= n - 1 + 1e-9) { const s = sample(m, tau); rows.push(s.row); src.push(s.src); last = tau; tau += 1 / slow(tau / fps); }
      if (last < n - 1 - 1e-6) { rows.push(rowOf(m, n - 1)); src.push(m.src[n - 1]); }
      return build(fps, rows, src);
    },
    // delete the range; a smooth bridge spans ramp seconds on both sides
    cut(m, op) {
      const [i0, i1] = needRange(m, op, 1), r = Math.round((op.ramp ?? 0.2) * m.fps);
      const a = Math.max(0, i0 - r), b = Math.min(m.n - 1, i1 + r), Mb = Math.max(1, (i0 - a) + (b - i1));
      const rows = [], src = [];
      for (let f = 0; f <= a; f++) { rows.push(rowOf(m, f)); src.push(m.src[f]); }
      for (let k = 1; k < Mb; k++) {
        const s = k / Mb, row = new Float64Array(W);
        for (let c = 0; c < W; c++) row[c] = hermite(m.data[a * W + c], vel(m, a, c), m.data[b * W + c], vel(m, b, c), Mb, s);
        rows.push(row); src.push((1 - s) * m.src[a] + s * m.src[b]);
      }
      for (let f = b; f < m.n; f++) { rows.push(rowOf(m, f)); src.push(m.src[f]); }
      return build(m.fps, rows, src);
    },
  };

  function apply(base, ops) {
    let m = clone(base);
    fixQuats(m);
    const errors = [];
    ops.forEach((op, i) => {
      if (op.off) return;
      try { const r = OPS[op.op](m, op); if (r) m = r; fixQuats(m); }
      catch (e) { errors.push({ i, msg: e.message }); }
    });
    return { motion: m, errors };
  }

  function fromRows(rows, fps) {
    const n = rows.length, data = new Float64Array(n * W), src = new Float64Array(n);
    rows.forEach((r, f) => { data.set(r, f * W); src[f] = f / fps; });
    return { fps, n, data, src };
  }
  function toCSV(m) {
    const lines = [];
    for (let f = 0; f < m.n; f++) lines.push(Array.from(m.data.subarray(f * W, f * W + W), v => v.toFixed(6)).join(","));
    return lines.join("\n") + "\n";
  }

  const API = { W, SDK, PARTS, OPS, apply, fromRows, toCSV, clone };
  if (typeof module !== "undefined" && module.exports) module.exports = API; else root.RefEdit = API;
})(typeof self !== "undefined" ? self : this);
