"""Apply edits from the Schuhplattler reference editor to a 30 Hz G1 CSV, then print a short check.

Usage: python ref_edit.py edits.json in_30hz.csv out_30hz.csv

edits.json is what the editor's "Copy edits" button gives: {"base": "...", "ops": [...]}.
Ops (times in seconds on the timeline as it is after the previous ops, same as in the editor):
  smooth  t0 t1 part sigma [ramp]   Gaussian low-pass of a body part
  reduce  t0 t1 part amount         pull a body part towards a smooth path between the poses at t0 and t1
  offset  t0 t1 joint delta [ramp]  add delta rad to one joint
  pose    t0 t1 deltas [ramp]       add {joint: delta rad} found by dragging a hand/foot (IK) in the editor
  speed   t0 t1 factor [ramp]       play the range factor times slower (<1 = faster)
  tempo   knots [[t, slow], ...]    smooth tempo curve over the dance (made by tools/tempo_smooth.py)
  cut     t0 t1 [ramp]              delete the range with a smooth bridge
This is a line-by-line port of edit_core.js in the editor page; keep the two in step.
Columns: root pos (3), quat xyzw (4), 29 joints in SDK order."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_check import SDK, load_urdf, quat_xyzw, axis_angle, SOLE

W = 36
PARTS = {
    "everything": range(0, 36), "joints": range(7, 36), "root": range(0, 7),
    "legs": range(7, 19), "left_leg": range(7, 13), "right_leg": range(13, 19),
    "waist": range(19, 22), "arms": range(22, 36), "left_arm": range(22, 29), "right_arm": range(29, 36),
}


def fix_quats(d):
    for f in range(len(d)):
        s = np.linalg.norm(d[f, 3:7]) or 1.0
        if f > 0 and d[f, 3:7] @ d[f - 1, 3:7] < 0:
            s = -s
        d[f, 3:7] /= s


def idx(d, fps, t):
    return int(max(0, min(len(d) - 1, np.floor(t * fps + 0.5))))  # JS Math.round


def jround(x):
    return int(np.floor(x + 0.5))


def weight(f, i0, i1, r):
    if i0 <= f <= i1:
        return 1.0
    if r <= 0:
        return 0.0
    if i0 - r < f < i0:
        return 0.5 - 0.5 * np.cos(np.pi * (f - (i0 - r)) / r)
    if i1 < f < i1 + r:
        return 0.5 - 0.5 * np.cos(np.pi * ((i1 + r) - f) / r)
    return 0.0


def vel(d, f, c):
    a, b = max(0, f - 1), min(len(d) - 1, f + 1)
    return (d[b, c] - d[a, c]) / (b - a) if b > a else 0.0


def hermite(p0, v0, p1, v1, T, s):
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * T * v1


def need_range(d, fps, op, min_frames):
    i0, i1 = idx(d, fps, op["t0"]), idx(d, fps, op["t1"])
    if i1 - i0 < min_frames:
        raise ValueError(f"range {op['t0']}-{op['t1']} s is too short")
    return i0, i1


def op_smooth(d, src, fps, op):
    i0, i1 = need_range(d, fps, op, 1)
    sf, r = op["sigma"] * fps, jround(op.get("ramp", 0.2) * fps)
    rad = int(np.ceil(3 * sf))
    ker = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sf) ** 2)
    ks, old, n = ker.sum(), d.copy(), len(d)
    for f in range(max(0, i0 - r), min(n - 1, i1 + r) + 1):
        w = weight(f, i0, i1, r)
        if w <= 0:
            continue
        rows = np.clip(np.arange(f - rad, f + rad + 1), 0, n - 1)
        for c in PARTS[op["part"]]:
            acc = ker @ old[rows, c]
            d[f, c] = old[f, c] + w * (acc / ks - old[f, c])
    return d, src


def op_reduce(d, src, fps, op):
    i0, i1 = need_range(d, fps, op, 2)
    T, a = i1 - i0, op["amount"]
    for c in PARTS[op["part"]]:
        p0, p1, v0, v1 = d[i0, c], d[i1, c], vel(d, i0, c), vel(d, i1, c)
        for f in range(i0 + 1, i1):
            h = hermite(p0, v0, p1, v1, T, (f - i0) / T)
            d[f, c] = h + (1 - a) * (d[f, c] - h)
    return d, src


def op_offset(d, src, fps, op):
    i0, i1 = idx(d, fps, op["t0"]), idx(d, fps, op["t1"])
    c, r = 7 + SDK.index(op["joint"]), jround(op.get("ramp", 0.3) * fps)
    for f in range(max(0, i0 - r), min(len(d) - 1, i1 + r) + 1):
        d[f, c] += weight(f, i0, i1, r) * op["delta"]
    return d, src


def op_pose(d, src, fps, op):
    i0, i1 = idx(d, fps, op["t0"]), idx(d, fps, op["t1"])
    r = jround(op.get("ramp", 0.3) * fps)
    for name, delta in op["deltas"].items():
        c = 7 + SDK.index(name)
        for f in range(max(0, i0 - r), min(len(d) - 1, i1 + r) + 1):
            d[f, c] += weight(f, i0, i1, r) * delta
    return d, src


def sample(d, src, tau):
    n = len(d)
    k = int(max(0, min(n - 1, np.floor(tau))))
    k1, u = min(n - 1, k + 1), tau - k
    return (1 - u) * d[k] + u * d[k1], (1 - u) * src[k] + u * src[k1]


def op_speed(d, src, fps, op):
    i0, i1 = need_range(d, fps, op, 2)
    L = i1 - i0
    Nn = max(1, jround(L * op["factor"]))
    r = min(jround(op.get("ramp", 0.3) * fps), Nn // 3)
    ss = lambda x: x * x * (3 - 2 * x)
    b = []
    for u in range(Nn):
        c = u + 0.5
        b.append(ss(c / r) if r > 0 and c < r else ss((Nn - c) / r) if r > 0 and c > Nn - r else 1.0)
    a = (Nn - L) / sum(b)
    if a > 0.95:
        raise ValueError(f"range {op['t0']}-{op['t1']} s is too short to slow down x{op['factor']}")
    rows, srcs, tau = list(d[:i0]), list(src[:i0]), float(i0)
    for u in range(Nn):
        row, s = sample(d, src, tau)
        rows.append(row); srcs.append(s)
        tau += 1 - a * b[u]
    rows += list(d[i1:]); srcs += list(src[i1:])
    return np.array(rows), np.array(srcs)


def op_tempo(d, src, fps, op):
    K, n = op["knots"], len(d)

    def slow(s):
        if not K or s <= K[0][0] or s >= K[-1][0]:
            return 1.0
        i = 0
        while K[i + 1][0] < s:
            i += 1
        (t0, a), (t1, b) = K[i], K[i + 1]
        return a + (b - a) * (s - t0) / (t1 - t0)

    rows, srcs, tau, last = [], [], 0.0, 0.0
    while tau <= n - 1 + 1e-9:
        row, s = sample(d, src, tau)
        rows.append(row); srcs.append(s)
        last = tau
        tau += 1 / slow(tau / fps)
    if last < n - 1 - 1e-6:
        rows.append(d[n - 1]); srcs.append(src[n - 1])
    return np.array(rows), np.array(srcs)


def op_cut(d, src, fps, op):
    i0, i1 = need_range(d, fps, op, 1)
    r, n = jround(op.get("ramp", 0.2) * fps), len(d)
    a, b = max(0, i0 - r), min(n - 1, i1 + r)
    Mb = max(1, (i0 - a) + (b - i1))
    va = np.array([vel(d, a, c) for c in range(W)]); vb = np.array([vel(d, b, c) for c in range(W)])
    rows, srcs = list(d[:a + 1]), list(src[:a + 1])
    for k in range(1, Mb):
        s = k / Mb
        rows.append(hermite(d[a], va, d[b], vb, Mb, s)); srcs.append((1 - s) * src[a] + s * src[b])
    rows += list(d[b:]); srcs += list(src[b:])
    return np.array(rows), np.array(srcs)


OPS = dict(smooth=op_smooth, reduce=op_reduce, offset=op_offset, pose=op_pose, speed=op_speed, tempo=op_tempo, cut=op_cut)


def apply(d, ops, fps=30):
    d = d.astype(float).copy()
    src = np.arange(len(d)) / fps
    fix_quats(d)
    errors = []
    for i, op in enumerate(ops):
        if op.get("off"):
            continue
        try:
            d, src = OPS[op["op"]](d, src, fps, op)
            fix_quats(d)
        except ValueError as e:
            errors.append((i, str(e)))
    return d, src, errors


def report(d, fps, label):
    U = next(p for p in [os.path.expanduser("~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
                         "/home/claude/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"] if os.path.exists(p))
    J, LIM = load_urdf(U)
    order, todo, known = [], dict(J), {"pelvis"}
    while todo:
        for c, j in list(todo.items()):
            if j["parent"] in known:
                order.append(c); known.add(c); del todo[c]
    I = {n: i for i, n in enumerate(SDK)}
    ee = ["left_rubber_hand", "right_rubber_hand"]
    P, soles = [], {"left": [], "right": []}
    for f in range(len(d)):
        T = {"pelvis": (d[f, :3], quat_xyzw(*d[f, 3:7]))}
        for c in order:
            j = J[c]; P0, R0 = T[j["parent"]]; Rc = R0 @ j["R"]
            if j["type"] == "revolute":
                Rc = Rc @ axis_angle(j["axis"], d[f, 7 + I[j["name"]]])
            T[c] = (P0 + R0 @ j["xyz"], Rc)
        P.append([T[e][0] for e in ee])
        for s in soles:
            p, R = T[f"{s}_ankle_roll_link"]; soles[s].append(p + (R @ SOLE.T).T)
    hv = np.linalg.norm(np.gradient(np.array(P), axis=0), axis=2) * fps
    qd = np.abs(np.gradient(d[:, 7:], axis=0)) * fps
    slide, low = [], min(np.array(soles[s])[:, :, 2].min() for s in soles)
    for s in soles:
        a3 = np.array(soles[s]); on = a3[:, :, 2].min(1) < 0.025
        v = np.linalg.norm(np.diff(a3[:, :, :2].mean(1), axis=0), axis=1) * fps
        slide.append(v[on[1:] & on[:-1]])
    sl = np.concatenate(slide)
    lim = [n for i, n in enumerate(SDK) if n in LIM and ((d[:, 7 + i] < LIM[n][0]) | (d[:, 7 + i] > LIM[n][1])).any()]
    print(f"{label}: {len(d)} frames = {len(d) / fps:.2f} s | hand speed max L {hv[:, 0].max():.2f} R {hv[:, 1].max():.2f} m/s,"
          f" p95 {np.percentile(hv, 95):.2f} | joint speed max {qd.max():.1f} rad/s ({SDK[qd.max(0).argmax()]})"
          f" | stance-foot slide mean {sl.mean() * 100:.0f} cm/s, p95 {np.percentile(sl, 95) * 100:.0f}"
          f" | lowest sole {low * 100:.1f} cm | joints past limits: {', '.join(lim) or 'none'}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    spec = json.load(open(sys.argv[1]))
    ops = spec["ops"] if isinstance(spec, dict) else spec
    m = np.loadtxt(sys.argv[2], delimiter=",")
    d, src, errors = apply(m, ops)
    for i, e in errors:
        print(f"edit {i + 1} skipped: {e}")
    np.savetxt(sys.argv[3], d, delimiter=",", fmt="%.6f")
    print(f"wrote {sys.argv[3]} ({len(ops) - len(errors)} edits applied)")
    report(m, 30, "before")
    report(d, 30, "after ")


if __name__ == "__main__":
    main()
