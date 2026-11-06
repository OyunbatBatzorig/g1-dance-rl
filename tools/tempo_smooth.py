"""Replace local speed edits with one smooth tempo curve, so the dance never jumps between fast and slow.

Usage: python tempo_smooth.py edits.json base_30hz.csv out_edits.json out_30hz.csv [--vmax 3.5] [--sigma 0.4]

1. Drops the "speed" ops from edits.json and applies the rest (pose, smooth, ...) to the base CSV.
2. Measures hand speed (forward kinematics). Where a hand is faster than --vmax, that part needs slowing by v / vmax.
3. That need is widened (max over +-0.2 s) and blurred with a Gaussian of --sigma seconds, then scaled so every
   peak still gets enough slowing. The result is a slow-factor curve that changes gently over ~1 s.
4. Writes it as one "tempo" op (knots every 0.1 s), so the editor page and ref_edit.py can apply and show it."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_edit as R
from motion_check import SDK, load_urdf, quat_xyzw, axis_angle

ap = argparse.ArgumentParser()
ap.add_argument("edits"); ap.add_argument("base"); ap.add_argument("out_edits"); ap.add_argument("out_csv")
ap.add_argument("--vmax", type=float, default=3.5, help="hand speed limit [m/s]")
ap.add_argument("--sigma", type=float, default=0.4, help="tempo blur [s]: larger = gentler tempo changes")
a = ap.parse_args()

U = next(p for p in [os.path.expanduser("~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
                     "/home/claude/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"] if os.path.exists(p))
J, _ = load_urdf(U)
order, todo, known = [], dict(J), {"pelvis"}
while todo:
    for c, j in list(todo.items()):
        if j["parent"] in known:
            order.append(c); known.add(c); del todo[c]
IDX = {n: i for i, n in enumerate(SDK)}


def hand_speed(d, fps=30):
    P = []
    for f in range(len(d)):
        T = {"pelvis": (d[f, :3], quat_xyzw(*d[f, 3:7]))}
        for c in order:
            j = J[c]; P0, R0 = T[j["parent"]]; Rc = R0 @ j["R"]
            if j["type"] == "revolute":
                Rc = Rc @ axis_angle(j["axis"], d[f, 7 + IDX[j["name"]]])
            T[c] = (P0 + R0 @ j["xyz"], Rc)
        P.append([T["left_rubber_hand"][0], T["right_rubber_hand"][0]])
    return (np.linalg.norm(np.gradient(np.array(P), axis=0), axis=2) * fps).max(1)


def tempo_stats(src, fps=30):
    rate = (np.gradient(src) * fps)[:-3]  # dance seconds per real second (1 = original tempo); end = still pose
    return rate.min(), np.abs(np.gradient(rate)).max() * fps  # slowest tempo, fastest tempo change [1/s]


spec = json.load(open(a.edits))
ops = [o for o in spec["ops"] if o["op"] not in ("speed", "tempo")]
base = np.loadtxt(a.base, delimiter=",")
d, _, err = R.apply(base, ops)
fps, n = 30, len(d)
v = hand_speed(d)
need = np.maximum(1.0, v / a.vmax)
w = int(0.2 * fps)
env = np.array([need[max(0, i - w):i + w + 1].max() for i in range(n)])
sf = a.sigma * fps
k = np.exp(-0.5 * (np.arange(-int(3 * sf), int(3 * sf) + 1) / sf) ** 2); k /= k.sum()
g = np.convolve(np.pad(env - 1, len(k) // 2, mode="edge"), k, mode="valid")
c = max(((need - 1) / np.maximum(g, 1e-9))[need > 1].max(), 1.0) if (need > 1).any() else 0.0
slow = 1 + c * g
t = np.arange(n) / fps
kt = np.arange(0, t[-1] + 1e-9, 0.1)
knots = [[round(float(x), 2), round(float(np.interp(x, t, slow)), 4)] for x in kt]
ops_out = ops + [{"op": "tempo", "vmax": a.vmax, "sigma": a.sigma, "knots": knots}]
d2, src2, err2 = R.apply(base, ops_out)
v2 = hand_speed(d2)
json.dump({"base": spec.get("base"), "ops": ops_out}, open(a.out_edits, "w"), indent=1)
np.savetxt(a.out_csv, d2, delimiter=",", fmt="%.6f")
_, src_old, _ = R.apply(base, spec["ops"])
d_old, _, _ = R.apply(base, spec["ops"])
for name, dd, ss in (("input edits ", d_old, src_old), ("smooth tempo", d2, src2)):
    vv = hand_speed(dd); lo, ch = tempo_stats(ss)
    print(f"{name}: {len(dd) / fps:.2f} s | hand max {vv.max():.2f} m/s, frames > {a.vmax}: {(vv > a.vmax).sum()}"
          f" | slowest tempo {lo:.2f}x, fastest tempo change {ch:.2f} per s")
print(f"wrote {a.out_edits} and {a.out_csv}; slow factor max {slow.max():.2f} at {t[slow.argmax()]:.2f} s")
