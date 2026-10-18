# ZMP (cart-table: no angular momentum) vs support polygon, per frame, for a 30 Hz CSV
import sys, numpy as np
sys.path.insert(0, '/mnt/project-files/g1_pilot/tools')
from motion_check import SDK, load_urdf, quat_xyzw, foot_sole
import xml.etree.ElementTree as ET
import os
U = os.path.expanduser(os.environ.get("G1_URDF", "~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"))  # G1 URDF from unitreerobotics/unitree_ros
J, LIM = load_urdf(U)
# full FK for CoM
MASS = {}
for l in ET.parse(U).getroot().findall('link'):
    i = l.find('inertial')
    if i is not None:
        o = i.find('origin'); MASS[l.get('name')] = (float(i.find('mass').get('value')), np.array([float(v) for v in (o.get('xyz') if o is not None else '0 0 0').split()]))
from motion_check import axis_angle
ORDER, todo, known = [], dict(J), {'pelvis'}
while todo:
    for c, j in list(todo.items()):
        if j['parent'] in known: ORDER.append(c); known.add(c); del todo[c]
IDX = {n: i for i, n in enumerate(SDK)}
def com(p, R, q):
    T = {'pelvis': (p, R)}
    for c in ORDER:
        j = J[c]; P0, R0 = T[j['parent']]; Rc = R0 @ j['R']
        if j['type'] == 'revolute': Rc = Rc @ axis_angle(j['axis'], q[IDX[j['name']]])
        T[c] = (P0 + R0 @ j['xyz'], Rc)
    s = sum(m * (T[n][0] + T[n][1] @ o) for n, (m, o) in MASS.items() if n in T); return s / sum(m for n, (m, o) in MASS.items() if n in T)
def hull_dist(pt, pts):  # signed distance: >0 outside convex hull (2D)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    cr = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in pts[::-1]:
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    H = np.array(lo[:-1] + up[:-1]); d = []
    for k in range(len(H)):
        a, b = H[k], H[(k+1) % len(H)]; e = b - a; n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        d.append((pt - a) @ n)
    return max(d)
from motion_check import load_motion
fps, rp, rq, qq = load_motion(sys.argv[1], 30)
if sys.argv[1].endswith('.npz'): rp, rq, qq = rp[3:], rq[3:], qq[3:]
m = np.c_[rp, rq, qq]; N = len(m)
c = np.array([com(m[f, :3], quat_xyzw(*m[f, 3:7]), m[f, 7:]) for f in range(N)])
w = max(1, int(round(fps / 6))) * 2 + 1; k = np.ones(w) / w; cs = np.stack([np.convolve(np.pad(c[:, i], w // 2, mode='edge'), k, mode='valid') for i in range(3)], 1)
acc = np.gradient(np.gradient(cs, axis=0), axis=0) * fps * fps
zmp = cs[:, :2] - cs[:, 2:3] / 9.81 * acc[:, :2]
soles = {s: np.array([foot_sole(J, m[f, :3], quat_xyzw(*m[f, 3:7]), dict(zip(SDK, m[f, 7:])), s) for f in range(N)]) for s in ('left', 'right')}
floor = np.percentile(np.minimum(soles['left'][:, :, 2].min(1), soles['right'][:, :, 2].min(1)), 5) - 0.005  # robust floor
out, sup = np.zeros(N), []
for f in range(N):
    pts = [soles[s][f] for s in soles if soles[s][f][:, 2].min() < floor + 0.025]
    if not pts: out[f] = np.nan; sup.append('-'); continue
    sup.append('B' if len(pts) == 2 else ('L' if soles['left'][f][:, 2].min() < floor + 0.025 else 'R'))
    out[f] = hull_dist(zmp[f], np.vstack(pts)[:, :2])
o = np.nan_to_num(out, nan=0) * 100
print(f"{sys.argv[1]}: ZMP outside support >2 cm: {(o > 2).sum()} frames ({(o > 2).sum()/fps:.2f} s), >5 cm: {(o > 5).sum()}, max {o.max():.1f} cm; static CoM outside >2cm: ", end='')
cout = np.array([0 if sup[f] == '-' else hull_dist(cs[f, :2], np.vstack([soles[s][f] for s in soles if soles[s][f][:, 2].min() < floor + 0.025])[:, :2]) for f in range(N)]) * 100
print((cout > 2).sum())
segs, f = [], 0
while f < N:
    if o[f] > 2:
        g = f
        while g < N and o[g] > 2: g += 1
        segs.append((f / fps, g / fps, o[f:g].max(), ''.join(sup[f:g])[:12])); f = g
    else: f += 1
for s in (segs if len(sys.argv) < 3 else []): print(f"  {s[0]:.2f}-{s[1]:.2f}s max {s[2]:.1f} cm support {s[3]}")
np.save(sys.argv[1].split('/')[-1].rsplit('.',1)[0] + '_zmpout.npy', o)
