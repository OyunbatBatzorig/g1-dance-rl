"""Usage: python balance_fix.py in_30hz.csv out.csv [--pin_left T0 T1] [--report report.txt]

Make a retargeted G1 CSV statically more balanced without changing the dance:
  1. Optional foot edits before balancing (the feet are IK targets, so the legs follow):
     --warp SIDE "tn:to,tn:to,...": re-time one foot. At new time tn the foot takes its original pose of time to
       (piecewise linear between knots, unchanged outside them). Used to land a foot earlier or lift it later.
     --pin SIDE T0 T1 TREF: hold the foot at its original pose of time TREF between T0 and T1, blended in and out
       over --pin_blend s (no jumps). --pin_left T0 T1 is the old hard pin (pose of T0, no blend).
  2. Whenever only one foot is on the floor, shift the pelvis sideways (in the heading frame) so the
     whole-body centre of mass is above the stance foot's sole centre. The shift is ramped in during the
     preceding double support (weight transfer) and out after the last single support.
  3. Leg joints are re-solved with IK so both ankles keep their original world pose (feet don't slide,
     swing foot path is unchanged). Arms, waist, head and root height/orientation are untouched.
Columns: root pos (3), quat xyzw (4), 29 joints in SDK order (same as add_ending.py)."""
import argparse, os, sys
import numpy as np
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_check import SDK, load_urdf, quat_xyzw, axis_angle, SOLE

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--pin_left", nargs=2, type=float, default=None)
ap.add_argument("--warp", nargs=2, action="append", default=[], metavar=("SIDE", "KNOTS"))
ap.add_argument("--pin", nargs=4, action="append", default=[], metavar=("SIDE", "T0", "T1", "TREF"))
ap.add_argument("--pin_blend", type=float, default=0.2, help="blend time into and out of a --pin [s]")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--ramp", type=float, default=0.6, help="max weight-transfer time [s] in double support")
ap.add_argument("--contact", type=float, default=0.025, help="sole within this of the floor = on the floor [m]")
ap.add_argument("--margin", type=float, default=0.015, help="CoM may sit this far sideways from the sole centre [m]")
ap.add_argument("--max_shift", type=float, default=0.10, help="largest pelvis shift [m]")
ap.add_argument("--min_single", type=float, default=0.15, help="shorter single-support runs count as double support [s]")
ap.add_argument("--leg_smooth", type=int, default=2, help="passes of a 5-tap binomial low-pass on the re-solved leg joints")
ap.add_argument("--report", default=None)
args = ap.parse_args()

U = os.path.expanduser(os.environ.get("G1_URDF", "~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"))  # G1 URDF from unitreerobotics/unitree_ros
J, LIM = load_urdf(U)
MASS = {}
for l in ET.parse(U).getroot().findall("link"):
    i = l.find("inertial")
    if i is not None:
        o = i.find("origin")
        MASS[l.get("name")] = (float(i.find("mass").get("value")),
                               np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]))
ORDER = []  # links in parent-before-child order
todo = dict(J)
known = {"pelvis"}
while todo:
    for c, j in list(todo.items()):
        if j["parent"] in known:
            ORDER.append(c); known.add(c); del todo[c]

m = np.loadtxt(args.src, delimiter=",")
rp, rq, q = m[:, :3].copy(), m[:, 3:7].copy(), m[:, 7:].copy()
N, fps = len(m), args.fps
IDX = {n: i for i, n in enumerate(SDK)}
LEG = {s: [IDX[f"{s}_{n}_joint"] for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
       for s in ("left", "right")}


def fk(p, R, qv):
    T = {"pelvis": (p, R)}
    for c in ORDER:
        j = J[c]; P0, R0 = T[j["parent"]]
        P = P0 + R0 @ j["xyz"]; Rc = R0 @ j["R"]
        if j["type"] == "revolute":
            Rc = Rc @ axis_angle(j["axis"], qv[IDX[j["name"]]])
        T[c] = (P, Rc)
    return T


def com_of(T):
    s, mt = np.zeros(3), 0.0
    for ln, (mm, o) in MASS.items():
        if ln in T:
            s += mm * (T[ln][0] + T[ln][1] @ o); mt += mm
    return s / mt


def sole(T, side):
    P, R = T[f"{side}_ankle_roll_link"]
    return P + (R @ SOLE.T).T


def rot_err(Ra, Rb):  # rotation vector taking Rb to Ra
    # atan2 form: the arccos-of-trace form returns 0 for small angles once the trace rounds above 3
    # (CSV quaternions have 6 digits), which silently switched off the orientation part of the IK
    E = Ra @ Rb.T
    w = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s = np.linalg.norm(w)
    if s < 1e-12:
        return np.zeros(3)
    return np.arctan2(s, (np.trace(E) - 1) / 2) * w / s


def ik_leg(p, R, qv, side, Pt, Rt, iters=80):
    ids = LEG[side]; link = f"{side}_ankle_roll_link"
    lo = np.array([LIM[SDK[i]][0] for i in ids]); hi = np.array([LIM[SDK[i]][1] for i in ids])
    qv = qv.copy()
    for _ in range(iters):
        P, Rc = fk(p, R, qv)[link]
        e = np.r_[Pt - P, 0.3 * rot_err(Rt, Rc)]
        if np.linalg.norm(e) < 1e-6:
            break
        Jm = np.zeros((6, 6)); h = 1e-5
        for k, i in enumerate(ids):
            qq = qv.copy(); qq[i] += h
            P2, R2 = fk(p, R, qq)[link]
            Jm[:, k] = np.r_[(P2 - P) / h, 0.3 * rot_err(R2, Rc) / h]
        dq = np.linalg.solve(Jm.T @ Jm + np.diag([1e-4, 1e-4, 3e-3, 1e-4, 1e-4, 1e-4]), Jm.T @ e)
        dq = np.clip(dq, -0.15, 0.15)
        qv[ids] = np.clip(qv[ids] + dq, lo, hi)
    P, Rc = fk(p, R, qv)[link]
    return qv, np.linalg.norm(Pt - P)


Rr = np.array([quat_xyzw(*(x / np.linalg.norm(x))) for x in rq])
yaw = np.array([np.arctan2(R[1, 0], R[0, 0]) for R in Rr])
lat = np.stack([-np.sin(yaw), np.cos(yaw)], 1)  # heading-frame left axis in world xy

# ---- original feet, contacts, CoM -------------------------------------------------------------
T0 = [fk(rp[f], Rr[f], q[f]) for f in range(N)]
foot = {s: [T0[f][f"{s}_ankle_roll_link"] for f in range(N)] for s in ("left", "right")}
def pose_at(side, t):  # original foot pose at time t [s], interpolated between frames
    x = np.clip(t * fps, 0, N - 1); i = min(int(x), N - 2); a = x - i
    (P0, R0), (P1, R1) = T0[i][f"{side}_ankle_roll_link"], T0[i + 1][f"{side}_ankle_roll_link"]
    return blend_pose((P0, R0), (P1, R1), a)


def blend_pose(A, B, w):
    rv = rot_err(B[1], A[1]); ang = np.linalg.norm(rv)
    R = axis_angle(rv / ang, w * ang) @ A[1] if ang > 1e-9 else A[1]
    return ((1 - w) * A[0] + w * B[0], R)


def ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


for sd, knots in args.warp:
    kn = np.array([[float(v) for v in k.split(":")] for k in knots.split(",")])
    for f in range(N):
        tn = f / fps
        if kn[0, 0] <= tn <= kn[-1, 0]:
            foot[sd][f] = pose_at(sd, np.interp(tn, kn[:, 0], kn[:, 1]))
for sd, a0, a1, ar in args.pin:
    a0, a1, ar, b = float(a0), float(a1), float(ar), args.pin_blend
    P = pose_at(sd, ar)
    for f in range(N):
        tn = f / fps
        w = ss((tn - (a0 - b)) / b) if tn < a0 else (1.0 if tn <= a1 else ss(1 - (tn - a1) / b))
        if w > 0:
            foot[sd][f] = blend_pose(foot[sd][f], P, w)
if args.pin_left:
    k0, k1 = int(round(args.pin_left[0] * fps)), int(round(args.pin_left[1] * fps))
    for f in range(k0, k1 + 1):
        foot["left"][f] = foot["left"][k0]
soles = {s: np.array([(foot[s][f][0] + (foot[s][f][1] @ SOLE.T).T)[:, 2].min() for f in range(N)]) for s in foot}
floor = min(soles["left"].min(), soles["right"].min())
down = {s: soles[s] < floor + args.contact for s in soles}
phase = np.where(down["left"] & down["right"], 0, np.where(down["left"], 1, np.where(down["right"], 2, 3)))
f, mn = 0, int(round(args.min_single * fps))
while f < N:  # drop single-support blips (contact flicker)
    g = f
    while g < N and phase[g] == phase[f]:
        g += 1
    if phase[f] in (1, 2) and g - f < mn:
        phase[f:g] = 0
    f = g
com0 = np.array([com_of(T) for T in T0])


def stance_centre(f, side):
    P, R = foot[side][f]
    return (P + (R @ SOLE.T).T).mean(0)


# lateral shift wanted on single-support frames: CoM above the stance sole centre
want = np.full(N, np.nan)
for f in range(N):
    if phase[f] in (1, 2):
        side = "left" if phase[f] == 1 else "right"
        e = (stance_centre(f, side)[:2] - com0[f, :2]) @ lat[f]
        want[f] = np.sign(e) * max(abs(e) - args.margin, 0.0)
# fill double support / flight: hold the previous value, then smoothstep to the next one over <= ramp s
nr = int(round(args.ramp * fps))
known_idx = np.where(~np.isnan(want))[0]
shift = np.zeros(N)
if len(known_idx):
    shift[known_idx] = want[known_idx]
    gaps, f = [], 0
    while f < N:
        if np.isnan(want[f]):
            g0 = f
            while f < N and np.isnan(want[f]):
                f += 1
            gaps.append((g0, f))  # [g0, f)
        else:
            f += 1
    for g0, g1 in gaps:
        a = want[g0 - 1] if g0 > 0 else 0.0
        b = want[g1] if g1 < N else 0.0
        L = g1 - g0
        r = min(L, nr)
        lead = g1 == N  # after the last single support: unload right away instead of at the end
        for i in range(L):
            s = (i + 1) / (r + 1) if lead else (i - (L - r) + 1) / (r + 1)
            s = min(s, 1.0)
            s = 0.0 if s <= 0 else s * s * (3 - 2 * s)
            shift[g0 + i] = a + (b - a) * s


KNEES = [IDX["left_knee_joint"], IDX["right_knee_joint"]]


def smooth(x, k=9):
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


# CoM follows the pelvis only partly (the legs stay put): measure the ratio per frame and scale up
gain = np.ones(N)
for f in range(N):
    if abs(shift[f]) > 0.005:
        p = rp[f].copy(); p[:2] += shift[f] * lat[f]; qv = q[f]
        for sd in ("left", "right"):
            qv, _ = ik_leg(p, Rr[f], qv, sd, *foot[sd][f])
        gain[f] = np.clip(((com_of(fk(p, Rr[f], qv)) - com0[f])[:2] @ lat[f]) / shift[f], 0.5, 1.0)
shift = smooth(np.clip(shift / gain, -args.max_shift, args.max_shift))

# a foot in the air moves with the pelvis (its path relative to the body is kept); a foot on the floor stays
for sd in ("left", "right"):
    for f in range(N):
        w = np.clip((soles[sd][f] - floor - args.contact) / 0.05, 0.0, 1.0)
        if w > 0:
            P, R = foot[sd][f]
            foot[sd][f] = (P + w * shift[f] * np.r_[lat[f], 0.0], R)

# ---- apply: move the pelvis, re-solve both legs; lower the pelvis a little where a leg can't reach ----
rp2, q2 = rp.copy(), q.copy()
ik_res, dz = np.zeros(N), np.zeros(N)
for f in range(N):
    p = rp[f].copy(); p[:2] += shift[f] * lat[f]
    for z in np.arange(0, 0.061, 0.005):
        pz = p.copy(); pz[2] -= z; qv = q[f]; res = 0.0
        for sd in ("left", "right"):
            qv, r = ik_leg(pz, Rr[f], qv, sd, *foot[sd][f]); res = max(res, r)
        # also keep the knees no straighter than in the original: a straight knee is an IK singularity
        # where the ankle flips between two solutions frame to frame (acceleration spikes)
        if res < 0.003 and all(qv[k] >= min(q[f, k], 0.15) - 0.01 for k in KNEES):
            break
    dz[f] = z
dz = np.array([dz[max(0, f - 12):f + 13].max() for f in range(N)])  # widen first, so smoothing never drops below what a frame needs
dz = smooth(smooth(smooth(dz, 9), 9), 9)  # the search is in 5 mm steps: smooth it into a curve, not a ramp with kinks
for f in range(N):
    p = rp[f].copy(); p[:2] += shift[f] * lat[f]; p[2] -= dz[f]
    qv = q[f].copy()
    res = 0.0
    for sd in ("left", "right"):
        qv, r = ik_leg(p, Rr[f], qv, sd, *foot[sd][f]); res = max(res, r)
    q2[f], rp2[f], ik_res[f] = qv, p, res

# low-pass the leg corrections (not the original motion) to remove frame-to-frame IK noise
legs = LEG["left"] + LEG["right"]
dq = q2[:, legs] - q[:, legs]
for _ in range(args.leg_smooth):
    dq = np.stack([np.convolve(np.pad(c, 2, mode="edge"), np.array([1, 4, 6, 4, 1]) / 16, mode="valid") for c in dq.T], 1)
q2[:, legs] = q[:, legs] + dq
if args.leg_smooth:  # put the feet back: a short IK from the smoothed pose stays on its (smooth) branch
    for f in range(N):
        for sd in ("left", "right"):
            q2[f], _ = ik_leg(rp2[f], Rr[f], q2[f], sd, *foot[sd][f], iters=60)
rot_res = np.zeros(N)
for f in range(N):  # final foot error (position and orientation: a tilted sole digs a toe or heel into the floor)
    T = fk(rp2[f], Rr[f], q2[f])
    ik_res[f] = max(np.linalg.norm(T[f"{sd}_ankle_roll_link"][0] - foot[sd][f][0]) for sd in ("left", "right"))
    rot_res[f] = max(np.linalg.norm(rot_err(foot[sd][f][1], T[f"{sd}_ankle_roll_link"][1])) for sd in ("left", "right"))
T2 = [fk(rp2[f], Rr[f], q2[f]) for f in range(N)]
com2 = np.array([com_of(T) for T in T2])
np.savetxt(args.dst, np.c_[rp2, rq, q2], delimiter=",", fmt="%.6f")


# ---- report ----------------------------------------------------------------------------------
def lat_err(com):
    e = []
    for f in range(N):
        if phase[f] in (1, 2):
            side = "left" if phase[f] == 1 else "right"
            e.append(abs((stance_centre(f, side)[:2] - com[f, :2]) @ lat[f]))
    return np.array(e) * 100


def acc(qq):
    return np.abs(np.diff(qq, 2, axis=0)) * fps * fps


lo = np.array([LIM[n][0] for n in SDK]); hi = np.array([LIM[n][1] for n in SDK])
e0, e2 = lat_err(com0), lat_err(com2)
leg = sorted(LEG["left"] + LEG["right"])
lines = [
    f"frames {N} @ {fps:.0f} Hz, single support {int((phase == 1).sum() + (phase == 2).sum())} frames, flight {int((phase == 3).sum())}",
    f"CoM sideways from stance sole centre [cm]: before median {np.median(e0):.1f} p90 {np.percentile(e0, 90):.1f} max {e0.max():.1f}"
    f"  ->  after median {np.median(e2):.1f} p90 {np.percentile(e2, 90):.1f} max {e2.max():.1f}",
    f"pelvis shift [cm]: max {100 * np.abs(shift).max():.1f}, max rate {100 * np.abs(np.diff(shift)).max() * fps:.0f} cm/s;"
    f" pelvis lowered max {100 * dz.max():.1f} cm",
    f"IK foot error [mm]: max {1000 * ik_res.max():.1f}, p99 {1000 * np.percentile(ik_res, 99):.1f};"
    f" orientation [deg]: max {np.degrees(rot_res.max()):.1f}, p99 {np.degrees(np.percentile(rot_res, 99)):.1f}",
    f"lowest sole point [cm]: {100 * (min(sole(T, sd)[:, 2].min() for T in T2 for sd in ('left', 'right')) - floor):.1f} (floor = 0)",
    f"leg joint acc p99 [rad/s^2]: before {np.percentile(acc(q[:, leg]), 99):.0f} after {np.percentile(acc(q2[:, leg]), 99):.0f}",
    f"joints within 0.02 rad of a limit: before {int(((q - lo < 0.02) | (hi - q < 0.02)).any(1).sum())} frames,"
    f" after {int(((q2 - lo < 0.02) | (hi - q2 < 0.02)).any(1).sum())} frames",
    "largest leg-joint changes [rad]: " + ", ".join(
        f"{SDK[i].replace('_joint', '')} {np.abs(q2[:, i] - q[:, i]).max():.2f}"
        for i in sorted(leg, key=lambda i: -np.abs(q2[:, i] - q[:, i]).max())[:4]),
]
print("\n".join(lines))
if args.report:
    open(args.report, "w").write("\n".join(lines) + "\n")
def lat_signed(com):  # CoM sideways from stance sole centre [m], NaN in double support
    e = np.full(N, np.nan)
    for f in range(N):
        if phase[f] in (1, 2):
            e[f] = (com[f, :2] - stance_centre(f, "left" if phase[f] == 1 else "right")[:2]) @ lat[f]
    return e


np.savez(os.path.splitext(args.dst)[0] + "_debug.npz", phase=phase, shift=shift, dz=dz, com0=com0, com2=com2,
         lat0=lat_signed(com0), lat2=lat_signed(com2))
