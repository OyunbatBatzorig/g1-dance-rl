"""Check a G1 29-DoF reference motion for problems before training.

Accepts either
  * a Unitree/GMR CSV: per row root pos (3) + root quat xyzw (4) + 29 joints in SDK order, or
  * an Isaac Lab motion .npz from csv_to_npz.py (joint_pos in sim order, body_pos_w/body_quat_w).

Usage:
  python motion_check.py <file.csv> --fps 30
  python motion_check.py <file.npz>
  (add --urdf if unitree_ros is not at ~/projects/unitree_ros)

Prints a PASS/WARN table and writes <file>_check.png (joint curves + foot heights) if matplotlib exists.
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

SDK = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
       "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
       "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
       "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
       "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
       "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
       "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
       "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
# Isaac Lab articulation order for g1_29dof_rev_1_0 (matches day1 joint_map.csv)
SIM = ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
       "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
       "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
       "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
       "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
       "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
       "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
       "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint"]
# sole corners in the ankle_roll_link frame (from the G1 foot mesh: sole at z=-0.035)
SOLE = np.array([[0.12, 0.025, -0.035], [0.12, -0.025, -0.035], [-0.05, 0.025, -0.035], [-0.05, -0.025, -0.035]])


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(ax, t):
    ax = np.asarray(ax) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def quat_xyzw(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_urdf(path):
    joints, limits = {}, {}
    for j in ET.parse(path).getroot().findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        joints[j.find("child").get("link")] = dict(
            name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
            xyz=np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3),
            R=rpy(*[float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.eye(3),
            axis=[float(v) for v in a.get("xyz").split()] if a is not None else [1, 0, 0])
        if lim is not None and lim.get("lower"):
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return joints, limits


def foot_sole(joints, root_p, root_R, q, side):
    chain, link = [], f"{side}_ankle_roll_link"
    while link in joints:
        chain.append(link)
        link = joints[link]["parent"]
    P, R = root_p.copy(), root_R.copy()
    for link in reversed(chain):
        j = joints[link]
        P = P + R @ j["xyz"]
        R = R @ j["R"]
        if j["type"] == "revolute":
            R = R @ axis_angle(j["axis"], q[j["name"]])
    return P + (R @ SOLE.T).T


def load_motion(path, fps):
    if path.endswith(".npz"):
        d = np.load(path)
        fps = int(d["fps"][0])
        q = d["joint_pos"][:, [SIM.index(n) for n in SDK]]
        root_p = d["body_pos_w"][:, 0]
        w, x, y, z = d["body_quat_w"][:, 0].T
        root_q = np.stack([x, y, z, w], 1)
    else:
        m = np.loadtxt(path, delimiter=",")
        if m.ndim != 2 or m.shape[1] != 36:
            raise SystemExit(f"expected 36 columns (3 pos + 4 quat + 29 joints), got shape {m.shape}")
        root_p, root_q, q = m[:, :3], m[:, 3:7], m[:, 7:]
    return fps, root_p.astype(float), root_q.astype(float), q.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--fps", type=int, default=30, help="CSV frame rate (ignored for .npz)")
    ap.add_argument("--urdf", default=os.path.expanduser(
        "~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"))
    a = ap.parse_args()
    fps, rp, rq, q = load_motion(a.file, a.fps)
    joints, limits = load_urdf(a.urdf)
    F = len(q)
    rows = []

    def add(name, value, ok, note):
        rows.append((name, value, "PASS" if ok else "WARN", note))

    finite = np.isfinite(np.c_[rp, rq, q]).all()
    add("frames / duration", f"{F} / {F / fps:.2f} s @ {fps} Hz", F > fps, "")
    add("NaN/Inf", "none" if finite else "FOUND", finite, "must be none")
    qn = np.linalg.norm(rq, axis=1)
    add("root quat norm", f"{qn.min():.3f}..{qn.max():.3f}", abs(qn - 1).max() < 1e-2, "should be 1")
    viol = {n: np.mean((q[:, i] < limits[n][0]) | (q[:, i] > limits[n][1])) for i, n in enumerate(SDK)}
    bad = {n: v for n, v in viol.items() if v > 0}
    add("joint limit violations", f"{len(bad)} joints", not bad,
        ", ".join(f"{n}:{v:.0%}" for n, v in sorted(bad.items(), key=lambda t: -t[1])[:4]))
    dq = np.abs(np.diff(q, axis=0)) * fps
    add("max joint speed", f"{dq.max():.1f} rad/s ({SDK[dq.max(0).argmax()]})", dq.max() < 20, "G1 limit ~20-30")
    jump = np.abs(np.diff(q, axis=0)).max()
    add("max frame-to-frame jump", f"{jump:.3f} rad", jump < 0.35, "big jumps = wrap-around / glitch")
    acc = np.sqrt(np.mean((np.diff(q, 2, axis=0) * fps * fps) ** 2))
    add("joint acc RMS", f"{acc:.1f} rad/s^2", acc < 60, "Dance-102 is ~20")

    soles = {s: [] for s in ("left", "right")}
    for f in range(F):
        R = quat_xyzw(*rq[f])
        qd = dict(zip(SDK, q[f]))
        for s in soles:
            soles[s].append(foot_sole(joints, rp[f], R, qd, s))
    lowest = np.minimum(*[np.array(soles[s])[:, :, 2].min(1) for s in soles])
    add("lowest foot height (median)", f"{np.median(lowest) * 100:.1f} cm", abs(np.median(lowest)) < 0.02,
        "support foot should sit on z=0 (Dance-102: -0.9 cm)")
    add("lowest foot below ground", f"{lowest.min() * 100:.1f} cm", lowest.min() > -0.03, "penetration")
    slides = []
    for s in soles:
        a3 = np.array(soles[s])
        z = a3[:, :, 2].min(1)
        c = z < lowest + 0.02  # this foot is (one of) the lowest -> support foot
        v = np.linalg.norm(np.diff(a3[:, :, :2].mean(1), axis=0), axis=1) * fps
        slides.append(v[c[1:] & c[:-1]])
    sl = np.concatenate(slides) if any(len(x) for x in slides) else np.zeros(1)
    add("support-foot slide (mean)", f"{sl.mean() * 100:.0f} cm/s", sl.mean() < 0.2, "Dance-102: 13-15 cm/s")
    add("pelvis height (median)", f"{np.median(rp[:, 2]):.3f} m", 0.65 < np.median(rp[:, 2]) < 0.80,
        "G1 standing ~0.76-0.79")
    add("root xy travel", f"{np.ptp(rp[:, 0]):.2f} x {np.ptp(rp[:, 1]):.2f} m", True, "")

    w = max(len(r[0]) for r in rows)
    for name, val, st, note in rows:
        print(f"{st:4s}  {name:<{w}}  {val:<34} {note}")
    n_warn = sum(r[2] == "WARN" for r in rows)
    print(f"\n{n_warn} warning(s)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(F) / fps
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for i in range(12):
            ax[0].plot(t, q[:, i], lw=0.8)
        ax[0].set_ylabel("legs q [rad]")
        for i in range(12, 29):
            ax[1].plot(t, q[:, i], lw=0.8)
        ax[1].set_ylabel("waist+arms q [rad]")
        for s in soles:
            ax[2].plot(t, np.array(soles[s])[:, :, 2].min(1) * 100, label=s)
        ax[2].axhline(0, color="k", lw=0.5)
        ax[2].set_ylabel("foot sole z [cm]")
        ax[2].set_xlabel("time [s]")
        ax[2].legend()
        out = os.path.splitext(a.file)[0] + "_check.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print("plot:", out)
    except ImportError:
        print("(matplotlib not installed: skipped plot)")


if __name__ == "__main__":
    main()
