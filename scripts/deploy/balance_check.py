"""Is the reference dance balanced on one leg when it is played slower?

For every frame of the training reference (npz), with the G1 MJCF:
  * stance: a foot is on the floor when its lowest contact sphere is < 1.5 cm above the lowest sole of the frame
  * support polygon: convex hull of the contact spheres of the feet on the floor
  * CoM: whole-body centre of mass; ZMP (point-mass model): p = c_xy - c_z / g * c_xy'' , where the dance played
    `s` times slower has accelerations x 1/s^2 (s = 2 -> half speed)
Prints, for one-foot phases, how far CoM and ZMP are from the support polygon at several speeds.
Static balance (very slow play) needs the CoM inside; at full speed the ZMP is what has to stay inside.

  conda run -n unitree_sim_env python deploy/balance_check.py
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser(os.environ.get("G1_DANCE_DATA", "~/projects/g1_pilot"))  # day1/, day2/ motion files
sys.path.insert(0, os.path.join(HERE, "..", "sim2sim"))
from sim2sim_dance import ISAAC  # noqa: E402

NPZ = os.path.join(DATA, "day2", "schuhplattler_gmr_v3s.npz")
XML = os.path.expanduser("~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.array(pts)
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and np.cross(np.subtract(lo[-1], lo[-2]), np.subtract(p, lo[-2])) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and np.cross(np.subtract(up[-1], up[-2]), np.subtract(p, up[-2])) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def signed_dist(p, poly):
    """> 0 inside (distance to the nearest edge), < 0 outside"""
    d, inside = np.inf, True
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        d = min(d, np.linalg.norm(p - (a + t * e)))
        if np.cross(e, p - a) < 0:
            inside = False
    return d if inside else -d


def main():
    z = np.load(NPZ)
    fps = float(np.asarray(z["fps"]).reshape(-1)[0])
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ISAAC]
    feet = {s: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == f"{s}_ankle_roll_link"
                and m.geom_contype[g]] for s in ("left", "right")}
    T = len(z["joint_pos"])
    com = np.zeros((T, 3))
    spheres = {s: np.zeros((T, len(g), 3)) for s, g in feet.items()}
    for k in range(T):
        d.qpos[:3], d.qpos[3:7], d.qpos[qadr] = z["body_pos_w"][k, 0], z["body_quat_w"][k, 0], z["joint_pos"][k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        com[k] = d.subtree_com[0]
        for s, g in feet.items():
            spheres[s][k] = d.geom_xpos[g] - [0, 0, 1] * m.geom_size[g, :1]
    acc = np.gradient(np.gradient(com[:, :2], 1 / fps, axis=0), 1 / fps, axis=0)
    low = np.minimum(spheres["left"][:, :, 2].min(1), spheres["right"][:, :, 2].min(1))
    on = {s: spheres[s][:, :, 2].min(1) - low < 0.015 for s in feet}
    single = on["left"] ^ on["right"]
    print(f"{T} frames ({(T - 1) / fps:.2f} s): one foot on the floor {100 * single.mean():.0f}% of the time, "
          f"both {100 * (on['left'] & on['right']).mean():.0f}%")
    runs, k = [], 0
    while k < T:
        if single[k]:
            j = k
            while j + 1 < T and single[j + 1] and on["left"][j + 1] == on["left"][k]:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    longest = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:5]
    print("longest one-foot phases: " + ", ".join(
        f"{a / fps:.2f}-{b / fps:.2f} s ({'left' if on['left'][a] else 'right'} foot, {(b - a + 1) / fps:.2f} s)" for a, b in longest))
    print(f"{'play speed':>11s} {'CoM inside %':>13s} {'ZMP inside %':>13s} {'worst ZMP margin':>17s}   (one-foot frames)")
    for s in (1.0, 0.75, 0.5, 0.25, 0.0):
        com_in, zmp_in, worst = [], [], np.inf
        for k in np.nonzero(single)[0]:
            st = "left" if on["left"][k] else "right"
            poly = hull(spheres[st][k][:, :2])
            c = com[k, :2]
            zmp = c - com[k, 2] / 9.81 * acc[k] * s * s
            com_in.append(signed_dist(c, poly) > 0)
            mz = signed_dist(zmp, poly)
            zmp_in.append(mz > 0)
            worst = min(worst, mz)
        label = "static" if s == 0 else f"x{s:.2f}"
        print(f"{label:>11s} {100 * np.mean(com_in):12.0f}% {100 * np.mean(zmp_in):12.0f}% {100 * worst:14.1f} cm")
    foot = hull(spheres["left"][0][:, :2])
    print(f"(foot contact polygon here: {100 * np.ptp(foot[:, 0]):.0f} x {100 * np.ptp(foot[:, 1]):.0f} cm, "
          "the 4 contact spheres of the MJCF foot)")


if __name__ == "__main__":
    main()
