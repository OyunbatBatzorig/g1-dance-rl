"""Remove slow vertical drift from a G1 CSV motion by keeping the support foot on the floor.

Video-based retargets (GVHMR -> GMR) often let the whole body creep upward over time, so the
standing foot ends up floating 10-20 cm above z=0. This script only shifts root z: per frame it
finds the lowest sole point (forward kinematics, same as motion_check.py), takes a rolling
minimum over --window seconds (so a lifted foot during a kick does not count), smooths it, and
subtracts it. Joint angles are untouched, so no new jerks are introduced.

Usage:
  python ground_fix.py in.csv out.csv --fps 30
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_check import SDK, foot_sole, load_urdf, quat_xyzw  # noqa: E402


def rolling_min(x, w):
    h = w // 2
    pad = np.r_[np.full(h, x[0]), x, np.full(h, x[-1])]
    return np.array([pad[i:i + w].min() for i in range(len(x))])


def smooth(x, w):
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    h = w // 2
    pad = np.r_[np.full(h, x[0]), x, np.full(w - 1 - h, x[-1])]
    return np.convolve(pad, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window", type=float, default=0.6, help="rolling-min window [s]")
    ap.add_argument("--urdf", default=os.path.expanduser(
        "~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"))
    a = ap.parse_args()

    m = np.loadtxt(a.inp, delimiter=",")
    joints, _ = load_urdf(a.urdf)
    lowest = []
    for row in m:
        R = quat_xyzw(*row[3:7])
        qd = dict(zip(SDK, row[7:]))
        lowest.append(min(foot_sole(joints, row[:3], R, qd, s)[:, 2].min() for s in ("left", "right")))
    lowest = np.array(lowest)
    w = max(3, int(a.window * a.fps) | 1)
    offset = smooth(rolling_min(lowest, w), w)
    m[:, 2] -= offset
    np.savetxt(a.out, m, delimiter=",", fmt="%.6f")
    print(f"root z shifted by {-offset.max() * 100:.1f} .. {-offset.min() * 100:.1f} cm "
          f"(median {-np.median(offset) * 100:.1f} cm); wrote {a.out}")


if __name__ == "__main__":
    main()
