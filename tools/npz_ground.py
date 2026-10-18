"""Ground the support foot in every frame of a motion npz (root z shift only, joints untouched).

Why: the Day 2 retarget leaves the lower (support) foot 2-9 cm above the floor in ~30% of frames,
and has short "hops" where both feet are up. Training starts episodes at random frames, so those
episodes start with the robot in the air, and the reference asks the robot to stand on nothing.

What it does: per frame, find the lower sole (ankle_roll_link z minus its standing height) and shift
every body down/up by that amount, so one foot always touches the floor. The shift is smoothed
(--smooth frames) to avoid jerks; linear z velocity is updated to match. --keep_hops_cm keeps real
flight phases: frames where BOTH feet are higher than that are left in the air.

Usage (any env with numpy):
  python npz_ground.py --src ~/projects/g1_pilot/day2/schuhplattler_gmr.npz \
                       --dst ~/projects/g1_pilot/day2/schuhplattler_gmr_grounded.npz
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--feet", type=int, nargs=2, default=[18, 19], help="body indexes of the ankle roll links (sim order)")
ap.add_argument("--smooth", type=int, default=5, help="moving-average window for the shift [frames], odd")
ap.add_argument("--keep_hops_cm", type=float, default=None, help="leave frames with both feet above this in the air")
a = ap.parse_args()

d = dict(np.load(os.path.expanduser(a.src)))
fps = float(np.asarray(d["fps"]).reshape(-1)[0])
pos = d["body_pos_w"].copy()
foot = pos[:, a.feet, 2]
standing = foot[0].min()  # frame 0 = standing on the floor
low = foot.min(1) - standing  # support-sole height above floor per frame [m]

shift = low.copy()
if a.keep_hops_cm is not None:
    shift[low > a.keep_hops_cm / 100] = 0.0
if a.smooth > 1:
    k = np.ones(a.smooth) / a.smooth
    pad = a.smooth // 2
    shift = np.convolve(np.pad(shift, pad, mode="edge"), k, mode="valid")
pos[:, :, 2] -= shift[:, None]
d["body_pos_w"] = pos
vel = d["body_lin_vel_w"].copy()
vel[:, :, 2] -= np.gradient(shift, 1 / fps)[:, None]
d["body_lin_vel_w"] = vel
np.savez(os.path.expanduser(a.dst), **d)


def report(name, z):
    lo = z.min(1) - standing
    print(f"{name:8s} support foot: median {np.median(lo) * 100:4.1f} cm, >2 cm in {(lo > 0.02).mean() * 100:4.1f}% "
          f"of frames, max {lo.max() * 100:4.1f} cm, min {lo.min() * 100:5.1f} cm")


report("before", foot)
report("after", pos[:, a.feet, 2])
print(f"root z shift: mean {shift.mean() * 100:.1f} cm, max {shift.max() * 100:.1f} cm; "
      f"pelvis z {d['body_pos_w'][:, 0, 2].min():.3f}-{d['body_pos_w'][:, 0, 2].max():.3f} m")
print("saved", a.dst)
