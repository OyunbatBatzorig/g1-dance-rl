"""Convert one GMR robot-motion pickle (unitree_g1) to the Unitree mimic CSV layout.

GMR's G1 joint order (assets/unitree_g1/g1_mocap_29dof.xml) is the SDK order and root_rot is xyzw,
which is exactly what scripts/mimic/csv_to_npz.py expects, so this only re-packs the arrays.

Usage: python gmr_to_csv.py <in.pkl> <out.csv>     (prints the fps to pass to csv_to_npz --input_fps)
"""
import pickle
import sys

import numpy as np

src, dst = sys.argv[1:3]
with open(src, "rb") as f:
    d = pickle.load(f)
dof = np.asarray(d["dof_pos"])
if dof.shape[1] != 29:
    raise SystemExit(f"expected 29 joints (unitree_g1), got {dof.shape[1]}; did you use --robot unitree_g1?")
m = np.concatenate([np.asarray(d["root_pos"]), np.asarray(d["root_rot"]), dof], axis=1)
np.savetxt(dst, m, delimiter=",", fmt="%.6f")
print(f"wrote {dst}: {len(m)} frames, fps = {d['fps']}  ->  use --input_fps {int(round(float(d['fps'])))}")
