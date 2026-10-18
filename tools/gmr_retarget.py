"""Headless GVHMR -> G1 retarget with GMR, using a chosen IK config, plus a human-vs-robot match report.

Same retarget as GMR's scripts/gvhmr_to_robot.py, but:
  * --ik_config lets you use a tuned copy of smplx_to_g1.json (GMR itself is not edited),
  * no viewer window, so it runs in a plain terminal,
  * it measures, per frame, how far each matched robot body (wrist, elbow, foot, ...) is from the
    scaled human body GMR was aiming at, and prints a table. Lower = robot imitates the human more closely.

Usage (gmr env):
  python gmr_retarget.py --gvhmr_pred_file .../hmr4d_results.pt --save_path out.pkl [--ik_config tuned.json]
Writes out.pkl (same format as gvhmr_to_robot.py) and out_match.npz (per-body errors, human + robot positions).
"""
import argparse
import json
import os
import pickle
import tempfile

import numpy as np

import general_motion_retargeting as gmr_pkg
from general_motion_retargeting import params
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

ap = argparse.ArgumentParser()
ap.add_argument("--gvhmr_pred_file", required=True)
ap.add_argument("--save_path", required=True)
ap.add_argument("--ik_config", default=None, help="tuned IK json (default: GMR's smplx_to_g1.json)")
ap.add_argument("--smplx_folder", default=None, help="folder holding smplx/SMPLX_NEUTRAL.npz (default: GMR assets)")
a = ap.parse_args()

gmr_root = os.path.dirname(os.path.dirname(os.path.abspath(gmr_pkg.__file__)))
default_cfg = str(params.IK_CONFIG_DICT["smplx"]["unitree_g1"])
cfg_path = a.ik_config or default_cfg
params.IK_CONFIG_DICT["smplx"]["unitree_g1"] = cfg_path  # same dict object GMR reads from
print("IK config:", cfg_path)

smplx_folder = a.smplx_folder or os.path.join(gmr_root, "assets", "body_models")
smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(a.gvhmr_pred_file, smplx_folder)
frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
print(f"human height estimate {height:.2f} m, {len(frames)} frames @ {fps} Hz")

retarget = gmr_pkg.GeneralMotionRetargeting(actual_human_height=height, src_human="smplx", tgt_robot="unitree_g1",
                                            verbose=False)
table = json.load(open(cfg_path))["ik_match_table2"]
pairs = [(robot_body, v[0]) for robot_body, v in table.items()]  # (robot body, human body)

qpos_list, human_pos, robot_pos = [], [], []
for i in range(1, len(frames)):  # GMR's script also starts at frame 1
    q = retarget.retarget(frames[i])
    qpos_list.append(q.copy())
    human_pos.append([retarget.scaled_human_data[h][0] for _, h in pairs])
    robot_pos.append([retarget.configuration.get_transform_frame_to_world(r, "body").translation() for r, _ in pairs])

qpos = np.array(qpos_list)
motion = {"fps": fps, "root_pos": qpos[:, :3], "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]], "dof_pos": qpos[:, 7:],
          "local_body_pos": None, "link_body_list": None}
os.makedirs(os.path.dirname(os.path.abspath(a.save_path)), exist_ok=True)
pickle.dump(motion, open(a.save_path, "wb"))

H, Rb = np.array(human_pos), np.array(robot_pos)
err = np.linalg.norm(H - Rb, axis=-1)  # (T, bodies)
names = [r for r, _ in pairs]
np.savez(os.path.splitext(a.save_path)[0] + "_match.npz", names=np.array(names), human=H, robot=Rb, err=err,
         dof=qpos[:, 7:], fps=fps)
acc = np.sqrt(np.mean((np.diff(qpos[:, 7:], 2, axis=0) * fps * fps) ** 2))
print(f"\n{'robot body':26s} {'mean err':>9s} {'p90 err':>9s}   (robot body vs scaled human body, cm)")
for j, n in enumerate(names):
    print(f"{n:26s} {err[:, j].mean() * 100:8.1f}  {np.percentile(err[:, j], 90) * 100:8.1f}")
arms = [j for j, n in enumerate(names) if "elbow" in n or "wrist" in n]
legs = [j for j, n in enumerate(names) if "knee" in n or "toe" in n]
print(f"\nARMS mean {err[:, arms].mean() * 100:.1f} cm | LEGS mean {err[:, legs].mean() * 100:.1f} cm | "
      f"joint acc RMS {acc:.1f} rad/s^2")
print("saved", a.save_path)
