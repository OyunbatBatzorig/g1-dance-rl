"""Build a Unitree deploy package (same layout as deploy/robots/g1_29dof/config/policy/mimic/dance_102/) for the
Schuhplattler v3s policy, and check the joint order three ways.

  conda run -n unitree_sim_env python deploy/build_deploy_pkg.py   (any env with mujoco + numpy + pyyaml)

Writes deploy/schuhplattler_v3s/
  params/deploy.yaml                   from the training run's params/deploy.yaml, cleaned (see below)
  params/schuhplattler_gmr_v3s_50hz.csv   the training reference (npz) as the deploy CSV: root pos, root quat xyzw,
                                          29 joints in SDK order, 50 Hz so every policy step lands on a frame
  params/schuhplattler_gmr_v3s_50hz_intro2.csv   the same with a 2 s blend from the default stance in front
  params/schuhplattler_v3s_arm_test_50hz.csv      first hardware run: arms up to the first dance pose and back (7 s)
  exported/policy.onnx                made by export_onnx.py
  config_snippet.yaml                  FSM entries to paste into deploy/robots/g1_29dof/config/config.yaml

deploy.yaml cleaning:
  * default_joint_pos / action offset: the training export took robot 0's *randomized* default pose
    (add_joint_default_pos DR, +-0.03 rad); replaced by the nominal UNITREE_G1_29DOF_MIMIC pose the policy was
    trained around (the DR centre). Unitree's own dance_102/deploy.yaml has the same randomized values.
  * residual-action fields (residual_*, ref_*, delay_substeps_range) dropped: the C++ JointPositionAction only
    reads scale / offset / clip, and the !!python/tuple tag is Python-only YAML.
"""
import csv
import os
import re
import sys

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser(os.environ.get("G1_DANCE_DATA", "~/projects/g1_pilot"))  # day1/, day2/ motion files
sys.path.insert(0, os.path.join(HERE, "..", "sim2sim"))
from sim2sim_dance import DEFAULT_Q, ISAAC, KD, KP, SCALE  # noqa: E402

RUN = os.path.expanduser("~/projects/unitree_rl_lab/logs/rsl_rl/g1_schuhplattler_residual/2026-10-31_22-40-36_sch_v3s_floor")
NPZ = os.path.join(DATA, "day2", "schuhplattler_gmr_v3s.npz")
XML = os.path.expanduser("~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
UNITREE = os.path.expanduser("~/projects/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/mimic/dance_102/params/deploy.yaml")
OUT = os.path.join(HERE, "schuhplattler_v3s")
CSV_NAME = "schuhplattler_gmr_v3s_50hz.csv"
CSV_INTRO = "schuhplattler_gmr_v3s_50hz_intro2.csv"
INTRO_S = 2.0
CSV_ARMS = "schuhplattler_v3s_arm_test_50hz.csv"
CSV_PART = "schuhplattler_v3s_part{:05.2f}_50hz.csv"
PART_CUTS = (1.66,)  # dance s, both feet down, calm. 11.04 failed: 29% real falls (fast motion there, deploy_sim2sim 512 runs)
RETURN_S = 2.0


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: list(l.construct_sequence(n)))


def main():
    os.makedirs(os.path.join(OUT, "params"), exist_ok=True)
    d = yaml.load(open(os.path.join(RUN, "params", "deploy.yaml")), Loader=_Loader)
    ids = d["joint_ids_map"]

    # ---- joint order: deploy.yaml map vs the MJCF (= SDK motor order) vs Day 1 joint_map.csv
    m = mujoco.MjModel.from_xml_path(XML)
    sdk = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    from_names = [sdk.index(n) for n in ISAAC]
    day1 = [r["joint_name"] for r in csv.DictReader(open(os.path.join(DATA, "day1", "joint_map.csv")))]
    print("joint_ids_map == SDK index of each Isaac joint (MJCF names):", ids == from_names)
    print("Isaac order == day1/joint_map.csv order:                ", day1 == ISAAC)
    print("joint_ids_map == Unitree dance_102 deploy.yaml:          ",
          ids == yaml.safe_load(open(UNITREE))["joint_ids_map"])
    assert ids == from_names and day1 == ISAAC

    # ---- gains / scale vs the sim2sim constants (3 significant digits in the export). stiffness / damping are
    # stored per motor (SDK order: State_Mimic sets lowcmd kp[i] = stiffness[i]); scale / offset per policy joint
    for key, ref in (("stiffness", KP), ("damping", KD)):
        ref_sdk = np.zeros(29)
        ref_sdk[ids] = ref
        print(f"{key:9s} (SDK order) max rel diff vs sim2sim: {np.max(np.abs(np.array(d[key]) - ref_sdk) / ref_sdk):.3%}")
    act = d["actions"]["JointPositionAction"]
    print(f"action scale max rel diff vs sim2sim: {np.max(np.abs(np.array(act['scale']) - SCALE) / SCALE):.3%}")
    print(f"exported default pose was off the nominal by up to {np.max(np.abs(np.array(d['default_joint_pos']) - DEFAULT_Q)):.3f} rad"
          " (robot 0's randomized default): replaced")

    nominal = [round(float(v), 4) for v in DEFAULT_Q]
    d["default_joint_pos"] = nominal
    act["offset"] = list(nominal)  # a separate list, so the YAML has no anchor / alias
    for k in [k for k in act if k.startswith(("residual", "ref_", "delay"))] + ["command_name"]:
        act.pop(k, None)
    obs = d["observations"]
    dims = {"motion_command": 58, "motion_anchor_ori_b": 6, "base_ang_vel": 3, "joint_pos_rel": 29,
            "joint_vel_rel": 29, "last_action": 29}
    assert list(obs) == list(dims) and all(len(obs[k]["scale"]) == n for k, n in dims.items()), "obs layout changed"
    print(f"observations: {list(obs)} = {sum(dims.values())} dims (same terms and order as dance_102)")

    text = yaml.safe_dump(d, sort_keys=False, default_flow_style=None, width=110)
    open(os.path.join(OUT, "params", "deploy.yaml"), "w").write(text)

    # ---- reference CSV at 50 Hz, SDK joint order
    z = np.load(NPZ)
    fps = float(np.asarray(z["fps"]).reshape(-1)[0])
    jp = z["joint_pos"]
    sdk_q = np.zeros_like(jp)
    sdk_q[:, ids] = jp
    rq = z["body_quat_w"][:, 0]  # wxyz
    rows = np.concatenate([z["body_pos_w"][:, 0], rq[:, [1, 2, 3, 0]], sdk_q], axis=1)
    np.savetxt(os.path.join(OUT, "params", CSV_NAME), rows, fmt="%.6f", delimiter=",")
    print(f"wrote {CSV_NAME}: {len(rows)} frames at {fps:.0f} Hz = {(len(rows) - 1) / fps:.2f} s, 36 columns")

    # same with a 2 s quintic blend from the default stance (arms down) to the first dance frame (arms out):
    # in deploy_sim2sim.py --start stand, 99.4% finished with it vs 98.8% (2 real falls) without
    q_stand = np.zeros(29)
    q_stand[ids] = DEFAULT_Q
    s = np.linspace(0, 1, int(round(INTRO_S * fps)), endpoint=False)
    h = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    intro = np.repeat(rows[:1], len(s), axis=0)
    intro[:, 7:] = q_stand[None] + h[:, None] * (rows[0, 7:] - q_stand)[None]
    rows_i = np.vstack([intro, rows])
    np.savetxt(os.path.join(OUT, "params", CSV_INTRO), rows_i, fmt="%.6f", delimiter=",")
    print(f"wrote {CSV_INTRO}: {len(rows_i)} frames = {INTRO_S:.0f} s intro + dance")

    # "smoke test" for the first hardware run: only the arms move (legs and waist stay in the default stance):
    # 1 s still, 2 s up to the first dance frame's arm pose, 1 s hold, 2 s back down, 1 s still = 7 s
    arms = np.arange(15, 29)  # SDK indices of both arms
    up = q_stand.copy()
    up[arms] = rows[0, 7 + arms]
    hold = lambda q, sec: np.repeat(q[None], int(round(sec * fps)), axis=0)
    blend = lambda q0, q1: q0[None] + h[:, None] * (q1 - q0)[None]
    seq = np.vstack([hold(q_stand, 1), blend(q_stand, up), hold(up, 1), blend(up, q_stand), hold(q_stand, 1)])
    rows_t = np.repeat(rows[:1], len(seq), axis=0)
    rows_t[:, 7:] = seq
    np.savetxt(os.path.join(OUT, "params", CSV_ARMS), rows_t, fmt="%.6f", delimiter=",")
    print(f"wrote {CSV_ARMS}: {len(rows_t)} frames = arms-only smoke test ({(len(rows_t) - 1) / fps:.2f} s)")

    # partial dances that end calmly: intro + dance up to a both-feet-down moment, then a 2 s velocity-matched
    # quintic back to the default stance (and an upright root), 1 s hold. Stopping the full CSV early instead lets
    # the Velocity policy snap the arms down (dance 1.66 s: arms 2.7 rad overhead).
    qd = np.gradient(rows[:, 7:], 1 / fps, axis=0)
    for t_cut in PART_CUTS:
        k = int(round(t_cut * fps))
        T = RETURN_S
        tt = np.arange(1, int(round(T * fps)) + 1) / fps
        sn = tt / T
        d = q_stand - rows[k, 7:]
        ret = (rows[k, 7:][None] + d[None] * (10 * sn ** 3 - 15 * sn ** 4 + 6 * sn ** 5)[:, None]
               + (qd[k] * T)[None] * (sn - 6 * sn ** 3 + 8 * sn ** 4 - 3 * sn ** 5)[:, None])
        w, x, y, zq = rq[k]
        yaw = np.arctan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        upright = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])  # xyzw
        tail = np.repeat(rows[k:k + 1], len(ret) + int(round(fps)), axis=0)
        tail[:, 7:] = np.vstack([ret, hold(q_stand, 1)])
        sm = np.concatenate([10 * sn ** 3 - 15 * sn ** 4 + 6 * sn ** 5, np.ones(int(round(fps)))])
        q0 = rows[k, 3:7]
        if np.dot(q0, upright) < 0:
            upright = -upright
        tail[:, 3:7] = q0[None] * (1 - sm[:, None]) + upright[None] * sm[:, None]  # small tilt: nlerp is enough
        tail[:, 3:7] /= np.linalg.norm(tail[:, 3:7], axis=1, keepdims=True)
        rows_p = np.vstack([intro, rows[:k + 1], tail])
        name = CSV_PART.format(t_cut)
        np.savetxt(os.path.join(OUT, "params", name), rows_p, fmt="%.6f", delimiter=",")
        print(f"wrote {name}: intro + dance 0-{t_cut:.2f} s + {T:.0f} s return + 1 s hold = {(len(rows_p) - 1) / fps:.2f} s")

    open(os.path.join(OUT, "config_snippet.yaml"), "w").write(f"""# Paste into deploy/robots/g1_29dof/config/config.yaml (copy this folder to config/policy/mimic/schuhplattler_v3s first).
# Keep fps at {fps:.0f}: playing the dance slower (lower fps) made the robot fall in MuJoCo (0.75x: 23%, 0.5x: ~100%).
# 1) under FSM._ (enabled fsms):
    Mimic_Schuhplattler:
      id: 103
      type: Mimic
    Mimic_Schuhplattler_ArmTest:
      id: 104
      type: Mimic
# 2) in FSM.Velocity.transitions, add button combos that are still free, e.g.:
      Mimic_Schuhplattler: LT(2s) + right.on_pressed
      Mimic_Schuhplattler_ArmTest: LT(2s) + up.on_pressed
# 3) the states themselves, next to Mimic_Dance_102.
#    First hardware run: the arm test (same policy, only the arms go up to the dance's first pose and back down).
  Mimic_Schuhplattler_ArmTest:
    transitions:
      Passive: LT + B.on_pressed
      Velocity: RB + X.on_pressed
    fps: {fps:.0f}
    motion_file: config/policy/mimic/schuhplattler_v3s/params/{CSV_ARMS}
    policy_dir: config/policy/mimic/schuhplattler_v3s/
    time_start: 0.0
    time_end: {(len(rows_t) - 1) / fps:.2f}
  Mimic_Schuhplattler:
    transitions:
      Passive: LT + B.on_pressed
      Velocity: RB + X.on_pressed
    fps: {fps:.0f}
    motion_file: config/policy/mimic/schuhplattler_v3s/params/{CSV_INTRO}
    policy_dir: config/policy/mimic/schuhplattler_v3s/
    time_start: 0.0
    time_end: {(len(rows_i) - 1) / fps:.2f}    # first tests: 3.66 then 13.04 (both feet on the floor there), see sim2real_runbook.md
""")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
