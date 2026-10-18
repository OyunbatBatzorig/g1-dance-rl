"""Sim2sim: play a dance policy trained in Isaac Lab inside MuJoCo, with the same observation, action and PD
as training, and count whether the robot finishes the clip.

Needs only mujoco, numpy and torch (onnxruntime for --onnx). No Isaac Sim.

Matches the training task (tracking_env_cfg.py + residual/residual_action.py):
  obs (154) = ref joint_pos (29) + ref joint_vel (29) at frame k, ref torso orientation in the robot torso frame
              (first two rotation columns, 6), pelvis angular velocity in the pelvis frame (3),
              q - default (29), qd (29), last action (29)          [Isaac joint order, no history, no noise]
  action    = wide:  q_ref[k+1] + clip(a * scale, +-1.0), PD velocity target qd_ref[k+1]
              mimic: default + a * scale, velocity target 0
  PD        = Unitree mimic gains (Kp = armature * (2 pi 10)^2, Kd = 2 * 2 * armature * 2 pi 10), armature on the
              joint, effort limits; 200 Hz physics, policy at 50 Hz (decimation 4)
  fall      = torso height off the reference by > 0.25 m, a foot or hand height off by > 0.25 m, or torso
              tilt (projected gravity z) off by > 0.8 -- the training terminations

Examples (from ~/projects/unitree_rl_lab, any env with mujoco + torch):
  # watch one run
  python ~/projects/g1_pilot/sim2sim/sim2sim_dance.py --checkpoint logs/rsl_rl/g1_schuhplattler_residual/<run>/model_4998.pt \
      --motion ~/projects/g1_pilot/day6/dance_102.npz --action wide --viewer
  # record an mp4 at the true speed (one frame per 20 ms policy step, however slow the PC is); needs imageio[ffmpeg]
  python ~/projects/g1_pilot/sim2sim/sim2sim_dance.py --checkpoint ... --motion ... --action wide --record ~/videos/d102_mujoco.mp4
  # nominal + Monte Carlo inside the training DR (gains, link masses, friction, 0-20 ms delay), 8 processes
  python ~/projects/g1_pilot/sim2sim/sim2sim_dance.py --checkpoint ... --motion ... --action wide --runs 64 --name p1_d102
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import mujoco
import numpy as np

ISAAC = [  # Isaac Lab joint order (= action / obs order, = npz joint order)
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def default_pos(n):
    """UNITREE_G1_29DOF_MIMIC_CFG init_state joint_pos (the action's default offset)."""
    for key, v in (("hip_pitch", -0.312), ("knee", 0.669), ("ankle_pitch", -0.363), ("elbow", 0.6),
                   ("left_shoulder_roll", 0.2), ("right_shoulder_roll", -0.2), ("shoulder_pitch", 0.2)):
        if key in n:
            return v
    return 0.0


W = 10 * 2 * np.pi  # 10 Hz natural frequency, damping ratio 2 (unitree.py)
ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}


def joint_params(n):
    """(armature, Kp, Kd, effort limit) of the Unitree mimic config for joint n."""
    if "hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint":
        a, e = ARM["7520_14"], 88.0
    elif "hip_roll" in n or "knee" in n:
        a, e = ARM["7520_22"], 139.0
    elif "ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint"):
        a, e = 2 * ARM["5020"], 50.0
    elif "wrist_pitch" in n or "wrist_yaw" in n:
        a, e = ARM["4010"], 5.0
    else:
        a, e = ARM["5020"], 25.0
    return a, a * W * W, 2 * 2.0 * a * W, e


ARMATURE, KP, KD, EFFORT = (np.array(v) for v in zip(*[joint_params(n) for n in ISAAC]))
SCALE = 0.25 * EFFORT / KP  # Unitree mimic action scale (= the Wide residual scale)
DEFAULT_Q = np.array([default_pos(n) for n in ISAAC])
EE = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
TRACK = ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
         "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link",
         "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mat(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, q); return m.reshape(3, 3)


def yaw_mat(q):
    R = quat_mat(q); y = np.arctan2(R[1, 0], R[0, 0]); c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Policy:
    def __init__(self, path):
        if path.endswith(".onnx"):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.inputs = [i.name for i in self.sess.get_inputs()]
            self.kind = "onnx"
        else:
            import torch
            sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
            self.layers = [(sd[f"mlp.{i}.weight"].numpy().astype(np.float64), sd[f"mlp.{i}.bias"].numpy().astype(np.float64))
                           for i in (0, 2, 4, 6)]
            self.kind = "torch"

    def __call__(self, obs, step):
        if self.kind == "onnx":
            feed = {self.inputs[0]: obs[None].astype(np.float32)}
            if len(self.inputs) > 1:  # BeyondMimic export also takes the time step
                feed[self.inputs[1]] = np.array([[step]], dtype=np.float32)
            return self.sess.run(None, feed)[0][0].astype(np.float64)
        x = obs
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < 3:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
        return x


def build_model(xml, dr, sim_dt):
    spec = mujoco.MjSpec.from_file(os.path.expanduser(xml))
    spec.option.timestep = sim_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = spec.compile()
    if not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):  # robot-only xml: add a floor
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], name="s2s_floor")
        model = spec.compile()
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ISAAC]
    qadr, vadr = np.array([model.jnt_qposadr[j] for j in jid]), np.array([model.jnt_dofadr[j] for j in jid])
    model.dof_armature[vadr] = ARMATURE
    model.dof_frictionloss[vadr] = 0.0
    kd = KD * dr["kd"]
    model.dof_damping[vadr] = kd  # damping toward qd = 0 is integrated implicitly; the velocity target is added explicitly
    for g in range(model.ngeom):  # foot spheres get the same friction as the floor (Isaac: random per robot)
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            model.geom_friction[g, 0] = dr["friction"]
    return model, qadr, vadr, kd


def run_once(cfg):
    a = argparse.Namespace(**cfg["args"])
    rng = np.random.default_rng(cfg["seed"])
    # domain randomization, the training ranges (DR v2) stretched by --widen; fixed values override
    dr = dict(kp=np.ones(29), kd=np.ones(29), mass=1.0, friction=1.0, delay=0)
    if cfg["random"]:
        w = a.widen
        dr["kp"], dr["kd"] = rng.uniform(1 - 0.25 * w, 1 + 0.25 * w, (2, 29))
        dr["mass"] = rng.uniform(1 - 0.3 * w, 1 + 0.3 * w, 64)  # per link, cut to the model's body count below
        dr["friction"] = float(rng.uniform(0.3, 2.0))
        dr["delay"] = int(rng.integers(0, 5))
    if a.kp_scale is not None: dr["kp"], dr["kd"] = np.full(29, a.kp_scale), np.full(29, a.kp_scale)
    if a.mass_scale is not None: dr["mass"] = a.mass_scale
    if a.friction is not None: dr["friction"] = a.friction
    if a.delay_substeps is not None: dr["delay"] = a.delay_substeps

    model, qadr, vadr, kd = build_model(a.xml, dr, a.sim_dt)
    m = np.broadcast_to(dr["mass"], (64,))[: model.nbody - 1] if np.ndim(dr["mass"]) else np.full(model.nbody - 1, dr["mass"])
    model.body_mass[1:] *= m
    model.body_inertia[1:] *= m[:, None]
    kp = KP * dr["kp"]
    data, ref = mujoco.MjData(model), mujoco.MjData(model)
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in set(EE + TRACK)}
    torso = bid["torso_link"]

    M = np.load(os.path.expanduser(a.motion))
    fps = float(np.asarray(M["fps"]).reshape(-1)[0])
    decim = int(round(1.0 / fps / a.sim_dt))
    jp, jv = M["joint_pos"].astype(np.float64), M["joint_vel"].astype(np.float64)
    rp, rq = M["body_pos_w"][:, 0].astype(np.float64), M["body_quat_w"][:, 0].astype(np.float64)
    rlv, rav = M["body_lin_vel_w"][:, 0].astype(np.float64), M["body_ang_vel_w"][:, 0].astype(np.float64)
    T = len(jp)
    k0 = int(round(a.start * fps)); k1 = T - 1 if a.end is None else min(T - 1, int(round(a.end * fps)))

    def set_ref(k):
        ref.qpos[:3], ref.qpos[3:7], ref.qpos[qadr] = rp[k], rq[k], jp[k]
        mujoco.mj_kinematics(model, ref)

    # start exactly on the reference (Isaac resets the robot onto the motion frame)
    data.qpos[:3], data.qpos[3:7], data.qpos[qadr] = rp[k0], rq[k0], jp[k0]
    data.qvel[:3] = rlv[k0]
    data.qvel[3:6] = quat_mat(rq[k0]).T @ rav[k0]  # MuJoCo free-joint angular velocity is in the body frame
    data.qvel[vadr] = jv[k0]
    mujoco.mj_forward(model, data)

    pol = Policy(a.checkpoint)
    last = np.zeros(29)
    q_t, qd_t = jp[k0].copy(), np.zeros(29)
    prev_t, prev_qd_t = q_t.copy(), qd_t.copy()
    def aim(cam):  # camera that follows the pelvis, so the dance stays in frame
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        # azimuth relative to the robot's first heading: 0 = in front of it, 90 = its left side
        w, x, y, z = rq[k0]
        yaw0 = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        cam.distance, cam.azimuth, cam.elevation = a.cam_distance, yaw0 + 180.0 + a.cam_azimuth, a.cam_elevation

    viewer = None
    if cfg.get("viewer"):
        from mujoco import viewer as mj_viewer
        state = dict(paused=not a.no_wait)

        def on_key(key):  # Space in the viewer window: start / pause / resume
            if key == 32:
                state["paused"] = not state["paused"]
                print("[sim2sim] paused" if state["paused"] else "[sim2sim] playing")

        viewer = mj_viewer.launch_passive(model, data, key_callback=on_key)
        aim(viewer.cam)
        viewer.sync()
        if state["paused"]:
            print("[sim2sim] viewer open and paused: set up the camera, then press Space in the viewer to start "
                  "(Space again = pause / resume, close the window to quit)")

        def wait_paused():
            while state["paused"] and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
        wait_paused()
        t_wall = time.time()
    rec = None
    if cfg.get("record"):
        import imageio.v2 as imageio  # pip install "imageio[ffmpeg]" in the s2s env
        model.vis.global_.offwidth, model.vis.global_.offheight = a.width, a.height
        renderer = mujoco.Renderer(model, a.height, a.width)
        rcam = mujoco.MjvCamera()
        aim(rcam)
        path = os.path.expanduser(cfg["record"])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rec = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)

        def grab():
            renderer.update_scene(data, camera=rcam)
            rec.append_data(renderer.render())
        grab()
    res = dict(seed=cfg["seed"], dr=dict(kp=float(dr["kp"].mean()), mass=float(np.mean(m)),
                                          friction=dr["friction"], delay_ms=dr["delay"] * a.sim_dt * 1000),
               success=True, t_fail=None, cause=None)
    jerr, berr = [], []
    t_wall = time.time()
    k = k0
    print(f"[sim2sim] playing {(k1 - k0) / fps:.2f} s of {os.path.basename(a.motion)}")
    while k < k1:
        set_ref(k)
        # ---- observation at frame k
        R_robot_t, R_ref_t = data.xquat[torso].copy(), ref.xquat[torso].copy()
        rel = quat_mat(quat_mul(quat_conj(R_robot_t), R_ref_t))
        ang_vel_b = data.qvel[3:6].copy()
        q, qd = data.qpos[qadr], data.qvel[vadr]
        obs = np.concatenate([jp[k], jv[k], rel[:, :2].reshape(-1), ang_vel_b, q - DEFAULT_Q, qd, last])
        act = pol(obs, k)
        last = act
        # ---- action -> PD targets
        prev_t, prev_qd_t = q_t, qd_t
        kn = min(k + 1, T - 1)
        if a.action == "wide":
            q_t = jp[kn] + np.clip(act * SCALE, -1.0, 1.0)
            qd_t = jv[kn]
        else:
            q_t = DEFAULT_Q + act * SCALE
            qd_t = np.zeros(29)
        for s in range(decim):
            tq, tqd = (prev_t, prev_qd_t) if s < dr["delay"] else (q_t, qd_t)
            # Kp (q* - q) + Kd qd*  explicitly; -Kd qd implicitly via dof_damping; clip like Isaac's effort limit
            tau = kp * (tq - data.qpos[qadr]) + kd * tqd
            tau = np.clip(tau - kd * data.qvel[vadr], -EFFORT, EFFORT) + kd * data.qvel[vadr]
            data.qfrc_applied[vadr] = tau
            mujoco.mj_step(model, data)
        if viewer is not None:  # draw once per 20 ms policy step, on a wall clock (no drift from slow frames)
            if state["paused"]:
                wait_paused()
                t_wall = time.time() - (k + 1 - k0) / fps / a.speed
            viewer.sync()
            dt = t_wall + (k + 1 - k0) / fps / a.speed - time.time()
            if dt > 0: time.sleep(dt)
        k += 1
        if rec is not None:
            grab()
        # ---- training terminations against frame k
        set_ref(k)
        cause = None
        if abs(data.xpos[torso, 2] - ref.xpos[torso, 2]) > 0.25: cause = "anchor_pos"
        elif max(abs(data.xpos[bid[b], 2] - ref.xpos[bid[b], 2]) for b in EE) > 0.25: cause = "ee_body_pos"
        else:
            gz_r = quat_mat(data.xquat[torso])[2, 2]; gz_m = quat_mat(ref.xquat[torso])[2, 2]
            if abs(gz_r - gz_m) > 0.8: cause = "anchor_ori"
        # tracking error: joints, and key bodies relative to the torso in each one's heading frame
        jerr.append(np.sqrt(np.mean((data.qpos[qadr] - jp[k]) ** 2)))
        Yr, Ym = yaw_mat(data.xquat[torso]), yaw_mat(ref.xquat[torso])
        pr = np.array([Yr.T @ (data.xpos[bid[b]] - data.xpos[torso]) for b in TRACK])
        pm = np.array([Ym.T @ (ref.xpos[bid[b]] - ref.xpos[torso]) for b in TRACK])
        berr.append(np.mean(np.linalg.norm(pr - pm, axis=1)))
        if cause and res["success"]:
            res.update(success=False, t_fail=round((k - k0) / fps, 2), cause=cause)
            if not a.no_fall_stop:
                break
        if viewer is not None and not viewer.is_running():
            break
    if rec is not None:
        rec.close()
        print(f"[sim2sim] video: {path} ({(k - k0) / fps:.2f} s at {fps:.0f} fps)")
    if viewer is not None:
        if not a.close_at_end and viewer.is_running():
            print("[sim2sim] done: the viewer stays open, close its window to exit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
        viewer.close()
    res["joint_rmse"] = float(np.mean(jerr)) if jerr else None
    res["body_err"] = float(np.mean(berr)) if berr else None
    res["duration"] = round((k1 - k0) / fps, 2)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="rsl-rl model_*.pt (actor MLP) or an exported policy .onnx")
    p.add_argument("--motion", required=True, help="npz from csv_to_npz (50 Hz, Isaac joint order)")
    p.add_argument("--action", choices=["wide", "mimic"], required=True,
                   help="wide = Residual-Wide tasks (q_ref + delta, velocity ff); mimic = MimicAction / Unitree tasks")
    p.add_argument("--xml", default="~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    p.add_argument("--start", type=float, default=0.0, help="clip start [s]")
    p.add_argument("--end", type=float, default=None, help="clip end [s]")
    p.add_argument("--sim_dt", type=float, default=0.005)
    p.add_argument("--viewer", action="store_true", help="watch one nominal run in the MuJoCo viewer")
    p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    p.add_argument("--no_wait", action="store_true", help="viewer: start at once instead of waiting for Space")
    p.add_argument("--close_at_end", action="store_true", help="viewer: close the window when the clip ends")
    p.add_argument("--record", default=None, help="write the nominal run to this .mp4 (offscreen, true speed)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cam_azimuth", type=float, default=0.0,
                   help="camera angle around the robot [deg]: 0 front, 45 front-left, 90 left side")
    p.add_argument("--cam_elevation", type=float, default=-10.0)
    p.add_argument("--cam_distance", type=float, default=3.5)
    p.add_argument("--no_fall_stop", action="store_true", help="keep going after a termination (to watch)")
    p.add_argument("--runs", type=int, default=0, help="Monte Carlo runs with random training DR (0 = nominal only)")
    p.add_argument("--widen", type=float, default=1.0, help="stretch the gains / mass ranges (1.5 = 1.5x wider)")
    p.add_argument("--kp_scale", type=float, default=None, help="fix Kp and Kd scale for every joint")
    p.add_argument("--mass_scale", type=float, default=None, help="fix every link mass scale")
    p.add_argument("--friction", type=float, default=None)
    p.add_argument("--delay_substeps", type=int, default=None, help="actuation delay in 5 ms sub-steps")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="save results to <out>/<name>.json")
    p.add_argument("--out", default="~/projects/g1_pilot/sim2sim/results")
    a = p.parse_args()

    base = dict(args=vars(a))
    nominal = run_once(dict(base, seed=a.seed, random=False, viewer=a.viewer, record=a.record))
    print(f"[sim2sim] {os.path.basename(a.motion)}  {a.action}  {a.checkpoint}")
    print(f"nominal: {'OK' if nominal['success'] else 'FELL at %.2f s (%s)' % (nominal['t_fail'], nominal['cause'])}"
          f"  of {nominal['duration']} s   joint rmse {nominal['joint_rmse']:.3f} rad   body err {100 * nominal['body_err']:.1f} cm")
    out = dict(nominal=nominal, runs=[])
    if a.runs:
        jobs = [dict(base, seed=a.seed + 1 + i, random=True) for i in range(a.runs)]
        with Pool(a.procs) as pool:
            out["runs"] = pool.map(run_once, jobs)
        ok = [r for r in out["runs"] if r["success"]]
        tf = sorted(r["t_fail"] for r in out["runs"] if not r["success"])
        print(f"Monte Carlo (training DR x{a.widen}): {len(ok)}/{a.runs} = {100 * len(ok) / a.runs:.1f}% finished")
        if tf:
            causes = {c: sum(r["cause"] == c for r in out["runs"]) for c in {r["cause"] for r in out["runs"] if r["cause"]}}
            print(f"  falls at [s]: median {np.median(tf):.2f}, min {tf[0]:.2f}, max {tf[-1]:.2f}   causes {causes}")
        if ok:
            print(f"  finished runs: joint rmse {np.mean([r['joint_rmse'] for r in ok]):.3f} rad, "
                  f"body err {100 * np.mean([r['body_err'] for r in ok]):.1f} cm")
    if a.name:
        d = os.path.expanduser(a.out); os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, a.name + ".json"), "w"), indent=1)
        print("saved", os.path.join(d, a.name + ".json"))


if __name__ == "__main__":
    main()
