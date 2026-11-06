"""Run a Unitree deploy package (deploy.yaml + exported/policy.onnx + motion CSV) in MuJoCo the way the C++
deploy stack does (deploy/robots/g1_29dof/src/State_Mimic.cpp), to check it before the real robot.

What is copied from the C++ code (and differs from sim2sim_dance.py, which reads the training npz):
  * MotionLoader_: CSV rows (root pos, quat xyzw, 29 joints in SDK order) at `fps`, frame = round(phase * (N-1))
    with a linear blend, joint velocity = forward difference; `fps` lower than the CSV rate plays it slower
    (that is how a slowed-down dance is set up on the robot: only `fps` in config.yaml changes)
  * motion_anchor_ori_b: torso = pelvis IMU quat * Rz(waist_yaw) Rx(waist_roll) Ry(waist_pitch), for the robot
    and the reference; the reference yaw is aligned to the robot's at the start (init_quat)
  * joint_ids_map between policy (Isaac) order and motor (SDK) order, stiffness / damping per motor from
    deploy.yaml, action = offset + scale * a, ONNX policy, 50 Hz policy / 200 Hz PD
  * --start stand: the robot stands still in the policy's default stance (deploy.yaml default_joint_pos, about
    the stance the Velocity policy holds) with both feet on the floor when the Mimic state starts -- like pressing
    the gamepad button. (FixStand itself is a plain PD pose without balance: in MuJoCo it tips over within ~1.5 s,
    on the robot it is used on the gantry.) --start ref: the robot starts exactly on the first reference frame
    (like Isaac training and sim2sim_dance.py).
  * --intro S: prepend an S-second smooth blend from the default stance to the first dance frame to the CSV.
Domain randomization and the MuJoCo model are the same as sim2sim_dance.py (same seeds -> same DR), so the two
can be compared run by run. A run fails on the training terminations; --no_fall_stop keeps going and the
result also says whether the robot really fell (pelvis < 0.45 m or torso tilt > 60 deg).

  conda run -n unitree_sim_env python deploy/deploy_sim2sim.py --pkg deploy/schuhplattler_v3s                 # nominal
  conda run -n unitree_sim_env python deploy/deploy_sim2sim.py --pkg deploy/schuhplattler_v3s --runs 512      # Monte Carlo
  conda run -n unitree_sim_env python deploy/deploy_sim2sim.py --pkg deploy/schuhplattler_v3s --start stand --intro 2 --fps 25
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim2sim"))
from sim2sim_dance import EE, EFFORT, ISAAC, TRACK, build_model, quat_conj, quat_mat, quat_mul, yaw_mat  # noqa: E402



def axis_quat(axis, a):
    q = np.zeros(4)
    q[0] = np.cos(a / 2)
    q["xyz".index(axis) + 1] = np.sin(a / 2)
    return q


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return axis_quat("z", yaw)


class Motion:
    """State_Mimic::MotionLoader_ (rows: root pos, quat xyzw, 29 joints SDK order)."""

    def __init__(self, rows, fps):
        self.dt = 1.0 / fps
        self.n = len(rows)
        self.duration = self.n * self.dt
        self.pos = rows[:, :3]
        self.quat = rows[:, [6, 3, 4, 5]]  # wxyz
        self.q = rows[:, 7:]
        self.qd = np.vstack([np.diff(self.q, axis=0) / self.dt, (self.q[-1] - self.q[-2])[None] / self.dt])
        self.update(0.0)

    def update(self, t):
        phase = np.clip(t / self.duration, 0.0, 1.0)
        self.i0 = int(np.round(phase * (self.n - 1)))
        self.i1 = min(self.i0 + 1, self.n - 1)
        self.b = np.round((t - self.i0 * self.dt) / self.dt * 1e5) / 1e5

    def joint_pos(self):
        return self.q[self.i0] * (1 - self.b) + self.q[self.i1] * self.b

    def joint_vel(self):
        return self.qd[self.i0] * (1 - self.b) + self.qd[self.i1] * self.b

    def root_quat(self):
        return slerp(self.quat[self.i0], self.quat[self.i1], self.b)


def torso_quat(root_q, q_sdk):
    """C++ torso_quat_w / anchor_quat_w: pelvis quat * waist yaw (z), roll (x), pitch (y); motors 12, 13, 14."""
    q = quat_mul(root_q, axis_quat("z", q_sdk[12]))
    q = quat_mul(q, axis_quat("x", q_sdk[13]))
    return quat_mul(q, axis_quat("y", q_sdk[14]))


def intro_rows(rows, seconds, fps, q_stand):
    """Prepend a quintic blend from the stance q_stand (SDK order; root of frame 0) to the first dance frame."""
    n = int(round(seconds * fps))
    s = np.linspace(0, 1, n, endpoint=False)
    h = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    out = np.repeat(rows[:1], n, axis=0)
    out[:, 7:] = q_stand[None] + h[:, None] * (rows[0, 7:] - q_stand)[None]
    return np.vstack([out, rows])


def run_once(cfg):
    a = argparse.Namespace(**cfg["args"])
    rng = np.random.default_rng(cfg["seed"])
    dr = dict(kp=np.ones(29), kd=np.ones(29), mass=1.0, friction=1.0, delay=0)
    if cfg["random"]:  # the same draws, in the same order, as sim2sim_dance.run_once
        w = a.widen
        dr["kp"], dr["kd"] = rng.uniform(1 - 0.25 * w, 1 + 0.25 * w, (2, 29))
        dr["mass"] = rng.uniform(1 - 0.3 * w, 1 + 0.3 * w, 64)
        dr["friction"] = float(rng.uniform(0.3, 2.0))
        dr["delay"] = int(rng.integers(0, 5))
    if a.friction is not None:
        dr["friction"] = a.friction

    pkg = yaml.safe_load(open(os.path.join(a.pkg, "params", "deploy.yaml")))
    ids = np.array(pkg["joint_ids_map"])  # SDK index of each policy (Isaac) joint
    act = pkg["actions"]["JointPositionAction"]
    offset, scale = np.array(act["offset"]), np.array(act["scale"])
    default = np.array(pkg["default_joint_pos"])
    kp_m, kd_m = np.array(pkg["stiffness"]), np.array(pkg["damping"])  # per motor (SDK order)
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(a.pkg, "exported", "policy.onnx"), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rows = np.loadtxt(os.path.join(a.pkg, "params", a.csv), delimiter=",")
    fps = a.fps or a.csv_fps
    q_stand = np.zeros(29)
    q_stand[ids] = default  # the policy's default stance, motor (SDK) order
    if a.intro > 0:
        rows = intro_rows(rows, a.intro, a.csv_fps, q_stand)
    motion = Motion(rows, fps)
    t_end = (len(rows) - 1) / fps

    model, qadr, vadr, kd_isaac = build_model(a.xml, dr, a.sim_dt)  # qadr / vadr in Isaac order
    m = np.broadcast_to(dr["mass"], (64,))[: model.nbody - 1] if np.ndim(dr["mass"]) else np.full(model.nbody - 1, dr["mass"])
    model.body_mass[1:] *= m
    model.body_inertia[1:] *= m[:, None]
    mq, mv = np.zeros(29, int), np.zeros(29, int)  # motor (SDK) index -> qpos / dof address
    mq[ids], mv[ids] = qadr, vadr
    data = mujoco.MjData(model)
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in set(EE + TRACK)}
    torso = bid["torso_link"]
    decim = int(round(0.02 / a.sim_dt))

    def pd_step(q_des_m, kp, kd, n):
        """motor PD (kd toward 0 velocity) with the DR gain scale; kd part implicit through dof_damping"""
        model.dof_damping[mv] = kd * dr["kd"][np.argsort(ids)] if np.ndim(dr["kd"]) else kd
        for _ in range(n):
            tau = kp * dr["kp"][np.argsort(ids)] * (q_des_m - data.qpos[mq])
            data.qfrc_applied[mv] = tau
            mujoco.mj_step(model, data)

    # ---- start: on the reference (training-like) or standing in FixStand first (gamepad-like)
    motion.update(0.0)
    if a.start == "ref":
        data.qpos[:3], data.qpos[3:7], data.qpos[mq] = motion.pos[0], motion.quat[0], motion.q[0]
        data.qvel[mv] = motion.qd[0]
        mujoco.mj_forward(model, data)
    else:  # standing still in the default stance, soles on the floor
        data.qpos[:3], data.qpos[3:7], data.qpos[mq] = [*motion.pos[0, :2], 1.0], yaw_quat(motion.quat[0]), q_stand
        mujoco.mj_forward(model, data)
        soles = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0 and model.geom_contype[g]
                 and model.body(model.geom_bodyid[g]).name.endswith("ankle_roll_link")]
        data.qpos[2] -= min(data.geom_xpos[g][2] - model.geom_size[g, 0] for g in soles)
        mujoco.mj_forward(model, data)

    # Mimic.enter(): yaw alignment of the reference to the robot
    q_m = data.qpos[mq].copy()
    init_quat = quat_mul(yaw_quat(torso_quat(data.qpos[3:7], q_m)), quat_conj(yaw_quat(motion.root_quat())))
    last = np.zeros(29)
    q_des_m = data.qpos[mq].copy()
    prev_m = q_des_m.copy()
    res = dict(seed=cfg["seed"], success=True, t_fail=None, cause=None, dr=dict(friction=dr["friction"], delay_ms=5 * dr["delay"]))
    low_z, max_tilt, jerr, berr, step = np.inf, 0.0, [], [], 0
    effort_m = np.zeros(29)
    effort_m[ids] = EFFORT
    peak_qd, peak_tau = np.zeros(29), np.zeros(29)
    z_floor = 0.0
    ref = mujoco.MjData(model)
    rec = None
    if cfg.get("record"):  # nominal run to mp4, one frame per 20 ms policy step, camera follows the pelvis
        import imageio.v2 as imageio
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        w, x, y, zq = motion.quat[0]
        cam.distance, cam.elevation = 3.5, -10.0
        cam.azimuth = np.degrees(np.arctan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))) + 180.0
        rec = imageio.get_writer(os.path.expanduser(cfg["record"]), fps=50, codec="libx264", quality=8, macro_block_size=8)

    while step * 0.02 <= t_end:
        if rec is not None:
            renderer.update_scene(data, camera=cam)
            rec.append_data(renderer.render())
        t = step * 0.02
        motion.update(t)
        q_m, qd_m = data.qpos[mq].copy(), data.qvel[mv].copy()
        # ---- observation (deploy.yaml order): motion_command, anchor ori, ang vel, joint pos/vel rel, last action
        ref_q = motion.joint_pos()[ids]
        ref_qd = motion.joint_vel()[ids]
        real = torso_quat(data.qpos[3:7], q_m)
        refq = quat_mul(init_quat, torso_quat(motion.root_quat(), motion.joint_pos()))
        rot = quat_mat(quat_mul(quat_conj(refq), real)).T
        obs = np.concatenate([ref_q, ref_qd, rot[:, :2].reshape(-1), data.qvel[3:6], q_m[ids] - default, qd_m[ids], last])
        action = sess.run(None, {in_name: obs[None].astype(np.float32)})[0][0].astype(np.float64)
        last = action
        prev_m = q_des_m
        q_des_m = np.zeros(29)
        q_des_m[ids] = offset + scale * action
        # 0-20 ms actuation delay (DR), then the new target
        if dr["delay"]:
            pd_step(prev_m, kp_m, kd_m, dr["delay"])
        pd_step(q_des_m, kp_m, kd_m, decim - dr["delay"])
        step += 1
        # how hard the motors work: joint speed, and PD torque against the motor's limit, per motor
        qd_now = np.abs(data.qvel[mv])
        tau_now = np.abs(kp_m * (q_des_m - data.qpos[mq]) - kd_m * data.qvel[mv])
        peak_qd, peak_tau = np.maximum(peak_qd, qd_now), np.maximum(peak_tau, tau_now / effort_m)
        # ---- checks against the reference (yaw/position aligned to where the dance started)
        motion.update(step * 0.02)
        ref.qpos[:3] = motion.pos[motion.i0]
        ref.qpos[3:7] = motion.root_quat()
        ref.qpos[mq] = motion.joint_pos()
        mujoco.mj_kinematics(model, ref)
        low_z = min(low_z, float(data.qpos[2]) - z_floor)
        max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(quat_mat(data.xquat[torso])[2, 2], -1, 1)))))
        cause = None
        if abs(data.xpos[torso, 2] - ref.xpos[torso, 2]) > 0.25:
            cause = "anchor_pos"
        elif max(abs(data.xpos[bid[b], 2] - ref.xpos[bid[b], 2]) for b in EE) > 0.25:
            cause = "ee_body_pos"
        jerr.append(np.sqrt(np.mean((data.qpos[mq] - motion.joint_pos()) ** 2)))
        Yr, Ym = yaw_mat(data.xquat[torso]), yaw_mat(ref.xquat[torso])
        pr = np.array([Yr.T @ (data.xpos[bid[b]] - data.xpos[torso]) for b in TRACK])
        pm = np.array([Ym.T @ (ref.xpos[bid[b]] - ref.xpos[torso]) for b in TRACK])
        berr.append(np.mean(np.linalg.norm(pr - pm, axis=1)))
        if cause and res["success"]:
            res.update(success=False, t_fail=round(t + 0.02, 2), cause=cause)
            if not a.no_fall_stop:
                break
    if rec is not None:
        rec.close()
    res.update(real_fall=bool(low_z < 0.45 or max_tilt > 60), min_pelvis_z=round(low_z, 3), max_tilt_deg=round(max_tilt, 1),
               joint_rmse=float(np.mean(jerr)), body_err=float(np.mean(berr)), duration=round(t_end, 2),
               peak_joint_speed=peak_qd.round(2).tolist(), peak_torque_frac=peak_tau.round(3).tolist())
    return res


SDK_INDEX = {}  # joint name -> motor index, filled from deploy.yaml's joint_ids_map in main()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", required=True, help="deploy package folder (params/deploy.yaml, exported/policy.onnx)")
    p.add_argument("--csv", default="schuhplattler_gmr_v3s_50hz.csv")
    p.add_argument("--csv_fps", type=float, default=50.0, help="frame rate the CSV was written at")
    p.add_argument("--fps", type=float, default=None, help="fps given to the deploy loader (config.yaml); "
                   "lower = slower dance, e.g. 25 = half speed")
    p.add_argument("--start", choices=("ref", "stand"), default="ref")
    p.add_argument("--intro", type=float, default=0.0, help="seconds of stand -> first frame blend before the dance")
    p.add_argument("--xml", default="~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    p.add_argument("--sim_dt", type=float, default=0.005)
    p.add_argument("--runs", type=int, default=0)
    p.add_argument("--widen", type=float, default=1.0)
    p.add_argument("--friction", type=float, default=None)
    p.add_argument("--no_fall_stop", action="store_true")
    p.add_argument("--procs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", default=None, help="write the nominal run to this .mp4")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    base = dict(args=vars(a))
    ids = yaml.safe_load(open(os.path.join(a.pkg, "params", "deploy.yaml")))["joint_ids_map"]
    SDK_INDEX.update({n: ids[i] for i, n in enumerate(ISAAC)})
    nom = run_once(dict(base, seed=a.seed, random=False, record=a.record))
    speed = (a.fps or a.csv_fps) / a.csv_fps
    print(f"[deploy-sim2sim] {a.pkg}  start={a.start} intro={a.intro}s speed x{speed:.2f}")
    print(f"nominal: {'OK' if nom['success'] else 'TERMINATED at %.2f s (%s)' % (nom['t_fail'], nom['cause'])}"
          f"{'' if nom['success'] or not a.no_fall_stop else (', real fall' if nom['real_fall'] else ', stayed upright')}"
          f"   joint rmse {nom['joint_rmse']:.3f} rad   body err {100 * nom['body_err']:.1f} cm")
    sdk = sorted(ISAAC, key=lambda n: SDK_INDEX[n])
    for key, unit, label in (("peak_joint_speed", "rad/s", "fastest joints"), ("peak_torque_frac", "of limit", "hardest-working motors")):
        v = np.array(nom[key])
        top = np.argsort(-v)[:5]
        print(f"  {label}: " + ", ".join(f"{sdk[i].replace('_joint', '')} {v[i]:.2f}" if unit == "rad/s"
                                         else f"{sdk[i].replace('_joint', '')} {100 * v[i]:.0f}%" for i in top)
              + (f" {unit}" if unit == "rad/s" else " of the torque limit"))
    out = dict(nominal=nom, runs=[])
    if a.runs:
        with Pool(a.procs) as pool:
            out["runs"] = pool.map(run_once, [dict(base, seed=a.seed + 1 + i, random=True) for i in range(a.runs)])
        r = out["runs"]
        ok = sum(x["success"] for x in r)
        tf = sorted(x["t_fail"] for x in r if not x["success"])
        line = f"Monte Carlo (training DR x{a.widen}): {ok}/{a.runs} = {100 * ok / a.runs:.1f}% finished"
        if a.no_fall_stop:
            fell = sum(x["real_fall"] for x in r)
            line += f", real falls {fell}/{a.runs} = {100 * fell / a.runs:.1f}%"
        print(line)
        if tf:
            causes = {c: sum(x["cause"] == c for x in r) for c in {x["cause"] for x in r if x["cause"]}}
            print(f"  terminations at [s]: median {np.median(tf):.2f}, min {tf[0]:.2f}, max {tf[-1]:.2f}   {causes}")
    if a.name:
        os.makedirs(a.out, exist_ok=True)
        json.dump(out, open(os.path.join(a.out, a.name + ".json"), "w"), indent=1)
        print("saved", os.path.join(a.out, a.name + ".json"))


if __name__ == "__main__":
    main()
