"""Day 4: evaluate a residual-RL policy the same way the Day 3 PD baseline was measured.

Same setup as day3_pd_baseline.py --mode free: one robot, every randomization off, the motion starts at
frame 0, the task's own terminations decide when it fell. The only difference is who picks the action:
  --checkpoint <model_N.pt | run folder>   the trained policy (q_des = q_ref[k+1] + clip(0.25 a, +-0.5))
  --zero_policy                            a = 0, i.e. q_des = q_ref[k+1]: must reproduce Day 3 free_1
                                           (fell at 1.0 s). Run it once to check the new action term.

Run from the unitree_rl_lab root with env_isaaclab activated:
  python ~/projects/g1_pilot/day4_eval.py --zero_policy --run zero --headless
  python ~/projects/g1_pilot/day4_eval.py --checkpoint logs/rsl_rl/g1_schuhplattler_residual/<run> --run full_1 --headless
Watch at the true speed when the PC is too slow to simulate live (e.g. a training run shares the GPU): run headless
first, then replay its log.npz in the window (no physics, only drawing the recorded robot states):
  python ~/projects/g1_pilot/day4_eval.py --checkpoint ... --run vid_d102 --headless
  python ~/projects/g1_pilot/day4_eval.py --checkpoint ... --run vid_d102 --replay ~/projects/g1_pilot/day4/vid_d102/log.npz
Video at the true speed (one frame per 20 ms env step, however slowly the PC simulates; camera follows the robot):
  python ~/projects/g1_pilot/day4_eval.py --checkpoint ... --run vid --video ~/videos/d102_isaac.mp4 --headless --enable_cameras
Outputs in ~/projects/g1_pilot/day4/<run>/: log.npz, summary.json, summary.txt, tracking.png
(same keys as Day 3, plus residual_* numbers), so tools/day3_compare.py can put both side by side.
"""

import argparse
import glob
import os
import re
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Unitree-G1-29dof-Mimic-Schuhplattler-Residual",
                    help="...-Residual (full dance) or ...-Residual-Stand (first 1.8 s)")
parser.add_argument("--checkpoint", default=None, help="model_N.pt, or a run folder (its newest model_N.pt is used)")
parser.add_argument("--zero_policy", action="store_true", help="no policy, residual = 0 (Day 3 check)")
parser.add_argument("--run", default=None, help="output folder name")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=os.path.expanduser("~/projects/g1_pilot/day4"))
parser.add_argument("--slowmo", type=float, default=1.0, help="with a window: play this many times slower")
parser.add_argument("--no_fall_stop", action="store_true", help="don't stop when it falls (to watch the fall)")
parser.add_argument("--clip", default=None,
                    help="multi-clip tasks (...-Multi2/3): which dance to play, by name or index (default 0)")
parser.add_argument("--stress", default=None,
                    help="fixed hard conditions: comma list of gains,mass,friction,noise,delay or 'all' "
                         "(Kp/Kd x0.75, link mass x1.3, friction 0.5, DR v2 sensor noise, 20 ms latency)")
parser.add_argument("--push", action="store_true",
                    help="keep the task's random pushes on (robustness test; vary --seed for repeats)")
parser.add_argument("--replay", default=None,
                    help="log.npz of an earlier run of this script: draw it at the true speed, no physics (same --task)")
parser.add_argument("--video", default=None,
                    help="write an .mp4 at the true speed (add --enable_cameras with --headless)")
parser.add_argument("--video_size", default="1280x720")
parser.add_argument("--cam_azimuth", type=float, default=0.0,
                    help="video camera angle around the robot [deg]: 0 front, 45 front-left, 90 left side")
parser.add_argument("--cam_distance", type=float, default=3.5)
parser.add_argument("--cam_height", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.zero_policy and not args.checkpoint and not args.replay:
    parser.error("give --checkpoint <model_N.pt or run folder>, or --zero_policy")
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

import json  # noqa: E402,F401
from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import unitree_rl_lab.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def resolve_checkpoint(p):
    p = os.path.expanduser(p)
    if os.path.isdir(p):
        files = glob.glob(os.path.join(p, "model_*.pt"))
        if not files:
            raise SystemExit(f"no model_*.pt in {p}")
        return max(files, key=lambda f: int(re.findall(r"model_(\d+)\.pt", f)[0]))
    if not os.path.isfile(p):
        raise SystemExit(f"checkpoint not found: {p}")
    return p


ckpt = None if (args.zero_policy or args.replay) else resolve_checkpoint(args.checkpoint)
policy_name = "zero" if ckpt is None else os.path.relpath(ckpt, os.path.dirname(os.path.dirname(ckpt)))
out = os.path.join(args.out, args.run or ("zero" if ckpt is None else "eval"))
os.makedirs(out, exist_ok=True)

# ---- config: same as Day 3 free mode -----------------------------------------------------------
cfg = parse_env_cfg(args.task, num_envs=1)
cfg.seed = args.seed
cfg.commands.motion.pose_range = {}
cfg.commands.motion.velocity_range = {}
cfg.commands.motion.joint_position_range = (0.0, 0.0)
for name in (("physics_material", "base_com", "add_joint_default_pos", "actuator_gains", "link_mass")
             + (() if args.push else ("push_robot",))):
    if hasattr(cfg.events, name):
        setattr(cfg.events, name, None)
cfg.observations.policy.enable_corruption = False
# the 30 s training cap would stop clips longer than 30 s (Gangnam 32.3 s) early; the loop below stops at the clip's last frame
cfg.episode_length_s = 3600.0
if args.video:
    cfg.viewer.resolution = tuple(int(v) for v in args.video_size.split("x"))

STRESS_ALL = ("gains", "mass", "friction", "noise", "delay")
stress = set(STRESS_ALL if args.stress == "all" else (args.stress.split(",") if args.stress else ()))
assert stress <= set(STRESS_ALL), f"unknown --stress items: {stress - set(STRESS_ALL)}"
act_cfg = cfg.actions.JointPositionAction
if hasattr(act_cfg, "delay_substeps_range"):
    act_cfg.delay_substeps_range = (4, 4) if "delay" in stress else (0, 0)
elif "delay" in stress:
    raise SystemExit("--stress delay needs the updated residual action (residual_action.py)")
if stress - {"delay"}:
    import isaaclab.envs.mdp as base_mdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

    if "gains" in stress:
        cfg.events.stress_gains = EventTerm(func=base_mdp.randomize_actuator_gains, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.75, 0.75),
            "damping_distribution_params": (0.75, 0.75), "operation": "scale", "distribution": "uniform"})
    if "mass" in stress:
        cfg.events.stress_mass = EventTerm(func=base_mdp.randomize_rigid_body_mass, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (1.3, 1.3),
            "operation": "scale", "distribution": "uniform", "recompute_inertia": True})
    if "friction" in stress:
        cfg.events.stress_friction = EventTerm(func=base_mdp.randomize_rigid_body_material, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.5, 0.5),
            "dynamic_friction_range": (0.5, 0.5), "restitution_range": (0.0, 0.0), "num_buckets": 1})
    if "noise" in stress:  # the same (DR v2) levels for every policy, so v1 and v2 face the same sensors
        pol = cfg.observations.policy
        pol.enable_corruption = True
        pol.motion_anchor_ori_b.noise = Unoise(n_min=-0.05, n_max=0.05)
        pol.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        pol.joint_pos_rel.noise = Unoise(n_min=-0.05, n_max=0.05)
        pol.joint_vel_rel.noise = Unoise(n_min=-1.0, n_max=1.0)
if args.no_fall_stop:
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        if hasattr(cfg.terminations, name):
            setattr(cfg.terminations, name, None)
motion_file = cfg.commands.motion.motion_file

ee_err = {}
if getattr(cfg.terminations, "ee_body_pos", None) is not None:
    _ee_func = cfg.terminations.ee_body_pos.func

    def _ee_func_logged(env, command_name, threshold, body_names=None):
        c = env.command_manager.get_term(command_name)
        idx = [i for i, nm in enumerate(c.cfg.body_names) if body_names is None or nm in body_names]
        z = (c.body_pos_relative_w[0, idx, 2] - c.robot_body_pos_w[0, idx, 2]).abs().cpu().numpy()
        ee_err.update(names=[c.cfg.body_names[i] for i in idx], z=z, threshold=threshold)
        return _ee_func(env, command_name, threshold, body_names)

    cfg.terminations.ee_body_pos.func = _ee_func_logged

env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
u = env.unwrapped
robot = u.scene["robot"]
cmd = u.command_manager.get_term("motion")
term = u.action_manager.get_term("JointPositionAction")
assert hasattr(term, "delta"), "this task's action is not the residual action (wrong --task?)"
res_lim_j = (term.residual_limit if hasattr(term, "residual_limit")
             else torch.full((term.action_dim,), float(term.cfg.residual_limit), device=u.device)).cpu().numpy()
res_lim = float(res_lim_j.max())


clip, clip_name, start = 0, None, 0
if hasattr(cmd, "set_clip_start"):  # several dances in one task
    # name, name prefix (schuhplattler_gmr matches schuhplattler_gmr_grounded) or index
    hits = [i for i, n in enumerate(cmd.clip_names) if args.clip is not None and n.startswith(str(args.clip))]
    clip = hits[0] if hits else int(args.clip or 0)
    clip_name = cmd.clip_names[clip]
    start = int(cmd.clip_start[clip])
    motion_file = cfg.commands.motion.motion_files[clip]
elif args.clip is not None:
    print("[day4] --clip ignored: this task has one motion")


def _start_at_zero(env_ids):
    if hasattr(cmd, "set_clip_start"):
        cmd.set_clip_start(env_ids, clip, 0)
    else:
        cmd.time_steps[env_ids] = 0


cmd._adaptive_sampling = _start_at_zero  # must be patched before the wrapper resets the env

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # resets the env (frame 0)

policy = None
if ckpt is not None:
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=u.device)
    print(f"[day4] policy: {ckpt}")
else:
    print("[day4] policy: zero residual (should match Day 3 free_1)")

names = robot.joint_names
T = cmd.motion.clip_lengths[clip] if clip_name is not None else cmd.motion.time_step_total
fps = float(np.asarray(cmd.motion.fps).reshape(-1)[0])
step_dt = u.step_dt
print(f"[day4] task={args.task} motion={motion_file} clip={clip_name} push={args.push} stress={sorted(stress) or None} seed={args.seed}")
print(f"[day4] motion frames={T} @ {fps:.0f} Hz, env step = {step_dt * 1000:.0f} ms, residual limit {res_lim} rad")

kp = robot.data.joint_stiffness[0].cpu().numpy()
kd = robot.data.joint_damping[0].cpu().numpy()
lim = robot.data.soft_joint_pos_limits[0].cpu().numpy()
ankles = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

assert int(cmd.time_steps[0]) == start, "start frame is not 0"
# same fix as Day 3: the command is not updated by reset, so fill frame 0 before the first step
cmd.time_steps[:] = start - 1
cmd._update_command()
assert int(cmd.time_steps[0]) == start, "start frame is not 0 after command update"
obs = venv.get_observations()
if isinstance(obs, tuple):  # rsl-rl 2.3 returns (obs, extras)
    obs = obs[0]

video = None
if args.video:
    import subprocess

    try:
        import imageio_ffmpeg  # comes with moviepy / imageio; has its own ffmpeg binary
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    vpath = os.path.expanduser(args.video)
    os.makedirs(os.path.dirname(vpath) or ".", exist_ok=True)
    vw, vh = cfg.viewer.resolution
    video = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{vw}x{vh}", "-r", f"{1.0 / step_dt:g}", "-i", "-", "-c:v", "libx264",
                              "-pix_fmt", "yuv420p", "-crf", "18", vpath], stdin=subprocess.PIPE)
    q0 = robot.data.root_quat_w[0].cpu().numpy()  # w x y z: the camera angle is relative to the first heading
    yaw0 = float(np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
    n_frames = 0


def grab():
    """Point the camera at the robot (it follows the pelvis on the ground plane) and write one frame."""
    global n_frames
    p = robot.data.root_pos_w[0].cpu().numpy()
    a = yaw0 + np.radians(args.cam_azimuth)
    eye = (p[0] + args.cam_distance * np.cos(a), p[1] + args.cam_distance * np.sin(a), args.cam_height)
    u.viewport_camera_controller.update_view_location(eye=eye, lookat=(p[0], p[1], 0.75))
    img = u.render()
    if img is not None:
        video.stdin.write(np.ascontiguousarray(img[..., :3], dtype=np.uint8).tobytes())
        n_frames += 1


if args.replay:
    # kinematic replay: put the robot into each recorded state and only render, so it runs at the true speed
    R = np.load(os.path.expanduser(args.replay))
    n = len(R["t"])
    print(f"[day4] replay {args.replay}: {n} frames = {R['t'][-1]:.2f} s. Close the window (or Ctrl+C) to exit.")
    dev = u.device
    zero_v = torch.zeros(1, 6, device=dev)
    while app.is_running():
        t0 = time.time()
        shown = 0
        while app.is_running():
            # pick the frame from the wall clock: if drawing is slower than 50 fps, frames are skipped
            # (shown at a lower frame rate) but the motion keeps its true speed
            i = int((time.time() - t0) / (step_dt * args.slowmo))
            if i >= n:
                break
            shown += 1
            pose = torch.tensor(np.concatenate([R["root"][i], R["quat"][i]]), dtype=torch.float32, device=dev)[None]
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(zero_v)
            q = torch.tensor(R["q"][i], dtype=torch.float32, device=dev)[None]
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            u.sim.forward()
            u.sim.render()
            time.sleep(max(0.0, t0 + (i + 1) * step_dt * args.slowmo - time.time()))
        took = time.time() - t0
        print(f"[day4] replay took {took:.2f} s for {R['t'][-1]:.2f} s of motion, drew {shown}/{n} frames "
              f"({shown / max(took, 1e-6):.0f} fps); looping (close window to exit)")
        th = time.time()
        while app.is_running() and time.time() - th < 1.0:  # hold the last pose for 1 s before looping
            u.sim.render()
    env.close()
    app.close()
    raise SystemExit(0)

log = {k: [] for k in ("t", "q_ref", "q", "qd", "tau", "root_ref", "root", "quat_ref", "quat", "ankle_pos", "delta")}


def record(t):
    d = robot.data
    log["t"].append(t)
    log["q_ref"].append(cmd.joint_pos[0].cpu().numpy())
    log["q"].append(d.joint_pos[0].cpu().numpy())
    log["qd"].append(d.joint_vel[0].cpu().numpy())
    log["tau"].append(d.applied_torque[0].cpu().numpy())
    log["root_ref"].append(cmd.body_pos_w[0, 0].cpu().numpy())
    log["root"].append(d.root_pos_w[0].cpu().numpy())
    log["quat_ref"].append(cmd.body_quat_w[0, 0].cpu().numpy())
    log["quat"].append(d.root_quat_w[0].cpu().numpy())
    log["ankle_pos"].append(d.body_pos_w[0, ankles].cpu().numpy())
    log["delta"].append(term.delta[0].cpu().numpy())


record(0.0)
if video:
    grab()
reason = "completed"
zero = torch.zeros(1, term.action_dim, device=u.device)
for k in range(T - 1):
    if not app.is_running():
        reason = "app closed"
        break
    t_wall = time.time()
    with torch.inference_mode():
        action = zero if policy is None else policy(obs)
        obs, _, dones, _ = venv.step(action)
    if video:
        grab()
    elif not args.headless:
        time.sleep(max(0.0, step_dt * args.slowmo - (time.time() - t_wall)))
    if bool(dones[0]):
        tm = u.termination_manager
        hit = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
        reason = "terminated: " + ",".join(hit) if hit else "terminated"
        print(f"[day4] stopped at t={(k + 1) * step_dt:.2f}s ({reason})")
        if "ee_body_pos" in hit and ee_err:
            print("[day4] ee z error at termination:",
                  {nm: round(float(z), 3) for nm, z in zip(ee_err["names"], ee_err["z"])},
                  f"(threshold {ee_err['threshold']})")
        break
    record((k + 1) * step_dt)
    if k % 50 == 0:
        e = np.abs(log["q"][-1] - log["q_ref"][-1])
        print(f"[day4] t={log['t'][-1]:5.2f}s  root_z={log['root'][-1][2]:.3f} (ref {log['root_ref'][-1][2]:.3f})"
              f"  |e|mean={e.mean():.3f} max={e.max():.3f} rad  |delta|mean={np.abs(log['delta'][-1]).mean():.3f}")

if video:
    video.stdin.close()
    video.wait()
    print(f"[day4] video: {vpath} ({n_frames} frames = {n_frames * step_dt:.2f} s at {1.0 / step_dt:g} fps)")

L = {k: np.array(v) for k, v in log.items()}
np.savez(os.path.join(out, "log.npz"), joint_names=np.array(names), kp=kp, kd=kd, **L)

# ---- metrics ----------------------------------------------------------------------------------
n = len(L["t"])
s = {"mode": "stress" if stress else ("push" if args.push else "free"), "stress": sorted(stress), "policy": policy_name, "task": args.task, "clip": clip_name,
     "motion": motion_file, "seed": args.seed,
     "result": reason,
     "tracked_s": round((n - 1) * step_dt, 2), "motion_s": round((T - 1) / fps, 2),
     "completed_pct": round(100 * (n - 1) / (T - 1), 1)}  # n - 1: sample 0 is the t=0 state
if reason.startswith("terminated") and ee_err:
    s["ee_z_err_at_end_m"] = {nm: round(float(z), 4) for nm, z in zip(ee_err["names"], ee_err["z"])}
if n > 0:
    e = L["q"] - L["q_ref"]
    rmse = np.sqrt((e ** 2).mean(0))
    s["joint_rmse_mean_rad"] = round(float(rmse.mean()), 4)
    s["joint_abs_err_max_rad"] = round(float(np.abs(e).max()), 4)
    s["worst_joints_rmse"] = {names[i]: round(float(rmse[i]), 4) for i in np.argsort(-rmse)[:5]}
    s["per_joint_rmse"] = {nm: round(float(r), 4) for nm, r in zip(names, rmse)}
    s["root_pos_err_mean_m"] = round(float(np.linalg.norm(L["root"] - L["root_ref"], axis=1).mean()), 4)
    s["root_z_err_mean_m"] = round(float(np.abs(L["root"][:, 2] - L["root_ref"][:, 2]).mean()), 4)
    viol = (L["q"] < lim[:, 0]) | (L["q"] > lim[:, 1])
    s["joint_limit_violation_frames"] = int(viol.any(1).sum())
    s["max_torque_Nm"] = round(float(np.abs(L["tau"]).max()), 1)
    az = L["ankle_pos"][:, :, 2]
    on = az < az.min() + 0.03  # ankle within 3 cm of its lowest height = in contact
    v = np.linalg.norm(np.diff(L["ankle_pos"][:, :, :2], axis=0), axis=-1) / step_dt
    c = on[1:] & on[:-1]
    s["foot_slide_mean_cm_s"] = round(float(v[c].mean() * 100), 1) if c.any() else None
    dl = np.abs(L["delta"])
    s["residual_abs_mean_rad"] = round(float(dl.mean()), 4)
    s["residual_abs_max_rad"] = round(float(dl.max()), 4)
    s["residual_at_limit_pct"] = round(float(100 * (dl > 0.98 * res_lim_j).mean()), 1)
    groups = {"legs": [i for i, nm in enumerate(names) if any(p in nm for p in ("hip", "knee", "ankle"))],
              "waist": [i for i, nm in enumerate(names) if "waist" in nm],
              "arms": [i for i, nm in enumerate(names) if any(p in nm for p in ("shoulder", "elbow", "wrist"))]}
    s["residual_abs_mean_by_group_rad"] = {g: round(float(dl[:, ix].mean()), 4) for g, ix in groups.items()}
    s["joint_rmse_by_group_rad"] = {g: round(float(rmse[ix].mean()), 4) for g, ix in groups.items()}
json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=2)
with open(os.path.join(out, "summary.txt"), "w") as f:
    for key, val in s.items():
        if key not in ("per_joint_rmse",):
            f.write(f"{key:28s} {val}\n")
print(open(os.path.join(out, "summary.txt")).read())

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if n > 0:
        show = ["left_hip_pitch_joint", "left_knee_joint", "right_knee_joint",
                "left_ankle_pitch_joint", "left_shoulder_pitch_joint", "right_elbow_joint"]
        fig, ax = plt.subplots(4, 2, figsize=(12, 11))
        for a, nm in zip(ax.flat[:6], show):
            i = names.index(nm)
            a.plot(L["t"], L["q_ref"][:, i], "k--", lw=1, label="q_ref")
            a.plot(L["t"], L["q"][:, i], lw=1.2, label="q actual")
            a.set_title(nm, fontsize=9)
        ax.flat[0].legend(fontsize=8)
        ax[3, 0].plot(L["t"], L["root_ref"][:, 2], "k--", lw=1, label="ref")
        ax[3, 0].plot(L["t"], L["root"][:, 2], lw=1.2, label="actual")
        ax[3, 0].set_title("pelvis z [m]", fontsize=9)
        ax[3, 0].legend(fontsize=8)
        order = np.argsort(-rmse)
        ax[3, 1].barh([names[i].replace("_joint", "") for i in order[:12]][::-1], rmse[order[:12]][::-1])
        ax[3, 1].set_title("per-joint RMSE [rad] (worst 12)", fontsize=9)
        ax[3, 1].tick_params(labelsize=7)
        fig.suptitle(f"residual policy={policy_name}: {reason}, {s['tracked_s']}/{s['motion_s']} s")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "tracking.png"), dpi=100)
        print("[day4] plot:", os.path.join(out, "tracking.png"))
except ImportError:
    print("[day4] matplotlib not installed: skipped plot")

if not args.headless:
    print("[day4] done. The window stays open: close it (or Ctrl+C) to exit.")
    while app.is_running():
        u.sim.render()
env.close()
app.close()
